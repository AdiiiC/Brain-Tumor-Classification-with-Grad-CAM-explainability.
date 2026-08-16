"""
End-to-end flow with a stubbed model.

Exercises inference → OOD gate → persistence → timeline → PDF report without
requiring TensorFlow or the trained weights.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from api.model_service import CLASS_NAMES


class FakeModel:
    """Stands in for the Keras model handed to the patch detector."""

    def predict(self, batch, verbose=0):
        return np.tile(np.array([[0.05, 0.05, 0.85, 0.05]], dtype=np.float32), (len(batch), 1))


class FakeModelService:
    """Deterministic replacement for ModelService."""

    def __init__(self, probs=(0.92, 0.04, 0.03, 0.01), logits=(9.0, 1.0, 0.5, 0.2)):
        self.model_type = "keras"
        self.model = FakeModel()
        self.is_loaded = True
        self._probs = np.array(probs, dtype=np.float32)
        self._logits = np.array(logits, dtype=np.float32)

    def set_calibration_temperature(self, temp):
        self._temp = temp

    def preprocess_image(self, image_bytes):
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return cv2.resize(img, (240, 240)).astype(np.float32)

    def predict(self, img):
        return self._probs

    def predict_calibrated(self, img):
        return self._probs

    def predict_with_tta(self, img, n_augments=10):
        return self._probs

    def predict_logits(self, img):
        return self._logits

    def extract_features(self, img):
        return np.ones(8, dtype=np.float32)

    def predict_with_uncertainty(self, img, n_iter=30):
        return self._probs, np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)

    def gradcam_plus_plus(self, img):
        cam = np.zeros((240, 240), dtype=np.float32)
        cv2.circle(cam, (120, 120), 35, 1.0, -1)
        heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = heatmap.copy()
        return heatmap, overlay, int(np.argmax(self._probs))

    @staticmethod
    def encode_image(img_rgb):
        _, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")


@pytest.fixture
def stub_model(client, monkeypatch):
    """Swap the real service for the stub after the app has started."""
    from api import main

    fake = FakeModelService()
    monkeypatch.setattr(main.svc, "model", fake)
    return client


class TestPredictFlow:
    def test_predict_returns_class(self, stub_model, upload):
        body = stub_model.post("/predict", files=upload()).json()
        assert body["predicted_class"] in set(CLASS_NAMES.values())

    def test_predict_includes_build_metadata(self, stub_model, upload):
        body = stub_model.post("/predict", files=upload()).json()
        assert body["model_version"]
        assert body["git_sha"] == "testsha00000"

    def test_predict_includes_ood_check(self, stub_model, upload):
        body = stub_model.post("/predict", files=upload()).json()
        assert body["out_of_distribution"]["method"] == "energy"

    def test_predict_persists_study(self, stub_model, upload):
        study_id = stub_model.post("/predict", files=upload()).json()["study_id"]
        assert study_id
        assert stub_model.get(f"/studies/{study_id}").status_code == 200

    def test_probabilities_cover_all_classes(self, stub_model, upload):
        body = stub_model.post("/predict", files=upload()).json()
        assert set(body["probabilities"]) == set(CLASS_NAMES.values())


class TestOODGate:
    def test_ood_input_is_flagged(self, client, monkeypatch, upload):
        from api import main

        # Flat logits produce high energy, which the default threshold rejects.
        monkeypatch.setattr(main.svc, "model", FakeModelService(logits=(0.01, 0.01, 0.01, 0.01)))

        body = client.post("/analyze", files=upload()).json()
        assert body["out_of_distribution"]["is_out_of_distribution"] is True
        assert body["clinical"]["confidence_level"] == "rejected"
        assert body["clinical"]["flagged_for_review"] is True

    def test_ood_recorded_on_study(self, client, monkeypatch, upload):
        from api import main

        monkeypatch.setattr(main.svc, "model", FakeModelService(logits=(0.01, 0.01, 0.01, 0.01)))
        study_id = client.post("/analyze", files=upload()).json()["study_id"]
        assert client.get(f"/studies/{study_id}").json()["is_ood"] is True

    def test_ood_counted_in_metrics(self, client, monkeypatch, upload):
        from api import main

        monkeypatch.setattr(main.svc, "model", FakeModelService(logits=(0.01, 0.01, 0.01, 0.01)))
        client.post("/analyze", files=upload())
        assert "brainscan_ood_rejections_total 1" in client.get("/metrics").text

    def test_in_distribution_not_flagged(self, stub_model, upload):
        body = stub_model.post("/analyze", files=upload()).json()
        assert body["out_of_distribution"]["is_out_of_distribution"] is False


class TestAnalyzeFlow:
    def test_returns_gradcam(self, stub_model, upload):
        body = stub_model.post("/analyze", files=upload()).json()
        assert body["explainability"]["gradcam_overlay"]

    def test_high_confidence_not_flagged(self, stub_model, upload):
        body = stub_model.post("/analyze", files=upload()).json()
        assert body["clinical"]["confidence_level"] == "high"
        assert body["clinical"]["flagged_for_review"] is False

    def test_prediction_counted_in_metrics(self, stub_model, upload):
        stub_model.post("/analyze", files=upload())
        assert "brainscan_predictions_total" in stub_model.get("/metrics").text


class TestSegmentFlow:
    def test_returns_measurements(self, stub_model, upload):
        body = stub_model.post("/segment", files=upload()).json()
        assert body["measurements"]["volume_mm3"] > 0
        assert body["approximate"] is True

    def test_persists_volume(self, stub_model, upload):
        study_id = stub_model.post("/segment?patient_id=PT-SEG", files=upload()).json()["study_id"]
        assert stub_model.get(f"/studies/{study_id}").json()["tumor_volume_mm3"] > 0

    def test_explicit_spacing_scales_volume(self, stub_model, upload):
        small = stub_model.post("/segment?pixel_spacing=0.5", files=upload()).json()
        large = stub_model.post("/segment?pixel_spacing=1.0", files=upload()).json()
        assert large["measurements"]["area_mm2"] > small["measurements"]["area_mm2"]

    def test_spacing_flagged_as_estimated(self, stub_model, upload):
        body = stub_model.post("/segment", files=upload()).json()
        assert body["spacing_estimated"] is True


class TestFullPatientJourney:
    def test_segment_twice_then_track_and_report(self, stub_model, upload):
        first = stub_model.post("/segment?patient_id=PT-JOURNEY", files=upload()).json()
        second = stub_model.post("/segment?patient_id=PT-JOURNEY", files=upload()).json()
        assert first["study_id"] != second["study_id"]

        timeline = stub_model.get("/patients/PT-JOURNEY/timeline").json()
        assert timeline["num_scans"] == 2

        review = stub_model.post(
            f"/studies/{second['study_id']}/feedback",
            json={"corrected_class": "Meningioma", "reviewer": "Dr. Rao"},
        )
        assert review.status_code == 200

        report = stub_model.get(f"/studies/{second['study_id']}/report")
        assert report.content.startswith(b"%PDF")

    def test_report_embeds_gradcam(self, stub_model, upload):
        from api.database import Study, session_scope

        study_id = stub_model.post("/analyze", files=upload()).json()["study_id"]

        # The overlay is intentionally absent from StudySummary, so read the row.
        with session_scope() as session:
            assert session.get(Study, study_id).gradcam_overlay

        report = stub_model.get(f"/studies/{study_id}/report")
        assert report.status_code == 200
        assert report.headers["content-type"] == "application/pdf"
        assert report.content.startswith(b"%PDF")
        assert b"/Image" in report.content


class TestBatchFlow:
    def test_processes_mixed_batch(self, stub_model, png_bytes):
        import io

        files = [
            ("files", ("good.png", io.BytesIO(png_bytes), "image/png")),
            ("files", ("bad.png", io.BytesIO(b"not an image"), "image/png")),
        ]
        body = stub_model.post("/predict/batch", files=files).json()
        assert len(body) == 2
        assert body[0]["result"]["predicted_class"] != "Error"
        assert body[1]["result"]["predicted_class"] == "Error"
