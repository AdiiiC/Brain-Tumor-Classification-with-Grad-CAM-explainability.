"""
Tests for the ONNX serving backend and backend selection.

A real .onnx graph is built on the fly when onnxruntime is installed, so these run in CI
without TensorFlow and without the trained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.onnx_service import ONNX_AVAILABLE, OnnxModelService
from api.preprocessing import preprocess_image, softmax

onnx_only = pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")


def _build_onnx(tmp_path, with_logits: bool):
    """Minimal graph: input → (probabilities) or (probabilities, logits)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Scaled so the summed activations land in a realistic logit range rather than
    # saturating the softmax, which would mask the effect of temperature.
    weights = numpy_helper.from_array(
        np.tile(np.array([[3.0, 0.5, 0.1, 0.2]], dtype=np.float32), (240 * 240 * 3, 1)) * 1e-6,
        name="W",
    )
    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    outputs = [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 4])]
    if with_logits:
        outputs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 4]))

    graph = helper.make_graph(
        nodes,
        "brainscan",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 240, 240, 3])],
        outputs,
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / ("dual.onnx" if with_logits else "probs.onnx")
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def onnx_dual(tmp_path):
    return OnnxModelService(_build_onnx(tmp_path, with_logits=True))


@pytest.fixture
def onnx_probs_only(tmp_path):
    return OnnxModelService(_build_onnx(tmp_path, with_logits=False))


class TestLoading:
    def test_missing_file_is_not_loaded(self, tmp_path):
        service = OnnxModelService(str(tmp_path / "absent.onnx"))
        assert service.is_loaded is False
        assert service.model_type == "none"

    def test_missing_model_does_not_raise_on_construction(self, tmp_path):
        OnnxModelService(str(tmp_path / "absent.onnx"))

    @onnx_only
    def test_loads_existing_graph(self, onnx_dual):
        assert onnx_dual.is_loaded
        assert onnx_dual.model_type == "onnx"

    @onnx_only
    def test_detects_logit_output(self, onnx_dual):
        assert onnx_dual.has_logits is True

    @onnx_only
    def test_handles_probability_only_graph(self, onnx_probs_only):
        assert onnx_probs_only.is_loaded
        assert onnx_probs_only.has_logits is False


@onnx_only
class TestInference:
    def test_predict_returns_distribution(self, onnx_dual, png_bytes):
        probs = onnx_dual.predict(onnx_dual.preprocess_image(png_bytes))
        assert probs.shape == (4,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_logits_are_not_probabilities(self, onnx_dual, png_bytes):
        img = onnx_dual.preprocess_image(png_bytes)
        assert onnx_dual.predict_logits(img).sum() != pytest.approx(1.0, abs=1e-3)

    def test_softmax_of_logits_matches_probabilities(self, onnx_dual, png_bytes):
        img = onnx_dual.preprocess_image(png_bytes)
        np.testing.assert_allclose(
            softmax(onnx_dual.predict_logits(img)), onnx_dual.predict(img), atol=1e-5
        )

    def test_calibration_applies_temperature_to_logits(self, onnx_dual, png_bytes):
        img = onnx_dual.preprocess_image(png_bytes)
        logits = onnx_dual.predict_logits(img)
        onnx_dual.set_calibration_temperature(2.5)
        np.testing.assert_allclose(
            onnx_dual.predict_calibrated(img), softmax(logits, 2.5), atol=1e-6
        )

    def test_uncalibrated_matches_raw_probabilities(self, onnx_dual, png_bytes):
        img = onnx_dual.preprocess_image(png_bytes)
        onnx_dual.set_calibration_temperature(1.0)
        np.testing.assert_allclose(
            onnx_dual.predict_calibrated(img), onnx_dual.predict(img), atol=1e-5
        )

    def test_probability_only_graph_falls_back_to_log_probs(self, onnx_probs_only, png_bytes):
        img = onnx_probs_only.preprocess_image(png_bytes)
        assert np.all(onnx_probs_only.predict_logits(img) <= 0)

    def test_tta_returns_valid_distribution(self, onnx_dual, png_bytes):
        probs = onnx_dual.predict_with_tta(onnx_dual.preprocess_image(png_bytes))
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)

    def test_uncertainty_is_zero_without_dropout(self, onnx_dual, png_bytes):
        _, std = onnx_dual.predict_with_uncertainty(onnx_dual.preprocess_image(png_bytes))
        assert np.all(std == 0)

    def test_features_unavailable_for_mahalanobis(self, onnx_dual, png_bytes):
        assert onnx_dual.extract_features(onnx_dual.preprocess_image(png_bytes)) is None

    def test_gradcam_is_refused(self, onnx_dual, png_bytes):
        with pytest.raises(RuntimeError, match="Keras backend"):
            onnx_dual.gradcam_plus_plus(onnx_dual.preprocess_image(png_bytes))


class TestPreprocessingParity:
    def test_output_shape_and_dtype(self, png_bytes):
        img = preprocess_image(png_bytes)
        assert img.shape == (240, 240, 3)
        assert img.dtype == np.float32

    def test_rejects_undecodable_bytes(self):
        with pytest.raises(ValueError, match="Could not decode"):
            preprocess_image(b"definitely not an image")

    @onnx_only
    def test_onnx_uses_identical_preprocessing(self, onnx_dual, png_bytes):
        np.testing.assert_array_equal(
            onnx_dual.preprocess_image(png_bytes), preprocess_image(png_bytes)
        )


class TestSoftmaxHelper:
    def test_sums_to_one(self):
        assert softmax(np.array([3.0, 1.0, 0.2, 0.1])).sum() == pytest.approx(1.0)

    def test_is_stable_for_large_logits(self):
        assert np.isfinite(softmax(np.array([1000.0, 999.0, 1.0, 0.0]))).all()

    def test_higher_temperature_reduces_peak(self):
        logits = np.array([5.0, 1.0, 0.5, 0.2])
        assert softmax(logits, 4.0).max() < softmax(logits, 1.0).max()

    def test_preserves_ranking(self):
        logits = np.array([1.0, 5.0, 0.5, 0.2])
        assert softmax(logits, 2.5).argmax() == logits.argmax()


class TestBackendSelection:
    def test_auto_prefers_keras_when_available(self, monkeypatch):
        from api import main

        monkeypatch.setattr(main, "MODEL_BACKEND", "auto")

        class Loaded:
            is_loaded = True
            model_type = "keras"

        monkeypatch.setattr(main, "ModelService", lambda **kwargs: Loaded())
        assert main.build_model_service().model_type == "keras"

    def test_auto_falls_back_to_onnx(self, monkeypatch):
        from api import main

        monkeypatch.setattr(main, "MODEL_BACKEND", "auto")

        class NotLoaded:
            is_loaded = False
            model_type = "none"

        class OnnxLoaded:
            is_loaded = True
            model_type = "onnx"

        monkeypatch.setattr(main, "ModelService", lambda **kwargs: NotLoaded())
        monkeypatch.setattr(main, "OnnxModelService", lambda path: OnnxLoaded())
        assert main.build_model_service().model_type == "onnx"

    def test_explicit_onnx_skips_keras(self, monkeypatch):
        from api import main

        monkeypatch.setattr(main, "MODEL_BACKEND", "onnx")
        monkeypatch.setattr(
            main, "ModelService", lambda **kwargs: pytest.fail("Keras must not be constructed")
        )
        monkeypatch.setattr(main, "OnnxModelService", lambda path: "onnx-service")
        assert main.build_model_service() == "onnx-service"


class TestExplainabilityGating:
    def test_gradcam_rejected_on_onnx_backend(self, client, monkeypatch, upload):
        from api import main

        class OnnxStub:
            is_loaded = True
            model_type = "onnx"

        monkeypatch.setattr(main.svc, "model", OnnxStub())
        response = client.post("/explain/gradcam", files=upload())
        assert response.status_code == 400
        assert "Keras backend" in response.json()["detail"]
