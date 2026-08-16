"""Endpoint behaviour: contracts, error handling, observability."""

from __future__ import annotations

import io

CLASSES = {"Glioma", "Meningioma", "No Tumor", "Pituitary"}


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_reports_build_metadata(self, client):
        body = client.get("/health").json()
        assert body["api_version"]
        assert body["model_version"]
        assert body["git_sha"] == "testsha00000"

    def test_lists_all_classes(self, client):
        assert set(client.get("/health").json()["classes"]) == CLASSES

    def test_calibration_defaults_to_documented_temperature(self, client):
        body = client.get("/health").json()
        assert body["calibration_temperature"] == 1.5
        assert body["calibration_enabled"] is True

    def test_reports_model_missing(self, client):
        assert client.get("/health").json()["model_loaded"] is False


class TestMetrics:
    def test_exposes_prometheus_text(self, client):
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "brainscan_requests_total" in response.text

    def test_counts_requests(self, client):
        client.get("/health")
        assert 'path="/health"' in client.get("/metrics").text

    def test_request_id_header_present(self, client):
        assert client.get("/health").headers.get("X-Request-ID")

    def test_request_id_echoed(self, client):
        response = client.get("/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"


class TestUploadRejection:
    def test_rejects_non_image_payload(self, client):
        files = {"file": ("evil.png", io.BytesIO(b"#!/bin/sh\nrm -rf /"), "image/png")}
        assert client.post("/assess/quality", files=files).status_code == 400

    def test_rejects_empty_file(self, client):
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        assert client.post("/assess/quality", files=files).status_code == 400

    def test_rejects_truncated_image(self, client, png_bytes):
        files = {"file": ("t.png", io.BytesIO(png_bytes[:30]), "image/png")}
        assert client.post("/assess/quality", files=files).status_code == 400

    def test_missing_file_is_422(self, client):
        assert client.post("/assess/quality").status_code == 422


class TestModelUnavailable:
    def test_predict_returns_503(self, client, upload):
        assert client.post("/predict", files=upload()).status_code == 503

    def test_analyze_returns_503(self, client, upload):
        assert client.post("/analyze", files=upload()).status_code == 503

    def test_segment_returns_503(self, client, upload):
        assert client.post("/segment", files=upload()).status_code == 503

    def test_batch_returns_503(self, client, png_bytes):
        files = [("files", ("a.png", io.BytesIO(png_bytes), "image/png"))]
        assert client.post("/predict/batch", files=files).status_code == 503


class TestQueryValidation:
    def test_rejects_negative_age(self, client, upload):
        response = client.post("/assess/pediatric?patient_age=-5", files=upload())
        assert response.status_code == 422

    def test_rejects_implausible_age(self, client, upload):
        response = client.post("/assess/pediatric?patient_age=500", files=upload())
        assert response.status_code == 422

    def test_rejects_bad_sensitivity(self, client, upload):
        response = client.post("/detect/small-tumors?sensitivity=extreme", files=upload())
        assert response.status_code == 422

    def test_rejects_bad_sex(self, client, upload):
        response = client.post("/assess/pediatric?patient_sex=X", files=upload())
        assert response.status_code == 422

    def test_rejects_non_positive_pixel_spacing(self, client, upload):
        response = client.post("/segment?pixel_spacing=0", files=upload())
        assert response.status_code == 422


class TestOpenAPI:
    def test_schema_generates(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_new_endpoints_registered(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/segment", "/studies", "/metrics", "/patients/{patient_id}/timeline"):
            assert path in paths
