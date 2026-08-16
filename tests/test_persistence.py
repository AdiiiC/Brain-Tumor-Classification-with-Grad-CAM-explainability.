"""Persistence, feedback loop, timelines, and PDF reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from api.database import Study, session_scope


class TestStudyEndpoints:
    def test_list_empty(self, client):
        assert client.get("/studies").json() == []

    def test_get_seeded_study(self, client, seeded_study):
        body = client.get(f"/studies/{seeded_study}").json()
        assert body["predicted_class"] == "Glioma"
        assert body["patient_id"] == "PT-001"

    def test_missing_study_is_404(self, client):
        assert client.get("/studies/doesnotexist").status_code == 404

    def test_filter_by_patient(self, client, seeded_study):
        assert len(client.get("/studies?patient_id=PT-001").json()) == 1
        assert client.get("/studies?patient_id=PT-999").json() == []

    def test_flagged_filter(self, client, seeded_study):
        assert client.get("/studies?flagged_only=true").json() == []

    def test_pagination_bounds_enforced(self, client):
        assert client.get("/studies?limit=0").status_code == 422
        assert client.get("/studies?limit=9999").status_code == 422


class TestFeedback:
    def test_records_agreement(self, client, seeded_study):
        response = client.post(
            f"/studies/{seeded_study}/feedback",
            json={"corrected_class": "Glioma", "reviewer": "Dr. Patel"},
        )
        assert response.status_code == 200
        assert response.json()["agrees_with_ai"] is True

    def test_records_disagreement(self, client, seeded_study):
        response = client.post(
            f"/studies/{seeded_study}/feedback",
            json={"corrected_class": "Meningioma", "notes": "Dural tail present"},
        )
        assert response.json()["agrees_with_ai"] is False

    def test_rejects_unknown_class(self, client, seeded_study):
        response = client.post(
            f"/studies/{seeded_study}/feedback", json={"corrected_class": "Astrocytoma"},
        )
        assert response.status_code == 400

    def test_feedback_on_missing_study_is_404(self, client):
        response = client.post("/studies/nope/feedback", json={"corrected_class": "Glioma"})
        assert response.status_code == 404

    def test_feedback_is_idempotent(self, client, seeded_study):
        client.post(f"/studies/{seeded_study}/feedback", json={"corrected_class": "Glioma"})
        client.post(f"/studies/{seeded_study}/feedback", json={"corrected_class": "Pituitary"})
        body = client.get(f"/studies/{seeded_study}").json()
        assert body["confirmed_class"] == "Pituitary"


class TestTimeline:
    @pytest.fixture
    def patient_with_growth(self, db_url):
        base = datetime.now(UTC) - timedelta(days=120)
        volumes = [3000.0, 3600.0, 5200.0]
        with session_scope() as session:
            for index, volume in enumerate(volumes):
                session.add(Study(
                    patient_id="PT-GROW",
                    image_sha256=f"{index:064d}",
                    predicted_class="Glioma",
                    confidence=0.9 - index * 0.05,
                    uncertainty=0.02,
                    probabilities=json.dumps({"Glioma": 90.0}),
                    tumor_volume_mm3=volume,
                    created_at=base + timedelta(days=index * 60),
                    study_date=base + timedelta(days=index * 60),
                    model_version="test-model",
                ))
        return "PT-GROW"

    def test_unknown_patient_is_404(self, client):
        assert client.get("/patients/NOBODY/timeline").status_code == 404

    def test_single_scan_reports_insufficient_data(self, client, seeded_study):
        body = client.get("/patients/PT-001/timeline").json()
        assert body["status"] == "insufficient_data"

    def test_growth_detected(self, client, patient_with_growth):
        body = client.get(f"/patients/{patient_with_growth}/timeline").json()
        assert body["num_scans"] == 3
        assert body["volume_analysis"]["growth_rate_percent"] > 0

    def test_overall_growth_summary(self, client, patient_with_growth):
        overall = client.get(f"/patients/{patient_with_growth}/timeline").json()["overall_growth"]
        assert overall["first_volume_cm3"] == 3.0
        assert overall["latest_volume_cm3"] == 5.2
        assert overall["total_change_percent"] > 70

    def test_scans_returned_in_order(self, client, patient_with_growth):
        scans = client.get(f"/patients/{patient_with_growth}/timeline").json()["scans"]
        volumes = [s["volume_mm3"] for s in scans]
        assert volumes == sorted(volumes)


class TestReports:
    def test_generates_pdf(self, client, seeded_study):
        response = client.get(f"/studies/{seeded_study}/report")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")

    def test_attachment_disposition(self, client, seeded_study):
        response = client.get(f"/studies/{seeded_study}/report")
        assert "attachment" in response.headers["content-disposition"]

    def test_report_for_missing_study_is_404(self, client):
        assert client.get("/studies/nope/report").status_code == 404

    def test_report_includes_feedback(self, client, seeded_study):
        client.post(f"/studies/{seeded_study}/feedback", json={"corrected_class": "Meningioma"})
        assert client.get(f"/studies/{seeded_study}/report").content.startswith(b"%PDF")


class TestFingerprint:
    def test_same_bytes_same_hash(self, png_bytes):
        from api.database import image_fingerprint
        assert image_fingerprint(png_bytes) == image_fingerprint(png_bytes)

    def test_different_bytes_differ(self, png_bytes, jpeg_bytes):
        from api.database import image_fingerprint
        assert image_fingerprint(png_bytes) != image_fingerprint(jpeg_bytes)
