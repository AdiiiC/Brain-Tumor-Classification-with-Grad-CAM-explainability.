"""Shared pytest fixtures.

Tests run against a temporary SQLite database and never require a trained
model — endpoints that need one are expected to return 503.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Configure the environment before api.* is imported anywhere.
os.environ.setdefault("MODEL_PATH", "__no_such_model__.keras")
os.environ.setdefault("TFLITE_PATH", "__no_such_model__.tflite")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("GIT_SHA", "testsha00000")
os.environ.pop("API_KEYS", None)


@pytest.fixture(scope="function")
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", url)

    from api import database
    database.reset_engine()
    database.init_db()
    yield url
    database.reset_engine()


@pytest.fixture(scope="function")
def client(db_url, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    from api.observability import metrics
    from api.security import reset_all_limiters

    reset_all_limiters()
    metrics.reset()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    """A small synthetic brain-like PNG."""
    import cv2

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 90, (140, 140, 140), -1)
    cv2.circle(img, (150, 110), 22, (220, 220, 220), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def jpeg_bytes() -> bytes:
    import cv2

    img = np.full((128, 128, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def upload(png_bytes):
    def _make(name: str = "scan.png", data: bytes | None = None, content_type: str = "image/png"):
        return {"file": (name, io.BytesIO(data if data is not None else png_bytes), content_type)}

    return _make


@pytest.fixture
def seeded_study(db_url):
    """Insert a completed study so read endpoints have something to return."""
    import json

    from api.database import Study, session_scope

    with session_scope() as session:
        study = Study(
            patient_id="PT-001",
            filename="scan.png",
            image_sha256="a" * 64,
            predicted_class="Glioma",
            confidence=0.93,
            uncertainty=0.02,
            probabilities=json.dumps({"Glioma": 93.0, "Meningioma": 4.0, "No Tumor": 2.0, "Pituitary": 1.0}),
            tumor_volume_mm3=4200.0,
            tumor_area_px=840.0,
            quality_score=88.0,
            sequence_type="T1CE",
            who_grade="Grade IV",
            flagged_for_review=False,
            is_ood=False,
            model_version="test-model",
            git_sha="testsha00000",
        )
        session.add(study)
        session.flush()
        return study.id
