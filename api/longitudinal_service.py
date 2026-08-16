"""
Patient timeline analysis backed by the studies table.

Wraps the existing LongitudinalTracker (upgrades/longitudinal.py), which had no
data source until studies became persistent. Volumes come from the segmentation
service, so growth rates are real measurements rather than placeholders.
"""

from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import Study

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upgrades.longitudinal import LongitudinalTracker  # noqa: E402

# RECIST-inspired thresholds also used by LongitudinalTracker.
RAPID_GROWTH_PCT = 20.0


def fetch_timeline(session: Session, patient_id: str) -> list[Study]:
    stmt = (
        select(Study)
        .where(Study.patient_id == patient_id)
        .order_by(Study.created_at.asc())
    )
    return list(session.execute(stmt).scalars())


def _scan_datetime(study: Study):
    dt = study.study_date or study.created_at
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def analyze_patient(session: Session, patient_id: str) -> dict:
    """Run progression analysis over every stored study for a patient."""
    studies = fetch_timeline(session, patient_id)

    if not studies:
        return {"status": "not_found", "patient_id": patient_id, "num_scans": 0}

    tracker = LongitudinalTracker()
    for study in studies:
        tracker.add_scan_result(
            patient_id=patient_id,
            scan_date=_scan_datetime(study),
            predicted_class=study.predicted_class,
            confidence=study.confidence,
            tumor_volume=study.tumor_volume_mm3,
            uncertainty=study.uncertainty or 0.0,
        )

    analysis = tracker.analyze_progression(patient_id)
    analysis["scans"] = [
        {
            "study_id": s.id,
            "date": _scan_datetime(s).isoformat(),
            "predicted_class": s.predicted_class,
            "confidence": round(s.confidence, 4),
            "uncertainty": round(s.uncertainty, 4) if s.uncertainty is not None else None,
            "volume_mm3": s.tumor_volume_mm3,
            "volume_cm3": round(s.tumor_volume_mm3 / 1000, 3) if s.tumor_volume_mm3 else None,
            "flagged_for_review": s.flagged_for_review,
            "confirmed_class": s.feedback.corrected_class if s.feedback else None,
        }
        for s in studies
    ]
    analysis["patient_id"] = patient_id
    analysis["num_scans"] = len(studies)

    volumes = [(s, s.tumor_volume_mm3) for s in studies if s.tumor_volume_mm3]
    if len(volumes) >= 2:
        first_study, first_vol = volumes[0]
        last_study, last_vol = volumes[-1]
        days = max(1, (_scan_datetime(last_study) - _scan_datetime(first_study)).days)
        total_change = ((last_vol - first_vol) / first_vol) * 100
        analysis["overall_growth"] = {
            "first_volume_cm3": round(first_vol / 1000, 3),
            "latest_volume_cm3": round(last_vol / 1000, 3),
            "total_change_percent": round(total_change, 1),
            "days_observed": days,
            "change_per_month_percent": round(total_change / days * 30, 2),
        }

    return analysis
