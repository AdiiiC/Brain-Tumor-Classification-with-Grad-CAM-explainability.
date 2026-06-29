"""
Upgrade #19 — Longitudinal Tracking.

Compare MRIs over time to track tumor progression or treatment response.
Detects growth patterns, measures volume changes, and alerts on concerning trends.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ScanRecord:
    """Single MRI scan entry in a patient's timeline."""
    scan_date: datetime
    predicted_class: str
    confidence: float
    tumor_volume_mm2: Optional[float] = None  # From segmentation
    uncertainty: float = 0.0
    notes: str = ""
    scan_id: str = ""


@dataclass
class PatientTimeline:
    """Patient's longitudinal MRI history."""
    patient_id: str
    scans: list[ScanRecord] = field(default_factory=list)

    def add_scan(self, record: ScanRecord):
        self.scans.append(record)
        self.scans.sort(key=lambda x: x.scan_date)

    @property
    def latest(self) -> Optional[ScanRecord]:
        return self.scans[-1] if self.scans else None

    @property
    def num_scans(self) -> int:
        return len(self.scans)


class LongitudinalTracker:
    """
    Track tumor progression across multiple MRI scans over time.

    Features:
    - Growth rate calculation
    - Classification stability monitoring
    - Treatment response assessment
    - Alert generation for concerning changes
    """

    def __init__(self):
        self.patients: dict[str, PatientTimeline] = {}

    def register_patient(self, patient_id: str) -> PatientTimeline:
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientTimeline(patient_id=patient_id)
        return self.patients[patient_id]

    def add_scan_result(
        self,
        patient_id: str,
        scan_date: datetime,
        predicted_class: str,
        confidence: float,
        tumor_volume: Optional[float] = None,
        uncertainty: float = 0.0,
    ) -> dict:
        """
        Add a new scan result and return analysis.
        """
        timeline = self.register_patient(patient_id)
        record = ScanRecord(
            scan_date=scan_date,
            predicted_class=predicted_class,
            confidence=confidence,
            tumor_volume_mm2=tumor_volume,
            uncertainty=uncertainty,
        )
        timeline.add_scan(record)

        return self.analyze_progression(patient_id)

    def analyze_progression(self, patient_id: str) -> dict:
        """
        Analyze tumor progression for a patient.

        Returns growth metrics, stability assessment, and alerts.
        """
        timeline = self.patients.get(patient_id)
        if not timeline or timeline.num_scans < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 scans for progression analysis",
                "num_scans": timeline.num_scans if timeline else 0,
            }

        scans = timeline.scans
        latest = scans[-1]
        previous = scans[-2]

        # Classification stability
        class_changes = sum(
            1 for i in range(1, len(scans)) if scans[i].predicted_class != scans[i-1].predicted_class
        )
        classification_stable = class_changes == 0

        # Volume tracking
        growth_rate = None
        volume_trend = "unknown"
        if latest.tumor_volume_mm2 is not None and previous.tumor_volume_mm2 is not None:
            days_elapsed = (latest.scan_date - previous.scan_date).days
            if days_elapsed > 0 and previous.tumor_volume_mm2 > 0:
                volume_change = latest.tumor_volume_mm2 - previous.tumor_volume_mm2
                growth_rate = (volume_change / previous.tumor_volume_mm2) * 100  # percent
                daily_rate = growth_rate / days_elapsed

                if growth_rate > 20:
                    volume_trend = "rapid_growth"
                elif growth_rate > 5:
                    volume_trend = "slow_growth"
                elif growth_rate < -20:
                    volume_trend = "significant_shrinkage"
                elif growth_rate < -5:
                    volume_trend = "mild_shrinkage"
                else:
                    volume_trend = "stable"

        # Confidence tracking
        confidence_trend = "stable"
        if len(scans) >= 3:
            recent_confs = [s.confidence for s in scans[-3:]]
            if all(recent_confs[i] < recent_confs[i-1] for i in range(1, len(recent_confs))):
                confidence_trend = "declining"
            elif all(recent_confs[i] > recent_confs[i-1] for i in range(1, len(recent_confs))):
                confidence_trend = "improving"

        # Generate alerts
        alerts = self._generate_alerts(
            latest, previous, growth_rate, classification_stable, confidence_trend
        )

        return {
            "status": "analyzed",
            "patient_id": patient_id,
            "num_scans": timeline.num_scans,
            "latest_prediction": latest.predicted_class,
            "latest_confidence": latest.confidence,
            "classification_stable": classification_stable,
            "class_changes": class_changes,
            "volume_analysis": {
                "current_volume": latest.tumor_volume_mm2,
                "previous_volume": previous.tumor_volume_mm2,
                "growth_rate_percent": growth_rate,
                "trend": volume_trend,
            },
            "confidence_trend": confidence_trend,
            "alerts": alerts,
            "recommendation": self._get_recommendation(volume_trend, classification_stable, alerts),
        }

    def _generate_alerts(self, latest, previous, growth_rate, stable, conf_trend) -> list[str]:
        """Generate clinical alerts based on progression data."""
        alerts = []

        if not stable:
            alerts.append("⚠️ CLASSIFICATION CHANGE: Tumor type reclassified — urgent review needed")

        if growth_rate is not None and growth_rate > 20:
            alerts.append(f"🔴 RAPID GROWTH: {growth_rate:.1f}% volume increase since last scan")

        if latest.uncertainty > 0.08:
            alerts.append("⚠️ HIGH UNCERTAINTY: Model confidence is low — consider re-scan or specialist")

        if latest.confidence < 0.7:
            alerts.append("⚠️ LOW CONFIDENCE: Below 70% threshold — manual review required")

        if conf_trend == "declining":
            alerts.append("📉 DECLINING CONFIDENCE: Model becoming less certain over time")

        if growth_rate is not None and growth_rate < -30:
            alerts.append("✅ SIGNIFICANT RESPONSE: >30% volume reduction — treatment appears effective")

        return alerts

    def _get_recommendation(self, volume_trend, stable, alerts) -> str:
        """Generate clinical recommendation."""
        if any("RAPID GROWTH" in a for a in alerts):
            return "Immediate specialist consultation recommended. Consider updated treatment plan."
        if any("CLASSIFICATION CHANGE" in a for a in alerts):
            return "Urgent radiology review — tumor reclassification detected."
        if volume_trend == "significant_shrinkage":
            return "Positive treatment response. Continue current protocol and schedule follow-up."
        if volume_trend == "stable" and stable:
            return "Stable findings. Continue monitoring per standard schedule."
        return "Schedule follow-up scan and discuss findings with treating physician."

    def get_patient_summary(self, patient_id: str) -> dict:
        """Get full timeline summary for a patient."""
        timeline = self.patients.get(patient_id)
        if not timeline:
            return {"error": "Patient not found"}

        return {
            "patient_id": patient_id,
            "total_scans": timeline.num_scans,
            "date_range": {
                "first": timeline.scans[0].scan_date.isoformat() if timeline.scans else None,
                "last": timeline.scans[-1].scan_date.isoformat() if timeline.scans else None,
            },
            "scan_history": [
                {
                    "date": s.scan_date.isoformat(),
                    "prediction": s.predicted_class,
                    "confidence": s.confidence,
                    "volume": s.tumor_volume_mm2,
                }
                for s in timeline.scans
            ],
        }
