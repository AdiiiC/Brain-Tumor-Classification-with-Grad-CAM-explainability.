"""
Tumor Grading module — WHO Grade estimation (Grade I-IV).

Uses the tumor classification probabilities + spatial analysis from Grad-CAM
to estimate likely WHO grade. This is an approximate estimate, NOT a
histopathological diagnosis.

Grade Estimation Logic:
- Glioma: Grade I-IV based on confidence, heterogeneity, and size indicators
- Meningioma: Typically Grade I-II
- Pituitary: Typically Grade I (benign adenoma)
- No Tumor: N/A
"""

import numpy as np
import cv2
from typing import Optional


# WHO Grade descriptions
GRADE_INFO = {
    "I": {
        "name": "WHO Grade I",
        "description": "Low-grade, slow-growing, often benign",
        "prognosis": "Generally favorable with complete surgical resection",
        "typical_treatment": "Surgical resection; monitoring",
    },
    "II": {
        "name": "WHO Grade II",
        "description": "Low-grade but infiltrative, may progress",
        "prognosis": "Variable; may transform to higher grade over years",
        "typical_treatment": "Surgery + monitoring; possible radiation",
    },
    "III": {
        "name": "WHO Grade III",
        "description": "Malignant, anaplastic features",
        "prognosis": "Guarded; median survival varies by type",
        "typical_treatment": "Surgery + radiation + chemotherapy",
    },
    "IV": {
        "name": "WHO Grade IV",
        "description": "Highly malignant (e.g., glioblastoma)",
        "prognosis": "Poor; median survival 12-18 months with treatment",
        "typical_treatment": "Maximal safe resection + Stupp protocol (TMZ + RT)",
    },
}


class TumorGrader:
    """
    Estimate WHO tumor grade from classification output and image features.

    NOTE: This is a computational estimate for clinical decision support.
    Definitive grading requires histopathological analysis of tissue biopsy.
    """

    def estimate_grade(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        image: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate WHO grade based on tumor type and image characteristics.

        Args:
            predicted_class: The predicted tumor class name
            confidence: Prediction confidence (0-1)
            probabilities: Dict of class probabilities
            image: Original preprocessed image (240x240 RGB float32)
            heatmap: Grad-CAM heatmap if available

        Returns:
            Dict with grade estimation, confidence, and supporting evidence
        """
        if predicted_class == "No Tumor":
            return {
                "grade": None,
                "grade_description": "No tumor detected — grading not applicable",
                "confidence": confidence,
                "note": "No tumorous tissue identified in this scan",
            }

        # Get image-based features if available
        heterogeneity = 0.5
        tumor_size_ratio = 0.3
        enhancement_pattern = "unknown"

        if image is not None:
            heterogeneity = self._compute_heterogeneity(image)
            tumor_size_ratio = self._estimate_tumor_size(image, heatmap)
            enhancement_pattern = self._classify_enhancement(image, heatmap)

        # Grade estimation by tumor type
        if predicted_class == "Glioma":
            grade = self._grade_glioma(confidence, heterogeneity, tumor_size_ratio, enhancement_pattern)
        elif predicted_class == "Meningioma":
            grade = self._grade_meningioma(confidence, heterogeneity, tumor_size_ratio)
        elif predicted_class == "Pituitary":
            grade = self._grade_pituitary(confidence, tumor_size_ratio)
        else:
            grade = {"estimated_grade": "II", "grade_confidence": 0.3}

        estimated = grade["estimated_grade"]
        return {
            "grade": estimated,
            "grade_name": GRADE_INFO[estimated]["name"],
            "grade_description": GRADE_INFO[estimated]["description"],
            "grade_confidence": round(grade["grade_confidence"], 2),
            "prognosis": GRADE_INFO[estimated]["prognosis"],
            "typical_treatment": GRADE_INFO[estimated]["typical_treatment"],
            "supporting_evidence": grade.get("evidence", []),
            "image_features": {
                "heterogeneity": round(heterogeneity, 3),
                "tumor_size_ratio": round(tumor_size_ratio, 3),
                "enhancement_pattern": enhancement_pattern,
            },
            "disclaimer": "Grade estimation is approximate. Definitive grading requires histopathological analysis.",
        }

    def _grade_glioma(self, confidence: float, heterogeneity: float,
                      size_ratio: float, enhancement: str) -> dict:
        """Grade glioma based on imaging features."""
        evidence = []
        # Higher heterogeneity suggests higher grade
        # Larger size suggests higher grade
        # Ring enhancement suggests GBM (Grade IV)

        score = 0.0  # 0 = Grade I, 1 = Grade IV

        if heterogeneity > 0.6:
            score += 0.3
            evidence.append("High tissue heterogeneity (suggests higher grade)")
        elif heterogeneity < 0.3:
            evidence.append("Homogeneous appearance (suggests lower grade)")

        if size_ratio > 0.3:
            score += 0.25
            evidence.append("Large tumor volume relative to brain")
        elif size_ratio < 0.1:
            evidence.append("Small, well-circumscribed lesion")

        if enhancement == "ring":
            score += 0.35
            evidence.append("Ring-like enhancement pattern (classic GBM feature)")
        elif enhancement == "homogeneous":
            evidence.append("Homogeneous enhancement (lower grade pattern)")

        if confidence > 0.9:
            score += 0.1
            evidence.append("High classification confidence")

        # Map score to grade
        if score >= 0.7:
            grade = "IV"
            conf = min(0.7, 0.4 + score * 0.3)
        elif score >= 0.45:
            grade = "III"
            conf = 0.5
        elif score >= 0.2:
            grade = "II"
            conf = 0.55
        else:
            grade = "I"
            conf = 0.5

        return {"estimated_grade": grade, "grade_confidence": conf, "evidence": evidence}

    def _grade_meningioma(self, confidence: float, heterogeneity: float,
                          size_ratio: float) -> dict:
        """Most meningiomas are Grade I; atypical features suggest Grade II."""
        evidence = []

        if heterogeneity > 0.5 or size_ratio > 0.25:
            grade = "II"
            conf = 0.45
            evidence.append("Atypical features detected (heterogeneity or large size)")
            if heterogeneity > 0.5:
                evidence.append("Heterogeneous tissue pattern")
        else:
            grade = "I"
            conf = 0.7
            evidence.append("Typical meningioma appearance — likely benign")
            evidence.append("Well-defined borders expected")

        return {"estimated_grade": grade, "grade_confidence": conf, "evidence": evidence}

    def _grade_pituitary(self, confidence: float, size_ratio: float) -> dict:
        """Pituitary tumors are almost always Grade I (benign adenomas)."""
        evidence = ["Pituitary tumors are typically benign adenomas (Grade I)"]

        if size_ratio > 0.15:
            evidence.append("Macroadenoma (>10mm) — may require surgical intervention")
        else:
            evidence.append("Likely microadenoma — monitoring may be sufficient")

        return {"estimated_grade": "I", "grade_confidence": 0.8, "evidence": evidence}

    def _compute_heterogeneity(self, image: np.ndarray) -> float:
        """Measure tissue heterogeneity via local intensity variance."""
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Use local standard deviation as heterogeneity proxy
        mean = cv2.blur(gray.astype(np.float32), (16, 16))
        sqmean = cv2.blur((gray.astype(np.float32)) ** 2, (16, 16))
        local_var = sqmean - mean ** 2
        local_var = np.clip(local_var, 0, None)
        # Normalize to 0-1
        max_var = local_var.max() if local_var.max() > 0 else 1.0
        return float(local_var.mean() / max_var)

    def _estimate_tumor_size(self, image: np.ndarray, heatmap: Optional[np.ndarray] = None) -> float:
        """Estimate tumor size as fraction of total brain area using heatmap."""
        if heatmap is not None:
            # Use Grad-CAM activation as tumor region proxy
            if len(heatmap.shape) == 3:
                gray_heat = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
            else:
                gray_heat = heatmap
            # Threshold at 50% activation
            _, active = cv2.threshold(gray_heat, 127, 255, cv2.THRESH_BINARY)
            return float(active.sum() / 255) / (gray_heat.shape[0] * gray_heat.shape[1])
        else:
            # Fallback: estimate from image intensity variance
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            return float(bright.sum() / 255) / (gray.shape[0] * gray.shape[1])

    def _classify_enhancement(self, image: np.ndarray, heatmap: Optional[np.ndarray] = None) -> str:
        """Classify enhancement pattern: ring, homogeneous, or heterogeneous."""
        if heatmap is None:
            return "unknown"

        if len(heatmap.shape) == 3:
            gray_heat = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        else:
            gray_heat = heatmap

        # Check if activation forms a ring (high at edges, low in center)
        h, w = gray_heat.shape
        center_region = gray_heat[h//4:3*h//4, w//4:3*w//4]
        edge_region = gray_heat.mean() - center_region.mean()

        if edge_region > 30:
            return "ring"
        elif float(gray_heat.std()) < 40:
            return "homogeneous"
        else:
            return "heterogeneous"
