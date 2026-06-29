"""
Pediatric Brain Tumor Support module.

Addresses differences between adult and pediatric brain tumors:
- Different tumor type distributions (posterior fossa dominance)
- Age-adjusted confidence calibration
- Pediatric-specific clinical recommendations
- Growth chart context for tumor size assessment
"""

import numpy as np
from typing import Optional


# Pediatric brain tumor epidemiology
PEDIATRIC_TUMOR_INFO = {
    "Glioma": {
        "pediatric_subtypes": [
            "Pilocytic astrocytoma (most common, Grade I)",
            "Diffuse midline glioma (H3 K27M-mutant)",
            "Ependymoma",
            "Optic pathway glioma",
        ],
        "typical_locations": ["Posterior fossa", "Brainstem", "Optic pathway", "Cerebellum"],
        "pediatric_note": "Pediatric gliomas differ significantly from adult — many are low-grade and curable",
        "prognosis_modifier": "Generally better prognosis than adult gliomas of same grade",
    },
    "Meningioma": {
        "pediatric_subtypes": [
            "Pediatric meningioma (rare in children)",
            "May be associated with NF2 or radiation exposure",
        ],
        "typical_locations": ["Cerebral convexity", "Posterior fossa"],
        "pediatric_note": "Very rare in children (<3% of pediatric CNS tumors). Consider alternative diagnoses.",
        "prognosis_modifier": "When present, may indicate genetic predisposition (e.g., NF2)",
    },
    "Pituitary": {
        "pediatric_subtypes": [
            "Craniopharyngioma (most common sellar tumor in children)",
            "Pituitary adenoma (rare before puberty)",
            "Rathke's cleft cyst",
        ],
        "typical_locations": ["Sellar/suprasellar region"],
        "pediatric_note": "Craniopharyngioma is more likely than pituitary adenoma in children",
        "prognosis_modifier": "Consider endocrine evaluation — growth hormone effects",
    },
    "No Tumor": {
        "pediatric_subtypes": [],
        "typical_locations": [],
        "pediatric_note": "Normal findings. If clinical suspicion remains, consider age-appropriate differentials.",
        "prognosis_modifier": "",
    },
}

# Age groups with different tumor epidemiology
AGE_GROUPS = {
    "infant": {"range": (0, 2), "label": "Infant (0-2 years)"},
    "young_child": {"range": (2, 6), "label": "Young child (2-6 years)"},
    "child": {"range": (6, 12), "label": "Child (6-12 years)"},
    "adolescent": {"range": (12, 18), "label": "Adolescent (12-18 years)"},
    "adult": {"range": (18, 200), "label": "Adult (18+ years)"},
}


class PediatricAssessor:
    """
    Provide pediatric-specific tumor assessment and recommendations.

    Addresses the limitation that the model was primarily trained on adult
    MRI data by providing context, confidence adjustments, and
    pediatric-specific clinical information.
    """

    def assess(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
    ) -> dict:
        """
        Generate pediatric-adjusted assessment.

        Args:
            predicted_class: Model's predicted class
            confidence: Model confidence (0-1)
            probabilities: Class probability distribution
            patient_age: Patient age in years (None if unknown)
            patient_sex: 'M' or 'F' (None if unknown)

        Returns:
            Pediatric assessment with adjusted confidence and recommendations
        """
        is_pediatric = patient_age is not None and patient_age < 18
        age_group = self._get_age_group(patient_age)

        result = {
            "is_pediatric": is_pediatric,
            "age_group": age_group,
            "patient_age": patient_age,
        }

        if not is_pediatric:
            result["note"] = "Adult patient — standard model predictions apply"
            result["confidence_adjustment"] = 1.0
            result["adjusted_confidence"] = confidence
            return result

        # Pediatric-specific assessment
        tumor_info = PEDIATRIC_TUMOR_INFO.get(predicted_class, {})
        confidence_adj = self._compute_confidence_adjustment(predicted_class, patient_age, confidence)

        result.update({
            "confidence_adjustment": round(confidence_adj, 2),
            "adjusted_confidence": round(confidence * confidence_adj, 3),
            "reliability_warning": self._get_reliability_warning(patient_age),
            "pediatric_considerations": tumor_info.get("pediatric_note", ""),
            "likely_subtypes": tumor_info.get("pediatric_subtypes", []),
            "typical_locations": tumor_info.get("typical_locations", []),
            "prognosis_note": tumor_info.get("prognosis_modifier", ""),
            "differential_diagnoses": self._get_differentials(predicted_class, patient_age),
            "recommended_workup": self._get_workup(predicted_class, patient_age),
            "specialist_referral": "Pediatric neuro-oncology consultation recommended",
        })

        return result

    def _get_age_group(self, age: Optional[int]) -> str:
        """Determine age group category."""
        if age is None:
            return "unknown"
        for group, info in AGE_GROUPS.items():
            if info["range"][0] <= age < info["range"][1]:
                return info["label"]
        return "adult"

    def _compute_confidence_adjustment(self, predicted_class: str, age: int, confidence: float) -> float:
        """
        Adjust model confidence for pediatric cases.

        The model was trained on adult data, so pediatric predictions
        are inherently less reliable. Apply a discount factor.
        """
        # Base discount for pediatric use
        base_discount = 0.85

        # Additional discounts based on tumor type rarity in children
        type_discount = {
            "Meningioma": 0.6,   # Very rare in children — likely misclassification
            "Pituitary": 0.75,   # Possible but uncommon before puberty
            "Glioma": 0.9,       # Common in children, model more reliable
            "No Tumor": 0.95,    # Negative prediction fairly reliable
        }.get(predicted_class, 0.8)

        # Age factor: younger = less reliable (more developmental differences)
        age_factor = min(1.0, 0.7 + (age / 18) * 0.3)

        return base_discount * type_discount * age_factor

    def _get_reliability_warning(self, age: int) -> str:
        """Generate age-appropriate reliability warning."""
        if age < 2:
            return (
                "CRITICAL: Model not validated for infants. Brain anatomy and tumor "
                "types differ significantly. Dedicated pediatric neuro-radiology review required."
            )
        elif age < 6:
            return (
                "WARNING: Limited reliability for young children. Posterior fossa tumors "
                "(medulloblastoma, ependymoma) are common at this age and NOT in training data."
            )
        elif age < 12:
            return (
                "CAUTION: Moderate reliability. Some pediatric tumor types (DIPG, "
                "craniopharyngioma) may be misclassified."
            )
        else:
            return (
                "NOTE: Adolescent brain approaching adult morphology. Predictions "
                "more reliable but pediatric-specific tumors should still be considered."
            )

    def _get_differentials(self, predicted_class: str, age: int) -> list:
        """Get pediatric differential diagnoses not in the model's vocabulary."""
        common_pediatric = [
            "Medulloblastoma (posterior fossa)",
            "ATRT (atypical teratoid rhabdoid tumor)",
            "DIPG (diffuse intrinsic pontine glioma)",
            "Craniopharyngioma",
            "Ependymoma",
            "Choroid plexus papilloma/carcinoma",
        ]

        if age < 3:
            return [
                "ATRT (most common malignant CNS tumor in infants)",
                "Choroid plexus tumors",
                "Teratoma",
                "Desmoplastic infantile ganglioglioma",
            ] + common_pediatric[:3]
        elif age < 10:
            return [
                "Medulloblastoma (peak age 5-9)",
                "Pilocytic astrocytoma",
                "Ependymoma",
                "Craniopharyngioma",
                "DIPG",
            ]
        else:
            return [
                "Pilocytic astrocytoma",
                "Medulloblastoma",
                "Craniopharyngioma",
                "Diffuse glioma",
                "Ganglioglioma",
            ]

    def _get_workup(self, predicted_class: str, age: int) -> list:
        """Recommended pediatric-specific workup."""
        workup = [
            "Pediatric neuro-oncology consultation",
            "Full neuraxis MRI (brain + spine) to rule out drop metastases",
            "Endocrine evaluation (growth, puberty, pituitary function)",
        ]

        if predicted_class == "Glioma":
            workup.extend([
                "Consider genetic testing (BRAF V600E, H3 K27M, IDH status)",
                "CSF cytology if leptomeningeal spread suspected",
            ])
        elif predicted_class == "Pituitary":
            workup.extend([
                "Visual field assessment",
                "Complete anterior pituitary hormone panel",
                "Consider craniopharyngioma vs. adenoma",
            ])

        if age < 5:
            workup.append("Ophthalmological exam (fundoscopy for papilledema)")
            workup.append("Developmental assessment baseline")

        return workup
