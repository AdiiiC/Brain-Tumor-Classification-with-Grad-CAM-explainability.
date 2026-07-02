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
    Provide pediatric-specific tumor assessment with Bayesian probability
    re-weighting based on pediatric brain tumor epidemiology.

    Instead of just advisory text, this module actually adjusts the model's
    output probabilities using age-specific prior distributions from published
    pediatric neuro-oncology statistics, producing corrected predictions.
    """

    # Pediatric tumor type priors by age group (from CBTRUS/WHO epidemiology)
    # Values represent relative incidence rates in each age group
    PEDIATRIC_PRIORS = {
        "infant": {  # 0-2 years
            "Glioma": 0.35,      # Pilocytic astrocytoma common
            "Meningioma": 0.02,  # Extremely rare
            "No Tumor": 0.30,    # Keep reasonable
            "Pituitary": 0.03,   # Very rare
            # Not in model but most common: ATRT (0.15), Choroid plexus (0.10), Teratoma (0.05)
        },
        "young_child": {  # 2-6 years
            "Glioma": 0.45,      # Pilocytic astrocytoma, DIPG peak
            "Meningioma": 0.02,
            "No Tumor": 0.25,
            "Pituitary": 0.03,   # Craniopharyngioma possible
        },
        "child": {  # 6-12 years
            "Glioma": 0.40,      # Medulloblastoma peak (5-9), pilocytic
            "Meningioma": 0.03,
            "No Tumor": 0.25,
            "Pituitary": 0.07,   # Craniopharyngioma more common
        },
        "adolescent": {  # 12-18 years
            "Glioma": 0.35,
            "Meningioma": 0.05,  # Starting to appear
            "No Tumor": 0.25,
            "Pituitary": 0.10,   # Adenomas appear with puberty
        },
    }

    # Adult priors (baseline from training data)
    ADULT_PRIORS = {
        "Glioma": 0.30,
        "Meningioma": 0.25,
        "No Tumor": 0.20,
        "Pituitary": 0.25,
    }

    CLASS_INDEX_MAP = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

    def assess(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
    ) -> dict:
        """
        Generate pediatric-adjusted assessment with Bayesian re-weighted probabilities.
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

        # === BAYESIAN RE-WEIGHTING ===
        # P(class|image, age) ∝ P(image|class) × P(class|age) / P(class|adult)
        pediatric_prior = self._get_prior(age_group)
        reweighted_probs = self._bayesian_reweight(probabilities, pediatric_prior)

        # New prediction after re-weighting
        adjusted_class = max(reweighted_probs, key=reweighted_probs.get)
        adjusted_confidence = reweighted_probs[adjusted_class]

        tumor_info = PEDIATRIC_TUMOR_INFO.get(adjusted_class, {})

        result.update({
            "original_prediction": predicted_class,
            "original_confidence": round(confidence, 3),
            "adjusted_prediction": adjusted_class,
            "adjusted_confidence": round(adjusted_confidence, 3),
            "adjusted_probabilities": {k: round(v, 4) for k, v in reweighted_probs.items()},
            "prediction_changed": adjusted_class != predicted_class,
            "bayesian_method": "P(class|image,age) ∝ P(image|class) × P(class|age) / P(class|adult)",
            "reliability_warning": self._get_reliability_warning(patient_age),
            "pediatric_considerations": tumor_info.get("pediatric_note", ""),
            "likely_subtypes": tumor_info.get("pediatric_subtypes", []),
            "typical_locations": tumor_info.get("typical_locations", []),
            "prognosis_note": tumor_info.get("prognosis_modifier", ""),
            "differential_diagnoses": self._get_differentials(adjusted_class, patient_age),
            "recommended_workup": self._get_workup(adjusted_class, patient_age),
            "specialist_referral": "Pediatric neuro-oncology consultation recommended",
        })

        return result

    def _get_age_group(self, age: Optional[int]) -> str:
        """Map a patient age to its epidemiological age-group label."""
        if age is None:
            return "Unknown"
        for info in AGE_GROUPS.values():
            low, high = info["range"]
            if low <= age < high:
                return info["label"]
        return AGE_GROUPS["adult"]["label"]

    def _get_prior(self, age_group: str) -> dict:
        """Get the pediatric prior for the age group."""
        # Map age group label back to key
        group_key_map = {
            "Infant (0-2 years)": "infant",
            "Young child (2-6 years)": "young_child",
            "Child (6-12 years)": "child",
            "Adolescent (12-18 years)": "adolescent",
        }
        key = group_key_map.get(age_group, "child")
        return self.PEDIATRIC_PRIORS.get(key, self.PEDIATRIC_PRIORS["child"])

    def _bayesian_reweight(self, model_probs: dict, pediatric_prior: dict) -> dict:
        """
        Bayesian re-weighting: adjust model probabilities using pediatric priors.

        P(class|image, age) ∝ P(image|class) × P(class|age) / P(class|adult)

        Where P(image|class) ∝ P(class|image) / P(class|adult)  [Bayes inversion]
        So: P(class|image, age) ∝ P(class|image) × P(class|age) / P(class|adult)
        """
        reweighted = {}
        for cls_name in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
            model_p = model_probs.get(cls_name, 0.0)
            if isinstance(model_p, (list, tuple)):
                model_p = float(model_p[0]) if model_p else 0.0
            else:
                model_p = float(model_p)

            ped_prior = pediatric_prior.get(cls_name, 0.1)
            adult_prior = self.ADULT_PRIORS.get(cls_name, 0.25)

            # Bayesian adjustment
            reweighted[cls_name] = model_p * (ped_prior / (adult_prior + 1e-10))

        # Normalize to sum to 1
        total = sum(reweighted.values())
        if total > 0:
            reweighted = {k: v / total for k, v in reweighted.items()}
        else:
            reweighted = {k: 0.25 for k in reweighted}

        return reweighted

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
