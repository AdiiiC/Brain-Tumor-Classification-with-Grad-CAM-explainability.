"""
MRI Sequence Detection module.

Auto-detects the MRI sequence type from image characteristics:
- T1-weighted (with/without contrast)
- T2-weighted
- FLAIR
- DWI (Diffusion-Weighted)
- Unknown

Adjusts preprocessing and confidence interpretation accordingly.
"""


import cv2
import numpy as np

# Sequence characteristics for preprocessing adaptation
SEQUENCE_INFO = {
    "T1": {
        "name": "T1-weighted",
        "description": "Standard anatomical imaging. CSF appears dark, gray matter darker than white matter.",
        "model_affinity": "high",
        "preprocessing": "standard (CLAHE + crop)",
        "note": "Primary training modality — highest expected accuracy",
    },
    "T1CE": {
        "name": "T1-weighted with Contrast Enhancement",
        "description": "Gadolinium-enhanced T1. Tumors with broken blood-brain barrier enhance brightly.",
        "model_affinity": "high",
        "preprocessing": "standard (CLAHE + crop)",
        "note": "Excellent for tumor boundary delineation",
    },
    "T2": {
        "name": "T2-weighted",
        "description": "CSF appears bright, edema visible. Good for detecting lesions.",
        "model_affinity": "moderate",
        "preprocessing": "standard (CLAHE + crop)",
        "note": "Good for edema detection; moderate accuracy expected",
    },
    "FLAIR": {
        "name": "FLAIR (Fluid-Attenuated Inversion Recovery)",
        "description": "CSF suppressed (dark). Periventricular lesions and edema highlighted.",
        "model_affinity": "moderate",
        "preprocessing": "resize only (no crop — FLAIR has different contrast)",
        "note": "Model trained with FLAIR examples; good for glioma detection",
    },
    "DWI": {
        "name": "Diffusion-Weighted Imaging",
        "description": "Shows areas of restricted diffusion. Useful for acute stroke and tumor cellularity.",
        "model_affinity": "low",
        "preprocessing": "resize only",
        "note": "Limited training data — results may be less reliable",
    },
    "unknown": {
        "name": "Unknown Sequence",
        "description": "Could not determine MRI sequence from image characteristics.",
        "model_affinity": "unknown",
        "preprocessing": "standard (CLAHE + crop)",
        "note": "Using default preprocessing; accuracy may vary",
    },
}


class SequenceDetector:
    """
    Detect MRI sequence type from image intensity characteristics.

    Uses statistical features of the image histogram to distinguish
    between T1, T2, FLAIR, and DWI sequences.
    """

    def detect(self, image: np.ndarray, dicom_metadata: dict | None = None) -> dict:
        """
        Detect MRI sequence from image and/or DICOM metadata.

        Args:
            image: Input image (BGR or RGB, uint8 or float32)
            dicom_metadata: Optional DICOM metadata dict with sequence info

        Returns:
            Dict with detected sequence, confidence, and preprocessing advice
        """
        # Priority 1: Use DICOM metadata if available
        if dicom_metadata:
            seq = self._from_dicom_metadata(dicom_metadata)
            if seq != "unknown":
                info = SEQUENCE_INFO[seq]
                return {
                    "detected_sequence": seq,
                    "sequence_name": info["name"],
                    "confidence": 0.95,
                    "source": "DICOM metadata",
                    "model_affinity": info["model_affinity"],
                    "preprocessing_advice": info["preprocessing"],
                    "accuracy_note": info["note"],
                    "description": info["description"],
                }

        # Priority 2: Infer from image statistics
        seq, conf = self._from_image_stats(image)
        info = SEQUENCE_INFO[seq]

        return {
            "detected_sequence": seq,
            "sequence_name": info["name"],
            "confidence": round(conf, 2),
            "source": "image analysis",
            "model_affinity": info["model_affinity"],
            "preprocessing_advice": info["preprocessing"],
            "accuracy_note": info["note"],
            "description": info["description"],
        }

    def _from_dicom_metadata(self, metadata: dict) -> str:
        """Extract sequence type from DICOM metadata fields."""
        # Common DICOM fields for sequence identification
        desc = str(metadata.get("SeriesDescription", "")).lower()
        seq_name = str(metadata.get("SequenceName", "")).lower()
        protocol = str(metadata.get("ProtocolName", "")).lower()
        scanning_seq = str(metadata.get("ScanningSequence", "")).lower()

        combined = f"{desc} {seq_name} {protocol} {scanning_seq}"

        if "flair" in combined:
            return "FLAIR"
        elif "dwi" in combined or "diffusion" in combined or "dw" in combined:
            return "DWI"
        elif "t2" in combined and "flair" not in combined:
            return "T2"
        elif ("t1" in combined and "gad" in combined) or "t1ce" in combined or "t1+c" in combined or "post" in combined:
            return "T1CE"
        elif "t1" in combined:
            return "T1"

        return "unknown"

    def _from_image_stats(self, image: np.ndarray) -> tuple:
        """
        Infer sequence from intensity distribution.

        Heuristics:
        - T1: CSF dark (low mean in ventricle region), bimodal histogram
        - T2: CSF bright (high mean overall), bright fluid areas
        - FLAIR: CSF suppressed, high contrast at brain-CSF boundary
        - DWI: Generally bright signal, low dynamic range
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8) if image.dtype != np.uint8 else image,
                                cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.uint8) if image.dtype != np.uint8 else image

        h, w = gray.shape
        # Analyze different regions
        center = gray[h//4:3*h//4, w//4:3*w//4]
        overall_mean = float(gray.mean())
        center_mean = float(center.mean())
        overall_std = float(gray.std())

        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()

        # Check for CSF brightness (top 10% intensity region)
        bright_fraction = float(hist[200:].sum())
        dark_fraction = float(hist[:50].sum())

        # Bimodality check
        peaks = self._find_histogram_peaks(hist)

        # Decision logic
        scores = {"T1": 0.0, "T2": 0.0, "FLAIR": 0.0, "DWI": 0.0}

        # T1: moderate overall brightness, CSF dark, bimodal
        if overall_mean < 120 and dark_fraction > 0.3:
            scores["T1"] += 0.4
        if len(peaks) >= 2:
            scores["T1"] += 0.2

        # T2: high overall brightness, CSF bright
        if bright_fraction > 0.1 and overall_mean > 100:
            scores["T2"] += 0.4
        if overall_std > 50:
            scores["T2"] += 0.1

        # FLAIR: moderate brightness, low bright_fraction (suppressed CSF)
        if 60 < overall_mean < 140 and bright_fraction < 0.05:
            scores["FLAIR"] += 0.4
        if overall_std > 40 and bright_fraction < 0.08:
            scores["FLAIR"] += 0.2

        # DWI: generally bright, low dynamic range
        if overall_mean > 100 and overall_std < 40:
            scores["DWI"] += 0.3

        # Default to T1 if unclear (most common in training)
        scores["T1"] += 0.15  # prior

        best_seq = max(scores, key=scores.get)
        best_score = scores[best_seq]

        # Normalize confidence
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.25

        # If confidence is too low, return unknown
        if confidence < 0.3:
            return "unknown", confidence

        return best_seq, confidence

    def _find_histogram_peaks(self, hist: np.ndarray, min_distance: int = 30) -> list:
        """Find prominent peaks in histogram."""
        peaks = []
        smoothed = np.convolve(hist, np.ones(5) / 5, mode='same')
        for i in range(min_distance, len(smoothed) - min_distance):
            if smoothed[i] > smoothed[i - min_distance:i].max() and \
               smoothed[i] > smoothed[i + 1:i + min_distance].max() and \
               smoothed[i] > 0.005:
                peaks.append(i)
        return peaks

    def get_preprocessing_for_sequence(self, sequence: str) -> dict:
        """Get recommended preprocessing parameters for a detected sequence."""
        if sequence in ("T1", "T1CE"):
            return {
                "use_clahe": True,
                "use_crop": True,
                "clahe_clip_limit": 3.0,
                "normalize": False,  # EfficientNet handles this
            }
        elif sequence == "T2":
            return {
                "use_clahe": True,
                "use_crop": True,
                "clahe_clip_limit": 2.0,
                "normalize": False,
            }
        elif sequence == "FLAIR":
            return {
                "use_clahe": False,  # FLAIR has good contrast already
                "use_crop": False,   # Different brain appearance
                "normalize": False,
            }
        elif sequence == "DWI":
            return {
                "use_clahe": False,
                "use_crop": False,
                "normalize": False,
            }
        else:
            return {
                "use_clahe": True,
                "use_crop": True,
                "clahe_clip_limit": 3.0,
                "normalize": False,
            }
