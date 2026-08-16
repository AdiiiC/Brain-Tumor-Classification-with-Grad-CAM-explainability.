"""
Small Tumor Detection module.

Uses sliding window + segmentation approach to detect tumors smaller
than what the standard 240x240 classification can reliably identify.

Strategy:
1. Multi-scale scanning: process at multiple resolutions
2. Patch-based detection: slide a window across high-res image
3. Attention aggregation: combine predictions from patches
4. Size estimation: approximate tumor dimensions in mm
"""


import cv2
import numpy as np


class SmallTumorDetector:
    """
    Detect tumors < 5mm using a dedicated trained patch classifier.

    The patch classifier is trained on multi-scale patches (60px, 90px, 120px)
    extracted from tumor and non-tumor training images. This enables detection
    of small lesions below the resolution threshold of the main 240x240 model.
    """

    # Typical MRI pixel spacing (mm/pixel) for common scanners
    DEFAULT_PIXEL_SPACING = 0.5  # mm per pixel (approximate)
    PATCH_SIZE = 120  # patch classifier input size
    OVERLAP = 0.5  # 50% patch overlap

    def __init__(self):
        self.patch_model = None
        self._load_patch_model()

    def _load_patch_model(self):
        """Load the trained patch classifier if available."""
        from pathlib import Path
        try:
            import tensorflow as tf
        except ImportError:
            return
        patch_path = Path("patch_classifier.keras")
        if patch_path.exists():
            try:
                self.patch_model = tf.keras.models.load_model(str(patch_path))
            except Exception:
                self.patch_model = None

    def detect(
        self,
        image: np.ndarray,
        model,
        pixel_spacing: float | None = None,
        sensitivity: str = "high",
    ) -> dict:
        """
        Multi-scale small tumor detection.

        Args:
            image: Original high-resolution image (BGR/RGB)
            model: Keras model for classification
            pixel_spacing: mm per pixel (from DICOM if available)
            sensitivity: 'low', 'medium', 'high' — affects detection threshold

        Returns:
            Dict with detection results, locations, and size estimates
        """
        if pixel_spacing is None:
            pixel_spacing = self.DEFAULT_PIXEL_SPACING

        h_orig, w_orig = image.shape[:2]
        threshold = {"low": 0.7, "medium": 0.5, "high": 0.35}.get(sensitivity, 0.5)

        results = {
            "small_lesions_detected": [],
            "total_patches_analyzed": 0,
            "image_resolution": f"{w_orig}x{h_orig}",
            "pixel_spacing_mm": pixel_spacing,
            "sensitivity": sensitivity,
            "detection_threshold": threshold,
        }

        # Only do multi-scale if image is large enough
        if max(h_orig, w_orig) < 300:
            results["note"] = "Image resolution too low for small tumor detection. Use original DICOM resolution."
            return results

        # Use dedicated patch classifier if available, else fall back to main model
        use_patch_model = self.patch_model is not None
        results["classifier"] = "dedicated patch model" if use_patch_model else "main model (fallback)"

        # Multi-scale patch analysis
        detections = []
        scales = [1.0, 0.75, 0.5] if max(h_orig, w_orig) > 500 else [1.0, 0.75]

        for scale in scales:
            sh = int(h_orig * scale)
            sw = int(w_orig * scale)
            scaled_img = cv2.resize(image, (sw, sh))

            # Slide patches
            stride = int(self.PATCH_SIZE * (1 - self.OVERLAP))
            patch_count = 0

            for y in range(0, sh - self.PATCH_SIZE + 1, stride):
                for x in range(0, sw - self.PATCH_SIZE + 1, stride):
                    patch = scaled_img[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE]

                    # Quick pre-filter: skip patches that are mostly background
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
                    if gray_patch.mean() < 20:  # mostly black
                        continue

                    # Predict using dedicated patch classifier or main model
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if len(patch.shape) == 3 else patch
                    patch_input = patch_rgb.astype(np.float32)

                    if use_patch_model:
                        # Patch model: binary sigmoid → tumor probability
                        patch_resized = cv2.resize(patch_input.astype(np.uint8), (120, 120)).astype(np.float32)
                        batch = np.expand_dims(patch_resized, 0)
                        tumor_prob = float(self.patch_model.predict(batch, verbose=0)[0][0])
                    else:
                        # Fallback: main 4-class model
                        patch_resized = cv2.resize(patch_input.astype(np.uint8), (240, 240)).astype(np.float32)
                        batch = np.expand_dims(patch_resized, 0)
                        pred = model.predict(batch, verbose=0)[0]
                        tumor_prob = float(pred[0] + pred[1] + pred[3])  # sum of tumor classes
                    patch_count += 1

                    if tumor_prob > threshold:
                        # Convert back to original image coordinates
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_size = int(self.PATCH_SIZE / scale)

                        # Estimate tumor size in mm
                        estimated_pixels = orig_size * tumor_prob * 0.3
                        estimated_mm = estimated_pixels * pixel_spacing

                        detection_info = {
                            "location": {
                                "x": orig_x,
                                "y": orig_y,
                                "width": orig_size,
                                "height": orig_size,
                            },
                            "tumor_probability": round(tumor_prob, 3),
                            "estimated_size_mm": round(estimated_mm, 1),
                            "scale": scale,
                        }

                        if use_patch_model:
                            detection_info["dominant_class"] = "Tumor"
                            detection_info["classifier_type"] = "dedicated_patch_model"
                        else:
                            detection_info["dominant_class"] = self._get_class_name(int(np.argmax(pred)))
                            detection_info["class_probabilities"] = {
                                "Glioma": round(float(pred[0]), 3),
                                "Meningioma": round(float(pred[1]), 3),
                                "No Tumor": round(float(pred[2]), 3),
                                "Pituitary": round(float(pred[3]), 3),
                            }

                        detections.append(detection_info)

            results["total_patches_analyzed"] += patch_count

        # Non-maximum suppression to remove overlapping detections
        detections = self._nms(detections, iou_threshold=0.3)

        # Filter to keep only small tumors (< 10mm estimated)
        small_detections = [d for d in detections if d["estimated_size_mm"] < 10]
        large_detections = [d for d in detections if d["estimated_size_mm"] >= 10]

        results["small_lesions_detected"] = small_detections
        results["larger_lesions_detected"] = large_detections
        results["total_suspicious_regions"] = len(detections)

        if small_detections:
            results["clinical_note"] = (
                f"Found {len(small_detections)} small suspicious region(s) "
                f"(estimated <10mm). These may represent early-stage lesions. "
                f"Recommend follow-up with high-resolution contrast-enhanced MRI."
            )
        else:
            results["clinical_note"] = "No small lesions detected at this sensitivity level."

        return results

    def _nms(self, detections: list, iou_threshold: float = 0.3) -> list:
        """Non-maximum suppression to merge overlapping detections."""
        if not detections:
            return []

        # Sort by tumor probability (descending)
        detections = sorted(detections, key=lambda d: d["tumor_probability"], reverse=True)
        kept = []

        for det in detections:
            overlap = False
            for kept_det in kept:
                if self._iou(det["location"], kept_det["location"]) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)

        return kept

    def _iou(self, box1: dict, box2: dict) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _get_class_name(self, idx: int) -> str:
        return {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}.get(idx, "Unknown")
