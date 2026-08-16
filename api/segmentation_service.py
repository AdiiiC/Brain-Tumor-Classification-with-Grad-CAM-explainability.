"""
Tumor segmentation and volumetry.

Produces a pixel-level tumor mask plus physical measurements. "Tumor is
3.2 cm3" is far more actionable than a heatmap blob, and a volume series is
what longitudinal progression tracking is built on.

Uses the trained U-Net from upgrades/segmentation.py when weights are present.
Without weights it falls back to thresholding the Grad-CAM++ activation, which
is approximate and labelled as such in the response.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np

UNET_PATH = os.getenv("UNET_PATH", "unet_segmentation.keras")

# Assumed acquisition geometry when DICOM does not supply it.
DEFAULT_PIXEL_SPACING_MM = 1.0
DEFAULT_SLICE_THICKNESS_MM = 5.0


class SegmentationService:
    def __init__(self, unet_path: str = UNET_PATH):
        self.model = None
        self.source = "gradcam_fallback"

        weights = Path(unet_path)
        if weights.exists():
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(str(weights), compile=False)
                self.source = "unet"
            except Exception:
                self.model = None

    @property
    def is_trained_model(self) -> bool:
        return self.model is not None

    # ── Mask generation ───────────────────────────────────────────────────

    def _mask_from_unet(self, img: np.ndarray, threshold: float) -> np.ndarray:
        batch = np.expand_dims(img, 0).astype("float32")
        prob = self.model.predict(batch, verbose=0)[0]
        if prob.ndim == 3 and prob.shape[-1] > 1:
            prob = prob[..., 1:].max(axis=-1)
        prob = np.squeeze(prob)
        return (prob >= threshold).astype(np.uint8)

    def _mask_from_heatmap(self, heatmap_rgb: np.ndarray, threshold: float) -> np.ndarray:
        """
        Derive an approximate mask from a Grad-CAM++ overlay.

        The heatmap is JET-coloured, so red channel dominance tracks activation.
        """
        gray = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gray.max() > gray.min():
            gray = (gray - gray.min()) / (gray.max() - gray.min())
        mask = (gray >= threshold).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the dominant lesion, discarding speckle."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest).astype(np.uint8)

    # ── Measurement ───────────────────────────────────────────────────────

    @staticmethod
    def measure(
        mask: np.ndarray,
        pixel_spacing_mm: float,
        slice_thickness_mm: float,
        original_shape: tuple[int, int] | None = None,
    ) -> dict:
        """Convert a binary mask into physical measurements."""
        area_px = int(mask.sum())

        # Mask is computed at model resolution; rescale spacing to the source image.
        spacing = pixel_spacing_mm
        if original_shape and mask.shape[0] > 0:
            spacing = pixel_spacing_mm * (original_shape[0] / mask.shape[0])

        area_mm2 = area_px * (spacing ** 2)
        volume_mm3 = area_mm2 * slice_thickness_mm

        max_diameter_mm = 0.0
        if area_px > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                (_, _), radius = cv2.minEnclosingCircle(largest)
                max_diameter_mm = float(2 * radius * spacing)

        return {
            "area_pixels": area_px,
            "area_mm2": round(float(area_mm2), 2),
            "volume_mm3": round(float(volume_mm3), 2),
            "volume_cm3": round(float(volume_mm3) / 1000.0, 3),
            "max_diameter_mm": round(max_diameter_mm, 2),
            "pixel_spacing_mm": round(float(spacing), 4),
            "slice_thickness_mm": float(slice_thickness_mm),
        }

    # ── Public entry point ────────────────────────────────────────────────

    def segment(
        self,
        img: np.ndarray,
        heatmap_rgb: np.ndarray | None = None,
        pixel_spacing_mm: float | None = None,
        slice_thickness_mm: float | None = None,
        threshold: float = 0.5,
        original_shape: tuple[int, int] | None = None,
    ) -> dict:
        """
        Segment the tumor and measure it.

        Supply heatmap_rgb to enable the fallback path when no U-Net is loaded.
        """
        spacing = pixel_spacing_mm or DEFAULT_PIXEL_SPACING_MM
        thickness = slice_thickness_mm or DEFAULT_SLICE_THICKNESS_MM
        spacing_estimated = pixel_spacing_mm is None

        if self.model is not None:
            mask = self._mask_from_unet(img, threshold)
            method = "unet"
        elif heatmap_rgb is not None:
            mask = self._mask_from_heatmap(heatmap_rgb, threshold)
            method = "gradcam_threshold"
        else:
            raise ValueError("No segmentation model loaded and no heatmap supplied")

        mask = self._largest_component(mask)
        measurements = self.measure(mask, spacing, thickness, original_shape)

        return {
            "method": method,
            "approximate": method != "unet",
            "spacing_estimated": spacing_estimated,
            "measurements": measurements,
            "mask_base64": self.encode_mask(mask),
            "overlay_base64": self.encode_overlay(img, mask),
            "note": (
                "Mask derived from Grad-CAM++ activation — an approximation suitable "
                "for relative trend tracking, not for surgical planning."
                if method != "unet" else
                "Mask produced by the trained U-Net segmentation model."
            ),
        }

    # ── Rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def encode_mask(mask: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def encode_overlay(img: np.ndarray, mask: np.ndarray) -> str:
        """Green contour of the mask drawn over the scan."""
        base = img.astype(np.float32)
        if base.max() > base.min():
            base = (base - base.min()) / (base.max() - base.min())
        base = (base * 255).astype(np.uint8)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        resized = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = base.copy()
        overlay[resized > 0] = (0.5 * overlay[resized > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

        contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")
