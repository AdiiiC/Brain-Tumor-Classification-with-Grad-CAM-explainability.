"""Segmentation mask handling and volumetry maths."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from api.segmentation_service import SegmentationService


@pytest.fixture
def service():
    return SegmentationService(unet_path="__no_such_unet__.keras")


@pytest.fixture
def heatmap():
    """JET-coloured activation with a hot blob in the centre."""
    cam = np.zeros((240, 240), dtype=np.float32)
    cv2.circle(cam, (120, 120), 30, 1.0, -1)
    coloured = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


@pytest.fixture
def scan():
    img = np.zeros((240, 240, 3), dtype=np.float32)
    cv2.circle(img, (120, 120), 90, (120, 120, 120), -1)
    return img


class TestFallbackMode:
    def test_uses_gradcam_when_no_unet(self, service):
        assert service.is_trained_model is False

    def test_requires_heatmap_without_model(self, service, scan):
        with pytest.raises(ValueError):
            service.segment(img=scan, heatmap_rgb=None)

    def test_marks_result_approximate(self, service, scan, heatmap):
        result = service.segment(img=scan, heatmap_rgb=heatmap)
        assert result["approximate"] is True
        assert result["method"] == "gradcam_threshold"

    def test_produces_non_empty_mask(self, service, scan, heatmap):
        result = service.segment(img=scan, heatmap_rgb=heatmap)
        assert result["measurements"]["area_pixels"] > 0

    def test_outputs_decodable_png(self, service, scan, heatmap):
        result = service.segment(img=scan, heatmap_rgb=heatmap)
        for key in ("mask_base64", "overlay_base64"):
            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(result[key]), np.uint8), cv2.IMREAD_COLOR,
            )
            assert decoded is not None


class TestMeasurements:
    def test_area_matches_pixel_count(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        assert SegmentationService.measure(mask, 1.0, 1.0)["area_pixels"] == 100

    def test_volume_uses_slice_thickness(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        result = SegmentationService.measure(mask, pixel_spacing_mm=1.0, slice_thickness_mm=5.0)
        assert result["volume_mm3"] == pytest.approx(500.0)

    def test_spacing_scales_area_quadratically(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        one = SegmentationService.measure(mask, 1.0, 1.0)["area_mm2"]
        two = SegmentationService.measure(mask, 2.0, 1.0)["area_mm2"]
        assert two == pytest.approx(one * 4)

    def test_cm3_conversion(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        result = SegmentationService.measure(mask, 1.0, 1.0)
        assert result["volume_cm3"] == pytest.approx(result["volume_mm3"] / 1000)

    def test_empty_mask_is_zero(self):
        result = SegmentationService.measure(np.zeros((50, 50), dtype=np.uint8), 1.0, 5.0)
        assert result["area_pixels"] == 0
        assert result["volume_mm3"] == 0.0
        assert result["max_diameter_mm"] == 0.0

    def test_diameter_approximates_circle(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 25, 1, -1)
        diameter = SegmentationService.measure(mask, 1.0, 1.0)["max_diameter_mm"]
        assert diameter == pytest.approx(50, abs=4)

    def test_rescales_spacing_to_source_resolution(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        native = SegmentationService.measure(mask, 1.0, 1.0)
        upscaled = SegmentationService.measure(mask, 1.0, 1.0, original_shape=(480, 480))
        assert upscaled["area_mm2"] == pytest.approx(native["area_mm2"] * 4)


class TestLargestComponent:
    def test_discards_speckle(self, service, scan):
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(mask, (60, 60), 20, 1, -1)
        mask[200, 200] = 1
        cleaned = SegmentationService._largest_component(mask)
        assert cleaned[200, 200] == 0
        assert cleaned.sum() > 100

    def test_handles_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert SegmentationService._largest_component(mask).sum() == 0
