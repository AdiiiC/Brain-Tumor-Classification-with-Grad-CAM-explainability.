"""
Image Quality Assessment module.

Pre-inference quality checks:
- Resolution adequacy
- Noise level (SNR)
- Blur detection (Laplacian variance)
- Compression artifact detection
- Brain region coverage
"""


import cv2
import numpy as np


class ImageQualityAssessor:
    """Assess MRI image quality before model inference."""

    # Minimum acceptable values
    MIN_RESOLUTION = 100  # pixels per dimension
    MIN_LAPLACIAN_VAR = 30.0  # blur threshold
    MIN_SNR = 5.0  # signal-to-noise ratio
    MIN_BRAIN_COVERAGE = 0.15  # brain should cover 15%+ of image

    def assess(self, image: np.ndarray) -> dict:
        """
        Run all quality checks on an image (BGR or RGB uint8).

        Returns dict with:
            - overall_score: 0-100
            - pass: bool
            - issues: list of detected problems
            - metrics: detailed measurements
            - recommendations: list of suggestions
        """
        issues = []
        recommendations = []
        metrics = {}

        # 1. Resolution check
        h, w = image.shape[:2]
        metrics["resolution"] = f"{w}x{h}"
        metrics["resolution_adequate"] = bool(min(w, h) >= self.MIN_RESOLUTION)
        if not metrics["resolution_adequate"]:
            issues.append(f"Low resolution ({w}x{h}). Minimum recommended: {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}")
            recommendations.append("Re-acquire scan at higher resolution or use original DICOM files")

        # 2. Blur detection (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["sharpness"] = round(float(laplacian_var), 2)
        metrics["is_blurry"] = bool(laplacian_var < self.MIN_LAPLACIAN_VAR)
        if metrics["is_blurry"]:
            issues.append(f"Image appears blurry (sharpness={laplacian_var:.1f}, min={self.MIN_LAPLACIAN_VAR})")
            recommendations.append("Use original uncompressed scan or check for motion artifacts")

        # 3. Noise estimation (SNR via median absolute deviation)
        snr = self._estimate_snr(gray)
        metrics["snr"] = round(float(snr), 2)
        metrics["noisy"] = bool(snr < self.MIN_SNR)
        if metrics["noisy"]:
            issues.append(f"High noise level detected (SNR={snr:.1f}, min={self.MIN_SNR})")
            recommendations.append("Check scanner calibration or use denoising preprocessing")

        # 4. Compression artifacts (JPEG block detection)
        compression_score = self._detect_compression_artifacts(gray)
        metrics["compression_artifact_score"] = round(float(compression_score), 2)
        metrics["heavily_compressed"] = bool(compression_score > 0.5)
        if metrics["heavily_compressed"]:
            issues.append("Significant compression artifacts detected")
            recommendations.append("Use lossless image format (PNG, TIFF, or DICOM)")

        # 5. Brain coverage (is there actually brain tissue?)
        brain_coverage = self._estimate_brain_coverage(gray)
        metrics["brain_coverage"] = round(float(brain_coverage), 3)
        metrics["sufficient_brain"] = bool(brain_coverage > self.MIN_BRAIN_COVERAGE)
        if not metrics["sufficient_brain"]:
            issues.append(f"Insufficient brain tissue visible ({brain_coverage*100:.1f}% coverage)")
            recommendations.append("Ensure the scan shows brain parenchyma, not just skull or background")

        # 6. Dynamic range
        dynamic_range = float(gray.max()) - float(gray.min())
        metrics["dynamic_range"] = round(dynamic_range, 1)
        metrics["low_contrast"] = bool(dynamic_range < 50)
        if metrics["low_contrast"]:
            issues.append("Very low contrast — image may be washed out")
            recommendations.append("Check window/level settings or use CLAHE enhancement")

        # Overall quality score (0-100)
        score = 100.0
        if not metrics["resolution_adequate"]:
            score -= 25
        if metrics["is_blurry"]:
            score -= 20
        if metrics["noisy"]:
            score -= 15
        if metrics["heavily_compressed"]:
            score -= 15
        if not metrics["sufficient_brain"]:
            score -= 20
        if metrics["low_contrast"]:
            score -= 10

        score = max(0, min(100, score))

        return {
            "overall_score": round(score),
            "pass": score >= 50 and metrics["sufficient_brain"],
            "issues": issues,
            "metrics": metrics,
            "recommendations": recommendations,
            "confidence_impact": self._estimate_confidence_impact(score),
        }

    def _estimate_snr(self, gray: np.ndarray) -> float:
        """Estimate SNR using signal mean / noise std in background region."""
        # Use top-left corner as background estimate
        h, w = gray.shape
        bg_region = gray[:h//8, :w//8]
        signal_region = gray[h//4:3*h//4, w//4:3*w//4]

        bg_std = float(bg_region.std()) + 1e-7
        signal_mean = float(signal_region.mean())

        return signal_mean / bg_std

    def _detect_compression_artifacts(self, gray: np.ndarray) -> float:
        """Detect JPEG block artifacts via 8x8 grid edge analysis."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0

        # Check for 8x8 block boundaries (JPEG artifact)
        horizontal_edges = np.abs(np.diff(gray.astype(np.float32), axis=1))
        vertical_edges = np.abs(np.diff(gray.astype(np.float32), axis=0))

        # Energy at 8-pixel intervals vs overall
        h_block_energy = float(horizontal_edges[:, 7::8].mean()) if w > 8 else 0
        h_overall_energy = float(horizontal_edges.mean()) + 1e-7

        v_block_energy = float(vertical_edges[7::8, :].mean()) if h > 8 else 0
        v_overall_energy = float(vertical_edges.mean()) + 1e-7

        ratio = ((h_block_energy / h_overall_energy) + (v_block_energy / v_overall_energy)) / 2.0
        # Normalize: ratio near 1.0 = uniform edges (no artifacts), much > 1 = block artifacts
        return max(0, min(1, (ratio - 1.0) * 2))

    def _estimate_brain_coverage(self, gray: np.ndarray) -> float:
        """Estimate what fraction of the image contains brain tissue."""
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return float(thresh.sum() / 255) / (gray.shape[0] * gray.shape[1])

    def _estimate_confidence_impact(self, quality_score: float) -> str:
        """How much image quality might affect prediction confidence."""
        if quality_score >= 80:
            return "minimal — image quality is good"
        elif quality_score >= 60:
            return "moderate — some accuracy loss possible"
        elif quality_score >= 40:
            return "significant — results may be unreliable"
        else:
            return "severe — recommend re-acquisition before diagnosis"
