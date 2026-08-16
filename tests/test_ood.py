"""Out-of-distribution scoring."""

from __future__ import annotations

import numpy as np
import pytest

from api.ood import OODDetector, free_energy


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def fitted_detector(rng):
    """Four tight clusters standing in for the four tumor classes."""
    centres = np.array([
        [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0],
    ])
    features = np.vstack([centre + rng.normal(0, 0.4, (80, 3)) for centre in centres])
    labels = np.repeat(np.arange(4), 80)

    detector = OODDetector(stats_path="__missing__.npz", mahalanobis_threshold=6.0)
    detector.fit(features, labels)
    return detector, centres


class TestFreeEnergy:
    def test_confident_logits_have_lower_energy(self):
        confident = free_energy(np.array([12.0, 0.0, 0.0, 0.0]))
        uniform = free_energy(np.array([0.1, 0.1, 0.1, 0.1]))
        assert confident < uniform

    def test_is_finite_for_large_logits(self):
        assert np.isfinite(free_energy(np.array([900.0, 0.0, 0.0, 0.0])))

    def test_temperature_scales_output(self):
        logits = np.array([3.0, 1.0, 0.5, 0.2])
        assert free_energy(logits, 1.0) != free_energy(logits, 2.0)


class TestFitting:
    def test_unfitted_detector(self):
        detector = OODDetector(stats_path="__missing__.npz")
        assert detector.is_fitted is False

    def test_fitting_sets_state(self, fitted_detector):
        detector, _ = fitted_detector
        assert detector.is_fitted
        assert detector.means.shape == (4, 3)

    def test_rejects_mismatched_lengths(self):
        detector = OODDetector(stats_path="__missing__.npz")
        with pytest.raises(ValueError):
            detector.fit(np.zeros((10, 3)), np.zeros(5))

    def test_rejects_1d_features(self):
        detector = OODDetector(stats_path="__missing__.npz")
        with pytest.raises(ValueError):
            detector.fit(np.zeros(10), np.zeros(10))

    def test_roundtrip_save_load(self, fitted_detector, tmp_path):
        detector, _ = fitted_detector
        path = tmp_path / "stats.npz"
        detector.save(str(path))

        reloaded = OODDetector(stats_path=str(path))
        assert reloaded.is_fitted
        np.testing.assert_allclose(reloaded.means, detector.means)

    def test_save_before_fit_raises(self, tmp_path):
        detector = OODDetector(stats_path="__missing__.npz")
        with pytest.raises(RuntimeError):
            detector.save(str(tmp_path / "x.npz"))


class TestScoring:
    def test_in_distribution_accepted(self, fitted_detector):
        detector, centres = fitted_detector
        result = detector.score(np.array([9.0, 0.1, 0.1, 0.1]), centres[0])
        assert result.is_ood is False
        assert result.method == "mahalanobis"

    def test_far_input_rejected(self, fitted_detector):
        detector, _ = fitted_detector
        result = detector.score(np.array([2.0, 1.9, 1.8, 1.7]), np.array([400.0, -300.0, 250.0]))
        assert result.is_ood is True
        assert result.mahalanobis_distance > detector.mahalanobis_threshold

    def test_falls_back_to_energy_without_features(self, fitted_detector):
        detector, _ = fitted_detector
        result = detector.score(np.array([5.0, 1.0, 0.5, 0.2]), None)
        assert result.method == "energy"

    def test_unfitted_uses_energy(self):
        detector = OODDetector(stats_path="__missing__.npz")
        result = detector.score(np.array([5.0, 1.0, 0.5, 0.2]), np.array([1.0, 2.0]))
        assert result.method == "energy"

    def test_score_is_normalised(self, fitted_detector):
        detector, centres = fitted_detector
        result = detector.score(np.array([9.0, 0.1, 0.1, 0.1]), centres[1])
        assert 0.0 <= result.score <= 1.0

    def test_serialises_to_dict(self, fitted_detector):
        detector, centres = fitted_detector
        payload = detector.score(np.array([9.0, 0.1, 0.1, 0.1]), centres[0]).to_dict()
        assert set(payload) == {
            "is_out_of_distribution", "score", "mahalanobis_distance",
            "energy", "method", "message",
        }
