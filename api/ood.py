"""
Out-of-distribution detection.

The classifier has a 4-way softmax, so a chest X-ray or a photo of a cat still
produces a confident tumor label. This module gates inference: anything that
does not look like the training distribution is rejected before a clinical
result is reported.

Two scores are combined:
  1. Mahalanobis distance over penultimate-layer features (needs fitting).
  2. Free energy over the logits (needs no fitting, works out of the box).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_STATS_PATH = os.getenv("OOD_STATS_PATH", "ood_stats.npz")

# Tuned on held-out in-distribution data; override per deployment.
DEFAULT_MAHALANOBIS_THRESHOLD = float(os.getenv("OOD_MAHALANOBIS_THRESHOLD", "120.0"))
DEFAULT_ENERGY_THRESHOLD = float(os.getenv("OOD_ENERGY_THRESHOLD", "-5.0"))


@dataclass
class OODResult:
    is_ood: bool
    score: float
    mahalanobis_distance: float | None
    energy: float
    method: str
    message: str

    def to_dict(self) -> dict:
        return {
            "is_out_of_distribution": self.is_ood,
            "score": round(self.score, 4),
            "mahalanobis_distance": (
                round(self.mahalanobis_distance, 3) if self.mahalanobis_distance is not None else None
            ),
            "energy": round(self.energy, 4),
            "method": self.method,
            "message": self.message,
        }


def free_energy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """
    Energy score: E(x) = -T * logsumexp(logits / T).

    In-distribution inputs sit at lower energy than OOD inputs.
    """
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    max_logit = np.max(scaled)
    lse = max_logit + np.log(np.sum(np.exp(scaled - max_logit)))
    return float(-temperature * lse)


class OODDetector:
    """
    Class-conditional Gaussian model over penultimate features.

    Fit on in-distribution training images, then score new inputs by the
    minimum Mahalanobis distance to any class centroid.
    """

    def __init__(
        self,
        stats_path: str = DEFAULT_STATS_PATH,
        mahalanobis_threshold: float = DEFAULT_MAHALANOBIS_THRESHOLD,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ):
        self.means: np.ndarray | None = None          # (n_classes, n_features)
        self.precision: np.ndarray | None = None      # (n_features, n_features)
        self.mahalanobis_threshold = mahalanobis_threshold
        self.energy_threshold = energy_threshold
        self._load(stats_path)

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Estimate per-class means and a shared covariance (tied-covariance LDA).

        features: (n_samples, n_features), labels: (n_samples,)
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels).ravel()
        if features.ndim != 2:
            raise ValueError("features must be 2-D (n_samples, n_features)")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same length")

        classes = np.unique(labels)
        means, centered = [], []
        for cls in classes:
            subset = features[labels == cls]
            mean = subset.mean(axis=0)
            means.append(mean)
            centered.append(subset - mean)

        self.means = np.vstack(means)
        pooled = np.vstack(centered)

        n_samples, n_features = pooled.shape
        cov = (pooled.T @ pooled) / max(1, n_samples - len(classes))
        # Shrinkage keeps the covariance invertible when features outnumber samples.
        shrinkage = 1e-3 * np.trace(cov) / n_features
        cov += np.eye(n_features) * max(shrinkage, 1e-6)
        self.precision = np.linalg.pinv(cov)

    def save(self, path: str = DEFAULT_STATS_PATH) -> None:
        if self.means is None or self.precision is None:
            raise RuntimeError("Nothing to save — call fit() first")
        np.savez_compressed(
            path,
            means=self.means,
            precision=self.precision,
            mahalanobis_threshold=self.mahalanobis_threshold,
            energy_threshold=self.energy_threshold,
        )

    def _load(self, path: str) -> None:
        stats_file = Path(path)
        if not stats_file.exists():
            return
        try:
            data = np.load(stats_file)
            self.means = data["means"]
            self.precision = data["precision"]
            if "mahalanobis_threshold" in data:
                self.mahalanobis_threshold = float(data["mahalanobis_threshold"])
            if "energy_threshold" in data:
                self.energy_threshold = float(data["energy_threshold"])
        except (OSError, KeyError, ValueError):
            self.means = None
            self.precision = None

    @property
    def is_fitted(self) -> bool:
        return self.means is not None and self.precision is not None

    # ── Scoring ───────────────────────────────────────────────────────────

    def mahalanobis(self, feature: np.ndarray) -> float:
        """Minimum Mahalanobis distance to any class centroid."""
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted")
        assert self.means is not None and self.precision is not None

        diff = self.means - np.asarray(feature, dtype=np.float64).ravel()
        distances = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return float(np.sqrt(max(0.0, float(np.min(distances)))))

    def score(self, logits: np.ndarray, feature: np.ndarray | None = None) -> OODResult:
        """Decide whether an input is out of distribution."""
        energy = free_energy(logits)

        if self.is_fitted and feature is not None:
            distance = self.mahalanobis(feature)
            is_ood = distance > self.mahalanobis_threshold
            normalised = min(1.0, distance / max(self.mahalanobis_threshold, 1e-6))
            return OODResult(
                is_ood=is_ood,
                score=normalised,
                mahalanobis_distance=distance,
                energy=energy,
                method="mahalanobis",
                message=(
                    "Input does not resemble the brain-MRI training distribution. "
                    "Classification suppressed."
                    if is_ood else "Input is consistent with the training distribution."
                ),
            )

        is_ood = energy > self.energy_threshold
        span = abs(self.energy_threshold) + 1e-6
        normalised = float(np.clip((energy - self.energy_threshold) / span + 0.5, 0.0, 1.0))
        return OODResult(
            is_ood=is_ood,
            score=normalised,
            mahalanobis_distance=None,
            energy=energy,
            method="energy",
            message=(
                "Input has abnormally high energy — likely not a brain MRI. "
                "Interpret the classification with caution."
                if is_ood else "Input energy is within the expected range."
            ),
        )
