"""
Confidence Calibration — Upgrade #9.

Implements temperature scaling: a simple post-hoc calibration method.
After training, a single temperature parameter T is optimized on the
validation set so that confidence scores match actual accuracy.

Usage:
    calibrator = TemperatureCalibrator()
    calibrator.fit(model, val_data)
    T = calibrator.temperature  # apply via model_service.set_calibration_temperature(T)
"""

import numpy as np

try:
    import tensorflow as tf
    from scipy.optimize import minimize_scalar
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class TemperatureCalibrator:
    """
    Temperature Scaling for neural network calibration.

    A perfectly calibrated model means: when it says "90% confidence",
    it is correct 90% of the time. Temperature scaling achieves this
    by dividing logits by T before softmax.
    """

    def __init__(self):
        self.temperature = 1.0
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def fit(self, model, val_generator, max_samples: int = 2000) -> float:
        """
        Optimize temperature T on validation data using NLL loss.

        Returns the optimal temperature value.
        """
        if not DEPS_AVAILABLE:
            raise ImportError("tensorflow and scipy required for calibration")

        # Collect logits and labels from validation set
        logits_list = []
        labels_list = []

        # Build logit model (remove final softmax)
        logit_model = tf.keras.Model(model.input, model.layers[-1].output)

        samples = 0
        for x_batch, y_batch in val_generator:
            logits = logit_model.predict(x_batch, verbose=0)
            logits_list.append(logits)
            labels_list.append(np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch)
            samples += len(x_batch)
            if samples >= max_samples:
                break

        all_logits = np.concatenate(logits_list, axis=0).astype(np.float64)
        all_labels = np.concatenate(labels_list, axis=0).astype(np.int64)

        def nll_loss(T):
            """Negative log-likelihood with temperature T."""
            scaled = all_logits / T
            # Stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
            # Cross-entropy
            correct_probs = probs[np.arange(len(all_labels)), all_labels]
            return -np.log(correct_probs + 1e-12).mean()

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self._is_calibrated = True
        return self.temperature

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits and return calibrated probabilities."""
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error — measures how well-calibrated a model is.

    Returns a float between 0 (perfect) and 1 (worst).
    ECE < 0.05 is considered well-calibrated.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.sum() * abs(bin_accuracy - bin_confidence)

    return ece / len(labels)
