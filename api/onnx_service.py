"""
ONNX Runtime inference backend.

Serving from ONNX drops TensorFlow from the runtime image (~1 GB) and cuts cold start,
which matters on Render's free tier where the container is evicted when idle.

Scope: this backend serves classification only. Grad-CAM++ needs gradients with respect
to intermediate activations, which ONNX Runtime does not provide — explainability
endpoints stay on the Keras backend and return 400 here rather than silently degrading
to a different attribution method.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from api.preprocessing import preprocess_image, softmax

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where onnxruntime is absent
    ort = None
    ONNX_AVAILABLE = False

# Exported graphs expose logits when available; probabilities are the fallback.
_LOGIT_OUTPUT_NAMES = ("logits", "logit", "pre_softmax")


class OnnxModelService:
    """Drop-in replacement for ModelService covering the classification endpoints."""

    def __init__(self, onnx_path: str = "brain_tumor_model.onnx", threads: int | None = None):
        self.session = None
        self.model_type = "none"
        self._calibration_temp = 1.0
        self._input_name = None
        self._prob_output = None
        self._logit_output = None

        path = Path(onnx_path)
        if not ONNX_AVAILABLE or not path.exists():
            return

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads

        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.model_type = "onnx"
        self._input_name = self.session.get_inputs()[0].name

        outputs = [o.name for o in self.session.get_outputs()]
        self._logit_output = next(
            (o for o in outputs if o.lower() in _LOGIT_OUTPUT_NAMES), None
        )
        self._prob_output = next((o for o in outputs if o != self._logit_output), outputs[0])

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    @property
    def has_logits(self) -> bool:
        """Temperature scaling and energy-based OOD are only meaningful with logits."""
        return self._logit_output is not None

    def set_calibration_temperature(self, temp: float):
        self._calibration_temp = temp

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        return preprocess_image(image_bytes)

    def _run(self, img: np.ndarray, output: str) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("No ONNX model loaded")
        batch = np.expand_dims(img, 0).astype(np.float32)
        return self.session.run([output], {self._input_name: batch})[0][0].astype(np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self._run(img, self._prob_output)

    def predict_logits(self, img: np.ndarray) -> np.ndarray:
        """
        Pre-softmax logits.

        When the exported graph only emits probabilities we return log-probabilities.
        Those are shifted logits: argmax and temperature ordering survive, but the
        free-energy OOD score does not, because log-probabilities always sum-exp to 1.
        Export with logits to keep OOD detection meaningful.
        """
        if self._logit_output is not None:
            return self._run(img, self._logit_output)
        probs = self.predict(img)
        return np.log(np.clip(probs, 1e-12, None)).astype(np.float32)

    def predict_calibrated(self, img: np.ndarray) -> np.ndarray:
        if self._logit_output is None:
            return self.predict(img)
        return softmax(self.predict_logits(img), self._calibration_temp)

    def extract_features(self, img: np.ndarray) -> np.ndarray | None:
        """Penultimate embeddings are not exported, so Mahalanobis OOD is unavailable."""
        return None

    def predict_with_uncertainty(self, img: np.ndarray, n_iter: int = 50):
        """MC Dropout needs stochastic forward passes; ONNX graphs are frozen."""
        probs = self.predict(img)
        return probs, np.zeros_like(probs)

    def predict_with_tta(self, img: np.ndarray, n_augments: int = 10) -> np.ndarray:
        """Deterministic flip/rotate TTA, avoiding the Keras augmentation pipeline."""
        views = [img, cv2.flip(img, 1)]
        for angle in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            views.append(cv2.resize(cv2.rotate(img, angle), (img.shape[1], img.shape[0])))
        return np.mean([self.predict(v) for v in views[:max(1, n_augments)]], axis=0).astype(np.float32)

    def gradcam_plus_plus(self, img: np.ndarray):
        raise RuntimeError("Grad-CAM++ requires the Keras backend")

    @staticmethod
    def encode_image(img_rgb: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")
