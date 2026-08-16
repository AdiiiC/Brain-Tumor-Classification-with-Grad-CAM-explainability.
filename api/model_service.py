"""
Model service: loading, inference, Grad-CAM++, MC Dropout, TTA.

Supports Keras (.keras) and TFLite (.tflite) models.
"""

import base64
from pathlib import Path

import cv2
import numpy as np

from api.preprocessing import IMG_SIZE, preprocess_image, softmax

# TensorFlow is optional so the app (and its test suite) can be imported without
# it. Endpoints that need a model return 503 when it is unavailable.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True

    class _AlwaysOnDropout(tf.keras.layers.Dropout):
        """Dropout that keeps sampling at inference time (Monte-Carlo dropout)."""

        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

except ImportError:  # pragma: no cover - exercised only in TF-less environments
    tf = None
    Model = load_model = ImageDataGenerator = None
    _AlwaysOnDropout = None
    TF_AVAILABLE = False

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

__all__ = ["CLASS_NAMES", "IMG_SIZE", "TF_AVAILABLE", "ModelService"]


class ModelService:
    def __init__(self, model_path: str = "model_best.keras", tflite_path: str | None = None):
        self.model = None
        self.interpreter = None
        self.model_type = "none"
        self._calibration_temp = 1.0  # temperature scaling (upgrade #9)
        self._logit_model = None
        self._feature_model = None
        self._mc_model = None

        if not TF_AVAILABLE:
            return

        model_p = Path(model_path)
        tflite_p = Path(tflite_path) if tflite_path else None

        if model_p.exists():
            self.model = load_model(str(model_p))
            self.model_type = "keras"
        elif tflite_p and tflite_p.exists():
            self.interpreter = tf.lite.Interpreter(model_path=str(tflite_p))
            self.interpreter.allocate_tensors()
            self.model_type = "tflite"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None or self.interpreter is not None

    def set_calibration_temperature(self, temp: float):
        """Set temperature for confidence calibration (upgrade #9)."""
        self._calibration_temp = temp

    # ── Preprocessing ─────────────────────────────────────────────────────

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode, crop, CLAHE-enhance, resize to 240×240."""
        return preprocess_image(image_bytes)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Raw softmax prediction."""
        batch = np.expand_dims(img, 0)
        if self.model_type == "keras":
            return self.model.predict(batch, verbose=0)[0].astype(np.float32)
        elif self.model_type == "tflite":
            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]
            self.interpreter.set_tensor(inp["index"], batch)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(out["index"])[0].astype(np.float32)
        raise RuntimeError("No model loaded")

    def _build_logit_model(self):
        """Sub-model returning pre-softmax activations of the final dense layer."""
        if self._logit_model is None:
            final = self.model.layers[-1]
            penultimate = self.model.layers[-2].output
            logits = tf.keras.layers.Dense(
                final.units, activation=None, name="logits",
            )(penultimate)
            logit_model = Model(self.model.inputs, logits)
            logit_model.get_layer("logits").set_weights(final.get_weights())
            self._logit_model = logit_model
        return self._logit_model

    def predict_logits(self, img: np.ndarray) -> np.ndarray:
        """Pre-softmax logits. Falls back to log-probabilities for TFLite."""
        if self.model_type != "keras":
            probs = self.predict(img)
            return np.log(np.clip(probs, 1e-12, None)).astype(np.float32)
        batch = np.expand_dims(img, 0)
        return self._build_logit_model().predict(batch, verbose=0)[0].astype(np.float32)

    def extract_features(self, img: np.ndarray) -> np.ndarray | None:
        """Penultimate-layer embedding, used for OOD scoring."""
        if self.model_type != "keras":
            return None
        if self._feature_model is None:
            self._feature_model = Model(self.model.inputs, self.model.layers[-2].output)
        batch = np.expand_dims(img, 0)
        return self._feature_model.predict(batch, verbose=0)[0].astype(np.float32).ravel()

    def predict_calibrated(self, img: np.ndarray) -> np.ndarray:
        """Prediction with temperature scaling (upgrade #9)."""
        if self.model_type != "keras":
            return self.predict(img)
        return softmax(self.predict_logits(img), self._calibration_temp)

    def _build_mc_model(self):
        """Clone of the model with dropout left permanently active.

        Calling the original model with ``training=True`` would also flip every
        BatchNormalization layer to batch statistics; on a single image that
        collapses the output to a near-uniform distribution.
        """
        if self._mc_model is None:
            def clone_layer(layer):
                if isinstance(layer, tf.keras.layers.Dropout):
                    return _AlwaysOnDropout.from_config(layer.get_config())
                return layer.__class__.from_config(layer.get_config())

            mc_model = tf.keras.models.clone_model(self.model, clone_function=clone_layer)
            mc_model.set_weights(self.model.get_weights())
            self._mc_model = mc_model
        return self._mc_model

    def predict_with_uncertainty(self, img: np.ndarray, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo Dropout uncertainty estimation."""
        if self.model_type != "keras":
            probs = self.predict(img)
            return probs, np.zeros_like(probs)
        batch = np.expand_dims(img, 0)
        mc_model = self._build_mc_model()
        preds = np.array([
            mc_model(batch, training=False).numpy()[0]
            for _ in range(n_iter)
        ], dtype=np.float32)
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_with_tta(self, img: np.ndarray, n_augments: int = 10) -> np.ndarray:
        """Test-Time Augmentation."""
        augmenter = ImageDataGenerator(
            rotation_range=10, zoom_range=0.1,
            horizontal_flip=True, brightness_range=[0.9, 1.1],
        )
        batch = np.expand_dims(img, 0)
        preds = [self.predict(next(augmenter.flow(batch, batch_size=1))[0])
                 for _ in range(n_augments)]
        return np.mean(preds, axis=0)

    # ── Grad-CAM++ ────────────────────────────────────────────────────────

    def gradcam_plus_plus(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Returns (heatmap_rgb, overlay_rgb, predicted_class_index).
        """
        if self.model_type != "keras":
            raise RuntimeError("Grad-CAM++ requires a Keras model")

        # Find last conv layer
        last_conv = next(
            x for x in self.model.layers[::-1]
            if isinstance(x, tf.keras.layers.Conv2D)
        )
        # `model.outputs[0]` (not `model.output`) — Keras 3 returns the output
        # struct, which is a list even for single-output models.
        grad_model = Model(self.model.inputs, [last_conv.output, self.model.outputs[0]])
        batch = np.expand_dims(img, 0).astype("float32")

        with tf.GradientTape() as tape3:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:
                    conv_out, preds = grad_model(batch)
                    pred_idx = tf.argmax(preds[0])
                    loss = preds[:, pred_idx]
                grads1 = tape1.gradient(loss, conv_out)
            grads2 = tape2.gradient(grads1, conv_out)
        grads3 = tape3.gradient(grads2, conv_out)

        global_sum = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
        alpha_denom = 2.0 * grads2 + global_sum * grads3 + 1e-7
        alpha = grads2 / alpha_denom
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads1), axis=(1, 2))[0]

        cam = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy().astype(np.float32)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig = np.uint8((img - img.min()) / (img.max() - img.min() + 1e-7) * 255)
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        overlay = np.uint8(orig * 0.5 + heatmap_resized * 0.5)

        return heatmap, overlay, int(pred_idx)

    # ── Encoding helpers ──────────────────────────────────────────────────

    @staticmethod
    def encode_image(img_rgb: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", img_bgr)
        return base64.b64encode(buf).decode("utf-8")
