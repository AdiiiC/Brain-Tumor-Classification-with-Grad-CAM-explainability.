"""
Model service: loading, inference, Grad-CAM++, MC Dropout, TTA.

Supports Keras (.keras) and TFLite (.tflite) models.
"""

import io
import base64
from typing import Optional
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
IMG_SIZE = (240, 240)


class ModelService:
    def __init__(self, model_path: str = "model_best.keras", tflite_path: Optional[str] = None):
        self.model = None
        self.interpreter = None
        self.model_type = "none"
        self._calibration_temp = 1.0  # temperature scaling (upgrade #9)

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
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        img = self._clahe_enhance(img)
        img = self._crop_brain(img)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def _clahe_enhance(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    def _crop_brain(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return image
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = image[y:y + h, x:x + w]
        return cropped if cropped.size > 0 else image

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

    def predict_calibrated(self, img: np.ndarray) -> np.ndarray:
        """Prediction with temperature scaling (upgrade #9)."""
        batch = np.expand_dims(img, 0)
        if self.model_type != "keras":
            return self.predict(img)

        # Get logits before softmax
        logit_model = Model(self.model.input, self.model.layers[-1].output)
        logits = logit_model.predict(batch, verbose=0)[0].astype(np.float32)
        scaled = logits / self._calibration_temp
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()

    def predict_with_uncertainty(self, img: np.ndarray, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo Dropout uncertainty estimation."""
        if self.model_type != "keras":
            probs = self.predict(img)
            return probs, np.zeros_like(probs)
        batch = np.expand_dims(img, 0)
        preds = np.array([
            self.model(batch, training=True).numpy()[0]
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
        grad_model = Model(self.model.inputs, [last_conv.output, self.model.output])
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
