"""
Image preprocessing shared by every inference backend.

Kept separate from the model services so the TensorFlow and ONNX paths cannot drift
apart — identical pixels must reach whichever backend is serving, or their predictions
are not comparable.
"""

from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = (240, 240)


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # noqa: E741 - OpenCV channel naming
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def crop_brain(image: np.ndarray) -> np.ndarray:
    """Crop to the largest bright contour, discarding black scanner border."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = image[y:y + h, x:x + w]
    return cropped if cropped.size > 0 else image


def preprocess_array(image_bgr: np.ndarray) -> np.ndarray:
    """Decoded BGR image → 240×240 RGB float32 ready for the network."""
    img = clahe_enhance(image_bgr)
    img = crop_brain(img)
    img = cv2.resize(img, IMG_SIZE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Encoded bytes → 240×240 RGB float32."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return preprocess_array(img)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature scaling."""
    scaled = np.asarray(logits, dtype=np.float32) / temperature
    exp = np.exp(scaled - np.max(scaled))
    return (exp / exp.sum()).astype(np.float32)
