"""
Upgrade #18 — Tumor Segmentation with U-Net.

Adds a segmentation branch to delineate exact tumor boundaries.
Surgeons need precise tumor masks for surgical planning.

Architecture: U-Net with EfficientNet encoder (shared with classification).
Output: Binary mask highlighting tumor region pixels.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB1


def build_unet(img_size=240, num_classes=1) -> Model:
    """
    U-Net segmentation model with EfficientNetB1 encoder.

    Uses skip connections from encoder layers to preserve spatial detail.
    Output: pixel-wise tumor probability map.

    num_classes=1 for binary (tumor/no-tumor)
    num_classes=4 for multi-class segmentation (by tumor type)
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # ── Encoder (EfficientNetB1) ──────────────────────────────────────────
    encoder = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # Extract skip connections at different resolutions
    skip_names = [
        "block2a_expand_activation",   # 120×120
        "block3a_expand_activation",   # 60×60
        "block4a_expand_activation",   # 30×30
        "block6a_expand_activation",   # 15×15
    ]

    skip_outputs = [encoder.get_layer(name).output for name in skip_names]
    encoder_output = encoder.output

    encoder_model = Model(inputs=encoder.input, outputs=[encoder_output] + skip_outputs)
    outputs = encoder_model(inputs)
    x = outputs[0]
    skips = outputs[1:]

    # ── Decoder ───────────────────────────────────────────────────────────
    decoder_filters = [256, 128, 64, 32]

    for i, filters in enumerate(decoder_filters):
        x = layers.UpSampling2D(size=(2, 2))(x)

        # Resize skip connection to match if needed
        skip = skips[-(i + 1)]
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            skip = layers.Resizing(x.shape[1], x.shape[2])(skip)

        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, filters)

    # Final upsampling to original size
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = _conv_block(x, 16)

    # Resize to exact input size if needed
    x = layers.Resizing(img_size, img_size)(x)

    # Output layer
    activation = "sigmoid" if num_classes == 1 else "softmax"
    mask_output = layers.Conv2D(num_classes, 1, activation=activation, name="segmentation_mask")(x)

    return Model(inputs, mask_output, name="UNet_BrainTumor")


def _conv_block(x, filters):
    """Double convolution block for U-Net decoder."""
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_joint_model(img_size=240) -> Model:
    """
    Joint classification + segmentation model.

    Shared encoder with two decoder heads:
    - Classification: tumor type (4 classes)
    - Segmentation: tumor mask (binary)
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Shared encoder
    encoder = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    features = encoder(inputs)

    # Classification head
    cls_pool = layers.GlobalAveragePooling2D()(features)
    cls_x = layers.Dense(256, activation="relu")(cls_pool)
    cls_x = layers.Dropout(0.4)(cls_x)
    cls_output = layers.Dense(4, activation="softmax", dtype="float32", name="classification")(cls_x)

    # Segmentation head (simplified decoder)
    seg_x = layers.Conv2D(128, 3, padding="same", activation="relu")(features)
    seg_x = layers.UpSampling2D(size=(2, 2))(seg_x)
    seg_x = layers.Conv2D(64, 3, padding="same", activation="relu")(seg_x)
    seg_x = layers.UpSampling2D(size=(2, 2))(seg_x)
    seg_x = layers.Conv2D(32, 3, padding="same", activation="relu")(seg_x)
    seg_x = layers.UpSampling2D(size=(4, 4))(seg_x)  # Upsample to ~240
    seg_x = layers.Resizing(img_size, img_size)(seg_x)
    seg_output = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation")(seg_x)

    return Model(inputs, [cls_output, seg_output], name="Joint_Cls_Seg")


# ── Loss Functions ────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss — better than BCE for imbalanced segmentation."""
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return 1 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    )


def combined_loss(y_true, y_pred):
    """BCE + Dice loss for robust segmentation training."""
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# ── Utilities ─────────────────────────────────────────────────────────────────

def compute_tumor_volume(mask: np.ndarray, pixel_spacing_mm: float = 1.0) -> float:
    """
    Compute tumor volume from segmentation mask.

    Returns volume in mm³ (or pixels if spacing unknown).
    """
    tumor_pixels = np.sum(mask > 0.5)
    volume = tumor_pixels * (pixel_spacing_mm ** 2)  # 2D approximation
    return float(volume)


def extract_tumor_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract tumor boundary contour from binary mask."""
    import cv2
    binary = (mask > 0.5).astype(np.uint8) * 255
    if binary.ndim == 3:
        binary = binary[:, :, 0]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(binary)
    cv2.drawContours(boundary, contours, -1, 255, 2)
    return boundary
