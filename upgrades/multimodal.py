"""
Upgrade #17 — Multi-Modal Fusion.

Combines MRI images with clinical metadata (patient age, symptoms, blood markers)
using a fusion architecture. Multi-modal inputs significantly improve diagnostic
accuracy, especially for ambiguous cases.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB1


def build_multimodal_model(
    img_size=240,
    num_clinical_features=12,
    num_classes=4,
    fusion_type="concat",  # "concat", "attention", "bilinear"
) -> Model:
    """
    Multi-modal model combining MRI images with clinical tabular data.

    Image branch: EfficientNetB1 features
    Clinical branch: MLP on tabular features
    Fusion: Concatenation / Cross-attention / Bilinear

    Clinical features could include:
    - Patient age, sex
    - Symptom duration
    - Headache severity (1-10)
    - Seizure history (binary)
    - Visual disturbances (binary)
    - Cognitive changes (binary)
    - Blood markers (e.g., GFAP, S100B levels)
    - Tumor location (from radiology report)
    - Edema presence (binary)
    """
    # ── Image Branch ──────────────────────────────────────────────────────
    img_input = layers.Input(shape=(img_size, img_size, 3), name="mri_image")

    backbone = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    img_features = backbone(img_input)
    img_features = layers.GlobalAveragePooling2D()(img_features)
    img_features = layers.BatchNormalization()(img_features)
    img_features = layers.Dense(256, activation="relu", name="img_embed")(img_features)
    img_features = layers.Dropout(0.3)(img_features)

    # ── Clinical Branch ───────────────────────────────────────────────────
    clinical_input = layers.Input(shape=(num_clinical_features,), name="clinical_data")

    clin_features = layers.Dense(64, activation="relu")(clinical_input)
    clin_features = layers.BatchNormalization()(clin_features)
    clin_features = layers.Dense(128, activation="relu")(clin_features)
    clin_features = layers.Dropout(0.2)(clin_features)
    clin_features = layers.Dense(64, activation="relu", name="clin_embed")(clin_features)

    # ── Fusion ────────────────────────────────────────────────────────────
    if fusion_type == "concat":
        fused = layers.Concatenate()([img_features, clin_features])
        fused = layers.Dense(256, activation="relu")(fused)
        fused = layers.Dropout(0.4)(fused)

    elif fusion_type == "attention":
        # Cross-attention: clinical features attend to image features
        img_expanded = layers.Reshape((1, 256))(img_features)
        clin_expanded = layers.Reshape((1, 64))(clin_features)

        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        attended = attn(query=clin_expanded, value=img_expanded, key=img_expanded)
        attended = layers.Flatten()(attended)
        fused = layers.Concatenate()([img_features, attended])
        fused = layers.Dense(256, activation="relu")(fused)
        fused = layers.Dropout(0.4)(fused)

    elif fusion_type == "bilinear":
        # Bilinear pooling: outer product of features
        img_proj = layers.Dense(64)(img_features)
        fused = layers.Multiply()([img_proj, clin_features])
        fused = layers.Dense(256, activation="relu")(fused)
        fused = layers.Dropout(0.4)(fused)

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    # ── Classification Head ───────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=[img_input, clinical_input], outputs=outputs, name=f"MultiModal_{fusion_type}")
    return model


def prepare_clinical_features(patient_data: dict) -> np.ndarray:
    """
    Convert patient clinical data to normalized feature vector.

    Expected keys:
        age, sex, symptom_duration_days, headache_severity,
        seizure_history, visual_disturbances, cognitive_changes,
        gfap_level, s100b_level, edema_present, tumor_size_mm,
        midline_shift
    """
    feature_order = [
        "age", "sex", "symptom_duration_days", "headache_severity",
        "seizure_history", "visual_disturbances", "cognitive_changes",
        "gfap_level", "s100b_level", "edema_present", "tumor_size_mm",
        "midline_shift",
    ]

    features = []
    for key in feature_order:
        val = patient_data.get(key, 0)
        if isinstance(val, bool):
            val = float(val)
        features.append(float(val))

    features = np.array(features, dtype=np.float32)

    # Normalize (approximate clinical ranges)
    normalization = {
        0: (0, 100),       # age
        1: (0, 1),         # sex (binary)
        2: (0, 365),       # symptom duration
        3: (0, 10),        # headache severity
        4: (0, 1),         # seizure
        5: (0, 1),         # visual
        6: (0, 1),         # cognitive
        7: (0, 10),        # GFAP (ng/mL)
        8: (0, 2),         # S100B (μg/L)
        9: (0, 1),         # edema
        10: (0, 100),      # tumor size mm
        11: (0, 20),       # midline shift mm
    }

    for i, (low, high) in normalization.items():
        if high > low:
            features[i] = (features[i] - low) / (high - low)

    return features
