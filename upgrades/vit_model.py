"""
Upgrade #1 — Vision Transformer (ViT) for Brain Tumor Classification.

Replaces EfficientNet with a ViT backbone. Transformers capture long-range
spatial dependencies in MRI scans through self-attention, often outperforming
CNNs on medical imaging tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ── ViT Components ─────────────────────────────────────────────────────────────

class PatchEmbedding(layers.Layer):
    """Split image into patches and embed them."""

    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size
        )
        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        return x


class TransformerBlock(layers.Layer):
    """Multi-Head Self-Attention + MLP block."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, norm_x, training=training)
        x = x + self.drop1(attn_out, training=training)
        # MLP
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def build_vit_model(
    img_size=240,
    patch_size=16,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    mlp_dim=512,
    num_classes=4,
    dropout=0.1,
) -> Model:
    """
    Build a Vision Transformer for brain tumor classification.

    Architecture:
        Input → Patch Embedding → [CLS] token → Positional Embedding
        → N × Transformer Blocks → [CLS] → MLP Head → 4 classes
    """
    num_patches = (img_size // patch_size) ** 2

    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Patch embedding
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    # [CLS] token
    cls_token = tf.Variable(tf.zeros([1, 1, embed_dim]), trainable=True, name="cls_token")
    cls_tokens = tf.repeat(cls_token, tf.shape(x)[0], axis=0)
    x = tf.concat([cls_tokens, x], axis=1)

    # Positional embedding
    pos_embed = tf.Variable(
        tf.random.normal([1, num_patches + 1, embed_dim], stddev=0.02),
        trainable=True, name="pos_embed"
    )
    x = x + pos_embed
    x = layers.Dropout(dropout)(x)

    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f"transformer_{i}")(x)

    # Classification head (use [CLS] token)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_output = x[:, 0]  # [CLS] token
    x = layers.Dense(128, activation="gelu")(cls_output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return Model(inputs, outputs, name="ViT_BrainTumor")


def train_vit(train_data, val_data, epochs=50, lr=1e-4):
    """Train the ViT model with cosine decay schedule."""
    model = build_vit_model()

    # Cosine decay with warmup
    total_steps = epochs * len(train_data)
    warmup_steps = total_steps // 10
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint("model_vit_best.keras", monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=10, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, mode="max"),
    ]

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
    )
    return model, history
