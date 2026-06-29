"""
Model architecture: EfficientNetB1 with transfer learning.
Corresponds to Notebook Cells 26-27.
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization)

from config import IMG_SIZE


def build_model(num_classes=4, freeze_base=True):
    """
    Build EfficientNetB1 model with custom classification head.
    
    Args:
        num_classes: Number of output classes.
        freeze_base: If True, freeze all base layers (Phase 1).
    
    Returns:
        Compiled Keras Model.
    """
    effenet = EfficientNetB1(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    if freeze_base:
        for layer in effenet.layers:
            layer.trainable = False

    x = effenet.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    # dtype='float32' required when using mixed_float16 policy
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=effenet.input, outputs=outputs)
    return model, effenet


def unfreeze_top_layers(effenet, n_layers=30):
    """Unfreeze top n layers of base model for fine-tuning (Phase 2)."""
    for layer in effenet.layers[-n_layers:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True


if __name__ == "__main__":
    model, effenet = build_model()
    model.summary()
