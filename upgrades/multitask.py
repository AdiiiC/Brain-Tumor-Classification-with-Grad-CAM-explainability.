"""
Upgrade #2 — Multi-Task Learning.

Shared backbone with multiple prediction heads:
- Head 1: Tumor class (Glioma, Meningioma, Pituitary, No Tumor)
- Head 2: Tumor grade (Low-grade vs High-grade) — when labels available
- Head 3: Tumor location (Frontal, Temporal, Parietal, etc.) — when labels available

Multi-task learning improves feature representation by forcing the model to
learn complementary signals from multiple objectives.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB1


def build_multitask_model(
    img_size=240,
    num_tumor_classes=4,
    num_grade_classes=3,  # low, high, unknown
    num_location_classes=6,  # frontal, temporal, parietal, occipital, cerebellum, other
) -> Model:
    """
    Multi-task EfficientNet model with 3 prediction heads.

    The shared backbone learns richer representations by jointly optimizing
    all tasks. Auxiliary tasks act as regularizers for the primary task.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Shared backbone
    backbone = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    features = backbone(inputs)

    # Shared feature processing
    shared = layers.GlobalAveragePooling2D()(features)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dense(512, activation="relu")(shared)
    shared = layers.Dropout(0.4)(shared)

    # Head 1: Tumor Classification (primary task)
    tumor_branch = layers.Dense(128, activation="relu", name="tumor_dense")(shared)
    tumor_branch = layers.Dropout(0.3)(tumor_branch)
    tumor_output = layers.Dense(
        num_tumor_classes, activation="softmax", dtype="float32", name="tumor_class"
    )(tumor_branch)

    # Head 2: Tumor Grade
    grade_branch = layers.Dense(64, activation="relu", name="grade_dense")(shared)
    grade_branch = layers.Dropout(0.3)(grade_branch)
    grade_output = layers.Dense(
        num_grade_classes, activation="softmax", dtype="float32", name="tumor_grade"
    )(grade_branch)

    # Head 3: Tumor Location
    location_branch = layers.Dense(64, activation="relu", name="location_dense")(shared)
    location_branch = layers.Dropout(0.3)(location_branch)
    location_output = layers.Dense(
        num_location_classes, activation="softmax", dtype="float32", name="tumor_location"
    )(location_branch)

    model = Model(inputs, [tumor_output, grade_output, location_output], name="MultiTask_BrainTumor")
    return model


def compile_multitask(model, primary_weight=1.0, aux_weight=0.3):
    """
    Compile with weighted losses.

    Primary task (classification) gets higher weight.
    Auxiliary tasks get lower weight — they help but shouldn't dominate.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "tumor_class": "categorical_crossentropy",
            "tumor_grade": "categorical_crossentropy",
            "tumor_location": "categorical_crossentropy",
        },
        loss_weights={
            "tumor_class": primary_weight,
            "tumor_grade": aux_weight,
            "tumor_location": aux_weight,
        },
        metrics={"tumor_class": "accuracy", "tumor_grade": "accuracy", "tumor_location": "accuracy"},
    )
    return model


class MultiTaskDataGenerator:
    """
    Custom generator for multi-task training.

    Wraps the standard ImageDataGenerator output and adds dummy labels
    for auxiliary tasks when those annotations aren't available.
    """

    def __init__(self, base_generator, num_grade=3, num_location=6):
        self.base = base_generator
        self.num_grade = num_grade
        self.num_location = num_location

    def __iter__(self):
        return self

    def __next__(self):
        x, y_tumor = next(self.base)
        batch_size = x.shape[0]
        # Placeholder labels for auxiliary tasks (uniform when unknown)
        y_grade = tf.ones((batch_size, self.num_grade)) / self.num_grade
        y_location = tf.ones((batch_size, self.num_location)) / self.num_location
        return x, {"tumor_class": y_tumor, "tumor_grade": y_grade, "tumor_location": y_location}

    def __len__(self):
        return len(self.base)
