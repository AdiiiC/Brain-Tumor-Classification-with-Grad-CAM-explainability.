"""
Upgrade #4 — K-Fold Cross-Validation.

Provides robust performance estimates by training on K different splits.
Reduces variance from a lucky/unlucky train-val split.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import cv2
import os


NUM_CLASSES = 4
IMG_SIZE = (240, 240)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_model():
    """Build fresh EfficientNetB1 model for each fold."""
    base = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(240, 240, 3))
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_all_images(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all images and labels from directory structure."""
    images = []
    labels = []

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            continue
        for img_file in cls_dir.iterdir():
            if img_file.name.startswith("."):
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32))
            labels.append(cls_idx)

    return np.array(images), np.array(labels)


def run_kfold(data_dir: str, k: int = 5, epochs: int = 15) -> dict:
    """
    Run K-Fold cross-validation.

    Returns per-fold and aggregate metrics.
    """
    print(f"Loading images from {data_dir}...")
    X, y = load_all_images(data_dir)
    print(f"Loaded {len(X)} images across {NUM_CLASSES} classes")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []
    all_val_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k}")
        print(f"{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # One-hot encode
        y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15, zoom_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=True, brightness_range=[0.9, 1.1],
        )
        datagen.fit(X_train)

        # Fresh model per fold
        model = build_model()

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=2),
        ]

        history = model.fit(
            datagen.flow(X_train, y_train_oh, batch_size=32),
            epochs=epochs,
            validation_data=(X_val, y_val_oh),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val_oh, verbose=0)
        all_val_acc.append(val_acc)

        fold_results.append({
            "fold": fold + 1,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
        })

        print(f"Fold {fold + 1} — Val Accuracy: {val_acc*100:.2f}%")

        # Save best fold model
        model.save(f"model_fold{fold+1}.keras")

    # Summary
    mean_acc = np.mean(all_val_acc)
    std_acc = np.std(all_val_acc)

    summary = {
        "k": k,
        "fold_results": fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "confidence_interval_95": (mean_acc - 1.96 * std_acc, mean_acc + 1.96 * std_acc),
    }

    print(f"\n{'='*50}")
    print(f"K-FOLD SUMMARY (K={k})")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"95% CI: [{summary['confidence_interval_95'][0]*100:.2f}%, {summary['confidence_interval_95'][1]*100:.2f}%]")

    return summary


if __name__ == "__main__":
    results = run_kfold("Crop-Brain-MRI", k=5, epochs=15)
