"""
Test-Time Augmentation (TTA) for accuracy boost.
Corresponds to Notebook Cell 48.
"""
import os

import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from config import CLASSES, CLASS_NAMES, TEST_DATA_DIR, IMG_SIZE


def predict_with_tta(model, image, n_augments=10):
    """Run inference with TTA (average over augmented versions)."""
    augmenter = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1]
    )
    img_batch = np.expand_dims(np.array(image, dtype='float32'), axis=0)
    preds = [
        model.predict(next(augmenter.flow(img_batch, batch_size=1)), verbose=0)
        for _ in range(n_augments)
    ]
    return np.mean(preds, axis=0)[0]


def evaluate_tta(model, samples_per_class=20):
    """Evaluate TTA accuracy on test set."""
    print("Computing TTA predictions on test set ...")
    tta_preds = []
    tta_labels = []

    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_dir = TEST_DATA_DIR / CLASSES[cls_idx]
        img_files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.')][:samples_per_class]
        for fname in tqdm(img_files, desc=cls_name):
            img = cv2.imread(str(cls_dir / fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            tta_p = predict_with_tta(model, img)
            tta_preds.append(np.argmax(tta_p))
            tta_labels.append(cls_idx)

    tta_accuracy = accuracy_score(tta_labels, tta_preds)
    print(f"\nTTA Accuracy ({samples_per_class} samples/class): {tta_accuracy*100:.2f}%")
    return tta_accuracy
