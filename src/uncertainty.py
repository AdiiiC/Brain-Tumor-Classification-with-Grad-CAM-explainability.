"""
Monte Carlo Dropout uncertainty quantification.
Corresponds to Notebook Cell 47.
"""
import os

import cv2
import numpy as np

from config import CLASSES, CLASS_NAMES, TEST_DATA_DIR, IMG_SIZE


def predict_with_uncertainty(model, image, n_iterations=50):
    """
    Run inference n times with dropout active (MC Dropout).
    Returns mean prediction and per-class uncertainty (std).
    """
    img_batch = np.expand_dims(np.array(image, dtype='float32'), axis=0)
    preds = np.array([
        model(img_batch, training=True).numpy()
        for _ in range(n_iterations)
    ])
    preds = preds[:, 0, :]
    return preds.mean(axis=0), preds.std(axis=0)


def demo_uncertainty(model):
    """Run MC Dropout uncertainty demo on one sample per class."""
    print("Monte Carlo Dropout Uncertainty (50 forward passes)\n")
    print(f"{'Class':<15} {'Predicted':<15} {'Confidence':>12} {'Uncertainty':>13}")
    print("-" * 58)

    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_dir = TEST_DATA_DIR / CLASSES[cls_idx]
        img_files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.')]
        if len(img_files) < 2:
            continue
        img_path = cls_dir / img_files[1]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        mean_pred, uncertainty = predict_with_uncertainty(model, img)
        pred_cls = CLASS_NAMES[np.argmax(mean_pred)]
        confidence = mean_pred.max() * 100
        unc_val = uncertainty[np.argmax(mean_pred)] * 100
        flag = " ⚠ UNCERTAIN — refer to specialist" if unc_val > 5.0 else ""
        print(f"{cls_name:<15} {pred_cls:<15} {confidence:>10.1f}%  ±{unc_val:>5.1f}%{flag}")
