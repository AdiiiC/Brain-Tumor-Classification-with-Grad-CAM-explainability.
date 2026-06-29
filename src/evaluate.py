"""
Model evaluation: confusion matrix, classification report, ROC curves, predictions.
Corresponds to Notebook Cells 32-44.
"""
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from config import CLASSES, CLASS_NAMES, TEST_DATA_DIR, IMG_SIZE


def evaluate_model(model, train_data, test_data):
    """Evaluate model on train and test data."""
    print("Training set evaluation:")
    model.evaluate(train_data)
    print("\nTest set evaluation:")
    model.evaluate(test_data)


def generate_predictions(model, test_data):
    """Generate predictions on test data."""
    y_test = test_data.classes
    y_test_hat = np.argmax(model.predict(test_data), axis=1)
    return y_test, y_test_hat


def plot_confusion_matrix(y_test, y_test_hat):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_test_hat)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['glioma', 'meningioma', 'notumor', 'pituitary']
    )
    cm_display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    return cm


def print_classification_report(y_test, y_test_hat):
    """Print detailed classification report."""
    print(classification_report(y_test, y_test_hat, target_names=CLASSES))


def plot_roc_curves(model, test_data, y_test):
    """Plot per-class ROC curves with AUC."""
    y_score = model.predict(test_data)
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    label_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

    for i, (cls_name, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_title(f'{cls_name}')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc='lower right')
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1.02])

    plt.suptitle('Per-Class ROC Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()


def predict_individual_images(model):
    """Predict on individual test images and compute accuracy."""
    CLASS_DICT = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    images = []
    prediction = []
    original = []

    for cls in os.listdir(str(TEST_DATA_DIR)):
        if cls.startswith('.'):
            continue
        cls_dir = TEST_DATA_DIR / cls
        for item in os.listdir(str(cls_dir)):
            if item.startswith('.'):
                continue
            img_path = cls_dir / item
            img = PIL.Image.open(str(img_path)).convert('RGB')
            images.append(img)
            img_arr = np.array(img.resize(IMG_SIZE))
            pred = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
            prediction.append(CLASS_DICT[np.argmax(pred)])
            original.append(cls)

    score = accuracy_score(original, prediction)
    print(f"Individual prediction accuracy: {score:.4f}")
    return images, prediction, original, score


def visualize_predictions(images, prediction, original, n=10):
    """Visualize random predictions (correct=green, wrong=red)."""
    fig = plt.figure(figsize=(20, 20))
    for i in range(n):
        j = random.randint(0, len(images) - 1)
        fig.add_subplot(5, 2, i + 1)
        correct = prediction[j] == original[j]
        color = 'green' if correct else 'red'
        plt.xlabel(f"Pred: {prediction[j]}   True: {original[j]}", color=color)
        plt.imshow(images[j])
        plt.axis('off')
    fig.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    plt.show()
