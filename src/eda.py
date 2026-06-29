"""
Exploratory Data Analysis: visualization, class distribution.
Corresponds to Notebook Cells 5-13.
"""
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img

from config import TRAIN_DIR, CLASSES


def get_filepath_dict():
    """Build dictionary of file paths per class."""
    filepath_dict = {}
    for c in CLASSES:
        filepath_dict[c] = [
            str(TRAIN_DIR / c / x)
            for x in os.listdir(TRAIN_DIR / c)
            if not x.startswith('.')
        ]
    return filepath_dict


def visualize_samples(filepath_dict, n_per_class=4):
    """Display n sample images per class."""
    plt.figure(figsize=(17, 17))
    index = 0
    for c in CLASSES:
        random.shuffle(filepath_dict[c])
        path_list = filepath_dict[c][:n_per_class + 1]
        for i in range(1, n_per_class + 1):
            index += 1
            plt.subplot(4, 4, index)
            plt.imshow(load_img(path_list[i]))
            plt.title(c)
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def class_distribution():
    """Print and plot class distribution."""
    counts = []
    names = []
    for cls in CLASSES:
        count = len([f for f in os.listdir(TRAIN_DIR / cls) if not f.startswith('.')])
        counts.append(count)
        names.append(cls)
        print(f"Number of images in {cls}: {count}")

    plt.figure(figsize=(8, 8))
    colors = sns.color_palette('pastel')
    plt.pie(counts, labels=names, autopct="%1.1f%%", colors=colors)
    plt.title("Class Distribution")
    plt.show()
    return names, counts


if __name__ == "__main__":
    fp_dict = get_filepath_dict()
    visualize_samples(fp_dict)
    class_distribution()
