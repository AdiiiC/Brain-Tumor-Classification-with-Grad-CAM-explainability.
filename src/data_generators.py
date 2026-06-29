"""
Data generators: augmentation pipelines for train/val/test.
Corresponds to Notebook Cells 21-25.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from config import CROP_DIR, TEST_DATA_DIR, IMG_SIZE, SEED


def create_generators(batch_size=32):
    """Create train, validation, and test data generators."""
    # Richer augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        str(CROP_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=SEED
    )

    valid_data = datagen.flow_from_directory(
        str(CROP_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=SEED
    )

    # No augmentation on test set
    test_datagen = ImageDataGenerator()
    test_data = test_datagen.flow_from_directory(
        str(TEST_DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, valid_data, test_data


def visualize_augmented(train_data, n=6):
    """Visualize augmented training samples."""
    sample_x, sample_y = next(train_data)
    plt.figure(figsize=(12, 9))
    for i in range(min(n, len(sample_x))):
        plt.subplot(2, 3, i + 1)
        sample = array_to_img(sample_x[i])
        plt.axis('off')
        plt.grid(False)
        plt.imshow(sample)
    plt.suptitle("Augmented Training Samples")
    plt.show()


if __name__ == "__main__":
    train_data, valid_data, test_data = create_generators()
    print(f"Train classes: {train_data.class_indices}")
    print(f"Test classes:  {test_data.class_indices}")
    visualize_augmented(train_data)
