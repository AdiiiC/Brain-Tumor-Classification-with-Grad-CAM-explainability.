"""
Upgrade #3 — 3D CNN for Volumetric MRI.

Uses full 3D MRI volumes instead of individual 2D slices. This captures
spatial context across adjacent slices — critical for understanding tumor
depth, shape, and boundary characteristics.

Requires NIfTI (.nii.gz) or stacked DICOM series input.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_3d_cnn(
    input_shape=(128, 128, 64, 1),  # depth=64 slices, single channel
    num_classes=4,
) -> Model:
    """
    3D Convolutional Neural Network for volumetric brain MRI.

    Architecture inspired by 3D-ResNet with residual connections.
    Input: 3D volume (H × W × D × C)
    Output: 4-class tumor classification
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: Initial convolution
    x = layers.Conv3D(32, (3, 3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((2, 2, 2))(x)

    # Block 2: Residual block
    x = _residual_block_3d(x, 64)
    x = layers.MaxPool3D((2, 2, 2))(x)

    # Block 3
    x = _residual_block_3d(x, 128)
    x = layers.MaxPool3D((2, 2, 2))(x)

    # Block 4
    x = _residual_block_3d(x, 256)
    x = layers.MaxPool3D((2, 2, 2))(x)

    # Block 5
    x = _residual_block_3d(x, 512)

    # Global pooling + classification head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return Model(inputs, outputs, name="3D_CNN_BrainTumor")


def _residual_block_3d(x, filters):
    """3D residual block with skip connection."""
    shortcut = layers.Conv3D(filters, (1, 1, 1), padding="same")(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv3D(filters, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def load_nifti_volume(filepath: str, target_shape=(128, 128, 64)) -> "np.ndarray":
    """
    Load a NIfTI file and resize to target shape.

    Requires nibabel: pip install nibabel
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    nii = nib.load(filepath)
    volume = nii.get_fdata().astype(np.float32)

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)

    # Resize to target
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape[:3])]
    volume = zoom(volume, zoom_factors, order=1)

    # Add channel dimension
    return volume[..., np.newaxis]


def load_dicom_series(directory: str, target_shape=(128, 128, 64)) -> "np.ndarray":
    """
    Load a DICOM series (folder of .dcm files) as a 3D volume.

    Requires pydicom: pip install pydicom
    """
    import pydicom
    import numpy as np
    from scipy.ndimage import zoom
    from pathlib import Path

    dcm_files = sorted(Path(directory).glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    # Sort by instance number
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        slices.append((int(getattr(ds, "InstanceNumber", 0)), ds.pixel_array))

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=-1).astype(np.float32)

    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)

    # Resize
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape[:3])]
    volume = zoom(volume, zoom_factors, order=1)

    return volume[..., np.newaxis]
