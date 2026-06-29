"""
Configuration: imports, reproducibility, paths, constants.
Corresponds to Notebook Cells 3-4.
"""
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ── Mixed Precision Training (2-3x GPU speedup) ──────────────────────────────
mixed_precision.set_global_policy('mixed_float16')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(".")
DATASET_DIR   = BASE_DIR / "brain-tumor-mri-dataset"
CROP_DIR      = BASE_DIR / "Crop-Brain-MRI"
TEST_DATA_DIR = BASE_DIR / "Test-Data"
CLASSES       = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_NAMES   = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
IMG_SIZE      = (240, 240)

# ── Create output directories ─────────────────────────────────────────────────
for cls in CLASSES:
    (CROP_DIR / cls).mkdir(parents=True, exist_ok=True)
    (TEST_DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR  = DATASET_DIR / "Testing"
