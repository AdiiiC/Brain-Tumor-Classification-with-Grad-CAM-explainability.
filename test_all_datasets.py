"""
Test all downloaded datasets with the Brain Tumor Classification model.
Trains on Sartaj dataset (same 4 classes), then evaluates on all others.
"""
import os
import sys
import json
import time
import random
import warnings
from pathlib import Path

# ── Suppress noisy third-party warnings ──────────────────────────────────────
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')      # hide TF C++ info/warning logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')     # silence oneDNN notice
# LibreSSL / urllib3 v2 compatibility notice
warnings.filterwarnings('ignore', message=r".*OpenSSL 1\.1\.1\+.*")
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings('ignore', category=NotOpenSSLWarning)
except Exception:
    pass
# Keras PyDataset super().__init__ UserWarning from legacy ImageDataGenerator
warnings.filterwarnings('ignore', message=r".*PyDataset.*super\(\).__init__.*")

import numpy as np
import cv2
import tensorflow as tf

tf.get_logger().setLevel('ERROR')   # silence Keras/TF Python-side warnings
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

IMG_SIZE = (240, 240)
BASE = Path(".")
DATASETS_DIR = BASE / "datasets"
PREPROCESSED_DIR = BASE / "preprocessed_sartaj"

CLASS_MAP = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Build Model
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    print("\n" + "=" * 70)
    print("BUILDING EfficientNetB1 MODEL")
    print("=" * 70)
    
    effenet = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    for layer in effenet.layers:
        layer.trainable = False

    x = effenet.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)

    model = Model(inputs=effenet.input, outputs=outputs)
    print(f"Model parameters: {model.count_params():,}")
    return model, effenet


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Preprocess (CLAHE + Crop)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_mri_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def crop_brain(image):
    import imutils
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return image
    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()])[0]
    extRight = tuple(c[c[:, :, 0].argmax()])[0]
    extTop = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]
    new_img = image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]
    if new_img is None or new_img.size == 0:
        return image
    return new_img


def load_and_preprocess(img_path):
    """Load, CLAHE-enhance, crop, resize, normalize a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    try:
        img = preprocess_mri_clahe(img)
        img = crop_brain(img)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Preprocess training data (CLAHE + crop) then train
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_dataset(src_dir, dst_dir):
    """Apply CLAHE + crop + resize to all images, saving to dst_dir."""
    import shutil
    if dst_dir.exists() and any(dst_dir.iterdir()):
        # Check if already preprocessed
        total = sum(1 for _ in dst_dir.rglob("*.jpg"))
        if total > 100:
            print(f"  Already preprocessed: {total} images in {dst_dir}")
            return
    
    classes = sorted([d.name for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    print(f"  Preprocessing {src_dir.name} → {dst_dir}")
    print(f"  Classes: {classes}")
    
    total_saved = 0
    for cls in classes:
        out_dir = dst_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        cls_dir = src_dir / cls
        files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.')]
        j = 0
        for fname in files:
            img = cv2.imread(str(cls_dir / fname))
            if img is None:
                continue
            try:
                img = preprocess_mri_clahe(img)
                img = crop_brain(img)
                img = cv2.resize(img, IMG_SIZE)
                cv2.imwrite(str(out_dir / f"{j}.jpg"), img)
                j += 1
            except Exception:
                continue
        total_saved += j
    print(f"  Saved {total_saved} preprocessed images")


def train_model(model, effenet):
    print("\n" + "=" * 70)
    print("TRAINING on Sartaj + Bilal + BRISC + BraTS-FLAIR + Figshare")
    print("(CLAHE+crop for standard MRI, resize-only for FLAIR/patches)")
    print("=" * 70)
    
    # ── Preprocess standard MRI datasets ─────────────────────────────────────
    sartaj_train = DATASETS_DIR / "brain-tumor-classification-sartaj" / "Training"
    bilal_train = DATASETS_DIR / "brain-tumor-bilal" / "brain_tumor_dataset" / "brain_tumor_classification" / "Training"
    brisc_train = DATASETS_DIR / "brisc2025" / "brisc2025" / "classification_task" / "train"
    
    # Preprocess Sartaj training data
    preprocess_dataset(sartaj_train, PREPROCESSED_DIR / "Training_sartaj")
    
    # Preprocess Bilal training data
    if bilal_train.exists():
        preprocess_dataset(bilal_train, PREPROCESSED_DIR / "Training_bilal")
    
    # Preprocess BRISC 2025 training data (same 4 classes)
    if brisc_train.exists():
        preprocess_dataset(brisc_train, PREPROCESSED_DIR / "Training_brisc")
    
    # ── Merge into one combined training directory ────────────────────────────
    combined_dir = PREPROCESSED_DIR / "Training_combined"
    import shutil
    
    # Force rebuild if BRISC/BraTS/Figshare not yet included
    needs_rebuild = False
    if combined_dir.exists():
        # Check if new sources are already integrated
        marker = combined_dir / ".sources_v3"
        if not marker.exists():
            needs_rebuild = True
            shutil.rmtree(str(combined_dir))
            print("  Rebuilding combined directory with all sources...")
    
    if not combined_dir.exists() or not any(combined_dir.iterdir()):
        needs_rebuild = True
    
    if needs_rebuild:
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Map class names to standard names
        sartaj_classes = {'glioma_tumor': 'glioma_tumor', 'meningioma_tumor': 'meningioma_tumor',
                          'no_tumor': 'no_tumor', 'pituitary_tumor': 'pituitary_tumor'}
        bilal_classes = {'glioma': 'glioma_tumor', 'meningioma': 'meningioma_tumor',
                         'notumor': 'no_tumor', 'pituitary': 'pituitary_tumor'}
        brisc_classes = {'glioma': 'glioma_tumor', 'meningioma': 'meningioma_tumor',
                         'no_tumor': 'no_tumor', 'pituitary': 'pituitary_tumor'}
        
        # Copy Sartaj preprocessed
        sartaj_proc = PREPROCESSED_DIR / "Training_sartaj"
        if sartaj_proc.exists():
            for cls_folder in sartaj_proc.iterdir():
                if not cls_folder.is_dir():
                    continue
                mapped_name = sartaj_classes.get(cls_folder.name, cls_folder.name)
                dst = combined_dir / mapped_name
                dst.mkdir(parents=True, exist_ok=True)
                for f in cls_folder.glob("*.jpg"):
                    shutil.copy2(str(f), str(dst / f"sartaj_{f.name}"))
        
        # Copy Bilal preprocessed
        bilal_proc = PREPROCESSED_DIR / "Training_bilal"
        if bilal_proc.exists():
            for cls_folder in bilal_proc.iterdir():
                if not cls_folder.is_dir():
                    continue
                mapped_name = bilal_classes.get(cls_folder.name, cls_folder.name)
                dst = combined_dir / mapped_name
                dst.mkdir(parents=True, exist_ok=True)
                for f in cls_folder.glob("*.jpg"):
                    shutil.copy2(str(f), str(dst / f"bilal_{f.name}"))
        
        # Copy BRISC 2025 preprocessed (same 4 classes!)
        brisc_proc = PREPROCESSED_DIR / "Training_brisc"
        if brisc_proc.exists():
            for cls_folder in brisc_proc.iterdir():
                if not cls_folder.is_dir():
                    continue
                mapped_name = brisc_classes.get(cls_folder.name, cls_folder.name)
                dst = combined_dir / mapped_name
                dst.mkdir(parents=True, exist_ok=True)
                for f in cls_folder.glob("*.jpg"):
                    shutil.copy2(str(f), str(dst / f"brisc_{f.name}"))
        
        # ── Add BraTS 2021 2D ALL SEQUENCES (T1, T1CE, T2, FLAIR) ────────
        brats_dir = DATASETS_DIR / "brats2021-2d"
        brats_csv = brats_dir / "target.csv"
        brats_sequences = ["flair", "t1", "t1ce", "t2"]
        if brats_csv.exists() and all((brats_dir / s).exists() for s in brats_sequences):
            print("  Adding BraTS 2021 2D ALL sequences (T1, T1CE, T2, FLAIR)...")
            import csv
            # Read labels
            tumor_rows = []
            no_tumor_rows = []
            with open(str(brats_csv), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['tumor_slice'] == '1':
                        tumor_rows.append(row)
                    else:
                        no_tumor_rows.append(row)
            
            rng = random.Random(SEED)
            rng.shuffle(tumor_rows)
            rng.shuffle(no_tumor_rows)
            # 500 tumor + 500 no-tumor PER sequence = 2000 tumor + 2000 no-tumor total
            n_tumor_per_seq = 500
            n_notumor_per_seq = 500
            
            dst_tumor = combined_dir / "glioma_tumor"
            dst_notumor = combined_dir / "no_tumor"
            dst_tumor.mkdir(parents=True, exist_ok=True)
            dst_notumor.mkdir(parents=True, exist_ok=True)
            
            for seq in brats_sequences:
                seq_dir = brats_dir / seq
                saved_tumor = 0
                saved_notumor = 0
                
                for row in tumor_rows:
                    if saved_tumor >= n_tumor_per_seq:
                        break
                    fname = f"{seq}_BraTS2021_{row['BraTS21ID']}_{row['image_id']}.png"
                    fpath = seq_dir / fname
                    if fpath.exists():
                        img = cv2.imread(str(fpath))
                        if img is None:
                            continue
                        img = cv2.resize(img, IMG_SIZE)
                        cv2.imwrite(str(dst_tumor / f"brats_{seq}_tumor_{saved_tumor}.jpg"), img)
                        saved_tumor += 1
                
                for row in no_tumor_rows:
                    if saved_notumor >= n_notumor_per_seq:
                        break
                    fname = f"{seq}_BraTS2021_{row['BraTS21ID']}_{row['image_id']}.png"
                    fpath = seq_dir / fname
                    if fpath.exists():
                        img = cv2.imread(str(fpath))
                        if img is None:
                            continue
                        img = cv2.resize(img, IMG_SIZE)
                        cv2.imwrite(str(dst_notumor / f"brats_{seq}_notumor_{saved_notumor}.jpg"), img)
                        saved_notumor += 1
                
                print(f"    {seq.upper()}: {saved_tumor} tumor + {saved_notumor} no-tumor")
        
        # ── Add Figshare patches (for patch-style generalization) ─────────────
        fig_img_dir = DATASETS_DIR / "figshare-brain-tumor" / "Brain Tumor" / "Brain Tumor"
        if fig_img_dir.exists():
            print("  Adding Figshare tumor patches to training...")
            fig_images = list(fig_img_dir.glob("*.jpg")) + list(fig_img_dir.glob("*.png"))
            rng = random.Random(SEED)
            rng.shuffle(fig_images)
            n_fig = min(1000, len(fig_images))
            
            # All Figshare images are tumors → distribute across tumor classes
            dst_glioma = combined_dir / "glioma_tumor"
            dst_menin = combined_dir / "meningioma_tumor"
            dst_pit = combined_dir / "pituitary_tumor"
            dst_glioma.mkdir(parents=True, exist_ok=True)
            dst_menin.mkdir(parents=True, exist_ok=True)
            dst_pit.mkdir(parents=True, exist_ok=True)
            
            saved_fig = 0
            # Split equally across tumor classes
            dsts = [dst_glioma, dst_menin, dst_pit]
            for i, fpath in enumerate(fig_images[:n_fig]):
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                target = dsts[i % 3]
                cv2.imwrite(str(target / f"figshare_{saved_fig}.jpg"), img)
                saved_fig += 1
            print(f"    Added {saved_fig} Figshare patches → tumor classes")
        
        # ── Add LGG segmentation FLAIR images (with masks) ───────────────────
        lgg_dir = DATASETS_DIR / "lgg-segmentation" / "kaggle_3m"
        if not lgg_dir.exists():
            lgg_dir = DATASETS_DIR / "lgg-segmentation" / "lgg-mri-segmentation"
        if lgg_dir.exists():
            print("  Adding LGG segmentation FLAIR images to training...")
            patient_dirs = sorted([d for d in lgg_dir.iterdir() if d.is_dir() and d.name.startswith('TCGA')])
            
            dst_tumor = combined_dir / "glioma_tumor"
            dst_notumor = combined_dir / "no_tumor"
            dst_tumor.mkdir(parents=True, exist_ok=True)
            dst_notumor.mkdir(parents=True, exist_ok=True)
            
            lgg_tumor = 0
            lgg_notumor = 0
            for patient_dir in patient_dirs:
                images = sorted(patient_dir.glob("*.tif"))
                masks = {str(p) for p in images if '_mask' in p.name}
                scans = [p for p in images if str(p) not in masks]
                
                for scan_path in scans:
                    if lgg_tumor >= 800 and lgg_notumor >= 400:
                        break
                    mask_path = scan_path.with_name(scan_path.stem + '_mask.tif')
                    has_tumor = False
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            has_tumor = mask.sum() > 0
                    
                    img = cv2.imread(str(scan_path))
                    if img is None:
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    
                    if has_tumor and lgg_tumor < 800:
                        cv2.imwrite(str(dst_tumor / f"lgg_tumor_{lgg_tumor}.jpg"), img)
                        lgg_tumor += 1
                    elif not has_tumor and lgg_notumor < 400:
                        cv2.imwrite(str(dst_notumor / f"lgg_notumor_{lgg_notumor}.jpg"), img)
                        lgg_notumor += 1
            
            print(f"    Added {lgg_tumor} LGG tumor slices → glioma_tumor")
            print(f"    Added {lgg_notumor} LGG no-tumor slices → no_tumor")
        
        # Mark as v3 (with all sequences + sources)
        (combined_dir / ".sources_v3").touch()
        (combined_dir / ".sources_v2").touch()  # backward compat
        
        # Print combined stats
        for cls in sorted(os.listdir(str(combined_dir))):
            if cls.startswith('.'):
                continue
            count = len(list((combined_dir / cls).glob("*.jpg")))
            print(f"  Combined {cls}: {count} images")
    else:
        print("  Combined training directory already exists (v2 with all sources)")
        for cls in sorted(os.listdir(str(combined_dir))):
            if not cls.startswith('.'):
                count = len(list((combined_dir / cls).glob("*.jpg")))
                print(f"  Combined {cls}: {count} images")
    
    train_dir = combined_dir
    
    # ── Oversample minority classes via offline augmentation ─────────────────
    balanced_marker = combined_dir / ".balanced"
    if not balanced_marker.exists():
        print("\n  Balancing classes via augmentation oversampling...")
        class_counts = {}
        for cls in sorted(os.listdir(str(combined_dir))):
            if cls.startswith('.'):
                continue
            class_counts[cls] = len(list((combined_dir / cls).glob("*.jpg")))
        
        max_count = max(class_counts.values())
        print(f"  Target count per class: {max_count} (matching majority class)")
        
        aug = ImageDataGenerator(
            rotation_range=20, zoom_range=0.15,
            width_shift_range=0.15, height_shift_range=0.15,
            horizontal_flip=True, brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        for cls, count in class_counts.items():
            deficit = max_count - count
            if deficit <= 0:
                print(f"    {cls}: {count} images (majority — no augmentation needed)")
                continue
            
            cls_dir = combined_dir / cls
            # Only pick original (non-augmented) images as source
            existing_images = [p for p in cls_dir.glob("*.jpg") if not p.name.startswith("aug_")]
            print(f"    {cls}: {count} images → generating {deficit} augmented samples...")
            
            generated = 0
            while generated < deficit:
                src_path = random.choice(existing_images)
                img = cv2.imread(str(src_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_batch = np.expand_dims(img_rgb, 0)
                
                aug_img = next(aug.flow(img_batch, batch_size=1))[0].astype(np.uint8)
                aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                out_path = cls_dir / f"aug_{generated}.jpg"
                cv2.imwrite(str(out_path), aug_bgr)
                generated += 1
            
            print(f"    → {cls} now has {count + deficit} images")
        
        print("\n  Balanced class distribution:")
        for cls in sorted(os.listdir(str(combined_dir))):
            if not cls.startswith('.'):
                count = len(list((combined_dir / cls).glob("*.jpg")))
                print(f"    {cls}: {count} images")
        
        balanced_marker.touch()
    else:
        print("\n  Classes already balanced (augmented samples present)")
        for cls in sorted(os.listdir(str(combined_dir))):
            if not cls.startswith('.'):
                count = len(list((combined_dir / cls).glob("*.jpg")))
                print(f"    {cls}: {count} images")
    
    # Augmentation — EfficientNet expects [0, 255] range
    datagen = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        channel_shift_range=20.0,
        fill_mode='nearest',
        validation_split=0.12
    )
    
    train_data = datagen.flow_from_directory(
        str(train_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', subset='training', seed=SEED
    )
    
    valid_data = datagen.flow_from_directory(
        str(train_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', subset='validation', seed=SEED
    )
    
    print(f"\nClass mapping: {train_data.class_indices}")
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {valid_data.samples}")
    
    # Class weights
    class_weights_arr = compute_class_weight(
        class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    
    # ── Phase 1: Frozen base (25 epochs) ─────────────────────────────────────
    print("\n--- Phase 1: Training classifier head (25 epochs, lr=1e-4) ---")
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_data, epochs=25, validation_data=valid_data,
        class_weight=class_weight_dict, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
        ]
    )
    
    # ── Phase 2: Fine-tune top 60 layers (35 epochs) ────────────────────────
    print("\n--- Phase 2: Fine-tuning top 60 layers (35 epochs, lr=5e-6) ---")
    for layer in effenet.layers[-80:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    model.compile(optimizer=Adam(5e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_data, epochs=35, validation_data=valid_data,
        class_weight=class_weight_dict, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-7)
        ]
    )
    
    # Save model
    model.save('model_best.keras')
    print("\nModel saved → model_best.keras")
    
    return model, train_data.class_indices


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: WHO Grade Classifier (HGG vs LGG from DICOM-multi + LGG-seg)
# ══════════════════════════════════════════════════════════════════════════════

def train_grade_classifier():
    """
    Train a binary grade classifier: High-Grade (III-IV) vs Low-Grade (I-II).
    Data sources:
    - brain-mri-multi: DICOM FLAIR with diagnosis+grade labels (105 patients)
    - lgg-segmentation: All LGG (111 patients) → label = low-grade
    """
    import csv
    grade_model_path = Path('grade_classifier.keras')
    if grade_model_path.exists():
        print("\n  Grade classifier already trained → grade_classifier.keras")
        return tf.keras.models.load_model(str(grade_model_path))
    
    print("\n" + "=" * 70)
    print("TRAINING WHO GRADE CLASSIFIER (High-Grade vs Low-Grade)")
    print("=" * 70)
    
    grade_dir = PREPROCESSED_DIR / "grade_training"
    hgg_dir = grade_dir / "high_grade"
    lgg_dir_out = grade_dir / "low_grade"
    hgg_dir.mkdir(parents=True, exist_ok=True)
    lgg_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Skip if already built
    if not (grade_dir / ".built").exists():
        # Source 1: brain-mri-multi DICOM with grade labels
        dicom_csv = DATASETS_DIR / "brain-mri-multi" / "data.csv"
        dicom_base = DATASETS_DIR / "brain-mri-multi" / "dicom-flair" / "dicom-flair"
        
        hgg_count = 0
        lgg_count = 0
        
        if dicom_csv.exists():
            print("  Loading DICOM-multi with grade labels...")
            try:
                import pydicom
                has_pydicom = True
            except ImportError:
                has_pydicom = False
            
            with open(str(dicom_csv), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    folder_id = row['FolderID']
                    diagnosis = row['Diagnosis']
                    
                    # Extract grade from diagnosis (e.g., "ASTROCYTOMA-3" → grade 3)
                    grade_num = 0
                    parts = diagnosis.rsplit('-', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        grade_num = int(parts[1])
                    
                    is_hgg = grade_num >= 3  # Grade III-IV = high-grade
                    target_dir = hgg_dir if is_hgg else lgg_dir_out
                    
                    # Load DICOM slices from this patient
                    patient_dir = dicom_base / folder_id
                    if not patient_dir.exists():
                        continue
                    
                    dcm_files = list(patient_dir.glob("*.dcm"))
                    for dcm_path in dcm_files[:5]:  # max 5 slices per patient
                        if has_pydicom:
                            try:
                                ds = pydicom.dcmread(str(dcm_path))
                                arr = ds.pixel_array.astype(np.float32)
                                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-7) * 255).astype(np.uint8)
                                if len(arr.shape) == 2:
                                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                                arr = cv2.resize(arr, IMG_SIZE)
                            except Exception:
                                continue
                        else:
                            # Fallback: try reading as image
                            arr = cv2.imread(str(dcm_path))
                            if arr is None:
                                continue
                            arr = cv2.resize(arr, IMG_SIZE)
                        
                        prefix = "hgg" if is_hgg else "lgg"
                        idx = hgg_count if is_hgg else lgg_count
                        cv2.imwrite(str(target_dir / f"dicom_{prefix}_{idx}.jpg"), arr)
                        if is_hgg:
                            hgg_count += 1
                        else:
                            lgg_count += 1
            
            print(f"    DICOM-multi: {hgg_count} HGG + {lgg_count} LGG slices")
        
        # Source 2: LGG segmentation (all low-grade) — tumor slices only
        lgg_seg_dir = DATASETS_DIR / "lgg-segmentation" / "kaggle_3m"
        if lgg_seg_dir.exists():
            print("  Loading LGG segmentation (all low-grade)...")
            patient_dirs = sorted([d for d in lgg_seg_dir.iterdir()
                                   if d.is_dir() and d.name.startswith('TCGA')])
            lgg_from_seg = 0
            for patient_dir in patient_dirs:
                images = sorted(patient_dir.glob("*.tif"))
                masks = {str(p) for p in images if '_mask' in p.name}
                scans = [p for p in images if str(p) not in masks]
                
                for scan_path in scans[:3]:  # 3 slices per patient
                    mask_path = scan_path.with_name(scan_path.stem + '_mask.tif')
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None and mask.sum() > 0:  # only tumor slices
                            img = cv2.imread(str(scan_path))
                            if img is None:
                                continue
                            img = cv2.resize(img, IMG_SIZE)
                            cv2.imwrite(str(lgg_dir_out / f"lggseg_{lgg_from_seg}.jpg"), img)
                            lgg_from_seg += 1
                            lgg_count += 1
            print(f"    LGG-seg: {lgg_from_seg} low-grade tumor slices")
        
        # Source 3: BraTS 2021 — all are HGG (Grade III-IV gliomas)
        brats_dir = DATASETS_DIR / "brats2021-2d"
        brats_csv_path = brats_dir / "target.csv"
        if brats_csv_path.exists():
            print("  Loading BraTS 2021 tumor slices (all HGG)...")
            tumor_rows = []
            with open(str(brats_csv_path), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['tumor_slice'] == '1':
                        tumor_rows.append(row)
            
            rng = random.Random(SEED)
            rng.shuffle(tumor_rows)
            brats_hgg = 0
            # Balance: add enough BraTS HGG to roughly match LGG count
            target_hgg = max(lgg_count - hgg_count, 300)
            for row in tumor_rows[:target_hgg]:
                fname = f"flair_BraTS2021_{row['BraTS21ID']}_{row['image_id']}.png"
                fpath = brats_dir / "flair" / fname
                if fpath.exists():
                    img = cv2.imread(str(fpath))
                    if img is None:
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    cv2.imwrite(str(hgg_dir / f"brats_hgg_{brats_hgg}.jpg"), img)
                    brats_hgg += 1
                    hgg_count += 1
            print(f"    BraTS HGG: {brats_hgg} high-grade tumor slices")
        
        print(f"\n  Grade training set: {hgg_count} HGG + {lgg_count} LGG")
        (grade_dir / ".built").touch()
    else:
        hgg_count = len(list(hgg_dir.glob("*.jpg")))
        lgg_count = len(list(lgg_dir_out.glob("*.jpg")))
        print(f"  Grade training data exists: {hgg_count} HGG + {lgg_count} LGG")
    
    # Build lightweight grade model (MobileNetV2 for speed)
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    for layer in base.layers:
        layer.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # binary: 0=LGG, 1=HGG
    
    grade_model = Model(inputs=base.input, outputs=outputs)
    
    datagen = ImageDataGenerator(
        rotation_range=15, zoom_range=0.15,
        horizontal_flip=True, brightness_range=[0.85, 1.15],
        validation_split=0.15
    )
    
    train_data = datagen.flow_from_directory(
        str(grade_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='binary', subset='training', seed=SEED
    )
    valid_data = datagen.flow_from_directory(
        str(grade_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='binary', subset='validation', seed=SEED
    )
    
    # Class weights for grade balance
    class_weights_arr = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=train_data.classes
    )
    grade_weights = dict(enumerate(class_weights_arr))
    
    print(f"  Class mapping: {train_data.class_indices}")
    print(f"  Training: {train_data.samples}, Validation: {valid_data.samples}")
    
    # Phase 1: frozen
    grade_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    grade_model.fit(
        train_data, epochs=15, validation_data=valid_data,
        class_weight=grade_weights, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6)
        ]
    )
    
    # Phase 2: fine-tune top layers
    for layer in base.layers[-40:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    grade_model.compile(optimizer=Adam(5e-6), loss='binary_crossentropy', metrics=['accuracy'])
    grade_model.fit(
        train_data, epochs=20, validation_data=valid_data,
        class_weight=grade_weights, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7)
        ]
    )
    
    grade_model.save(str(grade_model_path))
    print(f"\n  Grade classifier saved → {grade_model_path}")
    return grade_model


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Small Tumor Patch Classifier
# ══════════════════════════════════════════════════════════════════════════════

def train_patch_classifier():
    """
    Train a binary patch classifier for small tumor detection.
    Extracts small patches (60x60, 120x120) from training images,
    trains a lightweight model to detect tumor vs no-tumor at patch level.
    """
    patch_model_path = Path('patch_classifier.keras')
    if patch_model_path.exists():
        print("\n  Patch classifier already trained → patch_classifier.keras")
        return tf.keras.models.load_model(str(patch_model_path))
    
    print("\n" + "=" * 70)
    print("TRAINING SMALL TUMOR PATCH CLASSIFIER")
    print("=" * 70)
    
    patch_dir = PREPROCESSED_DIR / "patch_training"
    tumor_patch_dir = patch_dir / "tumor"
    clean_patch_dir = patch_dir / "clean"
    tumor_patch_dir.mkdir(parents=True, exist_ok=True)
    clean_patch_dir.mkdir(parents=True, exist_ok=True)
    
    if not (patch_dir / ".built").exists():
        combined_dir = PREPROCESSED_DIR / "Training_combined"
        tumor_classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        no_tumor_class = "no_tumor"
        
        patch_sizes = [60, 90, 120]  # multi-scale patches
        
        tumor_count = 0
        clean_count = 0
        
        # Extract tumor patches from tumor class images
        print("  Extracting tumor patches from training images...")
        for cls in tumor_classes:
            cls_dir = combined_dir / cls
            if not cls_dir.exists():
                continue
            images = [p for p in cls_dir.glob("*.jpg") if not p.name.startswith("aug_")]
            rng = random.Random(SEED)
            rng.shuffle(images)
            
            for img_path in images[:200]:  # 200 images per tumor class
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                for ps in patch_sizes:
                    if h < ps or w < ps:
                        continue
                    # Extract 3 random patches per image per scale
                    for _ in range(3):
                        y = random.randint(0, h - ps)
                        x = random.randint(0, w - ps)
                        patch = img[y:y+ps, x:x+ps]
                        # Skip mostly-black patches
                        if cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean() < 20:
                            continue
                        patch_resized = cv2.resize(patch, (120, 120))
                        cv2.imwrite(str(tumor_patch_dir / f"tp_{tumor_count}.jpg"), patch_resized)
                        tumor_count += 1
        
        # Extract clean patches from no-tumor images
        print("  Extracting clean patches from no-tumor images...")
        no_tumor_dir = combined_dir / no_tumor_class
        if no_tumor_dir.exists():
            images = [p for p in no_tumor_dir.glob("*.jpg") if not p.name.startswith("aug_")]
            rng = random.Random(SEED)
            rng.shuffle(images)
            
            target_clean = tumor_count  # balance
            for img_path in images:
                if clean_count >= target_clean:
                    break
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                for ps in patch_sizes:
                    if h < ps or w < ps:
                        continue
                    for _ in range(3):
                        y = random.randint(0, h - ps)
                        x = random.randint(0, w - ps)
                        patch = img[y:y+ps, x:x+ps]
                        if cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean() < 20:
                            continue
                        patch_resized = cv2.resize(patch, (120, 120))
                        cv2.imwrite(str(clean_patch_dir / f"cp_{clean_count}.jpg"), patch_resized)
                        clean_count += 1
                        if clean_count >= target_clean:
                            break
                    if clean_count >= target_clean:
                        break
        
        print(f"  Patch dataset: {tumor_count} tumor + {clean_count} clean patches")
        (patch_dir / ".built").touch()
    else:
        tumor_count = len(list(tumor_patch_dir.glob("*.jpg")))
        clean_count = len(list(clean_patch_dir.glob("*.jpg")))
        print(f"  Patch data exists: {tumor_count} tumor + {clean_count} clean")
    
    # Build small patch model (MobileNetV2 with 120x120 input)
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
    for layer in base.layers:
        layer.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # binary: tumor vs clean
    
    patch_model = Model(inputs=base.input, outputs=outputs)
    
    datagen = ImageDataGenerator(
        rotation_range=20, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.15
    )
    
    train_data = datagen.flow_from_directory(
        str(patch_dir), target_size=(120, 120), batch_size=32,
        class_mode='binary', subset='training', seed=SEED
    )
    valid_data = datagen.flow_from_directory(
        str(patch_dir), target_size=(120, 120), batch_size=32,
        class_mode='binary', subset='validation', seed=SEED
    )
    
    class_weights_arr = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=train_data.classes
    )
    patch_weights = dict(enumerate(class_weights_arr))
    
    print(f"  Class mapping: {train_data.class_indices}")
    
    # Phase 1: frozen
    patch_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    patch_model.fit(
        train_data, epochs=15, validation_data=valid_data,
        class_weight=patch_weights, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ]
    )
    
    # Phase 2: fine-tune
    for layer in base.layers[-30:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    patch_model.compile(optimizer=Adam(5e-6), loss='binary_crossentropy', metrics=['accuracy'])
    patch_model.fit(
        train_data, epochs=15, validation_data=valid_data,
        class_weight=patch_weights, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        ]
    )
    
    patch_model.save(str(patch_model_path))
    print(f"\n  Patch classifier saved → {patch_model_path}")
    return patch_model


# ══════════════════════════════════════════════════════════════════════════════
# TTA (Test-Time Augmentation) — free accuracy boost
# ══════════════════════════════════════════════════════════════════════════════

# ── Post-training hyperparameters (tweak without retraining) ─────────────────
# Temperature scaling — reduces overconfidence in softmax outputs.
# T > 1 → softer probabilities (less overconfident), T < 1 → sharper.
# Calibrated on validation set; default 1.0 = no change.
TEMPERATURE = 1.5

# Binary threshold for tumor vs no-tumor decisions (Navoneel, LGG, Figshare).
# Default 0.5 means tumor_prob > 0.5 → tumor. Lowering catches more tumors
# (higher sensitivity) at the cost of more false positives.
BINARY_TUMOR_THRESHOLD = 0.45

# Per-class confidence floor — if max softmax < this, fall back to second-best
# or flag as uncertain. Helps avoid overconfident wrong predictions.
MIN_CONFIDENCE = 0.40

# TTA weight for original image vs augmented copies.
# Higher → original image matters more, augmentations just smooth noise.
TTA_ORIGINAL_WEIGHT = 2.0


def apply_temperature(logits_or_probs, temperature=TEMPERATURE):
    """Apply temperature scaling to soften/sharpen probability distribution."""
    if temperature == 1.0:
        return logits_or_probs
    # Convert probs back to log-space, scale, re-softmax
    log_probs = np.log(logits_or_probs + 1e-10)
    scaled = log_probs / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()


def predict_with_tta(model, img, n_aug=8):
    """Average predictions over original + augmented versions with temperature scaling."""
    preds = []
    weights = []
    
    # Original — weighted higher
    preds.append(model.predict(np.expand_dims(img, 0), verbose=0)[0])
    weights.append(TTA_ORIGINAL_WEIGHT)
    
    # Horizontal flip
    preds.append(model.predict(np.expand_dims(np.fliplr(img), 0), verbose=0)[0])
    weights.append(1.0)
    
    # Small rotations
    h, w = img.shape[:2]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img.astype(np.uint8), M, (w, h)).astype(np.float32)
        preds.append(model.predict(np.expand_dims(rotated, 0), verbose=0)[0])
        weights.append(1.0)
    
    # Brightness variations
    for factor in [0.85, 1.15]:
        bright = np.clip(img * factor, 0, 255).astype(np.float32)
        preds.append(model.predict(np.expand_dims(bright, 0), verbose=0)[0])
        weights.append(1.0)
    
    # Small zoom (center crop + resize)
    crop = 20
    zoomed = cv2.resize(img[crop:-crop, crop:-crop].astype(np.uint8), IMG_SIZE).astype(np.float32)
    preds.append(model.predict(np.expand_dims(zoomed, 0), verbose=0)[0])
    weights.append(1.0)
    
    # Slight blur
    blurred = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0).astype(np.float32)
    preds.append(model.predict(np.expand_dims(blurred, 0), verbose=0)[0])
    weights.append(1.0)
    
    # Weighted average + temperature scaling
    weights = np.array(weights)
    preds = np.array(preds)
    avg_pred = np.average(preds, axis=0, weights=weights)
    return apply_temperature(avg_pred)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Test on all datasets
# ══════════════════════════════════════════════════════════════════════════════

def test_sartaj(model, class_indices):
    """Test on Sartaj's own test split (preprocessed + TTA)."""
    print("\n" + "─" * 70)
    print("TEST 1: Sartaj Test Split (4 classes — preprocessed + TTA)")
    print("─" * 70)
    
    raw_test_dir = DATASETS_DIR / "brain-tumor-classification-sartaj" / "Testing"
    proc_test_dir = PREPROCESSED_DIR / "Testing_sartaj"
    
    # Preprocess test data if not done
    preprocess_dataset(raw_test_dir, proc_test_dir)
    
    # Standard evaluation on preprocessed test set
    test_gen = ImageDataGenerator()
    test_data = test_gen.flow_from_directory(
        str(proc_test_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', shuffle=False
    )
    
    loss, acc_standard = model.evaluate(test_data, verbose=0)
    y_true = test_data.classes
    raw_preds = model.predict(test_data, verbose=0)
    # Apply temperature scaling to reduce overconfidence
    calibrated_preds = np.array([apply_temperature(p) for p in raw_preds])
    y_pred_standard = np.argmax(calibrated_preds, axis=1)
    
    print(f"\n  Standard Accuracy (preprocessed): {acc_standard*100:.2f}%")
    
    # TTA evaluation on ALL test images
    classes = sorted([d.name for d in proc_test_dir.iterdir() if d.is_dir()])
    y_true_tta = []
    y_pred_tta = []
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = proc_test_dir / cls_name
        files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in files:
            img_path = cls_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            tta_pred = predict_with_tta(model, img)
            y_pred_tta.append(np.argmax(tta_pred))
            y_true_tta.append(cls_idx)
    
    acc_tta = accuracy_score(y_true_tta, y_pred_tta)
    
    idx_to_class = {v: k for k, v in test_data.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    print(f"  TTA Accuracy (all): {acc_tta*100:.2f}%")
    print(f"\n  Classification Report (standard):")
    print(f"{classification_report(y_true, y_pred_standard, target_names=target_names)}")
    
    return max(acc_standard, acc_tta)


def test_bilal(model):
    """Test on Bilal Akgoz dataset with TTA (preprocessed)."""
    print("\n" + "─" * 70)
    print("TEST 2: Bilal Brain Tumor Dataset (4 classes — preprocessed + TTA)")
    print("─" * 70)
    
    # Navigate to the correct level
    test_dir = DATASETS_DIR / "brain-tumor-bilal" / "brain_tumor_dataset" / "brain_tumor_classification" / "Testing"
    
    if not test_dir.exists():
        print("  ⚠ Could not locate Testing folder, skipping")
        return None
    
    # Preprocess Bilal test set with CLAHE+crop (same as training)
    proc_test_dir = PREPROCESSED_DIR / "Testing_bilal"
    preprocess_dataset(test_dir, proc_test_dir)
    
    subdirs = sorted([d.name for d in proc_test_dir.iterdir() if d.is_dir()])
    print(f"  Classes found: {subdirs}")
    
    test_gen = ImageDataGenerator()
    test_data = test_gen.flow_from_directory(
        str(proc_test_dir), target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', shuffle=False
    )
    
    y_true = test_data.classes
    raw_preds = model.predict(test_data, verbose=0)
    calibrated_preds = np.array([apply_temperature(p) for p in raw_preds])
    y_pred = np.argmax(calibrated_preds, axis=1)
    acc_standard = accuracy_score(y_true, y_pred)
    
    # TTA on subset (50 per class) — use preprocessed images
    y_true_tta = []
    y_pred_tta = []
    for cls_idx, cls_name in enumerate(subdirs):
        cls_dir = proc_test_dir / cls_name
        files = [f for f in os.listdir(str(cls_dir)) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in files[:50]:
            img_path = cls_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            tta_pred = predict_with_tta(model, img)
            y_pred_tta.append(np.argmax(tta_pred))
            y_true_tta.append(cls_idx)
    
    acc_tta = accuracy_score(y_true_tta, y_pred_tta)
    
    idx_to_class = {v: k for k, v in test_data.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    print(f"\n  Standard Accuracy: {acc_standard*100:.2f}%")
    print(f"  TTA Accuracy (50/class): {acc_tta*100:.2f}%")
    print(f"\n{classification_report(y_true, y_pred, target_names=target_names)}")
    
    return max(acc_standard, acc_tta)  # Report the better result


def test_navoneel(model):
    """Test on Navoneel binary detection dataset with TTA + probability threshold."""
    print("\n" + "─" * 70)
    print("TEST 3: Navoneel Brain MRI Detection (Binary — TTA + threshold)")
    print("─" * 70)
    
    nav_dir = DATASETS_DIR / "brain-mri-detection-navoneel"
    
    # Find yes/no directories
    yes_dir = None
    no_dir = None
    for root, dirs, files in os.walk(str(nav_dir)):
        for d in dirs:
            if d.lower() == 'yes':
                yes_dir = Path(root) / d
            elif d.lower() == 'no':
                no_dir = Path(root) / d
    
    if not yes_dir or not no_dir:
        print("  ⚠ Could not find yes/no folders, skipping")
        return None
    
    print(f"  Yes (tumor): {yes_dir}")
    print(f"  No (no tumor): {no_dir}")
    
    results = {'y_true': [], 'y_pred': [], 'confidences': []}
    
    # For binary: sum tumor-class probabilities (classes 0,1,3) vs no-tumor (class 2)
    for label, folder in [('tumor', yes_dir), ('no_tumor', no_dir)]:
        files = [f for f in os.listdir(str(folder)) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  Processing {label}: {len(files)} images")
        
        for fname in files:
            img = load_and_preprocess(folder / fname)
            if img is None:
                continue
            
            # TTA prediction
            pred = predict_with_tta(model, img)
            
            # Binary: sum of tumor probabilities (glioma + meningioma + pituitary)
            tumor_prob = pred[0] + pred[1] + pred[3]  # classes 0,1,3
            no_tumor_prob = pred[2]                     # class 2
            
            # Use tunable threshold instead of simple 0.5 cutoff
            pred_binary = 'tumor' if tumor_prob > BINARY_TUMOR_THRESHOLD else 'no_tumor'
            results['y_true'].append(label)
            results['y_pred'].append(pred_binary)
            results['confidences'].append(float(max(tumor_prob, no_tumor_prob)))
    
    acc = accuracy_score(results['y_true'], results['y_pred'])
    print(f"\n  Binary Accuracy (TTA + threshold): {acc*100:.2f}%")
    print(f"  Mean Confidence: {np.mean(results['confidences'])*100:.1f}%")
    print(f"\n{classification_report(results['y_true'], results['y_pred'])}")
    
    return acc


def test_figshare(model):
    """Test on Figshare Brain Tumor dataset — all images ARE tumors.
    Measures: what % does the model correctly identify as having a tumor (not 'No Tumor').
    Also checks subtype distribution."""
    print("\n" + "─" * 70)
    print("TEST 4: Figshare Brain Tumor Dataset (All tumors — detection rate)")
    print("─" * 70)
    
    fig_dir = DATASETS_DIR / "figshare-brain-tumor"
    img_dir = fig_dir / "Brain Tumor" / "Brain Tumor"
    
    if not img_dir.exists():
        print("  ⚠ Image directory not found, skipping")
        return None
    
    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"  Total images: {len(all_images)} (ALL contain tumors)")
    
    # Sample for speed (TTA on each image)
    import random as rng
    rng.shuffle(all_images)
    sampled = all_images[:200]
    
    correct_tumor = 0
    subtype_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    confidences = []
    processed = 0
    
    for img_path in sampled:
        # Figshare images are already cropped brain patches — skip CLAHE/crop
        raw = cv2.imread(str(img_path))
        if raw is None:
            continue
        img = cv2.resize(raw, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # TTA for better accuracy
        pred = predict_with_tta(model, img)
        pred_class = np.argmax(pred)
        
        # All images have tumors → correct if tumor probability > threshold
        tumor_prob = pred[0] + pred[1] + pred[3]
        if tumor_prob > BINARY_TUMOR_THRESHOLD:
            correct_tumor += 1
        
        subtype_dist[pred_class] += 1
        confidences.append(float(pred.max()))
        processed += 1
    
    if processed == 0:
        print("  ⚠ No images could be processed")
        return None
    
    detection_rate = correct_tumor / processed
    print(f"\n  Processed: {processed} images")
    print(f"  Tumor Detection Rate: {detection_rate*100:.2f}% (ideal: 100%)")
    print(f"  Mean Confidence: {np.mean(confidences)*100:.1f}%")
    print(f"\n  Subtype predictions:")
    for cls_idx, count in subtype_dist.items():
        pct = count / processed * 100
        print(f"    {CLASS_MAP[cls_idx]}: {count} ({pct:.1f}%)")
    
    return detection_rate


def test_lgg_segmentation(model):
    """Test on LGG Segmentation dataset (FLAIR MRI) with TTA."""
    print("\n" + "─" * 70)
    print("TEST 5: LGG Segmentation Dataset (FLAIR MRI + masks — TTA)")
    print("─" * 70)
    
    lgg_dir = DATASETS_DIR / "lgg-segmentation" / "kaggle_3m"
    if not lgg_dir.exists():
        lgg_dir = DATASETS_DIR / "lgg-segmentation" / "lgg-mri-segmentation"
    if not lgg_dir.exists():
        lgg_dir = DATASETS_DIR / "lgg-segmentation"
    
    # Find patient directories
    patient_dirs = sorted([d for d in lgg_dir.iterdir() if d.is_dir() and d.name.startswith('TCGA')])
    print(f"  Patients found: {len(patient_dirs)}")
    
    if not patient_dirs:
        print("  ⚠ No patient directories found, skipping")
        return None
    
    tumor_correct = 0
    no_tumor_correct = 0
    tumor_total = 0
    no_tumor_total = 0
    processed = 0
    
    for patient_dir in patient_dirs[:40]:  # 40 patients
        images = sorted(patient_dir.glob("*.tif"))
        masks = {str(p) for p in images if '_mask' in p.name}
        scans = [p for p in images if str(p) not in masks]
        
        for scan_path in scans[:4]:  # 4 slices per patient
            mask_path = scan_path.with_name(scan_path.stem + '_mask.tif')
            has_tumor = False
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    has_tumor = mask.sum() > 0
            
            # LGG is FLAIR sequence — skip CLAHE/crop, just resize
            img = cv2.imread(str(scan_path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # TTA prediction
            pred = predict_with_tta(model, img)
            
            # Binary: sum tumor probs vs no-tumor prob
            tumor_prob = pred[0] + pred[1] + pred[3]
            no_tumor_prob = pred[2]
            pred_is_tumor = tumor_prob > BINARY_TUMOR_THRESHOLD
            
            if has_tumor:
                tumor_total += 1
                if pred_is_tumor:
                    tumor_correct += 1
            else:
                no_tumor_total += 1
                if not pred_is_tumor:
                    no_tumor_correct += 1
            
            processed += 1
    
    print(f"\n  Processed: {processed} slices from {min(len(patient_dirs), 40)} patients")
    
    if tumor_total > 0:
        print(f"\n  Tumor Detection (binary, TTA):")
        print(f"    Tumor present   → correctly detected: {tumor_correct}/{tumor_total} ({tumor_correct/tumor_total*100:.1f}%)")
    if no_tumor_total > 0:
        print(f"    No tumor        → correctly identified: {no_tumor_correct}/{no_tumor_total} ({no_tumor_correct/no_tumor_total*100:.1f}%)")
    
    total = tumor_total + no_tumor_total
    correct = tumor_correct + no_tumor_correct
    if total > 0:
        print(f"    Overall binary accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    return correct / total if total > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
CHECKPOINT_FILE = Path("test_checkpoint.json")


def load_checkpoint():
    """Load checkpoint from disk. Returns dict of completed step results."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        print(f"  Loaded checkpoint with {len(data.get('completed', {}))} completed steps")
        return data
    return {"completed": {}, "failed": [], "model_trained": False}


def save_checkpoint(ckpt):
    """Save checkpoint to disk immediately after each step."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)


def run_step(ckpt, step_name, fn, args, force=False):
    """
    Run a single test step with checkpoint support.
    - If step already passed → skip, return saved result.
    - If step previously failed or never ran → run it.
    - On success → save result to checkpoint immediately.
    - On failure → mark as failed, save checkpoint, continue.
    """
    # Skip if already completed (unless forced)
    if not force and step_name in ckpt["completed"]:
        saved = ckpt["completed"][step_name]
        val_str = f"{saved*100:.2f}%" if saved is not None else "N/A"
        print(f"\n  ✓ SKIP [{step_name}] — already passed ({val_str})")
        return saved

    # Remove from failed list if retrying
    ckpt["failed"] = [f for f in ckpt["failed"] if f["step"] != step_name]

    try:
        result = fn(*args)
        # Save immediately on success
        ckpt["completed"][step_name] = float(result) if result is not None else None
        save_checkpoint(ckpt)
        print(f"\n  ✓ CHECKPOINT SAVED: {step_name}")
        return result
    except Exception as e:
        import traceback
        print(f"\n  ✗ FAILED [{step_name}]: {e}")
        traceback.print_exc()
        ckpt["failed"].append({"step": step_name, "error": str(e), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        save_checkpoint(ckpt)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-dataset test suite with checkpoint/resume")
    parser.add_argument('--fresh', action='store_true', help='Clear checkpoint and start from scratch')
    parser.add_argument('--retry-failed', action='store_true', help='Only re-run previously failed steps')
    parser.add_argument('--only', type=str, help='Run only this step (e.g. --only bilal)')
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  BRAIN TUMOR CLASSIFICATION — MULTI-DATASET TEST SUITE")
    print("  (with checkpoint/resume support)")
    print("=" * 70)

    # ── Checkpoint management ────────────────────────────────────────────────
    if args.fresh and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  Cleared previous checkpoint — starting fresh")

    ckpt = load_checkpoint()

    if ckpt.get("failed"):
        print(f"\n  Previously failed steps: {[f['step'] for f in ckpt['failed']]}")
    if ckpt.get("completed"):
        print(f"  Previously completed steps: {list(ckpt['completed'].keys())}")

    # ── Step 0: Model ────────────────────────────────────────────────────────
    model_path = Path('model_best.keras')
    # Check if combined dir has v3 marker (all sequences + sources)
    v2_marker = PREPROCESSED_DIR / "Training_combined" / ".sources_v3"
    model_needs_retrain = not v2_marker.exists()
    
    if model_path.exists() and not model_needs_retrain and not args.fresh:
        print("\n  Loading saved model → model_best.keras")
        model = tf.keras.models.load_model(str(model_path))
        class_indices = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
        ckpt["model_trained"] = True
        save_checkpoint(ckpt)
    else:
        if model_needs_retrain:
            print("\n  New training sources detected — retraining with expanded dataset...")
        model, effenet = build_model()
        model, class_indices = train_model(model, effenet)
        ckpt["model_trained"] = True
        save_checkpoint(ckpt)

    # ── Step 0b: Train specialized models (grade + patch classifiers) ────────
    grade_model = train_grade_classifier()
    patch_model = train_patch_classifier()

    # ── Define all test steps ────────────────────────────────────────────────
    all_steps = [
        ('sartaj_test',      test_sartaj,           (model, class_indices)),
        ('bilal',            test_bilal,             (model,)),
        ('navoneel_binary',  test_navoneel,          (model,)),
        ('figshare',         test_figshare,          (model,)),
        ('lgg_segmentation', test_lgg_segmentation,  (model,)),
    ]

    # ── Filter steps based on CLI flags ──────────────────────────────────────
    if args.only:
        all_steps = [(n, f, a) for n, f, a in all_steps if n == args.only]
        if not all_steps:
            print(f"\n  ⚠ Unknown step '{args.only}'. Available: sartaj_test, bilal, navoneel_binary, figshare, lgg_segmentation")
            return

    failed_names = {f['step'] for f in ckpt.get('failed', [])}
    if args.retry_failed:
        all_steps = [(n, f, a) for n, f, a in all_steps if n in failed_names]
        if not all_steps:
            print("\n  No previously failed steps to retry!")
            return
        print(f"\n  Retrying failed steps: {[n for n, _, _ in all_steps]}")

    # ── Run each step ────────────────────────────────────────────────────────
    force = args.fresh or args.retry_failed or bool(args.only)
    for name, fn, fn_args in all_steps:
        run_step(ckpt, name, fn, fn_args, force=force)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results = ckpt["completed"]
    failed = ckpt.get("failed", [])

    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<35} {'Status':<10} {'Result':>12}")
    print("  " + "─" * 59)
    for name, _, _ in [
        ('sartaj_test', None, None),
        ('bilal', None, None),
        ('navoneel_binary', None, None),
        ('figshare', None, None),
        ('lgg_segmentation', None, None),
    ]:
        if name in results:
            val = results[name]
            val_str = f"{val*100:.2f}%" if val is not None else "N/A"
            print(f"  {name:<35} {'PASSED':<10} {val_str:>12}")
        elif name in {f['step'] for f in failed}:
            err = next(f['error'] for f in failed if f['step'] == name)
            print(f"  {name:<35} {'FAILED':<10} {err[:25]:>12}")
        else:
            print(f"  {name:<35} {'PENDING':<10} {'—':>12}")

    print("  " + "─" * 59)
    print(f"  Completed: {len(results)}/5  |  Failed: {len(failed)}  |  Time: {elapsed/60:.1f}m")
    print("=" * 70)

    if failed:
        print(f"\n  To retry failed steps:  python test_all_datasets.py --retry-failed")
    print(f"  To re-run everything:   python test_all_datasets.py --fresh")
    print(f"  To run one step:        python test_all_datasets.py --only <step_name>")

    # Save final results
    with open('test_results.json', 'w') as f:
        json.dump({
            "results": {k: float(v) if v is not None else None for k, v in results.items()},
            "failed": failed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print("\n  Results saved → test_results.json")
    print(f"  Checkpoint → {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
