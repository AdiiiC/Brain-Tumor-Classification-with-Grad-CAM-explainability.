"""
Image preprocessing: CLAHE enhancement, brain cropping, resizing, saving.
Corresponds to Notebook Cells 14-20.
"""
import os

import cv2
import imutils
import numpy as np
from tqdm import tqdm

from config import CLASSES, TRAIN_DIR, TEST_DIR, CROP_DIR, TEST_DATA_DIR, IMG_SIZE


def preprocess_mri_clahe(image):
    """CLAHE contrast enhancement — clinical radiology standard for MRI."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def crop_image(image, plot=False):
    """Crop brain region from MRI scan using contour detection."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Image")
        plt.show()

    return new_img


def save_cropped_class(src_dir, dst_dir, desc=""):
    """Crop, CLAHE-enhance, resize and save all images for one class."""
    failed = []
    j = 0
    for fname in tqdm(os.listdir(src_dir), desc=desc):
        if fname.startswith('.'):
            continue
        path = src_dir / fname
        img = cv2.imread(str(path))
        if img is None:
            failed.append(str(path))
            continue
        try:
            img = preprocess_mri_clahe(img)
            img = crop_image(img, plot=False)
            img = cv2.resize(img, IMG_SIZE)
            cv2.imwrite(str(dst_dir / f"{j}.jpg"), img)
            j += 1
        except Exception as e:
            failed.append((str(path), str(e)))
    print(f"  {desc}: {j} saved, {len(failed)} failed")
    return failed


def preprocess_all():
    """Run preprocessing on all train and test images."""
    print("Preprocessing training images...")
    all_failed = []
    for cls in CLASSES:
        fails = save_cropped_class(TRAIN_DIR / cls, CROP_DIR / cls, desc=f"Train/{cls}")
        all_failed.extend(fails)
    print(f"\nTotal train failed: {len(all_failed)}")

    print("\nPreprocessing test images...")
    test_failed = []
    for cls in CLASSES:
        fails = save_cropped_class(TEST_DIR / cls, TEST_DATA_DIR / cls, desc=f"Test/{cls}")
        test_failed.extend(fails)
    print(f"Total test failed: {len(test_failed)}")

    # Print counts
    for cls in CLASSES:
        print(f"{cls}: {len(os.listdir(CROP_DIR / cls))} cropped images")


if __name__ == "__main__":
    preprocess_all()
