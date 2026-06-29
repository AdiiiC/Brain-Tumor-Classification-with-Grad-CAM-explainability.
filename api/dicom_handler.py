"""
DICOM file handler — Upgrade #13.

Parses DICOM (.dcm) files from hospitals and converts them to numpy arrays
compatible with the model pipeline. Handles standard brain MRI DICOM series.
"""

import io
import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def is_dicom(file_bytes: bytes) -> bool:
    """Check if file bytes represent a DICOM file (magic bytes at offset 128)."""
    if len(file_bytes) < 132:
        return False
    return file_bytes[128:132] == b"DICM"


def dicom_to_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert DICOM bytes to a BGR image (uint8) suitable for the preprocessing pipeline.

    Handles:
    - Window/level adjustment via VOI LUT
    - Photometric interpretation (MONOCHROME1 inversion)
    - 16-bit to 8-bit normalization
    - Grayscale to 3-channel conversion
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM support. Install with: pip install pydicom"
        )

    ds = pydicom.dcmread(io.BytesIO(file_bytes))

    # Apply VOI LUT (windowing)
    pixel_array = apply_voi_lut(ds.pixel_array, ds)

    # Handle MONOCHROME1 (inverted grayscale)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Normalize to 0-255 uint8
    arr = pixel_array.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    img = arr.astype(np.uint8)

    # Convert grayscale to 3-channel BGR
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def extract_dicom_metadata(file_bytes: bytes) -> dict:
    """Extract clinically relevant metadata from DICOM headers."""
    if not PYDICOM_AVAILABLE:
        return {}

    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    return {
        "patient_id": str(getattr(ds, "PatientID", "Unknown")),
        "patient_name": str(getattr(ds, "PatientName", "Unknown")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "body_part": str(getattr(ds, "BodyPartExamined", "")),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
    }
