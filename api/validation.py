"""
Upload validation: content-based type detection and size limits.

Extension checks alone are not trustworthy — a caller controls the filename.
Everything here inspects the actual bytes.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_PIXELS = 64_000_000  # decompression-bomb guard (~8000x8000)
CHUNK_SIZE = 1024 * 1024

# Leading bytes that identify each accepted format.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

# DICOM stores "DICM" at byte 128, after the preamble.
_DICOM_OFFSET = 128
_DICOM_MAGIC = b"DICM"


class ValidationError(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)


def sniff_format(content: bytes) -> str | None:
    """Identify the file format from its magic bytes, or None if unrecognised."""
    if len(content) > _DICOM_OFFSET + 4 and content[_DICOM_OFFSET:_DICOM_OFFSET + 4] == _DICOM_MAGIC:
        return "dicom"
    for signature, name in _MAGIC_SIGNATURES:
        if content.startswith(signature):
            return name
    return None


async def read_upload_limited(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bytes:
    """
    Stream an upload into memory, aborting as soon as the limit is exceeded.

    Reading in chunks means an oversized body is rejected without buffering all of it.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise ValidationError(
                f"File too large (max {max_size // (1024 * 1024)} MB)", status_code=413
            )
        chunks.append(chunk)

    if total == 0:
        raise ValidationError("Uploaded file is empty")
    return b"".join(chunks)


def validate_content(content: bytes) -> str:
    """Confirm the bytes are an accepted image/DICOM format. Returns the format name."""
    fmt = sniff_format(content)
    if fmt is None:
        raise ValidationError(
            "Unrecognised file content. Accepted formats: JPEG, PNG, BMP, TIFF, DICOM."
        )
    return fmt


def decode_image(content: bytes) -> np.ndarray:
    """Decode non-DICOM image bytes, rejecting anything OpenCV cannot parse."""
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValidationError("Could not decode image — the file may be corrupt.")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValidationError("Image dimensions too large.", status_code=413)
    return img


async def read_validated(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> tuple[bytes, str]:
    """Read an upload and validate it by content. Returns (bytes, detected_format)."""
    content = await read_upload_limited(file, max_size)
    return content, validate_content(content)
