"""Upload validation: content sniffing, size limits, malformed input."""

from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from api.validation import (
    ValidationError,
    decode_image,
    read_upload_limited,
    read_validated,
    sniff_format,
    validate_content,
)


def _upload(data: bytes, name: str = "file.png") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data))


class TestSniffFormat:
    def test_detects_png(self, png_bytes):
        assert sniff_format(png_bytes) == "png"

    def test_detects_jpeg(self, jpeg_bytes):
        assert sniff_format(jpeg_bytes) == "jpeg"

    def test_detects_dicom_preamble(self):
        data = b"\x00" * 128 + b"DICM" + b"rest of file"
        assert sniff_format(data) == "dicom"

    def test_rejects_unknown(self):
        assert sniff_format(b"#!/bin/sh\nrm -rf /") is None

    def test_rejects_empty(self):
        assert sniff_format(b"") is None


class TestValidateContent:
    def test_accepts_png(self, png_bytes):
        assert validate_content(png_bytes) == "png"

    def test_rejects_script_disguised_as_image(self):
        with pytest.raises(ValidationError) as exc:
            validate_content(b"<?php system($_GET['c']); ?>")
        assert exc.value.status_code == 400

    def test_rejects_svg_with_script(self):
        payload = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        with pytest.raises(ValidationError):
            validate_content(payload)


class TestDecodeImage:
    def test_decodes_valid_png(self, png_bytes):
        assert decode_image(png_bytes).shape == (256, 256, 3)

    def test_rejects_truncated_png(self, png_bytes):
        with pytest.raises(ValidationError):
            decode_image(png_bytes[:20])


class TestSizeLimits:
    @pytest.mark.anyio
    async def test_rejects_oversized_upload(self):
        with pytest.raises(ValidationError) as exc:
            await read_upload_limited(_upload(b"x" * 5000), max_size=1024)
        assert exc.value.status_code == 413

    @pytest.mark.anyio
    async def test_rejects_empty_upload(self):
        with pytest.raises(ValidationError):
            await read_upload_limited(_upload(b""))

    @pytest.mark.anyio
    async def test_accepts_within_limit(self, png_bytes):
        content, fmt = await read_validated(_upload(png_bytes))
        assert fmt == "png"
        assert content == png_bytes

    @pytest.mark.anyio
    async def test_extension_does_not_override_content(self, png_bytes):
        """A .dcm filename on PNG bytes must still be classified as PNG."""
        _, fmt = await read_validated(_upload(png_bytes, name="scan.dcm"))
        assert fmt == "png"


@pytest.fixture
def anyio_backend():
    return "asyncio"
