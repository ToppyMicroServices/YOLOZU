"""Read image dimensions by parsing file headers.

Supports PNG, JPEG, BMP, GIF, TIFF, and WebP natively (no external
dependency).  Falls back to Pillow when available.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ImageSizeError", "get_image_size"]


class ImageSizeError(RuntimeError):
    pass


def get_image_size(path: str | Path) -> tuple[int, int]:
    """Return (width, height) for common image formats.

    Natively parses PNG, JPEG, BMP, GIF, TIFF, and WebP headers.
    Falls back to PIL/Pillow when available for other formats.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".png",):
        return _png_size(path)
    if suffix in (".jpg", ".jpeg"):
        return _jpeg_size(path)
    if suffix in (".bmp",):
        return _bmp_size(path)
    if suffix in (".gif",):
        return _gif_size(path)
    if suffix in (".tif", ".tiff"):
        return _tiff_size(path)
    if suffix in (".webp",):
        return _webp_size(path)
    # Fallback: try PIL/Pillow for any other format.
    return _pil_size(path)


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24:
        raise ImageSizeError("invalid PNG (too short)")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageSizeError("invalid PNG signature")
    # IHDR is the first chunk; width/height are big-endian uint32.
    if data[12:16] != b"IHDR":
        raise ImageSizeError("invalid PNG (missing IHDR)")
    width = int.from_bytes(data[16:20], "big", signed=False)
    height = int.from_bytes(data[20:24], "big", signed=False)
    if width <= 0 or height <= 0:
        raise ImageSizeError("invalid PNG dimensions")
    return width, height


def _jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 4:
        raise ImageSizeError("invalid JPEG (too short)")
    if data[0:2] != b"\xff\xd8":
        raise ImageSizeError("invalid JPEG signature")

    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue

        # Skip fill bytes 0xFF
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break

        marker = data[i]
        i += 1

        # Standalone markers (no length payload)
        if marker in (0xD8, 0xD9):  # SOI, EOI
            continue
        if marker == 0xDA:  # SOS: start of scan, size info is before this
            break

        if i + 2 > len(data):
            break
        seg_len = int.from_bytes(data[i : i + 2], "big", signed=False)
        if seg_len < 2:
            raise ImageSizeError("invalid JPEG segment length")
        seg_start = i + 2
        seg_end = seg_start + (seg_len - 2)
        if seg_end > len(data):
            break

        # SOF markers that contain width/height (baseline/progressive + variants)
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if seg_start + 7 > len(data):
                break
            height = int.from_bytes(data[seg_start + 1 : seg_start + 3], "big", signed=False)
            width = int.from_bytes(data[seg_start + 3 : seg_start + 5], "big", signed=False)
            if width <= 0 or height <= 0:
                raise ImageSizeError("invalid JPEG dimensions")
            return width, height

        i = seg_end

    raise ImageSizeError("could not determine JPEG size (no SOF marker found)")


def _bmp_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 26:
        raise ImageSizeError("invalid BMP (too short)")
    if data[:2] != b"BM":
        raise ImageSizeError("invalid BMP signature")
    # DIB header starts at offset 14; width/height are at +4/+8 as little-endian int32.
    header_size = int.from_bytes(data[14:18], "little", signed=False)
    if header_size < 12:
        raise ImageSizeError("invalid BMP DIB header")
    if header_size == 12:
        # OS/2 BITMAPCOREHEADER: 16-bit width/height.
        width = int.from_bytes(data[18:20], "little", signed=False)
        height = int.from_bytes(data[20:22], "little", signed=False)
    else:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        height = abs(height)  # top-down BMPs use negative height
    if width <= 0 or height <= 0:
        raise ImageSizeError("invalid BMP dimensions")
    return width, height


def _gif_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 10:
        raise ImageSizeError("invalid GIF (too short)")
    if data[:4] != b"GIF8":
        raise ImageSizeError("invalid GIF signature")
    width = int.from_bytes(data[6:8], "little", signed=False)
    height = int.from_bytes(data[8:10], "little", signed=False)
    if width <= 0 or height <= 0:
        raise ImageSizeError("invalid GIF dimensions")
    return width, height


def _tiff_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ImageSizeError("invalid TIFF (too short)")
    if data[:2] == b"II":
        endian = "little"
    elif data[:2] == b"MM":
        endian = "big"
    else:
        raise ImageSizeError("invalid TIFF byte-order marker")
    magic = int.from_bytes(data[2:4], endian, signed=False)
    if magic != 42:
        raise ImageSizeError("invalid TIFF magic number")
    ifd_offset = int.from_bytes(data[4:8], endian, signed=False)
    if ifd_offset + 2 > len(data):
        raise ImageSizeError("invalid TIFF IFD offset")
    num_entries = int.from_bytes(data[ifd_offset : ifd_offset + 2], endian, signed=False)
    width = height = None
    for i in range(num_entries):
        entry_offset = ifd_offset + 2 + i * 12
        if entry_offset + 12 > len(data):
            break
        tag = int.from_bytes(data[entry_offset : entry_offset + 2], endian, signed=False)
        dtype = int.from_bytes(data[entry_offset + 2 : entry_offset + 4], endian, signed=False)
        # Value is at offset +8; for SHORT (dtype=3) read 2 bytes, for LONG (dtype=4) read 4.
        if dtype == 3:
            val = int.from_bytes(data[entry_offset + 8 : entry_offset + 10], endian, signed=False)
        else:
            val = int.from_bytes(data[entry_offset + 8 : entry_offset + 12], endian, signed=False)
        if tag == 256:  # ImageWidth
            width = val
        elif tag == 257:  # ImageLength
            height = val
        if width is not None and height is not None:
            break
    if width is None or height is None or width <= 0 or height <= 0:
        raise ImageSizeError("could not determine TIFF dimensions")
    return width, height


def _webp_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 30:
        raise ImageSizeError("invalid WebP (too short)")
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ImageSizeError("invalid WebP signature")
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Lossy WebP: width/height at bytes 26-30 (little-endian 16-bit, lower 14 bits).
        if len(data) < 30:
            raise ImageSizeError("invalid VP8 chunk")
        width = int.from_bytes(data[26:28], "little", signed=False) & 0x3FFF
        height = int.from_bytes(data[28:30], "little", signed=False) & 0x3FFF
    elif chunk == b"VP8L":
        # Lossless WebP: 5 bytes starting at offset 21 encode width/height.
        if len(data) < 25:
            raise ImageSizeError("invalid VP8L chunk")
        b0 = data[21]
        b1 = data[22]
        b2 = data[23]
        b3 = data[24]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
    elif chunk == b"VP8X":
        # Extended WebP: canvas size at offset 24-30.
        if len(data) < 30:
            raise ImageSizeError("invalid VP8X chunk")
        width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
    else:
        raise ImageSizeError(f"unknown WebP chunk type: {chunk!r}")
    if width <= 0 or height <= 0:
        raise ImageSizeError("invalid WebP dimensions")
    return width, height


def _pil_size(path: Path) -> tuple[int, int]:
    """Fallback: use PIL/Pillow to read image dimensions."""
    try:
        from PIL import Image
    except ImportError:
        raise ImageSizeError(
            f"unsupported image type: {path.suffix} (install Pillow for broader format support)"
        )
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                raise ImageSizeError("invalid image dimensions from PIL")
            return w, h
    except ImageSizeError:
        raise
    except (OSError, ValueError) as exc:
        raise ImageSizeError(f"PIL failed to read {path}: {exc}") from exc

