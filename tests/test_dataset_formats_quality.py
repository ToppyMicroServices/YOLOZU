"""Tests for broadened dataset / image-format / letterbox support (quality hardening).

Covers:
  - image_size: BMP, GIF, TIFF, WebP native header parsers + PIL fallback
  - dataset loader: picks up .bmp/.webp/.tiff/.gif images
  - letterbox: rectangular (w,h) input_size
  - geometry: zero focal-length protection
  - calibration/distillation: preserving extra entry keys
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.image_size import ImageSizeError, get_image_size

try:
    import numpy as np
    from PIL import Image
except Exception:  # pragma: no cover
    np = None
    Image = None


# ---------------------------------------------------------------------------
# Helpers: write minimal valid image headers without Pillow dependency
# ---------------------------------------------------------------------------

def _write_minimal_bmp(path: Path, w: int = 4, h: int = 3) -> None:
    """Write a minimal 24-bit BMP that image_size can parse."""
    row_bytes = (w * 3 + 3) & ~3
    pixel_data = b"\x00" * (row_bytes * h)
    file_size = 54 + len(pixel_data)
    header = (
        b"BM"
        + struct.pack("<I", file_size)
        + b"\x00\x00\x00\x00"
        + struct.pack("<I", 54)
        + struct.pack("<I", 40)       # DIB header size
        + struct.pack("<i", w)        # width
        + struct.pack("<i", h)        # height (positive = bottom-up)
        + struct.pack("<HH", 1, 24)   # planes, bpp
        + b"\x00" * 24               # rest of DIB header
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + pixel_data)


def _write_minimal_gif(path: Path, w: int = 5, h: int = 3) -> None:
    """Write a minimal GIF87a header."""
    header = (
        b"GIF87a"
        + struct.pack("<HH", w, h)
        + b"\x00\x00\x00"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)


def _write_minimal_tiff_le(path: Path, w: int = 6, h: int = 4) -> None:
    """Write a minimal little-endian TIFF with just width/height IFD entries."""
    # Header: II + 42 + IFD offset (8)
    ifd_offset = 8
    num_entries = 2
    # Each IFD entry = 12 bytes: tag(2) + type(2) + count(4) + value(4)
    entry_width = struct.pack("<HHI I", 256, 4, 1, w)
    entry_height = struct.pack("<HHI I", 257, 4, 1, h)
    next_ifd = struct.pack("<I", 0)
    data = (
        b"II"
        + struct.pack("<H", 42)
        + struct.pack("<I", ifd_offset)
        + struct.pack("<H", num_entries)
        + entry_width
        + entry_height
        + next_ifd
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_minimal_webp_vp8(path: Path, w: int = 8, h: int = 6) -> None:
    """Write a minimal lossy (VP8) WebP."""
    # Lossy WebP: RIFF....WEBPVP8 ....
    # VP8 bitstream starts at offset 20; frame tag at +3 bytes, then
    # 3-byte start code 0x9D012A, then width(16le) height(16le).
    vp8_payload = (
        b"\x9d\x01\x2a"  # start code
        + struct.pack("<H", w)     # width (bits 13-0)
        + struct.pack("<H", h)     # height (bits 13-0)
    )
    vp8_payload = b"\x00\x00\x00" + vp8_payload  # 3-byte frame tag
    vp8_chunk = b"VP8 " + struct.pack("<I", len(vp8_payload)) + vp8_payload
    riff_size = 4 + len(vp8_chunk)
    data = b"RIFF" + struct.pack("<I", riff_size) + b"WEBP" + vp8_chunk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# ===========================================================================
# image_size: new format parsers
# ===========================================================================

class TestImageSizeBMP(unittest.TestCase):
    def test_bmp_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.bmp"
            _write_minimal_bmp(p, 12, 7)
            w, h = get_image_size(p)
            self.assertEqual(w, 12)
            self.assertEqual(h, 7)

    def test_bmp_topdown_negative_height(self):
        """Top-down BMPs encode height as negative."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "td.bmp"
            _write_minimal_bmp(p, 4, 3)
            data = bytearray(p.read_bytes())
            # Negate the height at offset 22..26
            data[22:26] = struct.pack("<i", -3)
            p.write_bytes(bytes(data))
            w, h = get_image_size(p)
            self.assertEqual(w, 4)
            self.assertEqual(h, 3)

    def test_bmp_invalid_signature(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.bmp"
            p.write_bytes(b"XX" + b"\x00" * 50)
            with self.assertRaises(ImageSizeError):
                get_image_size(p)


class TestImageSizeGIF(unittest.TestCase):
    def test_gif_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.gif"
            _write_minimal_gif(p, 10, 8)
            w, h = get_image_size(p)
            self.assertEqual(w, 10)
            self.assertEqual(h, 8)

    def test_gif_invalid_signature(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.gif"
            p.write_bytes(b"NOTG" + b"\x00" * 20)
            with self.assertRaises(ImageSizeError):
                get_image_size(p)


class TestImageSizeTIFF(unittest.TestCase):
    def test_tiff_le_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.tif"
            _write_minimal_tiff_le(p, 20, 15)
            w, h = get_image_size(p)
            self.assertEqual(w, 20)
            self.assertEqual(h, 15)

    def test_tiff_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tiff"
            p.write_bytes(b"ZZ" + b"\x00" * 20)
            with self.assertRaises(ImageSizeError):
                get_image_size(p)


class TestImageSizeWebP(unittest.TestCase):
    def test_webp_vp8_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.webp"
            _write_minimal_webp_vp8(p, 16, 12)
            w, h = get_image_size(p)
            self.assertEqual(w, 16)
            self.assertEqual(h, 12)

    def test_webp_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.webp"
            p.write_bytes(b"RIFF" + b"\x00" * 30)
            with self.assertRaises(ImageSizeError):
                get_image_size(p)


class TestImageSizePILFallback(unittest.TestCase):
    """Unsupported extensions should fall back to PIL if available."""

    @unittest.skipUnless(Image is not None, "Pillow not installed")
    def test_ppm_via_pil(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.ppm"
            Image.new("RGB", (7, 3)).save(p)
            w, h = get_image_size(p)
            self.assertEqual(w, 7)
            self.assertEqual(h, 3)


# ===========================================================================
# dataset loader: new image extensions
# ===========================================================================

@unittest.skipUnless(Image is not None, "Pillow not installed")
class TestDatasetBMPImages(unittest.TestCase):
    def test_bmp_images_discovered(self):
        from yolozu.dataset import build_manifest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "val"
            lbl_dir = root / "labels" / "val"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(img_dir / "0001.bmp")
            (lbl_dir / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            manifest = build_manifest(root, split="val")
            self.assertEqual(len(manifest["images"]), 1)
            self.assertTrue(manifest["images"][0]["image"].endswith(".bmp"))

    def test_webp_images_discovered(self):
        from yolozu.dataset import build_manifest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "val"
            lbl_dir = root / "labels" / "val"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(img_dir / "0001.webp")
            (lbl_dir / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            manifest = build_manifest(root, split="val")
            self.assertEqual(len(manifest["images"]), 1)
            self.assertTrue(manifest["images"][0]["image"].endswith(".webp"))

    def test_tiff_images_discovered(self):
        from yolozu.dataset import build_manifest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "val"
            lbl_dir = root / "labels" / "val"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            Image.new("RGB", (10, 10)).save(img_dir / "0001.tiff")
            (lbl_dir / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            manifest = build_manifest(root, split="val")
            self.assertEqual(len(manifest["images"]), 1)
            self.assertTrue(manifest["images"][0]["image"].endswith(".tiff"))


# ===========================================================================
# letterbox: rectangular (w,h)
# ===========================================================================

class TestLetterboxRectangular(unittest.TestCase):
    def test_square_int_still_works(self):
        from yolozu.letterbox import compute_letterbox

        lb = compute_letterbox(orig_w=640, orig_h=480, input_size=320)
        self.assertEqual(lb.input_size, 320)

    def test_rectangular_tuple(self):
        from yolozu.letterbox import compute_letterbox

        lb = compute_letterbox(orig_w=640, orig_h=480, input_size=(640, 320))
        # Scale limited by height: 320/480 = 0.6667
        self.assertAlmostEqual(lb.scale, 320 / 480, places=4)
        self.assertEqual(lb.input_size, 640)

    def test_rectangular_list(self):
        from yolozu.letterbox import compute_letterbox

        lb = compute_letterbox(orig_w=100, orig_h=100, input_size=[200, 100])
        self.assertAlmostEqual(lb.scale, 1.0, places=4)

    def test_zero_input_raises(self):
        from yolozu.letterbox import compute_letterbox

        with self.assertRaises(ValueError):
            compute_letterbox(orig_w=100, orig_h=100, input_size=(0, 100))


# ===========================================================================
# geometry: zero focal length
# ===========================================================================

class TestGeometryZeroFocal(unittest.TestCase):
    def test_zero_fx_no_crash(self):
        from yolozu.geometry import recover_translation

        x, y, z = recover_translation((320, 240), (0, 0), 1.5, (0, 500, 320, 240))
        self.assertIsInstance(x, float)
        self.assertEqual(z, 1.5)

    def test_zero_fy_no_crash(self):
        from yolozu.geometry import recover_translation

        x, y, z = recover_translation((320, 240), (0, 0), 2.0, (500, 0, 320, 240))
        self.assertIsInstance(y, float)

    def test_corrected_intrinsics_len_check(self):
        from yolozu.geometry import corrected_intrinsics

        with self.assertRaises(ValueError):
            corrected_intrinsics((1, 2, 3), (0, 0, 0, 0))


# ===========================================================================
# calibration: key preservation
# ===========================================================================

class TestCalibrationKeyPreservation(unittest.TestCase):
    def test_extra_keys_preserved(self):
        from yolozu.calibration import calibrate_predictions_entries

        entries = [
            {
                "image": "a.jpg",
                "image_size": {"width": 640, "height": 480},
                "preprocess": {"method": "resize"},
                "detections": [{"score": 0.9, "class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}],
            }
        ]
        out = calibrate_predictions_entries(entries, temperature=1.0)
        self.assertIn("image_size", out[0])
        self.assertIn("preprocess", out[0])
        self.assertEqual(out[0]["image_size"]["width"], 640)


# ===========================================================================
# distillation: key preservation
# ===========================================================================

class TestDistillationKeyPreservation(unittest.TestCase):
    def test_extra_keys_preserved(self):
        from yolozu.distillation import distill_predictions

        student = [
            {
                "image": "a.jpg",
                "image_size": {"width": 640, "height": 480},
                "detections": [{"score": 0.8, "class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}],
            }
        ]
        teacher = [
            {
                "image": "a.jpg",
                "detections": [{"score": 0.9, "class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}],
            }
        ]
        out, stats = distill_predictions(student, teacher)
        self.assertIn("image_size", out[0])
        self.assertEqual(out[0]["image_size"]["width"], 640)


# ===========================================================================
# constraints: r_mat validation
# ===========================================================================

class TestConstraintsRmatValidation(unittest.TestCase):
    def test_none_rmat_uses_identity(self):
        from yolozu.constraints import apply_constraints

        cfg = {"enabled": {"upright": True}, "upright": {"default": {"roll_deg": [-10, 10], "pitch_deg": [-10, 10]}}}
        result = apply_constraints(
            cfg, class_key=0, bbox_wh=(0.1, 0.1), size_wh=(0.1, 0.1),
            intrinsics_fx_fy=(500, 500), t_xyz=(0, 0, 1), r_mat=None, z_pred=1.0,
        )
        # Identity => roll=0, pitch=0 => within range => zero violation
        self.assertAlmostEqual(result["upright_violation"], 0.0, places=4)

    def test_invalid_rmat_falls_back(self):
        from yolozu.constraints import apply_constraints

        cfg = {"enabled": {"upright": False}}
        # 2x2 matrix (invalid) — should not crash
        result = apply_constraints(
            cfg, class_key=0, bbox_wh=(0.1, 0.1), size_wh=(0.1, 0.1),
            intrinsics_fx_fy=(500, 500), t_xyz=(0, 0, 1), r_mat=[[1, 0], [0, 1]], z_pred=1.0,
        )
        self.assertIn("plane_ok", result)


# ===========================================================================
# gates: safe dict fallback
# ===========================================================================

class TestGatesSafe(unittest.TestCase):
    def test_empty_weights_dict(self):
        from yolozu.gates import final_score

        score = final_score(0.9, 0.5, 0.1, 0.1, {})
        # defaults: 1.0*0.9 + 1.0*0.5 - 1.0*(0.1+0.1) = 1.2
        self.assertAlmostEqual(score, 1.2, places=4)

    def test_non_dict_weights(self):
        from yolozu.gates import final_score

        score = final_score(0.9, 0.5, 0.1, 0.1, None)
        self.assertAlmostEqual(score, 1.2, places=4)


if __name__ == "__main__":
    unittest.main()
