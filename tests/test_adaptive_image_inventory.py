from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from yolozu.adaptive.inventory import build_decoded_input_inventory


class TestAdaptiveImageInventory(unittest.TestCase):
    def _image(
        self,
        path: Path,
        *,
        color: tuple[int, int, int] = (255, 0, 0),
        size: tuple[int, int] = (2, 2),
        format_name: str | None = None,
    ) -> None:
        image = Image.new("RGB", size, color)
        image.save(path, format=format_name)

    def test_single_image_accepts_jpeg_png_and_single_frame_webp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (("a.jpg", "JPEG"), ("b.png", "PNG"), ("c.webp", "WEBP"))
            for name, format_name in cases:
                with self.subTest(format=format_name):
                    path = root / name
                    self._image(path, format_name=format_name)
                    inventory = build_decoded_input_inventory(
                        path,
                        input_mode="single_image",
                        workspace_root=root,
                        max_images=1,
                    )
                    self.assertEqual(inventory.input_count, 1)
                    self.assertEqual(inventory.inputs[0].width, 2)
                    self.assertEqual(inventory.inputs[0].height, 2)
                    self.assertEqual(inventory.inputs[0].color_mode, "RGB")
                    self.assertEqual(len(inventory.local_input_digest), 64)
                    local = json.dumps(inventory.to_local_provenance())
                    self.assertNotIn(name, local)
                    self.assertNotIn("source_sha256", local)

    def test_local_digest_changes_with_bytes_and_directory_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "a.png", color=(255, 0, 0))
            self._image(image_dir / "b.png", color=(0, 0, 255))
            first = build_decoded_input_inventory(
                image_dir,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            )
            self._image(image_dir / "a.png", color=(0, 0, 255))
            self._image(image_dir / "b.png", color=(255, 0, 0))
            second = build_decoded_input_inventory(
                image_dir,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            )
            self.assertNotEqual(first.local_input_digest, second.local_input_digest)
            self.assertEqual([item.index for item in first.inputs], [0, 1])

    def test_directory_uses_normalized_utf8_order_and_ignores_regular_nonimages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "z.png", size=(3, 2))
            self._image(image_dir / "a.png", size=(2, 2))
            (image_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            inventory = build_decoded_input_inventory(
                image_dir,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            )
            self.assertEqual([item.width for item in inventory.inputs], [2, 3])
            self.assertEqual(inventory.input_order, "normalized_basename_utf8_v1")

    def test_rejects_animation_archive_and_malformed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            animated = root / "animated.webp"
            first = Image.new("RGB", (2, 2), (255, 0, 0))
            second = Image.new("RGB", (2, 2), (0, 0, 255))
            try:
                first.save(animated, format="WEBP", save_all=True, append_images=[second], duration=20)
            except OSError as exc:
                self.skipTest(f"Pillow WebP animation unavailable: {exc}")
            archive = root / "archive.jpg"
            archive.write_bytes(b"PK\x03\x04" + b"x" * 32)
            malformed = root / "malformed.png"
            malformed.write_bytes(b"\x89PNG\r\n\x1a\ninvalid")
            for path in (animated, archive, malformed):
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    build_decoded_input_inventory(
                        path,
                        input_mode="single_image",
                        workspace_root=root,
                        max_images=1,
                    )

    def test_rejects_workspace_escape_symlink_nested_and_nonregular_entries(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_path = Path(outside) / "outside.png"
            self._image(outside_path)
            with self.assertRaises(ValueError):
                build_decoded_input_inventory(
                    outside_path,
                    input_mode="single_image",
                    workspace_root=root,
                    max_images=1,
                )

            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "ok.png")
            symlink = image_dir / "link.png"
            try:
                symlink.symlink_to(outside_path)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaises(ValueError):
                build_decoded_input_inventory(
                    image_dir,
                    input_mode="bounded_directory",
                    workspace_root=root,
                    max_images=2,
                )
            symlink.unlink()

            real_dir = root / "real"
            real_dir.mkdir()
            self._image(real_dir / "inside.png")
            directory_link = root / "directory-link"
            directory_link.symlink_to(real_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                build_decoded_input_inventory(
                    directory_link / "inside.png",
                    input_mode="single_image",
                    workspace_root=root,
                    max_images=1,
                )

            (image_dir / "nested").mkdir()
            with self.assertRaises(ValueError):
                build_decoded_input_inventory(
                    image_dir,
                    input_mode="bounded_directory",
                    workspace_root=root,
                    max_images=2,
                )
            (image_dir / "nested").rmdir()

            fifo = image_dir / "pipe"
            if hasattr(os, "mkfifo"):
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):
                    build_decoded_input_inventory(
                        image_dir,
                        input_mode="bounded_directory",
                        workspace_root=root,
                        max_images=2,
                    )

    def test_rejects_normalized_name_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "A.png")
            self._image(image_dir / "Ａ.png")
            with self.assertRaisesRegex(ValueError, "normalized basename collision"):
                build_decoded_input_inventory(
                    image_dir,
                    input_mode="bounded_directory",
                    workspace_root=root,
                    max_images=2,
                )

    def test_rejects_entry_and_image_count_overflow_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "a.png")
            self._image(image_dir / "b.png")
            with self.assertRaisesRegex(ValueError, "image count"):
                build_decoded_input_inventory(
                    image_dir,
                    input_mode="bounded_directory",
                    workspace_root=root,
                    max_images=1,
                )
            (image_dir / "a.png").unlink()
            (image_dir / "b.png").unlink()
            (image_dir / "a.txt").write_text("a", encoding="utf-8")
            (image_dir / "b.txt").write_text("b", encoding="utf-8")
            with mock.patch("yolozu.adaptive.inventory.MAX_DIRECTORY_ENTRIES", 1):
                with self.assertRaisesRegex(ValueError, "direct entries"):
                    build_decoded_input_inventory(
                        image_dir,
                        input_mode="bounded_directory",
                        workspace_root=root,
                        max_images=1,
                    )

    def test_enforces_source_byte_dimension_and_pixel_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = root / "one.png"
            self._image(one, size=(2, 2))
            source_size = one.stat().st_size
            with mock.patch("yolozu.adaptive.inventory.MAX_SOURCE_FILE_BYTES", source_size - 1):
                with self.assertRaisesRegex(ValueError, "source cap"):
                    build_decoded_input_inventory(
                        one,
                        input_mode="single_image",
                        workspace_root=root,
                        max_images=1,
                    )
            with mock.patch("yolozu.adaptive.inventory.MAX_IMAGE_DIMENSION", 1):
                with self.assertRaisesRegex(ValueError, "dimension"):
                    build_decoded_input_inventory(
                        one,
                        input_mode="single_image",
                        workspace_root=root,
                        max_images=1,
                    )
            with mock.patch("yolozu.adaptive.inventory.MAX_IMAGE_PIXELS", 3):
                with self.assertRaisesRegex(ValueError, "pixel cap"):
                    build_decoded_input_inventory(
                        one,
                        input_mode="single_image",
                        workspace_root=root,
                        max_images=1,
                    )

    def test_enforces_source_and_decoded_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._image(image_dir / "a.png", color=(255, 0, 0))
            self._image(image_dir / "b.png", color=(0, 0, 255))
            total = sum(path.stat().st_size for path in image_dir.iterdir())
            with mock.patch("yolozu.adaptive.inventory.MAX_SOURCE_TOTAL_BYTES", total - 1):
                with self.assertRaisesRegex(ValueError, "source-total"):
                    build_decoded_input_inventory(
                        image_dir,
                        input_mode="bounded_directory",
                        workspace_root=root,
                        max_images=2,
                    )
            with mock.patch("yolozu.adaptive.inventory.MAX_TOTAL_PIXELS", 7):
                with self.assertRaisesRegex(ValueError, "decoded-pixel"):
                    build_decoded_input_inventory(
                        image_dir,
                        input_mode="bounded_directory",
                        workspace_root=root,
                        max_images=2,
                    )


if __name__ == "__main__":
    unittest.main()
