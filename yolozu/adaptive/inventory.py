"""Bounded read-only decoded input inventory for adaptive recommendation."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError, __version__ as PILLOW_VERSION

from .canonical import canonical_sha256_v1

__all__ = [
    "DecodedInputInventory",
    "DecodedInputObservation",
    "build_decoded_input_inventory",
]


MAX_SOURCE_FILE_BYTES = 67_108_864
MAX_SOURCE_TOTAL_BYTES = 536_870_912
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 64_000_000
MAX_TOTAL_PIXELS = 512_000_000
MAX_DIRECTORY_ENTRIES = 1_024

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_PILLOW_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
_ARCHIVE_MAGICS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"BZh",
    b"7z\xbc\xaf\x27\x1c",
)


@dataclass(frozen=True)
class DecodedInputObservation:
    index: int
    width: int
    height: int
    color_mode: str
    orientation_policy: str = "exif_transpose_v1"

    def to_workload_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode,
            "orientation_policy": self.orientation_policy,
        }


@dataclass(frozen=True)
class DecodedInputInventory:
    input_mode: str
    input_count: int
    input_order: str
    inputs: tuple[DecodedInputObservation, ...]
    decoder_id: str
    decoder_version: str
    source_total_bytes: int
    local_input_digest: str

    def to_local_provenance(self) -> dict[str, Any]:
        """Return only the aggregate local digest, never filenames or member hashes."""

        return {
            "input_count": self.input_count,
            "local_input_digest": self.local_input_digest,
        }


def _workspace_path(path: str | Path, *, workspace_root: str | Path) -> Path:
    workspace_input = Path(workspace_root)
    workspace_lexical = Path(os.path.abspath(workspace_input))
    if workspace_lexical.is_symlink():
        raise ValueError("workspace_root: symlinks are invalid")
    try:
        workspace = workspace_lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError("workspace_root: unavailable") from exc
    if not workspace.is_dir():
        raise ValueError("workspace_root: expected directory")

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_lexical / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical_relative = lexical.relative_to(workspace_lexical)
    except (OSError, ValueError) as exc:
        raise ValueError("input path: unavailable or outside workspace") from exc

    current = workspace_lexical
    for component in lexical_relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("input path: symlink component is invalid")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise ValueError("input path: unavailable or outside workspace") from exc
    return resolved


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("input file: unable to open regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input file: expected regular file")
        if before.st_size > MAX_SOURCE_FILE_BYTES:
            raise ValueError("input file: exceeds 64 MiB source cap")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, MAX_SOURCE_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_SOURCE_FILE_BYTES:
                raise ValueError("input file: exceeds 64 MiB source cap")
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or total != after.st_size
        ):
            raise ValueError("input file: changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _check_dimensions(width: int, height: int) -> None:
    if not (1 <= width <= MAX_IMAGE_DIMENSION and 1 <= height <= MAX_IMAGE_DIMENSION):
        raise ValueError("decoded image: dimension exceeds v1 bounds")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("decoded image: exceeds 64 million pixel cap")


def _decode_image_bytes(source: bytes) -> tuple[int, int, str]:
    if any(source.startswith(magic) for magic in _ARCHIVE_MAGICS):
        raise ValueError("input file: archive content is invalid")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source)) as image:
                if image.format not in _PILLOW_FORMATS:
                    raise ValueError("decoded image: unsupported format")
                if int(getattr(image, "n_frames", 1)) != 1 or bool(
                    getattr(image, "is_animated", False)
                ):
                    raise ValueError("decoded image: animation is invalid")
                _check_dimensions(int(image.width), int(image.height))
                transposed = ImageOps.exif_transpose(image)
                transposed.load()
                width, height = int(transposed.width), int(transposed.height)
                _check_dimensions(width, height)
                color_mode = str(transposed.mode)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("decoded image: decompression-bomb condition") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("decoded image: malformed or unsupported data") from exc
    return width, height, color_mode


def _directory_image_paths(directory: Path, *, max_images: int) -> list[Path]:
    entries: list[tuple[bytes, Path]] = []
    normalized_names: set[str] = set()
    total_entries = 0
    try:
        iterator = os.scandir(directory)
    except OSError as exc:
        raise ValueError("bounded_directory: unable to enumerate") from exc
    with iterator:
        for entry in iterator:
            total_entries += 1
            if total_entries > MAX_DIRECTORY_ENTRIES:
                raise ValueError("bounded_directory: exceeds 1024 direct entries")
            normalized_name = unicodedata.normalize("NFKC", entry.name)
            if normalized_name in normalized_names:
                raise ValueError("bounded_directory: normalized basename collision")
            normalized_names.add(normalized_name)
            if entry.is_symlink():
                raise ValueError("bounded_directory: symlink entry is invalid")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError("bounded_directory: unable to inspect entry") from exc
            if stat.S_ISDIR(info.st_mode):
                raise ValueError("bounded_directory: nested directory is invalid")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("bounded_directory: non-regular entry is invalid")
            suffix = Path(normalized_name).suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                entries.append((normalized_name.encode("utf-8"), Path(entry.path)))
                if len(entries) > max_images:
                    raise ValueError("bounded_directory: image count exceeds max_images")
    if not entries:
        raise ValueError("bounded_directory: no candidate image files")
    entries.sort(key=lambda item: item[0])
    return [path for _, path in entries]


def build_decoded_input_inventory(
    input_path: str | Path,
    *,
    input_mode: str,
    workspace_root: str | Path,
    max_images: int,
) -> DecodedInputInventory:
    """Read and validate local images once without retaining paths or member hashes."""

    if input_mode not in {"single_image", "bounded_directory"}:
        raise ValueError("input_mode: unsupported value")
    if isinstance(max_images, bool) or not isinstance(max_images, int) or not (1 <= max_images <= 100):
        raise ValueError("max_images: expected 1..100")
    if input_mode == "single_image" and max_images != 1:
        raise ValueError("max_images: single_image requires exactly 1")

    resolved = _workspace_path(input_path, workspace_root=workspace_root)
    if input_mode == "single_image":
        if not resolved.is_file():
            raise ValueError("single_image: expected regular file")
        paths = [resolved]
        input_order = "single_image_v1"
    else:
        if not resolved.is_dir():
            raise ValueError("bounded_directory: expected directory")
        paths = _directory_image_paths(resolved, max_images=max_images)
        input_order = "normalized_basename_utf8_v1"

    observations: list[DecodedInputObservation] = []
    local_members: list[dict[str, Any]] = []
    source_total = 0
    decoded_total_pixels = 0
    for index, path in enumerate(paths):
        source = _read_regular_file(path)
        source_total += len(source)
        if source_total > MAX_SOURCE_TOTAL_BYTES:
            raise ValueError("input inventory: exceeds 512 MiB source-total cap")
        width, height, color_mode = _decode_image_bytes(source)
        decoded_total_pixels += width * height
        if decoded_total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError("input inventory: exceeds 512 million decoded-pixel cap")
        local_members.append(
            {
                "index": index,
                "source_byte_length": len(source),
                "source_sha256": hashlib.sha256(source).hexdigest(),
            }
        )
        observations.append(
            DecodedInputObservation(
                index=index,
                width=width,
                height=height,
                color_mode=color_mode,
            )
        )
    local_input_digest = canonical_sha256_v1(local_members)
    return DecodedInputInventory(
        input_mode=input_mode,
        input_count=len(observations),
        input_order=input_order,
        inputs=tuple(observations),
        decoder_id="pillow",
        decoder_version=str(PILLOW_VERSION),
        source_total_bytes=source_total,
        local_input_digest=local_input_digest,
    )
