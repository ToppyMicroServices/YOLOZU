from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from yolozu.contracts.synthgen import normalize_synthgen_sample


_ARRAY_FIELDS = (
    "image",
    "depth_ndc",
    "inst_id",
    "sem_id",
    "bbox2d_visible",
    "kpts2d",
    "kpts3d_object",
    "pose_obj2cam",
)


def _resolve_file_path(value: str, *, shard_dir: Path, root: Path) -> Path | None:
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    cand1 = shard_dir / p
    if cand1.exists():
        return cand1
    cand2 = root / p
    if cand2.exists():
        return cand2
    return None


def _load_path_value(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        with Image.open(path) as img:
            arr = np.asarray(img)
        return arr
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _decode_json_or_path(value: Any, *, shard_dir: Path, root: Path) -> Any:
    if isinstance(value, str):
        resolved = _resolve_file_path(value, shard_dir=shard_dir, root=root)
        if resolved is not None:
            return _load_path_value(resolved)
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _materialize_record(raw: dict[str, Any], *, root: Path, shard_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = dict(raw)

    for key in _ARRAY_FIELDS:
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, str):
            resolved = _resolve_file_path(value, shard_dir=shard_dir, root=root)
            if resolved is not None:
                out[key] = _load_path_value(resolved)
                continue
        out[key] = value

    out["scene_spec"] = _decode_json_or_path(out.get("scene_spec"), shard_dir=shard_dir, root=root)
    out["inst_map"] = _decode_json_or_path(out.get("inst_map"), shard_dir=shard_dir, root=root)

    asset_ids = out.get("asset_ids")
    if isinstance(asset_ids, str):
        resolved = _resolve_file_path(asset_ids, shard_dir=shard_dir, root=root)
        if resolved is not None and resolved.suffix.lower() == ".json":
            loaded = _load_path_value(resolved)
            if isinstance(loaded, list):
                out["asset_ids"] = loaded

    return normalize_synthgen_sample(out)


def _collect_shards(root: Path, shards: list[str] | None) -> list[Path]:
    if shards:
        out: list[Path] = []
        for shard in shards:
            p = Path(shard)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                raise ValueError(f"shard not found: {p}")
            out.append(p)
        return out

    paths = sorted((root / "shards").glob("*.jsonl")) if (root / "shards").is_dir() else []
    paths.extend(sorted(root.glob("*.jsonl")))
    if not paths:
        raise ValueError(f"no shard jsonl files found under {root}")
    return paths


@dataclass(frozen=True)
class _ShardEntry:
    shard_path: Path
    line_no: int
    raw: dict[str, Any]


class SynthGenShardDataset:
    """Load SynthGen samples from JSONL shard metadata + external blobs."""

    def __init__(
        self,
        root: str | Path,
        *,
        shards: list[str] | None = None,
        schema_id: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise ValueError(f"dataset root not found: {self.root}")
        self.schema_id = str(schema_id) if schema_id else None
        self._entries = self._read_entries(_collect_shards(self.root, shards), schema_id=self.schema_id)

    def _read_entries(self, shard_paths: list[Path], *, schema_id: str | None) -> list[_ShardEntry]:
        entries: list[_ShardEntry] = []
        for shard_path in shard_paths:
            for line_no, raw_line in enumerate(shard_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"{shard_path}:{line_no}: JSON object expected")
                if schema_id and str(data.get("schema_id")) != schema_id:
                    continue
                entries.append(_ShardEntry(shard_path=shard_path, line_no=line_no, raw=data))
        return entries

    def __len__(self) -> int:
        return int(len(self._entries))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self._entries):
            raise IndexError(index)
        entry = self._entries[int(index)]
        sample = _materialize_record(entry.raw, root=self.root, shard_dir=entry.shard_path.parent)
        sample.setdefault("sample_id", entry.raw.get("sample_id") or f"{entry.shard_path.stem}:{entry.line_no}")
        sample.setdefault("schema_id", entry.raw.get("schema_id"))
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self._entries)):
            yield self[idx]


def collate_synthgen_batch(
    samples: list[dict[str, Any]],
    *,
    pad_keypoints: bool = False,
    pad_instance_labels: bool | None = None,
) -> dict[str, Any]:
    """Collate samples, retaining ``pad_keypoints`` as a compatibility alias."""

    if not samples:
        return {}
    pad_instances = pad_keypoints if pad_instance_labels is None else pad_instance_labels

    out: dict[str, Any] = {}
    keys: set[str] = set()
    for sample in samples:
        keys.update(sample.keys())

    for key in sorted(keys):
        values = [sample.get(key) for sample in samples]
        if key in {"bbox2d_visible", "kpts3d_object", "pose_obj2cam"} and pad_instances:
            arrays = [v for v in values if isinstance(v, np.ndarray)]
            if len(arrays) != len(values):
                out[key] = values
                continue
            if key == "pose_obj2cam":
                arrays = [arr[None, ...] if arr.ndim == 2 else arr for arr in arrays]
            trailing_shapes = {tuple(arr.shape[1:]) for arr in arrays}
            if len(trailing_shapes) != 1:
                out[key] = values
                continue
            max_n = max(int(arr.shape[0]) for arr in arrays)
            trailing_shape = tuple(arrays[0].shape[1:])
            fill_value = {
                "bbox2d_visible": -1.0,
                "kpts3d_object": np.nan,
                "pose_obj2cam": 0.0,
            }[key]
            padded = np.full(
                (len(arrays), max_n, *trailing_shape),
                fill_value,
                dtype=np.float32,
            )
            for i, arr in enumerate(arrays):
                padded[i, : int(arr.shape[0])] = arr
            out[key] = padded
            continue
        if key == "kpts2d" and pad_instances:
            arrays = [v for v in values if isinstance(v, np.ndarray)]
            if len(arrays) != len(values):
                out[key] = values
                continue
            max_n = max(int(arr.shape[0]) for arr in arrays)
            keypoint_dim = int(arrays[0].shape[1]) if arrays and arrays[0].ndim >= 2 else 0
            padded = np.zeros((len(arrays), max_n, keypoint_dim, 3), dtype=np.float32)
            mask = np.zeros((len(arrays), max_n, keypoint_dim), dtype=bool)
            for i, arr in enumerate(arrays):
                n = int(arr.shape[0])
                padded[i, :n] = arr
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    mask[i, :n] = np.rint(arr[..., 2]).astype(np.int64) > 0
            out[key] = padded
            out["kpts2d_mask"] = mask
            instance_mask = np.zeros((len(arrays), max_n), dtype=bool)
            for i, arr in enumerate(arrays):
                instance_mask[i, : int(arr.shape[0])] = True
            out["instance_mask"] = instance_mask
            continue

        if values and all(isinstance(v, np.ndarray) for v in values):
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) == 1:
                out[key] = np.stack(values, axis=0)
            else:
                out[key] = values
            continue

        out[key] = values

    return out
