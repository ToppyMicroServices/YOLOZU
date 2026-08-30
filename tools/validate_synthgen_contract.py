import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.contracts.synthgen import validate_synthgen_sample  # noqa: E402


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(data)
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item[{i}] expected JSON object")
            out.append(item)
        return out
    raise ValueError(f"{path}: expected object/list/jsonl")


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _resolve_path(path_like: str, *, base_dir: Path, root_dir: Path) -> Path | None:
    p = Path(path_like)
    if p.is_absolute() and _path_exists(p):
        return p
    cand1 = base_dir / p
    if _path_exists(cand1):
        return cand1
    cand2 = root_dir / p
    if _path_exists(cand2):
        return cand2
    return None


def _load_artifact(value: Any, *, base_dir: Path, root_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    resolved = _resolve_path(value, base_dir=base_dir, root_dir=root_dir)
    if resolved is None:
        return value
    suffix = resolved.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        with Image.open(resolved) as img:
            return np.asarray(img)
    if suffix == ".npy":
        return np.load(resolved, allow_pickle=False)
    if suffix == ".json":
        return json.loads(resolved.read_text(encoding="utf-8"))
    return resolved.read_text(encoding="utf-8")


def _materialize_record(record: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    out = dict(record)
    root_dir = source_path.parent.parent if source_path.parent.name == "shards" else source_path.parent
    for field in (
        "image",
        "depth_ndc",
        "inst_id",
        "sem_id",
        "bbox2d_visible",
        "kpts2d",
        "kpts3d_object",
        "pose_obj2cam",
    ):
        if field in out:
            out[field] = _load_artifact(out[field], base_dir=source_path.parent, root_dir=root_dir)
    if "scene_spec" in out:
        out["scene_spec"] = _load_artifact(out["scene_spec"], base_dir=source_path.parent, root_dir=root_dir)
    if "inst_map" in out:
        out["inst_map"] = _load_artifact(out["inst_map"], base_dir=source_path.parent, root_dir=root_dir)
    if "asset_ids" in out and isinstance(out["asset_ids"], str):
        out["asset_ids"] = _load_artifact(out["asset_ids"], base_dir=source_path.parent, root_dir=root_dir)
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description="Validate SynthGen sample contract records.")
    p.add_argument("--input", required=True, help="Path to sample JSON/JSONL (inline values or shard metadata records).")
    p.add_argument("--max-samples", type=int, default=100, help="Max records to validate.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    path = Path(args.input)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise SystemExit(f"input not found: {path}")

    records = _load_records(path)
    limit = max(int(args.max_samples), 0)
    checked = 0
    errors: list[str] = []
    for i, sample in enumerate(records):
        if limit and checked >= limit:
            break
        checked += 1
        materialized = _materialize_record(sample, source_path=path)
        result = validate_synthgen_sample(materialized)
        if not result.ok:
            for err in result.errors:
                errors.append(f"record[{i}]: {err}")

    if errors:
        for err in errors[:100]:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print(f"OK: validated {checked} synthgen samples")


if __name__ == "__main__":
    main()
