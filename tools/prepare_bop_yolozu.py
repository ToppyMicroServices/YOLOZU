import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

_OWNERSHIP_FILE = ".yolozu_bop_output.json"
_OWNERSHIP_KIND = "yolozu_bop_conversion_output"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a BOP dataset split into a YOLOZU YOLO-format dataset with sidecars.")
    p.add_argument("--bop-root", required=True, help="Path to extracted BOP dataset root (e.g., /tmp/bop/tless).")
    p.add_argument("--split", required=True, help="BOP split folder name (e.g., train_primesense, val, test).")
    p.add_argument("--out", required=True, help="Output dataset root (YOLO images/ + labels/).")
    p.add_argument("--out-split", default="train2017", help="Output split name under images/ and labels/ (default: train2017).")
    p.add_argument(
        "--bbox-source",
        choices=("bbox_visib", "bbox_obj", "bbox_vis"),
        default="bbox_visib",
        help="Which BOP bbox field to use (default: bbox_visib). 'bbox_vis' is a legacy alias for bbox_visib.",
    )
    p.add_argument("--visib-fract-min", type=float, default=0.0, help="Minimum visibility fraction (default: 0.0).")
    p.add_argument("--max-scenes", type=int, default=None, help="Optional cap for scenes to convert.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for images to convert.")
    p.add_argument(
        "--partition-modulus",
        type=int,
        default=None,
        help="Optional deterministic frame partition modulus (for example 5).",
    )
    p.add_argument(
        "--partition-remainder",
        type=int,
        default=0,
        help="Selected remainder for --partition-modulus (default: 0).",
    )
    p.add_argument(
        "--partition-mode",
        choices=("include", "exclude"),
        default="include",
        help="Include or exclude frames matching image_id %% modulus (default: include).",
    )
    p.add_argument("--link-images", action="store_true", help="Symlink images instead of copying (recommended).")
    p.add_argument(
        "--class-map",
        default="obj_id_minus_1",
        choices=("obj_id_minus_1",),
        help="Class id mapping (default: obj_id_minus_1).",
    )
    p.add_argument("--t-scale", type=float, default=0.001, help="Scale for BOP translation units to meters (default: 0.001 for mm->m).")
    p.add_argument(
        "--models-dir",
        default=None,
        help="Optional BOP model directory; auto-detects models_eval, models_cad, then models under --bop-root.",
    )
    p.add_argument(
        "--cad-max-points",
        type=int,
        default=1000,
        help="Maximum deterministic CAD points stored per object for ADD/ADDS evaluation (default: 1000).",
    )
    output_mode = p.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only when its YOLOZU BOP ownership marker is valid.",
    )
    output_mode.add_argument(
        "--append-owned",
        action="store_true",
        help="Add a new output split only when the existing root has a valid YOLOZU BOP ownership marker.",
    )
    return p.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _owned_marker_valid(path: Path) -> bool:
    marker = path / _OWNERSHIP_FILE
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("kind") == _OWNERSHIP_KIND and payload.get("schema_version") == 1


def _prepare_owned_output(
    path: Path,
    *,
    overwrite: bool,
    append_owned: bool,
    source_root: Path,
    out_split: str,
) -> None:
    if path.is_symlink():
        raise SystemExit(f"refusing symlink output directory: {path}")
    resolved = path.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve(), source_root.resolve()}
    if resolved in protected:
        raise SystemExit(f"refusing protected output directory: {resolved}")
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(f"refusing output inside the BOP source: {resolved}")
    try:
        source_root.resolve().relative_to(resolved)
    except ValueError:
        pass
    else:
        raise SystemExit(f"refusing output that contains the BOP source: {resolved}")

    if path.exists() and not path.is_dir():
        raise SystemExit(f"refusing non-directory output path: {path}")

    if append_owned:
        if not path.is_dir() or not _owned_marker_valid(path):
            raise SystemExit(f"refusing to append to unowned output directory: {path}")
        for split_root in (path / "images" / out_split, path / "labels" / out_split):
            if split_root.exists():
                raise SystemExit(f"refusing to replace existing output split during append: {split_root}")
        return

    if path.exists():
        if not overwrite:
            raise SystemExit(f"output directory already exists: {path} (pass --overwrite only for YOLOZU-owned output)")
        if not _owned_marker_valid(path):
            raise SystemExit(f"refusing to delete unowned output directory: {path}")
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    (path / _OWNERSHIP_FILE).write_text(
        json.dumps({"schema_version": 1, "kind": _OWNERSHIP_KIND}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    _ensure_dir(dst.parent)
    if dst.exists():
        return
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _find_image(scene_dir: Path, subdir: str, image_id: int) -> Path | None:
    stem = f"{image_id:06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        cand = scene_dir / subdir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _read_ply_xyz(path: Path, *, scale: float, max_points: int) -> list[list[float]]:
    with path.open("rb") as handle:
        first = handle.readline()
        if first.rstrip(b"\r\n") != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = None
        vertex_count = None
        current_element = None
        vertex_properties: list[tuple[str, str]] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"truncated PLY header: {path}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"non-ASCII PLY header: {path}") from exc
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"unsupported vertex property in PLY: {line!r}")
                vertex_properties.append((parts[2], parts[1]))

        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"unsupported PLY format {fmt!r}: {path}")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        names = [name for name, _ in vertex_properties]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError(f"PLY vertex properties must contain x/y/z: {path}")
        xyz_indices = (names.index("x"), names.index("y"), names.index("z"))
        rows: list[list[float]] = []
        if fmt == "ascii":
            for _ in range(vertex_count):
                values = handle.readline().decode("ascii").split()
                if len(values) < len(vertex_properties):
                    raise ValueError(f"truncated PLY vertex row: {path}")
                rows.append(
                    [
                        float(values[xyz_indices[0]]) * scale,
                        float(values[xyz_indices[1]]) * scale,
                        float(values[xyz_indices[2]]) * scale,
                    ]
                )
        else:
            try:
                row_format = "<" + "".join(_PLY_SCALAR_FORMATS[data_type] for _, data_type in vertex_properties)
            except KeyError as exc:
                raise ValueError(f"unsupported PLY scalar type {exc.args[0]!r}: {path}") from exc
            row_size = struct.calcsize(row_format)
            for _ in range(vertex_count):
                raw = handle.read(row_size)
                if len(raw) != row_size:
                    raise ValueError(f"truncated PLY vertex data: {path}")
                values = struct.unpack(row_format, raw)
                rows.append(
                    [
                        float(values[xyz_indices[0]]) * scale,
                        float(values[xyz_indices[1]]) * scale,
                        float(values[xyz_indices[2]]) * scale,
                    ]
                )

    if len(rows) <= max_points:
        return rows
    indices = [(idx * (len(rows) - 1)) // (max_points - 1) for idx in range(max_points)]
    return [rows[idx] for idx in indices]


def _resolve_models_dir(bop_root: Path, value: str | None) -> Path | None:
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_dir():
            raise SystemExit(f"models directory not found: {candidate}")
        return candidate
    for name in ("models_eval", "models_cad", "models"):
        candidate = bop_root / name
        if candidate.is_dir():
            return candidate
    return None


def _bbox_xywh_to_cxcywh_norm(bbox_xywh: list[float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    bw = w / float(width)
    bh = h / float(height)
    return float(cx), float(cy), float(bw), float(bh)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    bop_root = Path(str(args.bop_root))
    if not bop_root.is_absolute():
        bop_root = Path.cwd() / bop_root
    split_dir = bop_root / str(args.split)
    if not split_dir.exists():
        raise SystemExit(f"split not found: {split_dir}")

    out_root = Path(str(args.out))
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    out_split = str(args.out_split)
    if not out_split or out_split in {".", ".."} or "/" in out_split or "\\" in out_split:
        raise SystemExit(f"--out-split must be one directory name: {out_split!r}")
    if args.partition_modulus is not None:
        if int(args.partition_modulus) < 2:
            raise SystemExit("--partition-modulus must be >= 2")
        if not 0 <= int(args.partition_remainder) < int(args.partition_modulus):
            raise SystemExit("--partition-remainder must be in [0, partition-modulus)")
    if int(args.cad_max_points) < 2:
        raise SystemExit("--cad-max-points must be >= 2")
    models_dir = _resolve_models_dir(bop_root, args.models_dir)
    models_info: dict[str, Any] = {}
    if models_dir is not None and (models_dir / "models_info.json").is_file():
        loaded_models_info = _load_json(models_dir / "models_info.json")
        if isinstance(loaded_models_info, dict):
            models_info = loaded_models_info
    _prepare_owned_output(
        out_root,
        overwrite=bool(args.overwrite),
        append_owned=bool(args.append_owned),
        source_root=bop_root,
        out_split=out_split,
    )
    out_images = out_root / "images" / out_split
    out_labels = out_root / "labels" / out_split
    _ensure_dir(out_images)
    _ensure_dir(out_labels)
    cad_output = out_root / "cad_points"
    cad_records: dict[int, dict[str, Any]] = {}

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: int(args.max_scenes)]

    converted_images = 0
    for scene_dir in scene_dirs:
        gt_path = scene_dir / "scene_gt.json"
        cam_path = scene_dir / "scene_camera.json"
        info_path = scene_dir / "scene_gt_info.json"
        if not (gt_path.exists() and cam_path.exists() and info_path.exists()):
            continue

        scene_gt = _load_json(gt_path)
        scene_cam = _load_json(cam_path)
        scene_info = _load_json(info_path)

        image_ids = sorted([int(k) for k in scene_gt.keys()])
        for image_id in image_ids:
            if args.partition_modulus is not None:
                matches = image_id % int(args.partition_modulus) == int(args.partition_remainder)
                if (args.partition_mode == "include" and not matches) or (
                    args.partition_mode == "exclude" and matches
                ):
                    continue
            if args.max_images is not None and converted_images >= int(args.max_images):
                break

            rgb_path = _find_image(scene_dir, "rgb", image_id)
            if rgb_path is None:
                continue
            depth_path = _find_image(scene_dir, "depth", image_id)

            width, height = _image_size(rgb_path)

            instances = scene_gt.get(str(image_id)) or []
            infos = scene_info.get(str(image_id)) or []
            cam = scene_cam.get(str(image_id)) or {}
            k = cam.get("cam_K")
            if not isinstance(k, list) or len(k) < 9:
                continue

            # Copy/link RGB.
            out_name = f"{scene_dir.name}_{image_id:06d}{rgb_path.suffix.lower()}"
            out_img = out_images / out_name
            _copy_or_link(rgb_path, out_img, link=bool(args.link_images))

            label_lines: list[str] = []
            t_list: list[list[float] | None] = []
            r_list: list[list[list[float]] | None] = []
            off_list: list[list[float] | None] = []
            cad_list: list[str | None] = []
            symmetry_list: list[dict[str, Any] | None] = []

            for inst, info in zip(instances, infos):
                if not isinstance(inst, dict) or not isinstance(info, dict):
                    continue
                try:
                    obj_id = int(inst.get("obj_id"))
                except Exception:
                    continue

                visib = info.get("visib_fract")
                if visib is not None and float(visib) < float(args.visib_fract_min):
                    continue

                bbox_key = str(args.bbox_source)
                if bbox_key == "bbox_vis":
                    bbox_key = "bbox_visib"
                bbox = info.get(bbox_key)
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                if float(bbox[2]) <= 1.0 or float(bbox[3]) <= 1.0:
                    continue

                cx, cy, bw, bh = _bbox_xywh_to_cxcywh_norm(bbox, width=width, height=height)
                if bw <= 0.0 or bh <= 0.0:
                    continue

                if args.class_map == "obj_id_minus_1":
                    class_id = int(obj_id) - 1
                else:
                    raise SystemExit(f"unsupported class_map: {args.class_map}")

                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                r = inst.get("cam_R_m2c")
                t = inst.get("cam_t_m2c")
                if not (isinstance(r, list) and len(r) == 9 and isinstance(t, list) and len(t) == 3):
                    r_list.append(None)
                    t_list.append(None)
                    off_list.append(None)
                    cad_list.append(None)
                    symmetry_list.append(None)
                    continue

                r3 = [
                    [float(r[0]), float(r[1]), float(r[2])],
                    [float(r[3]), float(r[4]), float(r[5])],
                    [float(r[6]), float(r[7]), float(r[8])],
                ]
                t3 = [float(t[0]) * float(args.t_scale), float(t[1]) * float(args.t_scale), float(t[2]) * float(args.t_scale)]
                r_list.append(r3)
                t_list.append(t3)
                off_list.append([0.0, 0.0])
                cad_record = cad_records.get(obj_id)
                if cad_record is None and models_dir is not None:
                    model_path = models_dir / f"obj_{obj_id:06d}.ply"
                    if model_path.is_file():
                        try:
                            points = _read_ply_xyz(
                                model_path,
                                scale=float(args.t_scale),
                                max_points=int(args.cad_max_points),
                            )
                        except (OSError, ValueError) as exc:
                            raise SystemExit(str(exc)) from exc
                        cad_output.mkdir(parents=True, exist_ok=True)
                        cad_path = cad_output / f"obj_{obj_id:06d}.json"
                        cad_path.write_text(json.dumps(points, separators=(",", ":")) + "\n", encoding="utf-8")
                        cad_record = {
                            "path": str(cad_path),
                            "points_sha256": _sha256(cad_path),
                            "source": str(model_path),
                            "source_sha256": _sha256(model_path),
                            "points": len(points),
                        }
                        cad_records[obj_id] = cad_record
                cad_list.append(str(cad_record["path"]) if cad_record is not None else None)
                info = models_info.get(str(obj_id))
                if isinstance(info, dict):
                    symmetry_list.append(
                        {
                            key: info[key]
                            for key in ("symmetries_discrete", "symmetries_continuous")
                            if key in info
                        }
                        or None
                    )
                else:
                    symmetry_list.append(None)

            if not label_lines:
                converted_images += 1
                continue

            stem = Path(out_name).stem
            (out_labels / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            sidecar: dict[str, Any] = {
                "K_gt": [float(k[i]) for i in range(9)],
                "depth_path": (str(depth_path) if depth_path is not None else None),
                "depth_scale": cam.get("depth_scale"),
                "scene_id": int(scene_dir.name),
                "image_id": int(image_id),
                "bop_root": str(bop_root),
                "bop_split": str(args.split),
            }
            if any(value is not None for value in r_list) and any(value is not None for value in t_list):
                sidecar["R_gt"] = r_list
                sidecar["t_gt"] = t_list
                sidecar["offsets_gt"] = off_list
            if any(value is not None for value in cad_list):
                sidecar["cad_points"] = cad_list
                sidecar["cad_points_unit"] = "meters"
            if any(value is not None for value in symmetry_list):
                sidecar["bop_symmetry"] = symmetry_list

            (out_labels / f"{stem}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            converted_images += 1

        if args.max_images is not None and converted_images >= int(args.max_images):
            break

    descriptor_path = out_root / "dataset.json"
    descriptor: dict[str, Any] = {"schema_version": 1, "kind": "bop_yolozu_dataset", "splits": {}}
    if descriptor_path.is_file():
        try:
            existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict):
            descriptor.update(existing)
    splits = descriptor.get("splits")
    if not isinstance(splits, dict):
        splits = {}
    splits[out_split] = {"images_dir": str(out_images), "labels_dir": str(out_labels)}
    descriptor["splits"] = splits
    descriptor["latest_split"] = out_split
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    license_info = None
    if bop_root.name.lower() == "tless":
        license_info = {
            "spdx": "CC-BY-4.0",
            "source": "https://bop.felk.cvut.cz/datasets/",
        }
    report_dir = out_root / "conversion_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{out_split}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bop_yolozu_conversion",
                "source_root": str(bop_root),
                "source_split": str(args.split),
                "output_split": out_split,
                "converted_images": converted_images,
                "translation_scale_to_meters": float(args.t_scale),
                "bbox_source": str(args.bbox_source),
                "visibility_fraction_min": float(args.visib_fract_min),
                "partition": {
                    "modulus": args.partition_modulus,
                    "remainder": int(args.partition_remainder),
                    "mode": str(args.partition_mode),
                },
                "models_dir": str(models_dir) if models_dir is not None else None,
                "cad_points_unit": "meters",
                "cad_max_points": int(args.cad_max_points),
                "cad_models": {str(obj_id): value for obj_id, value in sorted(cad_records.items())},
                "dataset_license": license_info,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
