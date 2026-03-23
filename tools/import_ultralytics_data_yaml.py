import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.config import simple_yaml_load
from yolozu.migrate import migrate_ultralytics_dataset_wrapper


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        data = simple_yaml_load(text)
        return data if isinstance(data, dict) else {}


def _extract_class_names(cfg: dict[str, Any]) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        parsed: list[tuple[int, str]] = []
        for key, value in names.items():
            try:
                parsed.append((int(key), str(value)))
            except (TypeError, ValueError):
                continue
        parsed.sort(key=lambda x: x[0])
        if parsed:
            upper = parsed[-1][0]
            out = [""] * (upper + 1)
            for index, label in parsed:
                if 0 <= index < len(out):
                    out[index] = label
            return [label if label else f"class_{i}" for i, label in enumerate(out)]
    nc = cfg.get("nc")
    if isinstance(nc, int) and nc > 0:
        return [f"class_{i}" for i in range(nc)]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import YOLO-style data.yaml into a YOLOZU dataset wrapper + classes mapping.")
    parser.add_argument("--data-yaml", required=True, help="Path to YOLO-style data.yaml")
    parser.add_argument("--split", default=None, help="Split override (default: auto)")
    parser.add_argument("--output", required=True, help="Output dataset root (writes dataset.json and labels/<split>/classes.json)")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs when present")
    args = parser.parse_args(argv)

    data_yaml = Path(args.data_yaml).expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    output_root = Path(args.output).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    dataset_json = migrate_ultralytics_dataset_wrapper(
        data_yaml=data_yaml,
        args_yaml=None,
        split=args.split,
        task=None,
        output=output_root,
        force=bool(args.force),
    )
    dataset_doc = json.loads(dataset_json.read_text(encoding="utf-8"))
    split_effective = str(dataset_doc.get("split") or args.split or "val")

    cfg = _load_yaml(data_yaml)
    class_names = _extract_class_names(cfg)
    class_id_to_category_id = {str(i): int(i) for i in range(len(class_names))}
    category_id_to_class_id = {str(i): int(i) for i in range(len(class_names))}
    classes_payload = {
        "class_names": class_names,
        "class_id_to_category_id": class_id_to_category_id,
        "category_id_to_class_id": category_id_to_class_id,
        "source": {"from": "ultralytics", "data_yaml": str(data_yaml)},
    }

    labels_dir = output_root / "labels" / split_effective
    labels_dir.mkdir(parents=True, exist_ok=True)
    classes_json = labels_dir / "classes.json"
    classes_txt = labels_dir / "classes.txt"
    if classes_json.exists() and not args.force:
        raise SystemExit(f"classes mapping exists: {classes_json} (use --force)")
    classes_json.write_text(json.dumps(classes_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if class_names:
        classes_txt.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    else:
        classes_txt.write_text("", encoding="utf-8")

    summary = {"dataset_json": str(dataset_json), "classes_json": str(classes_json), "classes_txt": str(classes_txt)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
