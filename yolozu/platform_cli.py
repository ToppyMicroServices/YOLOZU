"""CLI surface for release qualification packs and the adapter SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yolozu.adapter_sdk import (
    AdapterSDKError,
    check_adapter_plugin,
    discover_adapters,
    load_adapter_plugin,
    test_adapter_plugin,
)
from yolozu.qualification import (
    QualificationError,
    diff_qualification_packs,
    qualify_release,
    qualify_release_from_spec,
    verify_qualification_pack,
)

__all__ = ["add_platform_parsers", "handle_platform_command"]


def add_platform_parsers(sub: argparse._SubParsersAction) -> None:
    qualify = sub.add_parser(
        "qualify-release",
        help="Create, verify, or compare portable release qualification packs.",
    )
    qualify_sub = qualify.add_subparsers(dest="qualify_release_command", required=True)

    create = qualify_sub.add_parser(
        "create",
        help="Evaluate a candidate and atomically create a qualification pack.",
    )
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--spec",
        help="Strict YAML spec; relative paths resolve from the spec directory.",
    )
    source.add_argument("--dataset", help="YOLO-format dataset root.")
    create.add_argument("--predictions", help="Candidate predictions JSON.")
    create.add_argument("--output-dir", help="New qualification pack directory.")
    create.add_argument(
        "--baseline-predictions", help="Optional baseline predictions JSON."
    )
    create.add_argument("--split", default=None, help="Dataset split override.")
    create.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default=None,
        help="Prediction bbox format (direct mode default: cxcywh_norm).",
    )
    create.add_argument(
        "--max-images", type=int, default=None, help="Optional positive subset cap."
    )
    create.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate conversion without COCO metrics; decision will be hold.",
    )
    create.add_argument(
        "--min-map50-95",
        type=float,
        default=None,
        help="Minimum candidate mAP@[.50:.95].",
    )
    create.add_argument(
        "--max-map50-95-drop",
        type=float,
        default=None,
        help="Maximum allowed baseline minus candidate mAP@[.50:.95].",
    )
    create.add_argument(
        "--force",
        action="store_true",
        help="Replace only an existing pack that first passes verification.",
    )

    verify = qualify_sub.add_parser(
        "verify", help="Verify pack checksums and decision semantics."
    )
    verify.add_argument("pack", help="Qualification pack directory.")

    diff = qualify_sub.add_parser(
        "diff", help="Compare metrics from two verified compatible packs."
    )
    diff.add_argument("baseline", help="Baseline qualification pack directory.")
    diff.add_argument("candidate", help="Candidate qualification pack directory.")

    adapter = sub.add_parser(
        "adapter",
        help="Discover and check packaged third-party adapter plugins.",
    )
    adapter_sub = adapter.add_subparsers(dest="adapter_command", required=True)
    adapter_sub.add_parser(
        "list",
        help="List entry-point metadata without importing plugin code.",
    )
    doctor = adapter_sub.add_parser(
        "doctor",
        help="Explicitly load one plugin and validate its metadata interface contract.",
    )
    doctor.add_argument("name", help="Entry-point name in yolozu.adapters.v1.")
    doctor.add_argument(
        "--allow-plugin-load",
        action="store_true",
        help="Acknowledge that importing a third-party plugin executes its code.",
    )
    test = adapter_sub.add_parser(
        "test",
        help="Create one plugin adapter and run predictions interface conformance.",
    )
    test.add_argument("name", help="Entry-point name in yolozu.adapters.v1.")
    test.add_argument(
        "--config", default=None, help="Optional plugin config JSON object."
    )
    test.add_argument(
        "--records",
        default=None,
        help="Optional JSON list of input records; default uses a packaged 8x8 image.",
    )
    test.add_argument(
        "--allow-plugin-load",
        action="store_true",
        help="Acknowledge that importing and running a third-party plugin executes its code.",
    )


def _read_object(path: str | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterSDKError(f"could not read {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterSDKError(f"{label} JSON must contain an object")
    return payload


def _read_records(path: str | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterSDKError(f"could not read records JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("records")
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise AdapterSDKError(
            "records JSON must be a list of objects or an object with records"
        )
    return payload


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _handle_qualify_release(args: argparse.Namespace) -> int:
    if args.qualify_release_command == "create":
        if args.spec is not None:
            direct_values = {
                "--predictions": args.predictions,
                "--output-dir": args.output_dir,
                "--baseline-predictions": args.baseline_predictions,
                "--split": args.split,
                "--bbox-format": args.bbox_format,
                "--max-images": args.max_images,
                "--min-map50-95": args.min_map50_95,
                "--max-map50-95-drop": args.max_map50_95_drop,
                "--dry-run": True if args.dry_run else None,
                "--force": True if args.force else None,
            }
            mixed = [name for name, value in direct_values.items() if value is not None]
            if mixed:
                raise QualificationError(
                    "--spec cannot be combined with direct options: " + ", ".join(mixed)
                )
            result = qualify_release_from_spec(args.spec)
        else:
            if args.predictions is None or args.output_dir is None:
                raise QualificationError(
                    "direct mode requires --dataset, --predictions, and --output-dir"
                )
            result = qualify_release(
                args.dataset,
                args.predictions,
                args.output_dir,
                baseline_predictions=args.baseline_predictions,
                split=args.split,
                bbox_format=args.bbox_format or "cxcywh_norm",
                max_images=args.max_images,
                dry_run=bool(args.dry_run),
                min_map50_95=args.min_map50_95,
                max_map50_95_drop=args.max_map50_95_drop,
                force=bool(args.force),
            )
        _emit(result.to_dict())
        return {"pass": 0, "hold": 3, "fail": 4}[result.decision]
    if args.qualify_release_command == "verify":
        result = verify_qualification_pack(args.pack)
        _emit(result.to_dict())
        return 0 if result.ok else 1
    if args.qualify_release_command == "diff":
        payload = diff_qualification_packs(args.baseline, args.candidate)
        _emit(payload)
        if not payload.get("ok"):
            return 1
        return 0 if payload.get("compatible") else 3
    raise QualificationError("unknown qualify-release command")


def _handle_adapter(args: argparse.Namespace) -> int:
    if args.adapter_command == "list":
        adapters = [descriptor.to_dict() for descriptor in discover_adapters()]
        _emit(
            {
                "schema_version": 1,
                "ok": True,
                "entry_point_group": "yolozu.adapters.v1",
                "adapters": adapters,
                "count": len(adapters),
                "plugin_code_loaded": False,
            }
        )
        return 0
    if args.adapter_command == "doctor":
        plugin = load_adapter_plugin(
            args.name, allow_plugin_load=bool(args.allow_plugin_load)
        )
        checked = check_adapter_plugin(plugin, expected_name=args.name)
        _emit(
            {
                "schema_version": 1,
                "ok": True,
                "adapter_id": checked.adapter_id,
                "interface_version": checked.interface_version,
                "summary": checked.summary,
                "capabilities": dict(checked.capabilities),
                "plugin_code_loaded": True,
                "runtime_adapter_created": False,
            }
        )
        return 0
    if args.adapter_command == "test":
        result = test_adapter_plugin(
            args.name,
            config=_read_object(args.config, label="config"),
            records=_read_records(args.records),
            allow_plugin_load=bool(args.allow_plugin_load),
        )
        _emit(result.to_dict())
        return 0
    raise AdapterSDKError("unknown adapter command")


def handle_platform_command(args: argparse.Namespace) -> int | None:
    try:
        if args.command == "qualify-release":
            return _handle_qualify_release(args)
        if args.command == "adapter":
            return _handle_adapter(args)
    except (QualificationError, AdapterSDKError) as exc:
        _emit(
            {
                "schema_version": 1,
                "ok": False,
                "command": args.command,
                "error": {"category": type(exc).__name__, "message": str(exc)},
            }
        )
        return 2
    return None
