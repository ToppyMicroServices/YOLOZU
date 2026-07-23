#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = repo_root / "yolozu" / "data" / "manifest" / "benchmark_support.json"
DEFAULT_OUTPUT = repo_root / "docs" / "benchmark_support_matrix.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate docs/benchmark_support_matrix.md from benchmark support metadata."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Benchmark support metadata JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Generated Markdown output path.")
    parser.add_argument("--check", action="store_true", help="Fail if the output file is not up to date.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    return parser.parse_args(argv)


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"metadata not found: {_repo_rel(path)}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"metadata is not valid JSON: {_repo_rel(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("metadata root must be a JSON object")
    return payload


def _required_str(item: dict[str, Any], key: str, *, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{where} requires non-empty string field {key!r}")
    return value


def _required_str_list(
    item: dict[str, Any],
    key: str,
    *,
    where: str,
    allow_empty: bool = False,
) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "" if allow_empty else " non-empty"
        raise SystemExit(f"{where} requires{qualifier} string-list field {key!r}")
    if any(not isinstance(entry, str) or not entry for entry in value):
        raise SystemExit(f"{where} field {key!r} must contain non-empty strings")
    return value


def _validate_metadata(meta: dict[str, Any]) -> None:
    formats = meta.get("formats")
    tasks = meta.get("tasks")
    support = meta.get("support")
    legend = meta.get("legend")
    for key, value in (("formats", formats), ("tasks", tasks), ("support", support), ("legend", legend)):
        if not isinstance(value, list) or not value:
            raise SystemExit(f"metadata field {key!r} must be a non-empty list")

    format_ids = [_required_str(item, "id", where="formats[]") for item in formats if isinstance(item, dict)]
    task_ids = [_required_str(item, "id", where="tasks[]") for item in tasks if isinstance(item, dict)]
    if len(format_ids) != len(formats) or len(task_ids) != len(tasks):
        raise SystemExit("formats[] and tasks[] entries must be objects")
    if len(format_ids) != len(set(format_ids)):
        raise SystemExit("formats[] contains duplicate ids")
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("tasks[] contains duplicate ids")

    expected_pairs = {(fmt, task) for fmt in format_ids for task in task_ids}
    actual_pairs: set[tuple[str, str]] = set()
    for item in support:
        if not isinstance(item, dict):
            raise SystemExit("support[] entries must be objects")
        fmt = _required_str(item, "format", where="support[]")
        task = _required_str(item, "task", where="support[]")
        if fmt not in format_ids:
            raise SystemExit(f"support[] references unknown format: {fmt}")
        if task not in task_ids:
            raise SystemExit(f"support[] references unknown task: {task}")
        pair = (fmt, task)
        if pair in actual_pairs:
            raise SystemExit(f"support[] contains duplicate row: {fmt}/{task}")
        actual_pairs.add(pair)
        for key in ("inference_artifact", "eval_artifact", "parity_artifact", "notes"):
            _required_str(item, key, where=f"support[{fmt}/{task}]")

    missing = sorted(expected_pairs - actual_pairs)
    extra = sorted(actual_pairs - expected_pairs)
    if missing or extra:
        raise SystemExit(f"support matrix coverage mismatch: missing={missing} extra={extra}")

    flag_applicability = meta.get("flag_applicability")
    if not isinstance(flag_applicability, dict):
        raise SystemExit("metadata field 'flag_applicability' must be an object")
    defaults = flag_applicability.get("defaults")
    if not isinstance(defaults, dict) or not defaults:
        raise SystemExit("flag_applicability.defaults must be a non-empty object")
    artifact_tasks = _required_str_list(
        flag_applicability,
        "artifact_eval_tasks",
        where="flag_applicability",
    )
    if any(task not in task_ids for task in artifact_tasks):
        raise SystemExit("flag_applicability.artifact_eval_tasks references an unknown task")
    rejected_flags = _required_str_list(
        flag_applicability,
        "artifact_eval_rejected_nondefault_flags",
        where="flag_applicability",
    )
    if set(defaults) != set(rejected_flags):
        raise SystemExit(
            "flag_applicability.defaults and artifact_eval_rejected_nondefault_flags must name the same flags"
        )
    matrix = flag_applicability.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise SystemExit("flag_applicability.matrix must be a non-empty list")
    for index, item in enumerate(matrix):
        where = f"flag_applicability.matrix[{index}]"
        if not isinstance(item, dict):
            raise SystemExit(f"{where} must be an object")
        _required_str(item, "task_scope", where=where)
        _required_str_list(item, "requested_latency_sources", where=where)
        _required_str(item, "effective_latency_source", where=where)
        row_formats = _required_str_list(item, "formats", where=where)
        if any(fmt not in format_ids for fmt in row_formats):
            raise SystemExit(f"{where} references an unknown format")
        accepted = _required_str_list(
            item,
            "accepted_nondefault_flags",
            where=where,
            allow_empty=True,
        )
        rejected = _required_str_list(
            item,
            "rejected_nondefault_flags",
            where=where,
            allow_empty=True,
        )
        if set(accepted) & set(rejected):
            raise SystemExit(f"{where} accepts and rejects the same flag")
        if any(flag not in defaults for flag in accepted + rejected):
            raise SystemExit(f"{where} references a flag without a declared default")
        _required_str(item, "behavior", where=where)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_md_escape(cell) for cell in row) + " |")
    return out


def _code_list(values: list[str], *, flag_prefix: bool = False) -> str:
    if not values:
        return "none"
    prefix = "--" if flag_prefix else ""
    return ", ".join(f"`{prefix}{value}`" for value in values)


def _default_flag(name: str, value: Any) -> str:
    if value is False:
        return f"`--no-{name}`"
    if value is True:
        return f"`--{name}`"
    return f"`--{name} {value}`"


def render_markdown(meta: dict[str, Any], *, metadata_path: Path) -> str:
    _validate_metadata(meta)
    flag_defaults = meta["flag_applicability"]["defaults"]
    lines: list[str] = [
        "# Benchmark Support Matrix",
        "",
        "<!-- Generated by tools/generate_benchmark_support_matrix.py from "
        f"{_repo_rel(metadata_path)}. Do not edit by hand. -->",
        "",
        str(meta.get("scope") or "Canonical benchmark support metadata."),
        "",
        "This matrix describes benchmark artifacts, not every standalone exporter utility.",
        "",
        "## Legend",
        "",
    ]
    for item in meta["legend"]:
        term = _required_str(item, "term", where="legend[]")
        description = _required_str(item, "description", where=f"legend[{term}]")
        lines.append(f"- `{term}`: {description}")

    lines.extend(
        [
            "",
            "## Runtime Requirements",
            "",
            *_table(
                ["Format", "Support state", "Runtime requirements", "License/runtime notes"],
                [
                    [
                        f"`{item['id']}`",
                        item["support_state"],
                        item["runtime_requirements"],
                        item["license_runtime_notes"],
                    ]
                    for item in meta["formats"]
                ],
            ),
            "",
            "## Task Semantics",
            "",
            *_table(
                ["Task", "Metric family", "Surface", "Semantics"],
                [
                    [
                        f"`{item['id']}`",
                        item["metric_family"],
                        item["surface"],
                        item["semantics"],
                    ]
                    for item in meta["tasks"]
                ],
            ),
            "",
            "## Backend Flag Applicability",
            "",
            "Task/source validation is applied before per-format flag applicability, after `auto` resolves to an effective latency source.",
            "Within a valid task/source lane, default backend flag values are always accepted: "
            + ", ".join(
                _default_flag(name, flag_defaults[name])
                for name in sorted(flag_defaults)
            )
            + ".",
            "",
            *_table(
                [
                    "Task scope",
                    "Requested latency source",
                    "Effective latency source",
                    "Formats",
                    "Accepted non-default flags",
                    "Rejected non-default flags",
                    "Behavior",
                ],
                [
                    [
                        item["task_scope"],
                        _code_list(item["requested_latency_sources"]),
                        f"`{item['effective_latency_source']}`",
                        _code_list(item["formats"]),
                        _code_list(item["accepted_nondefault_flags"], flag_prefix=True),
                        _code_list(item["rejected_nondefault_flags"], flag_prefix=True),
                        item["behavior"],
                    ]
                    for item in meta["flag_applicability"]["matrix"]
                ],
            ),
            "",
            "## Artifact Support",
            "",
            *_table(
                ["Format", "Task", "Inference artifact", "Eval artifact", "Parity artifact", "Notes"],
                [
                    [
                        f"`{item['format']}`",
                        f"`{item['task']}`",
                        item["inference_artifact"],
                        item["eval_artifact"],
                        item["parity_artifact"],
                        item["notes"],
                    ]
                    for item in meta["support"]
                ],
            ),
            "",
            "## Sync Rule",
            "",
            "When CLI behavior changes, update the metadata source above and regenerate this page.",
            "Keep these files synchronized in the same PR:",
            "",
        ]
    )
    for target in meta.get("sync_targets") or []:
        lines.append(f"- `{target}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = repo_root / metadata_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    meta = _load_metadata(metadata_path)
    rendered = render_markdown(meta, metadata_path=metadata_path)
    current = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    drifted = current != rendered
    summary = {
        "ok": not drifted,
        "check": bool(args.check),
        "metadata": _repo_rel(metadata_path),
        "output": _repo_rel(output_path),
        "formats": len(meta.get("formats") or []),
        "tasks": len(meta.get("tasks") or []),
        "rows": len(meta.get("support") or []),
        "drifted": drifted,
    }

    if args.check:
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif drifted:
            print(f"benchmark support matrix drifted: {_repo_rel(output_path)}", file=sys.stderr)
        return 1 if drifted else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    summary["ok"] = True
    summary["drifted"] = False
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_repo_rel(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
