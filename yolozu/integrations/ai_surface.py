from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SUPPORTED_MCP_TOOL_IDS = (
    "doctor",
    "generate_config",
    "review_config",
    "validate_predictions",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_manifest_path(manifest_path: str | None = None) -> Path:
    if manifest_path:
        p = Path(manifest_path).expanduser()
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        return p
    return _repo_root() / "tools" / "manifest.json"


def load_tool_manifest(manifest_path: str | None = None) -> dict[str, Any]:
    path = _resolve_manifest_path(manifest_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("manifest must be a JSON object")
    return doc


def list_manifest_tools(
    *,
    manifest_path: str | None = None,
    only_supported: bool = False,
) -> list[dict[str, Any]]:
    doc = load_tool_manifest(manifest_path)
    tools = list(doc.get("tools") or [])
    items: list[dict[str, Any]] = []
    for tool in tools:
        tool_id = str(tool.get("id") or "")
        if not tool_id:
            continue
        if only_supported and tool_id not in _SUPPORTED_MCP_TOOL_IDS:
            continue
        items.append(
            {
                "id": tool_id,
                "summary": str(tool.get("summary") or ""),
                "inputs": list(tool.get("inputs") or []),
                "examples": list(tool.get("examples") or []),
                "effects": dict(tool.get("effects") or {}),
                "requires": dict(tool.get("requires") or {}),
            }
        )
    items.sort(key=lambda row: row["id"])
    return items


def generate_config(
    *,
    goal: str = "evaluate_predictions",
    dataset: str = "data/smoke",
    predictions: str = "data/smoke/predictions/predictions_dummy.json",
    split: str = "val",
    output: str = "reports/ai_eval.json",
    max_images: int = 50,
    dry_run: bool = True,
    network: bool = False,
    allow_gpu: bool = False,
    workspace_root: str = ".",
) -> dict[str, Any]:
    max_images = max(1, int(max_images))
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "goal": str(goal),
        "tool": "eval_coco",
        "arguments": {
            "dataset": str(dataset),
            "predictions": str(predictions),
            "split": str(split),
            "output": str(output),
            "max_images": max_images,
            "dry_run": bool(dry_run),
        },
        "safety": {
            "deterministic": True,
            "allow_network": bool(network),
            "allow_gpu": bool(allow_gpu),
            "workspace_root": str(workspace_root),
            "notes": [
                "prefer dry-run where available",
                "keep outputs under reports/",
                "avoid absolute writes by default",
            ],
        },
        "recommended_sequence": [
            {"tool": "doctor", "arguments": {"output": "reports/doctor.json"}},
            {"tool": "validate_predictions", "arguments": {"path": str(predictions), "strict": True}},
            {"tool": "eval_coco", "arguments": {"dataset": str(dataset), "predictions": str(predictions), "split": str(split), "dry_run": bool(dry_run), "output": str(output), "max_images": max_images}},
        ],
    }
    return cfg


def review_config(
    config: dict[str, Any],
    *,
    workspace_root: str = ".",
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    tool = str(config.get("tool") or "")
    args = dict(config.get("arguments") or {})
    safety = dict(config.get("safety") or {})

    if tool not in {"doctor", "validate_predictions", "validate_dataset", "eval_coco", "predict_images", "parity_check"}:
        issues.append({"code": "unsupported_tool", "message": f"tool `{tool}` is outside supported AI-first set"})

    allow_network = bool(safety.get("allow_network", False))
    if allow_network:
        warnings.append("allow_network=true; prefer false for deterministic CI runs")

    max_images = args.get("max_images")
    if max_images is not None:
        try:
            if int(max_images) > 1000:
                warnings.append("max_images is high; cap to <=1000 for CI determinism")
        except (TypeError, ValueError):
            issues.append({"code": "invalid_max_images", "message": "max_images must be an integer"})

    workspace = Path(workspace_root).resolve()
    output = args.get("output")
    if isinstance(output, str) and output:
        out_path = Path(output).expanduser()
        if out_path.is_absolute():
            try:
                out_path.relative_to(workspace)
            except ValueError:
                issues.append({"code": "unsafe_output_path", "message": f"output path escapes workspace: {output}"})
        elif output.startswith("../"):
            issues.append({"code": "unsafe_output_path", "message": f"relative output escapes workspace: {output}"})

    ok = len(issues) == 0
    return {
        "schema_version": 1,
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "summary": "config accepted" if ok else "config rejected",
    }


def supported_mcp_tool_ids() -> list[str]:
    return list(_SUPPORTED_MCP_TOOL_IDS)
