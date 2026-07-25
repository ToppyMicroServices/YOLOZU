from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .manifest_resources import (
    load_packaged_tool_reference,
    load_tool_manifest,
)

_AI_SURFACE_NAMES = (
    "mcp_live",
    "guaranteed_ai_safe",
    "config_review",
    "actions_public",
)


def _surface_sets_from_manifest(
    doc: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = doc.get("ai_surfaces")
    if not isinstance(raw, dict):
        raise ValueError("manifest is missing ai_surfaces")

    surfaces: dict[str, dict[str, Any]] = {}
    for name in _AI_SURFACE_NAMES:
        item = raw.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"manifest ai_surfaces is missing {name}")
        tool_ids = item.get("tool_ids")
        if not isinstance(tool_ids, list) or not all(
            isinstance(tool_id, str) and tool_id
            for tool_id in tool_ids
        ):
            raise ValueError(
                f"manifest ai_surfaces.{name}.tool_ids must be strings"
            )
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError(
                f"manifest ai_surfaces.{name}.tool_ids contains duplicates"
            )
        surfaces[name] = {
            "tool_ids": list(tool_ids),
            "availability": str(item.get("availability") or ""),
        }
    return surfaces


def ai_surface_sets(
    manifest_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return public AI surface sets from one machine-readable manifest."""
    return _surface_sets_from_manifest(load_tool_manifest(manifest_path))


def _as_filter_set(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _packaged_live_metadata(
    manifest_path: str | None,
) -> dict[str, dict[str, Any]]:
    if manifest_path is not None:
        return {}
    reference = load_packaged_tool_reference()
    rows = reference.get("mcp_live_tools")
    if not isinstance(rows, list):
        raise ValueError("packaged MCP reference is missing mcp_live_tools")
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("packaged MCP reference contains a non-object tool")
        name = str(row.get("name") or "")
        if not name or name in metadata:
            raise ValueError(
                "packaged MCP reference contains a missing or duplicate tool name"
            )
        input_schema = row.get("input_schema")
        if not isinstance(input_schema, dict):
            raise ValueError(
                f"packaged MCP reference tool {name} is missing input_schema"
            )
        metadata[name] = dict(row)
    return metadata


def _discover_manifest_tools(
    *,
    manifest_path: str | None = None,
    only_supported: bool = False,
    guaranteed: bool = False,
    supported: bool = False,
    maturity: str | Iterable[str] | None = None,
    tag: str | Iterable[str] | None = None,
    ids_only: bool = False,
) -> tuple[list[dict[str, Any]] | list[str], dict[str, Any]]:
    doc = load_tool_manifest(manifest_path)
    raw_tools = list(doc.get("tools") or [])
    tools_by_id = {
        str(tool.get("id")): tool
        for tool in raw_tools
        if isinstance(tool, dict) and str(tool.get("id") or "")
    }
    items: list[dict[str, Any]] = []
    maturity_filter = _as_filter_set(maturity)
    tag_filter = _as_filter_set(tag)
    surfaces = _surface_sets_from_manifest(doc)
    live_metadata = _packaged_live_metadata(manifest_path)
    guaranteed_ids = set(surfaces["guaranteed_ai_safe"]["tool_ids"])
    supported_ids = set(surfaces["mcp_live"]["tool_ids"])
    selected_ids = set(tools_by_id)
    if only_supported or guaranteed:
        selected_ids = guaranteed_ids
    if supported:
        selected_ids = (
            selected_ids & supported_ids
            if only_supported or guaranteed
            else supported_ids
        )

    excluded_unclassified_maturity = 0
    excluded_unclassified_tags = 0
    for tool_id in sorted(selected_ids):
        tool = tools_by_id.get(tool_id, {})
        live = live_metadata.get(tool_id)
        if live is not None:
            raw_maturity = live.get("maturity")
            tool_maturity = (
                str(raw_maturity)
                if isinstance(raw_maturity, str) and raw_maturity
                else None
            )
            tool_tags = [
                str(value)
                for value in list(live.get("tags") or [])
                if str(value)
            ]
            maturity_source = str(
                live.get("maturity_source") or "unclassified"
            )
            tags_source = str(
                live.get("tags_source") or "unclassified"
            )
        else:
            raw_maturity = tool.get("maturity")
            tool_maturity = (
                str(raw_maturity)
                if isinstance(raw_maturity, str) and raw_maturity
                else None
            )
            tool_tags = [
                str(value)
                for value in list(tool.get("tags") or [])
                if str(value)
            ]
            maturity_source = (
                "tools_manifest"
                if tool_maturity is not None
                else "unclassified"
            )
            tags_source = (
                "tools_manifest" if tool_tags else "unclassified"
            )
        if maturity_filter:
            if tool_maturity is None:
                excluded_unclassified_maturity += 1
                continue
            if tool_maturity not in maturity_filter:
                continue
        if tag_filter and not tool_tags:
            excluded_unclassified_tags += 1
            continue
        if tag_filter and not tag_filter.issubset(set(tool_tags)):
            continue
        source = (
            str(live.get("metadata_source") or "mcp_reference")
            if live is not None
            else "tools"
        )
        items.append(
            {
                "id": tool_id,
                "summary": str(
                    (live or {}).get("summary")
                    or tool.get("summary")
                    or ""
                ),
                "maturity": tool_maturity,
                "maturity_source": maturity_source,
                "tags": tool_tags,
                "tags_source": tags_source,
                "surface_tiers": list(
                    (live or {}).get("surface_tiers") or []
                ),
                "inputs": list(
                    (live or {}).get("inputs")
                    or tool.get("inputs")
                    or []
                ),
                "input_schema": (
                    dict(live["input_schema"])
                    if live is not None
                    else None
                ),
                "examples": list(
                    (live or {}).get("examples")
                    or tool.get("examples")
                    or []
                ),
                "effects": (
                    (live or {}).get("effects")
                    if live is not None
                    else dict(tool.get("effects") or {})
                ),
                "requires": (
                    (live or {}).get("requires")
                    if live is not None
                    else dict(tool.get("requires") or {})
                ),
                "metadata_source": source,
            }
        )
    diagnostics = {
        "maturity_filter": sorted(maturity_filter),
        "tag_filter": sorted(tag_filter),
        "excluded_unclassified_maturity": excluded_unclassified_maturity,
        "excluded_unclassified_tags": excluded_unclassified_tags,
        "semantics": (
            "maturity/tag filters match explicit metadata only; "
            "unclassified tools are excluded and counted"
        ),
    }
    if ids_only:
        return [row["id"] for row in items], diagnostics
    return items, diagnostics


def discover_manifest_tools(
    *,
    manifest_path: str | None = None,
    only_supported: bool = False,
    guaranteed: bool = False,
    supported: bool = False,
    maturity: str | Iterable[str] | None = None,
    tag: str | Iterable[str] | None = None,
    ids_only: bool = False,
) -> dict[str, Any]:
    """Return selected tool records plus explicit filter diagnostics."""
    tools, diagnostics = _discover_manifest_tools(
        manifest_path=manifest_path,
        only_supported=only_supported,
        guaranteed=guaranteed,
        supported=supported,
        maturity=maturity,
        tag=tag,
        ids_only=ids_only,
    )
    return {
        "tools": tools,
        "filter_diagnostics": diagnostics,
    }


def list_manifest_tools(
    *,
    manifest_path: str | None = None,
    only_supported: bool = False,
    guaranteed: bool = False,
    supported: bool = False,
    maturity: str | Iterable[str] | None = None,
    tag: str | Iterable[str] | None = None,
    ids_only: bool = False,
) -> list[dict[str, Any]] | list[str]:
    """Compatibility list view of :func:`discover_manifest_tools`."""
    return discover_manifest_tools(
        manifest_path=manifest_path,
        only_supported=only_supported,
        guaranteed=guaranteed,
        supported=supported,
        maturity=maturity,
        tag=tag,
        ids_only=ids_only,
    )["tools"]


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

    workspace = Path(workspace_root).expanduser()
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
    workspace = workspace.resolve()
    output = args.get("output")
    if isinstance(output, str) and output:
        out_path = Path(output).expanduser()
        candidate = out_path if out_path.is_absolute() else workspace / out_path
        try:
            candidate.resolve().relative_to(workspace)
        except ValueError:
            issues.append({"code": "unsafe_output_path", "message": f"output path escapes workspace: {output}"})

    ok = len(issues) == 0
    return {
        "schema_version": 1,
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "summary": "config accepted" if ok else "config rejected",
    }


def supported_mcp_tool_ids(
    manifest_path: str | None = None,
) -> list[str]:
    """Compatibility helper for the guaranteed AI-safe MCP subset."""
    return list(
        ai_surface_sets(manifest_path)["guaranteed_ai_safe"]["tool_ids"]
    )
