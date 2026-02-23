from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tool_runner


@dataclass(frozen=True)
class ActionsBinding:
    method: str
    path: str
    route_fn: str
    request_model: str | None = None


@dataclass(frozen=True)
class ToolSurfaceSpec:
    canonical_name: str
    category: str
    tool_runner_fn: str
    mcp_tool_fn: str
    actions: ActionsBinding
    tool_runner_aliases: tuple[str, ...] = ()
    mcp_aliases: tuple[str, ...] = ()
    actions_aliases: tuple[ActionsBinding, ...] = ()
    response_keys: tuple[str, ...] = ("ok", "tool", "summary", "exit_code")


TOOL_SURFACE_SPECS: tuple[ToolSurfaceSpec, ...] = (
    ToolSurfaceSpec(
        canonical_name="doctor",
        category="sync",
        tool_runner_fn="doctor",
        mcp_tool_fn="doctor_tool",
        actions=ActionsBinding("POST", "/doctor", "doctor_route", "DoctorRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="validate_predictions",
        category="sync",
        tool_runner_fn="validate_predictions",
        mcp_tool_fn="validate_predictions_tool",
        actions=ActionsBinding("POST", "/validate/predictions", "validate_predictions_route", "ValidatePredictionsRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="validate_dataset",
        category="sync",
        tool_runner_fn="validate_dataset",
        mcp_tool_fn="validate_dataset_tool",
        actions=ActionsBinding("POST", "/validate/dataset", "validate_dataset_route", "ValidateDatasetRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="eval_coco",
        category="sync",
        tool_runner_fn="eval_coco",
        mcp_tool_fn="eval_coco_tool",
        actions=ActionsBinding("POST", "/eval/coco", "eval_coco_route", "EvalCocoRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="predict_images",
        category="sync",
        tool_runner_fn="predict_images",
        mcp_tool_fn="predict_images_tool",
        actions=ActionsBinding("POST", "/predict/images", "predict_images_route", "PredictImagesRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="parity_check",
        category="sync",
        tool_runner_fn="parity_check",
        mcp_tool_fn="parity_check_tool",
        actions=ActionsBinding("POST", "/parity/check", "parity_check_route", "ParityCheckRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="calibrate_predictions",
        category="sync",
        tool_runner_fn="calibrate_predictions",
        mcp_tool_fn="calibrate_predictions_tool",
        actions=ActionsBinding("POST", "/calibrate/predictions", "calibrate_predictions_route", "CalibratePredictionsRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="eval_instance_seg",
        category="sync",
        tool_runner_fn="eval_instance_seg",
        mcp_tool_fn="eval_instance_seg_tool",
        actions=ActionsBinding("POST", "/eval/instance-seg", "eval_instance_seg_route", "EvalInstanceSegRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="eval_long_tail",
        category="sync",
        tool_runner_fn="eval_long_tail",
        mcp_tool_fn="eval_long_tail_tool",
        actions=ActionsBinding("POST", "/eval/long-tail", "eval_long_tail_route", "EvalLongTailRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="run_scenarios",
        category="sync",
        tool_runner_fn="run_scenarios",
        mcp_tool_fn="run_scenarios_tool",
        actions=ActionsBinding("POST", "/run/scenarios", "run_scenarios_route", "RunScenariosRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="convert_dataset",
        category="sync",
        tool_runner_fn="convert_dataset",
        mcp_tool_fn="convert_dataset_tool",
        actions=ActionsBinding("POST", "/convert/dataset", "convert_dataset_route", "ConvertDatasetRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="train_job",
        category="job",
        tool_runner_fn="train_job",
        mcp_tool_fn="train_job_tool",
        actions=ActionsBinding("POST", "/jobs/train", "train_job_route", "TrainJobRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="export_predictions_job",
        category="job",
        tool_runner_fn="export_predictions_job",
        mcp_tool_fn="export_predictions_job_tool",
        actions=ActionsBinding("POST", "/jobs/export-predictions", "export_predictions_job_route", "ExportPredictionsJobRequest"),
        tool_runner_aliases=("export_onnx_job",),
        mcp_aliases=("export_onnx_job_tool",),
        actions_aliases=(ActionsBinding("POST", "/jobs/export-onnx", "export_onnx_job_route", "ExportOnnxJobRequest"),),
    ),
    ToolSurfaceSpec(
        canonical_name="test_job",
        category="job",
        tool_runner_fn="test_job",
        mcp_tool_fn="test_job_tool",
        actions=ActionsBinding("POST", "/jobs/test", "test_job_route", "TestJobRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="ttt_job",
        category="job",
        tool_runner_fn="ttt_job",
        mcp_tool_fn="ttt_job_tool",
        actions=ActionsBinding("POST", "/jobs/ttt", "ttt_job_route", "TTTJobRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="ctta_job",
        category="job",
        tool_runner_fn="ctta_job",
        mcp_tool_fn="ctta_job_tool",
        actions=ActionsBinding("POST", "/jobs/ctta", "ctta_job_route", "CTTAJobRequest"),
    ),
    ToolSurfaceSpec(
        canonical_name="jobs_list",
        category="control",
        tool_runner_fn="jobs_list",
        mcp_tool_fn="jobs_list_tool",
        actions=ActionsBinding("GET", "/jobs", "jobs_list_route"),
    ),
    ToolSurfaceSpec(
        canonical_name="jobs_status",
        category="control",
        tool_runner_fn="jobs_status",
        mcp_tool_fn="jobs_status_tool",
        actions=ActionsBinding("GET", "/jobs/{job_id}", "jobs_status_route"),
    ),
    ToolSurfaceSpec(
        canonical_name="jobs_cancel",
        category="control",
        tool_runner_fn="jobs_cancel",
        mcp_tool_fn="jobs_cancel_tool",
        actions=ActionsBinding("POST", "/jobs/{job_id}/cancel", "jobs_cancel_route"),
    ),
    ToolSurfaceSpec(
        canonical_name="runs_list",
        category="control",
        tool_runner_fn="runs_list",
        mcp_tool_fn="runs_list_tool",
        actions=ActionsBinding("GET", "/runs", "runs_list_route"),
    ),
    ToolSurfaceSpec(
        canonical_name="runs_describe",
        category="control",
        tool_runner_fn="runs_describe",
        mcp_tool_fn="runs_describe_tool",
        actions=ActionsBinding("GET", "/runs/{run_id}", "runs_describe_route"),
    ),
)


def _integrations_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def _normalize_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_default(v) for v in value]
    if isinstance(value, list):
        return [_normalize_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_default(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalize_param(name: str, *, required: bool, default: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "required": bool(required)}
    if not required:
        out["default"] = _normalize_default(default)
    return out


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"


def _extract_ast_params(arguments: ast.arguments) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    positional = list(arguments.posonlyargs) + list(arguments.args)
    defaults = list(arguments.defaults)
    required_count = len(positional) - len(defaults)
    for idx, arg in enumerate(positional):
        if idx < required_count:
            params.append(_normalize_param(arg.arg, required=True))
            continue
        default = _ast_literal(defaults[idx - required_count])
        params.append(_normalize_param(arg.arg, required=False, default=default))
    for idx, arg in enumerate(arguments.kwonlyargs):
        default_node = arguments.kw_defaults[idx]
        if default_node is None:
            params.append(_normalize_param(arg.arg, required=True))
            continue
        params.append(_normalize_param(arg.arg, required=False, default=_ast_literal(default_node)))
    return params


def _parse_functions_with_route_metadata(path: Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        entry: dict[str, Any] = {"params": _extract_ast_params(node.args)}
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "app":
                continue
            if func.attr in {"post", "get"} and decorator.args:
                method = func.attr.upper()
                path_value = _ast_literal(decorator.args[0])
                entry["method"] = method
                entry["path"] = path_value
                break
            if func.attr == "tool":
                entry["tool"] = True
        out[node.name] = entry
    return out


def _parse_actions_models(path: Path) -> dict[str, list[dict[str, Any]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_info: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names: list[str] = []
        has_base_model = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                has_base_model = True
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                has_base_model = True
                base_names.append(base.attr)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        fields: list[dict[str, Any]] = []
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            if stmt.value is None:
                fields.append(_normalize_param(stmt.target.id, required=True))
                continue
            fields.append(_normalize_param(stmt.target.id, required=False, default=_ast_literal(stmt.value)))
        class_info[node.name] = {
            "bases": base_names,
            "is_base_model": has_base_model,
            "fields": fields,
        }

    out: dict[str, list[dict[str, Any]]] = {}
    changed = True
    while changed:
        changed = False
        for name, info in class_info.items():
            if name in out:
                continue
            if info["is_base_model"]:
                out[name] = list(info["fields"])
                changed = True
                continue
            for base_name in info["bases"]:
                inherited = out.get(base_name)
                if inherited is None:
                    continue
                fields = list(inherited)
                if info["fields"]:
                    by_name = {field["name"]: field for field in fields}
                    for field in info["fields"]:
                        by_name[field["name"]] = field
                    fields = [by_name[field["name"]] for field in inherited if field["name"] in by_name]
                    for field in info["fields"]:
                        if field["name"] not in {f["name"] for f in inherited}:
                            fields.append(field)
                out[name] = fields
                changed = True
                break
    return out


def _inspect_tool_runner_params(fn_name: str) -> list[dict[str, Any]]:
    fn = getattr(tool_runner, fn_name)
    sig = inspect.signature(fn)
    out: list[dict[str, Any]] = []
    for param in sig.parameters.values():
        if param.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}:
            continue
        if param.default is inspect.Parameter.empty:
            out.append(_normalize_param(param.name, required=True))
            continue
        out.append(_normalize_param(param.name, required=False, default=param.default))
    return out


def _param_key(param: dict[str, Any]) -> tuple[Any, ...]:
    return (param.get("name"), bool(param.get("required")), param.get("default"))


def _params_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    return all(_param_key(a) == _param_key(b) for a, b in zip(left, right))


def _binding_params(
    binding: ActionsBinding,
    *,
    routes: dict[str, dict[str, Any]],
    models: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    route = routes.get(binding.route_fn)
    if route is None:
        raise ValueError(f"missing Actions route function: {binding.route_fn}")
    method = route.get("method")
    path = route.get("path")
    if method != binding.method or path != binding.path:
        raise ValueError(
            f"Actions binding mismatch for {binding.route_fn}: "
            f"expected {binding.method} {binding.path}, got {method} {path}"
        )
    if binding.request_model is None:
        return list(route["params"])
    params = models.get(binding.request_model)
    if params is None:
        raise ValueError(f"missing Actions request model: {binding.request_model}")
    return list(params)


def build_tool_surface_reference() -> dict[str, Any]:
    mcp_functions = _parse_functions_with_route_metadata(_integrations_path("mcp_server.py"))
    actions_functions = _parse_functions_with_route_metadata(_integrations_path("actions_api.py"))
    actions_models = _parse_actions_models(_integrations_path("actions_api.py"))

    tools: list[dict[str, Any]] = []
    for spec in TOOL_SURFACE_SPECS:
        tool_params = _inspect_tool_runner_params(spec.tool_runner_fn)
        mcp_primary = mcp_functions.get(spec.mcp_tool_fn)
        if mcp_primary is None:
            raise ValueError(f"missing MCP tool function: {spec.mcp_tool_fn}")
        if not mcp_primary.get("tool"):
            raise ValueError(f"MCP function is not decorated with @app.tool(): {spec.mcp_tool_fn}")
        mcp_params = list(mcp_primary["params"])

        mcp_aliases: list[dict[str, Any]] = []
        for alias_name in spec.mcp_aliases:
            alias = mcp_functions.get(alias_name)
            if alias is None:
                raise ValueError(f"missing MCP alias function: {alias_name}")
            if not alias.get("tool"):
                raise ValueError(f"MCP alias function is not decorated with @app.tool(): {alias_name}")
            alias_params = list(alias["params"])
            mcp_aliases.append(
                {
                    "name": alias_name.removesuffix("_tool"),
                    "function": alias_name,
                    "params": alias_params,
                    "parity_with_tool_runner": _params_equal(alias_params, tool_params),
                }
            )

        actions_params = _binding_params(spec.actions, routes=actions_functions, models=actions_models)
        actions_aliases: list[dict[str, Any]] = []
        for alias in spec.actions_aliases:
            alias_params = _binding_params(alias, routes=actions_functions, models=actions_models)
            actions_aliases.append(
                {
                    "method": alias.method,
                    "path": alias.path,
                    "route_fn": alias.route_fn,
                    "request_model": alias.request_model,
                    "params": alias_params,
                    "parity_with_tool_runner": _params_equal(alias_params, tool_params),
                }
            )

        tool_aliases: list[dict[str, Any]] = []
        for alias_name in spec.tool_runner_aliases:
            alias_params = _inspect_tool_runner_params(alias_name)
            tool_aliases.append(
                {
                    "name": alias_name,
                    "params": alias_params,
                    "parity_with_tool_runner": _params_equal(alias_params, tool_params),
                }
            )

        tools.append(
            {
                "canonical_name": spec.canonical_name,
                "category": spec.category,
                "response_keys": list(spec.response_keys),
                "tool_runner": {
                    "function": spec.tool_runner_fn,
                    "params": tool_params,
                    "aliases": tool_aliases,
                },
                "mcp": {
                    "tool": spec.mcp_tool_fn.removesuffix("_tool"),
                    "function": spec.mcp_tool_fn,
                    "params": mcp_params,
                    "aliases": mcp_aliases,
                },
                "actions": {
                    "method": spec.actions.method,
                    "path": spec.actions.path,
                    "route_fn": spec.actions.route_fn,
                    "request_model": spec.actions.request_model,
                    "params": actions_params,
                    "aliases": actions_aliases,
                },
                "parity": {
                    "mcp_vs_tool_runner": _params_equal(mcp_params, tool_params),
                    "actions_vs_tool_runner": _params_equal(actions_params, tool_params),
                    "tool_runner_aliases": all(item["parity_with_tool_runner"] for item in tool_aliases) if tool_aliases else True,
                    "mcp_aliases": all(item["parity_with_tool_runner"] for item in mcp_aliases) if mcp_aliases else True,
                    "actions_aliases": all(item["parity_with_tool_runner"] for item in actions_aliases) if actions_aliases else True,
                },
            }
        )
    return {
        "schema_version": 1,
        "source": "yolozu.integrations",
        "tools": tools,
    }


def collect_surface_parity_errors(reference: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for tool in reference.get("tools", []):
        name = tool.get("canonical_name")
        parity = tool.get("parity", {})
        for key in ("mcp_vs_tool_runner", "actions_vs_tool_runner", "tool_runner_aliases", "mcp_aliases", "actions_aliases"):
            if not parity.get(key, False):
                errors.append(f"{name}: parity check failed for {key}")
        keys = set(tool.get("response_keys") or [])
        missing = [k for k in ("ok", "tool", "summary", "exit_code") if k not in keys]
        if missing:
            errors.append(f"{name}: missing response keys {missing}")
    return errors


def render_tool_surface_markdown(reference: dict[str, Any]) -> str:
    lines: list[str] = [
        "# MCP/Actions tool reference (generated)",
        "",
        "Source of truth: `yolozu.integrations.tool_runner`, `yolozu.integrations.mcp_server`, `yolozu.integrations.actions_api`.",
        "",
        "| Canonical | Category | MCP tool | Actions endpoint | Request model | Parity |",
        "|---|---|---|---|---|---|",
    ]
    for tool in reference.get("tools", []):
        parity = tool.get("parity", {})
        parity_ok = all(
            bool(parity.get(key, False))
            for key in ("mcp_vs_tool_runner", "actions_vs_tool_runner", "tool_runner_aliases", "mcp_aliases", "actions_aliases")
        )
        actions = tool.get("actions", {})
        lines.append(
            "| "
            + f"`{tool['canonical_name']}` | "
            + f"{tool['category']} | "
            + f"`{tool['mcp']['tool']}` | "
            + f"`{actions.get('method')} {actions.get('path')}` | "
            + f"`{actions.get('request_model') or '-'}` | "
            + ("✅" if parity_ok else "❌")
            + " |"
        )
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    for tool in reference.get("tools", []):
        lines.append(f"### `{tool['canonical_name']}`")
        lines.append("")
        lines.append("`tool_runner` params:")
        params = tool["tool_runner"]["params"]
        if not params:
            lines.append("- (none)")
        for param in params:
            if param.get("required"):
                lines.append(f"- `{param['name']}` (required)")
            else:
                lines.append(f"- `{param['name']}` (default: `{param.get('default')}`)")
        aliases: list[str] = []
        for alias in tool["tool_runner"].get("aliases", []):
            aliases.append(alias["name"])
        for alias in tool["mcp"].get("aliases", []):
            aliases.append(alias["name"])
        for alias in tool["actions"].get("aliases", []):
            aliases.append(f"{alias['method']} {alias['path']}")
        if aliases:
            deduped = list(dict.fromkeys(aliases))
            lines.append(f"- aliases: {', '.join(f'`{a}`' for a in deduped)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
