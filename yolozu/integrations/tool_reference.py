from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tool_runner
from .ai_surface import ai_surface_sets
from .manifest_resources import load_tool_manifest


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
    except (ValueError, SyntaxError):
        try:
            return ast.unparse(node)
        except (AttributeError, TypeError, ValueError):
            return "<expr>"


def _annotation_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _annotation_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _annotation_schema(node: ast.AST | None) -> dict[str, Any]:
    if node is None:
        return {}
    name = _annotation_name(node)
    primitive = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "None": {"type": "null"},
        "NoneType": {"type": "null"},
        "dict": {"type": "object"},
        "list": {"type": "array", "items": {}},
    }
    if name in primitive:
        return dict(primitive[name])
    if name in {"Any", "typing.Any"}:
        return {}
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        variants: list[dict[str, Any]] = []
        for side in (node.left, node.right):
            schema = _annotation_schema(side)
            nested = schema.get("anyOf")
            if isinstance(nested, list) and len(schema) == 1:
                variants.extend(dict(item) for item in nested)
            else:
                variants.append(schema)
        return {"anyOf": variants}
    if not isinstance(node, ast.Subscript):
        return {}

    base = (_annotation_name(node.value) or "").removeprefix("typing.")
    slice_node = node.slice
    if base in {"list", "List", "set", "Set", "tuple", "Tuple"}:
        item_node = slice_node
        if isinstance(slice_node, ast.Tuple) and slice_node.elts:
            item_node = slice_node.elts[0]
        return {"items": _annotation_schema(item_node), "type": "array"}
    if base in {"dict", "Dict", "Mapping"}:
        value_node: ast.AST | None = None
        if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
            value_node = slice_node.elts[1]
        schema: dict[str, Any] = {"type": "object"}
        value_schema = _annotation_schema(value_node)
        if value_schema:
            schema["additionalProperties"] = value_schema
        elif value_node is not None:
            schema["additionalProperties"] = True
        return schema
    if base in {"Optional"}:
        return {"anyOf": [_annotation_schema(slice_node), {"type": "null"}]}
    if base in {"Union"}:
        variants = (
            list(slice_node.elts)
            if isinstance(slice_node, ast.Tuple)
            else [slice_node]
        )
        return {"anyOf": [_annotation_schema(item) for item in variants]}
    if base in {"Literal"}:
        values = (
            [_ast_literal(item) for item in slice_node.elts]
            if isinstance(slice_node, ast.Tuple)
            else [_ast_literal(slice_node)]
        )
        schema = {"enum": values}
        value_types = {type(value) for value in values}
        if value_types == {str}:
            schema["type"] = "string"
        elif value_types == {int}:
            schema["type"] = "integer"
        return schema
    if base in {"Annotated"}:
        parts = (
            list(slice_node.elts)
            if isinstance(slice_node, ast.Tuple)
            else [slice_node]
        )
        schema = _annotation_schema(parts[0] if parts else None)
        for metadata in parts[1:]:
            if not isinstance(metadata, ast.Call):
                continue
            if (_annotation_name(metadata.func) or "").split(".")[-1] != "Field":
                continue
            constraint_names = {
                "ge": "minimum",
                "gt": "exclusiveMinimum",
                "le": "maximum",
                "lt": "exclusiveMaximum",
                "min_length": "minLength",
                "max_length": "maxLength",
                "pattern": "pattern",
            }
            for keyword in metadata.keywords:
                json_name = constraint_names.get(str(keyword.arg))
                if json_name:
                    schema[json_name] = _ast_literal(keyword.value)
        return schema
    return {}


def _field_schema(
    name: str,
    annotation: ast.AST | None,
    *,
    default_node: ast.AST | None,
) -> dict[str, Any]:
    schema = _annotation_schema(annotation)
    schema["title"] = name.replace("_", " ").title()
    if default_node is not None:
        schema["default"] = _ast_literal(default_node)
    return schema


def _arguments_input_schema(
    function_name: str,
    arguments: ast.arguments,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    positional = list(arguments.posonlyargs) + list(arguments.args)
    defaults = list(arguments.defaults)
    required_count = len(positional) - len(defaults)
    for idx, arg in enumerate(positional):
        default_node = None if idx < required_count else defaults[idx - required_count]
        properties[arg.arg] = _field_schema(
            arg.arg,
            arg.annotation,
            default_node=default_node,
        )
        if default_node is None:
            required.append(arg.arg)
    for idx, arg in enumerate(arguments.kwonlyargs):
        default_node = arguments.kw_defaults[idx]
        properties[arg.arg] = _field_schema(
            arg.arg,
            arg.annotation,
            default_node=default_node,
        )
        if default_node is None:
            required.append(arg.arg)
    schema: dict[str, Any] = {
        "properties": properties,
        "title": f"{function_name}Arguments",
        "type": "object",
    }
    if required:
        schema["required"] = required
    return schema


def _normalize_parameter_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def without_titles(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_titles(item)
                for key, item in value.items()
                if key != "title"
            }
        if isinstance(value, list):
            return [without_titles(item) for item in value]
        return value

    normalized = without_titles(schema)
    return {
        "properties": dict(normalized.get("properties") or {}),
        "required": list(normalized.get("required") or []),
    }


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
        docstring = (ast.get_docstring(node) or "").strip()
        entry: dict[str, Any] = {
            "params": _extract_ast_params(node.args),
            "input_schema": _arguments_input_schema(node.name, node.args),
            "summary": docstring.splitlines()[0] if docstring else "",
        }
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
                tool_name = None
                if decorator.args:
                    tool_name = _ast_literal(decorator.args[0])
                for keyword in decorator.keywords:
                    if keyword.arg == "name":
                        tool_name = _ast_literal(keyword.value)
                        break
                entry["tool_name"] = tool_name
        out[node.name] = entry
    return out


def _parse_actions_models(path: Path) -> dict[str, dict[str, Any]]:
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
                fields.append(
                    {
                        "param": _normalize_param(stmt.target.id, required=True),
                        "schema": _field_schema(
                            stmt.target.id,
                            stmt.annotation,
                            default_node=None,
                        ),
                    }
                )
                continue
            fields.append(
                {
                    "param": _normalize_param(
                        stmt.target.id,
                        required=False,
                        default=_ast_literal(stmt.value),
                    ),
                    "schema": _field_schema(
                        stmt.target.id,
                        stmt.annotation,
                        default_node=stmt.value,
                    ),
                }
            )
        class_info[node.name] = {
            "bases": base_names,
            "is_base_model": has_base_model,
            "fields": fields,
        }

    resolved_fields: dict[str, list[dict[str, Any]]] = {}
    changed = True
    while changed:
        changed = False
        for name, info in class_info.items():
            if name in resolved_fields:
                continue
            if info["is_base_model"]:
                resolved_fields[name] = list(info["fields"])
                changed = True
                continue
            for base_name in info["bases"]:
                inherited = resolved_fields.get(base_name)
                if inherited is None:
                    continue
                fields = list(inherited)
                if info["fields"]:
                    by_name = {
                        field["param"]["name"]: field
                        for field in fields
                    }
                    for field in info["fields"]:
                        by_name[field["param"]["name"]] = field
                    fields = [
                        by_name[field["param"]["name"]]
                        for field in inherited
                        if field["param"]["name"] in by_name
                    ]
                    for field in info["fields"]:
                        if field["param"]["name"] not in {
                            inherited_field["param"]["name"]
                            for inherited_field in inherited
                        }:
                            fields.append(field)
                resolved_fields[name] = fields
                changed = True
                break
    out: dict[str, dict[str, Any]] = {}
    for name, fields in resolved_fields.items():
        required = [
            field["param"]["name"]
            for field in fields
            if field["param"]["required"]
        ]
        schema: dict[str, Any] = {
            "properties": {
                field["param"]["name"]: field["schema"]
                for field in fields
            },
            "title": name,
            "type": "object",
        }
        if required:
            schema["required"] = required
        out[name] = {
            "params": [field["param"] for field in fields],
            "input_schema": schema,
        }
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
    models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        return list(route["params"]), dict(route["input_schema"])
    model = models.get(binding.request_model)
    if model is None:
        raise ValueError(f"missing Actions request model: {binding.request_model}")
    return list(model["params"]), dict(model["input_schema"])


def build_tool_surface_reference() -> dict[str, Any]:
    mcp_functions = _parse_functions_with_route_metadata(_integrations_path("mcp_server.py"))
    actions_functions = _parse_functions_with_route_metadata(_integrations_path("actions_api.py"))
    actions_models = _parse_actions_models(_integrations_path("actions_api.py"))
    surfaces = ai_surface_sets()
    manifest = load_tool_manifest()
    manifest_tools = {
        str(item.get("id")): item
        for item in list(manifest.get("tools") or [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    categories = {
        spec.canonical_name: spec.category
        for spec in TOOL_SURFACE_SPECS
    }
    for spec in TOOL_SURFACE_SPECS:
        for alias in spec.mcp_aliases:
            categories[alias.removesuffix("_tool")] = spec.category
    categories.update(
        {
            "ai_tools": "discovery",
            "generate_config": "config",
            "review_config": "config",
            "recommend_image_pipeline": "recommendation",
        }
    )

    mcp_live_tools: list[dict[str, Any]] = []
    for function_name, metadata in mcp_functions.items():
        if not metadata.get("tool"):
            continue
        tool_name = metadata.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(
                f"MCP tool must declare an explicit canonical name: {function_name}"
            )
        manifest_tool = manifest_tools.get(tool_name)
        explicit_maturity = (
            str(manifest_tool.get("maturity"))
            if isinstance(manifest_tool, dict)
            and str(manifest_tool.get("maturity") or "")
            else None
        )
        explicit_tags = (
            [
                str(tag)
                for tag in list(manifest_tool.get("tags") or [])
                if str(tag)
            ]
            if isinstance(manifest_tool, dict)
            else []
        )
        input_schema = dict(metadata["input_schema"])
        required = set(input_schema.get("required") or [])
        inputs = [
            {
                "name": name,
                "required": name in required,
                "schema": dict(schema),
            }
            for name, schema in dict(
                input_schema.get("properties") or {}
            ).items()
        ]
        surface_tiers = [
            name
            for name, surface in surfaces.items()
            if tool_name in set(surface["tool_ids"])
        ]
        mcp_live_tools.append(
            {
                "name": tool_name,
                "function": function_name,
                "summary": str(metadata.get("summary") or ""),
                "category": categories.get(tool_name, "mcp"),
                "surface_tiers": surface_tiers,
                "maturity": explicit_maturity,
                "maturity_source": (
                    "tools_manifest"
                    if explicit_maturity is not None
                    else "unclassified"
                ),
                "tags": explicit_tags,
                "tags_source": (
                    "tools_manifest"
                    if explicit_tags
                    else "unclassified"
                ),
                "params": list(metadata["params"]),
                "inputs": inputs,
                "input_schema": input_schema,
                "parameter_schema": _normalize_parameter_schema(
                    input_schema
                ),
                "examples": (
                    list(manifest_tool.get("examples") or [])
                    if isinstance(manifest_tool, dict)
                    else []
                ),
                "effects": (
                    dict(manifest_tool.get("effects") or {})
                    if isinstance(manifest_tool, dict)
                    else None
                ),
                "requires": (
                    dict(manifest_tool.get("requires") or {})
                    if isinstance(manifest_tool, dict)
                    else None
                ),
                "metadata_source": (
                    "tools_manifest+mcp_ast"
                    if isinstance(manifest_tool, dict)
                    else "mcp_ast"
                ),
            }
        )
    live_names = [item["name"] for item in mcp_live_tools]
    if len(live_names) != len(set(live_names)):
        raise ValueError("duplicate canonical MCP tool names")

    declared_live_names = list(surfaces["mcp_live"]["tool_ids"])
    if live_names != declared_live_names:
        raise ValueError(
            "MCP live surface does not match the declared AI surface: "
            f"registered={live_names}, declared={declared_live_names}"
        )

    actions_public_names = [spec.canonical_name for spec in TOOL_SURFACE_SPECS]
    declared_actions_names = list(surfaces["actions_public"]["tool_ids"])
    if actions_public_names != declared_actions_names:
        raise ValueError(
            "Actions public surface does not match the declared AI surface: "
            f"registered={actions_public_names}, declared={declared_actions_names}"
        )

    for subset_name in ("guaranteed_ai_safe", "config_review"):
        unknown = set(surfaces[subset_name]["tool_ids"]) - set(live_names)
        if unknown:
            raise ValueError(
                f"{subset_name} contains tools outside the live MCP surface: {sorted(unknown)}"
            )

    tools: list[dict[str, Any]] = []
    for spec in TOOL_SURFACE_SPECS:
        tool_params = _inspect_tool_runner_params(spec.tool_runner_fn)
        mcp_primary = mcp_functions.get(spec.mcp_tool_fn)
        if mcp_primary is None:
            raise ValueError(f"missing MCP tool function: {spec.mcp_tool_fn}")
        if not mcp_primary.get("tool"):
            raise ValueError(f"MCP function is not decorated with @app.tool(): {spec.mcp_tool_fn}")
        if mcp_primary.get("tool_name") != spec.canonical_name:
            raise ValueError(
                f"MCP canonical name mismatch for {spec.mcp_tool_fn}: "
                f"expected {spec.canonical_name}, got {mcp_primary.get('tool_name')}"
            )
        mcp_params = list(mcp_primary["params"])
        mcp_input_schema = dict(mcp_primary["input_schema"])

        mcp_aliases: list[dict[str, Any]] = []
        for alias_name in spec.mcp_aliases:
            alias = mcp_functions.get(alias_name)
            if alias is None:
                raise ValueError(f"missing MCP alias function: {alias_name}")
            if not alias.get("tool"):
                raise ValueError(f"MCP alias function is not decorated with @app.tool(): {alias_name}")
            expected_alias_name = alias_name.removesuffix("_tool")
            if alias.get("tool_name") != expected_alias_name:
                raise ValueError(
                    f"MCP alias canonical name mismatch for {alias_name}: "
                    f"expected {expected_alias_name}, got {alias.get('tool_name')}"
                )
            alias_params = list(alias["params"])
            alias_input_schema = dict(alias["input_schema"])
            mcp_aliases.append(
                {
                    "name": expected_alias_name,
                    "function": alias_name,
                    "params": alias_params,
                    "input_schema": alias_input_schema,
                    "parameter_schema": _normalize_parameter_schema(
                        alias_input_schema
                    ),
                    "parity_with_tool_runner": _params_equal(alias_params, tool_params),
                }
            )

        actions_params, actions_input_schema = _binding_params(
            spec.actions,
            routes=actions_functions,
            models=actions_models,
        )
        actions_aliases: list[dict[str, Any]] = []
        for alias in spec.actions_aliases:
            alias_params, alias_input_schema = _binding_params(
                alias,
                routes=actions_functions,
                models=actions_models,
            )
            actions_aliases.append(
                {
                    "method": alias.method,
                    "path": alias.path,
                    "route_fn": alias.route_fn,
                    "request_model": alias.request_model,
                    "params": alias_params,
                    "input_schema": alias_input_schema,
                    "parameter_schema": _normalize_parameter_schema(
                        alias_input_schema
                    ),
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
                    "tool": mcp_primary["tool_name"],
                    "function": spec.mcp_tool_fn,
                    "params": mcp_params,
                    "input_schema": mcp_input_schema,
                    "parameter_schema": _normalize_parameter_schema(
                        mcp_input_schema
                    ),
                    "aliases": mcp_aliases,
                },
                "actions": {
                    "method": spec.actions.method,
                    "path": spec.actions.path,
                    "route_fn": spec.actions.route_fn,
                    "request_model": spec.actions.request_model,
                    "params": actions_params,
                    "input_schema": actions_input_schema,
                    "parameter_schema": _normalize_parameter_schema(
                        actions_input_schema
                    ),
                    "aliases": actions_aliases,
                },
                "parity": {
                    "mcp_vs_tool_runner": _params_equal(mcp_params, tool_params),
                    "actions_vs_tool_runner": _params_equal(actions_params, tool_params),
                    "mcp_vs_actions_schema": _normalize_parameter_schema(
                        mcp_input_schema
                    )
                    == _normalize_parameter_schema(actions_input_schema),
                    "tool_runner_aliases": all(item["parity_with_tool_runner"] for item in tool_aliases) if tool_aliases else True,
                    "mcp_aliases": all(item["parity_with_tool_runner"] for item in mcp_aliases) if mcp_aliases else True,
                    "actions_aliases": all(item["parity_with_tool_runner"] for item in actions_aliases) if actions_aliases else True,
                },
            }
        )
    return {
        "schema_version": 3,
        "source": "yolozu.integrations",
        "surfaces": surfaces,
        "mcp_live_tools": mcp_live_tools,
        "tools": tools,
    }


def collect_surface_parity_errors(reference: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for tool in reference.get("tools", []):
        name = tool.get("canonical_name")
        parity = tool.get("parity", {})
        for key in (
            "mcp_vs_tool_runner",
            "actions_vs_tool_runner",
            "mcp_vs_actions_schema",
            "tool_runner_aliases",
            "mcp_aliases",
            "actions_aliases",
        ):
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
        "## Public surface sets",
        "",
        "| Surface | Tool ids | Availability |",
        "|---|---|---|",
    ]
    for name, surface in reference.get("surfaces", {}).items():
        tool_ids = ", ".join(f"`{tool_id}`" for tool_id in surface.get("tool_ids", []))
        lines.append(
            f"| `{name}` | {tool_ids} | {surface.get('availability', '')} |"
        )
    lines.extend(
        [
            "",
            (
                "The `mcp_live` set is the exact list returned by the installed "
                "MCP SDK's live tool-list API; registration is discovery evidence, "
                "not an execution guarantee. Only `guaranteed_ai_safe` carries "
                "the deterministic lightweight guarantee; `actions_public` lists "
                "canonical operations shared with the Actions API."
            ),
            "",
            "## MCP/Actions parity",
            "",
        "| Canonical | Category | MCP tool | Actions endpoint | Request model | Parity |",
        "|---|---|---|---|---|---|",
        ]
    )
    for tool in reference.get("tools", []):
        parity = tool.get("parity", {})
        parity_ok = all(
            bool(parity.get(key, False))
            for key in (
                "mcp_vs_tool_runner",
                "actions_vs_tool_runner",
                "mcp_vs_actions_schema",
                "tool_runner_aliases",
                "mcp_aliases",
                "actions_aliases",
            )
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
