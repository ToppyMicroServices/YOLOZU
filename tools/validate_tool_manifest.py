import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_RUNNERS = {"python3", "bash"}
_IO_KINDS = {"file", "dir", "string", "number", "json", "stdout"}
_EFFECT_KINDS = {"file", "dir"}
_EFFECT_SCOPES = {"path", "tree"}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_contracts(contracts: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    if contracts is None:
        return out, errors
    if not isinstance(contracts, dict):
        return out, ["manifest.contracts: expected object"]

    for name, spec in contracts.items():
        if not isinstance(name, str) or not name:
            errors.append("manifest.contracts: contract id must be non-empty string")
            continue
        if not isinstance(spec, dict):
            errors.append(f"manifest.contracts.{name}: expected object")
            continue

        schema = spec.get("schema")
        if schema is not None:
            if not isinstance(schema, str) or not schema:
                errors.append(f"manifest.contracts.{name}.schema: expected non-empty string")
            else:
                if schema.startswith("/"):
                    errors.append(f"manifest.contracts.{name}.schema: must be repo-relative, got absolute path")
                elif ".." in Path(schema).parts:
                    errors.append(f"manifest.contracts.{name}.schema: must not contain '..'")
                else:
                    schema_path = _resolve(schema)
                    if not schema_path.exists():
                        errors.append(f"manifest.contracts.{name}.schema: file not found: {schema}")
                    else:
                        try:
                            schema_obj = _load_json(schema_path)
                            if not isinstance(schema_obj, dict):
                                errors.append(f"manifest.contracts.{name}.schema: expected JSON object: {schema}")
                        except Exception as exc:
                            errors.append(f"manifest.contracts.{name}.schema: invalid JSON: {schema}: {exc}")

        out[name] = spec

    return out, errors


def _validate_tool(tool: Any, *, index: int, require_declarative: bool = False) -> list[str]:
    errors: list[str] = []
    where = f"tools[{index}]"
    if not isinstance(tool, dict):
        return [f"{where}: expected object"]

    tool_id = tool.get("id")
    if not isinstance(tool_id, str) or not tool_id:
        errors.append(f"{where}.id: required string")
    elif not _ID_RE.match(tool_id):
        errors.append(f"{where}.id: invalid format '{tool_id}'")

    entrypoint = tool.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        errors.append(f"{where}.entrypoint: required string")
    else:
        if entrypoint.startswith("/"):
            errors.append(f"{where}.entrypoint: must be repo-relative, got absolute path")
        if ".." in Path(entrypoint).parts:
            errors.append(f"{where}.entrypoint: must not contain '..'")
        ep = _resolve(entrypoint)
        if not ep.exists():
            errors.append(f"{where}.entrypoint: file not found: {entrypoint}")

    runner = tool.get("runner")
    if not isinstance(runner, str) or runner not in _RUNNERS:
        errors.append(f"{where}.runner: must be one of {sorted(_RUNNERS)}")

    summary = tool.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append(f"{where}.summary: required string")

    if require_declarative:
        required_top = ("platform", "inputs", "effects", "outputs", "examples")
        for field in required_top:
            if field not in tool:
                errors.append(f"{where}.{field}: required in declarative mode")

    docs = tool.get("docs")
    if docs is not None:
        if not isinstance(docs, list):
            errors.append(f"{where}.docs: must be a list of repo-relative paths")
        else:
            for j, doc in enumerate(docs):
                if not isinstance(doc, str) or not doc:
                    errors.append(f"{where}.docs[{j}]: must be non-empty string")
                    continue
                if doc.startswith("/"):
                    errors.append(f"{where}.docs[{j}]: must be repo-relative, got absolute path")
                    continue
                if ".." in Path(doc).parts:
                    errors.append(f"{where}.docs[{j}]: must not contain '..'")
                    continue
                p = _resolve(doc)
                if not p.exists():
                    errors.append(f"{where}.docs[{j}]: file not found: {doc}")

    contract_outputs = tool.get("contract_outputs")
    if contract_outputs is not None:
        if not isinstance(contract_outputs, dict):
            errors.append(f"{where}.contract_outputs: expected object")
        else:
            for k, v in contract_outputs.items():
                if not isinstance(k, str) or not k:
                    errors.append(f"{where}.contract_outputs: contract id must be non-empty string")
                    continue
                if not isinstance(v, str) or not v:
                    errors.append(f"{where}.contract_outputs.{k}: expected non-empty string (output name)")

    declared_flags: set[str] = set()
    inputs = tool.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict) and isinstance(item.get("flag"), str) and item.get("flag"):
                declared_flags.add(str(item["flag"]))

    effects = tool.get("effects")
    if effects is not None:
        if not isinstance(effects, dict):
            errors.append(f"{where}.effects: expected object")
        else:
            if require_declarative and "writes" not in effects:
                errors.append(f"{where}.effects.writes: required in declarative mode")
            if require_declarative and "fixed_writes" not in effects:
                errors.append(f"{where}.effects.fixed_writes: required in declarative mode")
            allow_unknown = bool(effects.get("allow_unknown_flags", False))
            writes = effects.get("writes")
            if writes is not None:
                if not isinstance(writes, list):
                    errors.append(f"{where}.effects.writes: expected list")
                else:
                    for j, w in enumerate(writes):
                        if not isinstance(w, dict):
                            errors.append(f"{where}.effects.writes[{j}]: expected object")
                            continue
                        flag = w.get("flag")
                        kind = w.get("kind")
                        scope = w.get("scope")
                        if not isinstance(flag, str) or not flag.startswith("--"):
                            errors.append(f"{where}.effects.writes[{j}].flag: expected '--foo' string")
                        elif (not allow_unknown) and declared_flags and flag not in declared_flags:
                            errors.append(f"{where}.effects.writes[{j}].flag: not declared in tool.inputs: {flag}")
                        if not isinstance(kind, str) or kind not in _EFFECT_KINDS:
                            errors.append(f"{where}.effects.writes[{j}].kind: must be one of {sorted(_EFFECT_KINDS)}")
                        if not isinstance(scope, str) or scope not in _EFFECT_SCOPES:
                            errors.append(f"{where}.effects.writes[{j}].scope: must be one of {sorted(_EFFECT_SCOPES)}")

            fixed = effects.get("fixed_writes")
            if fixed is not None:
                if not isinstance(fixed, list):
                    errors.append(f"{where}.effects.fixed_writes: expected list")
                else:
                    for j, fw in enumerate(fixed):
                        if not isinstance(fw, dict):
                            errors.append(f"{where}.effects.fixed_writes[{j}]: expected object")
                            continue
                        path = fw.get("path")
                        kind = fw.get("kind")
                        scope = fw.get("scope")
                        if not isinstance(path, str) or not path:
                            errors.append(f"{where}.effects.fixed_writes[{j}].path: required string")
                        else:
                            if path.startswith("/"):
                                errors.append(f"{where}.effects.fixed_writes[{j}].path: must be repo-relative")
                            elif ".." in Path(path).parts:
                                errors.append(f"{where}.effects.fixed_writes[{j}].path: must not contain '..'")
                        if not isinstance(kind, str) or kind not in _EFFECT_KINDS:
                            errors.append(f"{where}.effects.fixed_writes[{j}].kind: must be one of {sorted(_EFFECT_KINDS)}")
                        if not isinstance(scope, str) or scope not in _EFFECT_SCOPES:
                            errors.append(f"{where}.effects.fixed_writes[{j}].scope: must be one of {sorted(_EFFECT_SCOPES)}")

    for field in ("inputs", "outputs"):
        items = tool.get(field)
        if items is None:
            if require_declarative:
                errors.append(f"{where}.{field}: required in declarative mode")
            continue
        if not isinstance(items, list):
            errors.append(f"{where}.{field}: must be a list")
            continue
        for j, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{where}.{field}[{j}]: expected object")
                continue
            name = item.get("name")
            kind = item.get("kind")
            if not isinstance(name, str) or not name:
                errors.append(f"{where}.{field}[{j}].name: required string")
            if not isinstance(kind, str) or kind not in _IO_KINDS:
                errors.append(f"{where}.{field}[{j}].kind: must be one of {sorted(_IO_KINDS)}")
            flag = item.get("flag")
            if flag is not None and (not isinstance(flag, str) or (flag and not flag.startswith("--"))):
                errors.append(f"{where}.{field}[{j}].flag: expected '--foo' style flag string")

    examples = tool.get("examples")
    if examples is not None:
        if not isinstance(examples, list):
            errors.append(f"{where}.examples: must be a list")
        else:
            if require_declarative and len(examples) == 0:
                errors.append(f"{where}.examples: must contain at least one item in declarative mode")
            for j, ex in enumerate(examples):
                if not isinstance(ex, dict):
                    errors.append(f"{where}.examples[{j}]: expected object")
                    continue
                cmd = ex.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    errors.append(f"{where}.examples[{j}].command: required string")
    elif require_declarative:
        errors.append(f"{where}.examples: required in declarative mode")

    if require_declarative:
        platform_obj = tool.get("platform")
        if isinstance(platform_obj, dict):
            for key in ("cpu_ok", "gpu_required", "macos_ok", "linux_ok"):
                if key not in platform_obj:
                    errors.append(f"{where}.platform.{key}: required in declarative mode")
                elif not isinstance(platform_obj.get(key), bool):
                    errors.append(f"{where}.platform.{key}: must be bool in declarative mode")

    return list(dict.fromkeys(errors))


def validate_manifest(obj: Any, *, require_declarative: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["manifest: expected object"]

    mv = obj.get("manifest_version")
    if not isinstance(mv, int) or mv < 1:
        errors.append("manifest.manifest_version: required integer >= 1")

    tools = obj.get("tools")
    if not isinstance(tools, list) or not tools:
        errors.append("manifest.tools: required non-empty list")
        return errors

    contracts_map, contract_errors = _validate_contracts(obj.get("contracts"))
    errors.extend(contract_errors)

    seen: set[str] = set()
    for i, tool in enumerate(tools):
        errors.extend(_validate_tool(tool, index=i, require_declarative=require_declarative))

        # Validate tool.contract_outputs cross-references (if present)
        if isinstance(tool, dict) and isinstance(tool.get("contract_outputs"), dict):
            outputs = tool.get("outputs")
            output_names: set[str] = set()
            if isinstance(outputs, list):
                for out in outputs:
                    if isinstance(out, dict) and isinstance(out.get("name"), str) and out.get("name"):
                        output_names.add(str(out["name"]))

            for contract_id, output_name in tool["contract_outputs"].items():
                if not isinstance(contract_id, str) or not contract_id:
                    continue
                if contracts_map and contract_id not in contracts_map:
                    errors.append(f"tools[{i}].contract_outputs: unknown contract id '{contract_id}'")
                if not isinstance(output_name, str) or not output_name:
                    continue
                if output_names and output_name not in output_names:
                    errors.append(
                        f"tools[{i}].contract_outputs.{contract_id}: unknown output name '{output_name}' (not in tool.outputs)"
                    )

        if isinstance(tool, dict) and isinstance(tool.get("contracts"), dict):
            for direction in ("consumes", "produces"):
                refs = tool["contracts"].get(direction)
                if refs is None:
                    continue
                if not isinstance(refs, list) or not all(isinstance(x, str) and x for x in refs):
                    errors.append(f"tools[{i}].contracts.{direction}: expected list[str]")
                    continue
                for ref in refs:
                    if contracts_map and ref not in contracts_map:
                        errors.append(f"tools[{i}].contracts.{direction}: unknown contract id '{ref}'")
        tool_id = tool.get("id") if isinstance(tool, dict) else None
        if isinstance(tool_id, str) and tool_id:
            if tool_id in seen:
                errors.append(f"tools[{i}].id: duplicate id '{tool_id}'")
            seen.add(tool_id)

    return errors


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate tools/manifest.json structure and references.")
    p.add_argument("--manifest", default="tools/manifest.json", help="Manifest path (default: tools/manifest.json).")
    p.add_argument(
        "--require-declarative",
        action="store_true",
        help="Require declarative fields (platform/inputs/effects/outputs/examples) for each tool.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = _resolve(str(args.manifest))
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        obj = _load_json(manifest_path)
    except Exception as exc:
        print(f"error: failed to parse JSON: {manifest_path}: {exc}", file=sys.stderr)
        return 2

    errors = validate_manifest(obj, require_declarative=bool(args.require_declarative))
    if errors:
        print("error: invalid tool manifest:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 2

    print(f"OK: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
