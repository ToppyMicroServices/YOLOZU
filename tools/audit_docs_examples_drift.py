#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]

DEFAULT_DOCS = [
    "README.md",
    "Readme_jp.md",
    "Readme_zh.md",
    "docs/README.md",
    "docs/cpu_only_dod.md",
    "docs/external_inference.md",
    "docs/interop_detectron2_mmdet.md",
    "docs/interop_yolox.md",
    "examples/infer_cpp/README.md",
    "examples/infer_rust/README.md",
]

_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.DOTALL)
_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9\-]*")
_SHELL_PYTHON_DELEGATE_RE = re.compile(
    r'exec\s+python3\s+"\$\{REPO_ROOT\}/(?P<entrypoint>tools/[A-Za-z0-9_.\-/]+\.py)"\s+"\$@"'
)
_COMMAND_PREFIXES = ("yolozu", "python3 -m yolozu", "python -m yolozu", "bash scripts/", "python3 tools/")
_EXTERNAL_TRAIN_SUBCOMMANDS = {
    "yolox": "train-yolox",
    "detectron2": "train-detectron2",
    "mmdetection": "train-mmdetection",
    "mmpose": "train-mmpose",
    "mmseg": "train-mmseg",
    "tao": "train-tao",
    "ultralytics": "train-ultralytics",
    "hf-detr": "train-hf-detr",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit README/docs shell examples against CLI help and manifest drift gates."
    )
    p.add_argument(
        "--docs",
        action="append",
        default=None,
        help="Markdown doc to scan. Repeatable. Defaults to the maintained README and interop guide set.",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable used for yolozu/tool probes.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.add_argument("--skip-manual", action="store_true", help="Skip manual/CLI drift audit.")
    p.add_argument("--skip-manifest", action="store_true", help="Skip manifest inputs-vs-help audit.")
    return p.parse_args(argv)


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _run(cmd: list[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def _shell_lines_from_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    for block in _FENCE_RE.findall(text):
        pending = ""
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].strip()
            if pending:
                line = f"{pending} {line}"
                pending = ""
            if line.endswith("\\"):
                pending = line[:-1].strip()
                continue
            lines.append(line)
        if pending:
            lines.append(pending)
    return lines


def _interesting_command(line: str) -> bool:
    if "=" in line and not line.startswith(("python", "yolozu", "bash")):
        return False
    return line.startswith(_COMMAND_PREFIXES)


def _extract_flags(tokens: list[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in tokens if token.startswith("--")}


def _extract_help_flags(help_text: str) -> set[str]:
    out: set[str] = set()
    for line in help_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("-") or "--" not in stripped:
            continue
        head = re.split(r"\s{2,}", stripped, maxsplit=1)[0]
        out |= set(_FLAG_RE.findall(head))
    return out


def _option_value(tokens: list[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 < len(tokens):
                return tokens[index + 1]
            return None
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _external_train_help(
    args: list[str],
    *,
    python: str,
) -> tuple[list[str], str, str | None] | None:
    if not args or args[0] != "train":
        return None
    backend = _option_value(args, "--external-backend")
    subcommand = _EXTERNAL_TRAIN_SUBCOMMANDS.get(str(backend or "").strip().lower())
    if subcommand is None:
        return None

    parent_probe = [python, "-m", "yolozu", "train", "--help"]
    delegated_probe = [python, "tools/support_external_training.py", subcommand, "--help"]
    help_parts: list[str] = []
    for probe in (parent_probe, delegated_probe):
        proc = _run(probe)
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return delegated_probe, "", help_text[-1000:] or "external train help probe failed"
        help_parts.append(help_text)
    return delegated_probe, "\n".join(help_parts), None


def _first_existing_help(tokens: list[str], *, python: str) -> tuple[list[str], str, str | None]:
    probes: list[list[str]] = []
    if tokens[:3] in (["python3", "-m", "yolozu"], ["python", "-m", "yolozu"]):
        args = tokens[3:]
    elif tokens[:1] == ["yolozu"]:
        args = tokens[1:]
    else:
        return [], "", "not a yolozu command"

    external_train = _external_train_help(args, python=python)
    if external_train is not None:
        return external_train

    subcmd: list[str] = []
    for token in args:
        if token.startswith("-"):
            break
        if "/" in token or token.endswith((".json", ".yaml", ".yml", ".py")):
            break
        subcmd.append(token)

    for n in range(len(subcmd), 0, -1):
        probes.append([python, "-m", "yolozu", *subcmd[:n], "--help"])
    probes.append([python, "-m", "yolozu", "--help"])

    last_err = ""
    for probe in probes:
        proc = _run(probe)
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0:
            return probe, help_text, None
        last_err = help_text[-1000:]
    return probes[-1] if probes else [], "", last_err or "help probe failed"


def _audit_yolozu_command(line: str, *, python: str) -> dict[str, Any]:
    tokens = shlex.split(line)
    probe, help_text, error = _first_existing_help(tokens, python=python)
    flags = _extract_flags(tokens)
    help_flags = _extract_help_flags(help_text) if not error else set()
    missing_flags = sorted(flag for flag in flags if flag not in help_flags)
    return {
        "line": line,
        "kind": "yolozu",
        "help_probe": probe,
        "ok": error is None and not missing_flags,
        "error": error,
        "missing_flags": missing_flags,
    }


def _shell_python_delegate(path: Path) -> Path | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _SHELL_PYTHON_DELEGATE_RE.search(source)
    if match is None:
        return None
    candidate = (repo_root / match.group("entrypoint")).resolve()
    tools_root = (repo_root / "tools").resolve()
    if not candidate.is_relative_to(tools_root) or not candidate.is_file():
        return None
    return candidate


def _audit_script_command(line: str, *, python: str) -> dict[str, Any]:
    tokens = shlex.split(line)
    script = tokens[1] if len(tokens) >= 2 and tokens[0] == "bash" else ""
    path = _resolve(script)
    if not path.is_file():
        return {"line": line, "kind": "script", "ok": False, "error": f"script not found: {script}"}
    proc = _run(["bash", str(path), "--help"])
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    delegated_probe: list[str] | None = None
    delegate = _shell_python_delegate(path)
    delegate_error: str | None = None
    if proc.returncode == 0 and delegate is not None:
        delegated_probe = [python, str(delegate.relative_to(repo_root)), "--help"]
        delegated = _run(delegated_probe)
        delegated_text = (delegated.stdout or "") + "\n" + (delegated.stderr or "")
        if delegated.returncode == 0:
            help_text = f"{help_text}\n{delegated_text}"
        else:
            delegate_error = delegated_text[-1000:] or f"delegated --help exited {delegated.returncode}"
    flags = _extract_flags(tokens)
    help_flags = _extract_help_flags(help_text)
    missing_flags = sorted(flag for flag in flags if flag not in help_flags)
    result = {
        "line": line,
        "kind": "script",
        "help_probe": ["bash", script, "--help"],
        "ok": proc.returncode == 0 and delegate_error is None and not missing_flags,
        "error": delegate_error if delegate_error is not None else (None if proc.returncode == 0 else help_text[-1000:]),
        "missing_flags": missing_flags,
    }
    if delegated_probe is not None:
        result["delegated_help_probe"] = delegated_probe
    return result


def _audit_tool_command(line: str, *, manifest: dict[str, Any]) -> dict[str, Any]:
    tokens = shlex.split(line)
    entrypoint = tokens[1] if len(tokens) >= 2 and tokens[0].startswith("python") else ""
    tool = None
    for item in manifest.get("tools") or []:
        if isinstance(item, dict) and item.get("entrypoint") == entrypoint:
            tool = item
            break
    if tool is None:
        return {"line": line, "kind": "tool", "ok": False, "error": f"tool not found in manifest: {entrypoint}"}
    declared = {
        str(inp.get("flag"))
        for inp in tool.get("inputs") or []
        if isinstance(inp, dict) and isinstance(inp.get("flag"), str)
    }
    missing_flags = sorted(flag for flag in _extract_flags(tokens) if flag not in declared and flag != "--help")
    return {
        "line": line,
        "kind": "tool",
        "tool_id": tool.get("id"),
        "ok": not missing_flags,
        "error": None,
        "missing_flags": missing_flags,
    }


def _scan_docs(doc_paths: list[Path], *, python: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in doc_paths:
        if not path.is_file():
            results.append({"doc": str(path), "line": "", "kind": "doc", "ok": False, "error": "doc not found"})
            continue
        for line in _shell_lines_from_markdown(path):
            if not _interesting_command(line):
                continue
            if line.startswith(("yolozu", "python3 -m yolozu", "python -m yolozu")):
                item = _audit_yolozu_command(line, python=python)
            elif line.startswith("bash scripts/"):
                item = _audit_script_command(line, python=python)
            else:
                item = _audit_tool_command(line, manifest=manifest)
            item["doc"] = str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path)
            results.append(item)
    return results


def _load_manifest() -> dict[str, Any]:
    return json.loads((repo_root / "tools" / "manifest.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    docs = [_resolve(p) for p in (args.docs or DEFAULT_DOCS)]
    manifest = _load_manifest()

    doc_results = _scan_docs(docs, python=str(args.python), manifest=manifest)
    failures = [item for item in doc_results if not item.get("ok")]

    subchecks: list[dict[str, Any]] = []
    if not args.skip_manual:
        proc = _run([str(args.python), "tools/audit_manual_cli_drift.py", "--json"])
        subchecks.append(
            {
                "name": "manual_cli_drift",
                "ok": proc.returncode == 0,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        )
    if not args.skip_manifest:
        proc = _run([str(args.python), "tools/audit_manifest_inputs_vs_help.py"])
        subchecks.append(
            {
                "name": "manifest_inputs_vs_help",
                "ok": proc.returncode == 0,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        )

    ok = not failures and all(item.get("ok") for item in subchecks)
    payload = {
        "kind": "docs_examples_drift_audit",
        "schema_version": 1,
        "ok": ok,
        "docs": [str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p) for p in docs],
        "checked_examples": len(doc_results),
        "failures": failures,
        "subchecks": subchecks,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"checked_examples {len(doc_results)}")
        print(f"failures {len(failures)}")
        for item in failures:
            print(f"FAIL {item.get('doc')}: {item.get('line')}")
            if item.get("missing_flags"):
                print("  missing flags: " + ", ".join(item["missing_flags"]))
            if item.get("error"):
                print(f"  error: {item.get('error')}")
        for item in subchecks:
            print(f"subcheck {item['name']}: {'ok' if item.get('ok') else 'fail'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
