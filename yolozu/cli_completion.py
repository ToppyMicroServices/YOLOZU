"""Shell completion helpers for the yolozu CLI."""

from __future__ import annotations

from pathlib import Path

TOP_LEVEL_COMMANDS: tuple[str, ...] = (
    "doctor",
    "list",
    "fetch",
    "export",
    "predict-images",
    "eval-coco",
    "calibrate",
    "eval-long-tail",
    "long-tail-recipe",
    "parity",
    "predictions",
    "validate",
    "eval-instance-seg",
    "onnxrt",
    "resources",
    "migrate",
    "import",
    "train",
    "test",
    "demo",
    "completion",
)

NESTED_COMMANDS: dict[str, tuple[str, ...]] = {
    "doctor": ("import",),
    "list": ("models",),
    "predictions": ("migrate",),
    "onnxrt": ("export", "quantize"),
    "import": ("dataset", "config"),
    "demo": ("overview", "instance-seg", "continual", "keypoints", "pose", "depth", "train"),
}


def _func_name_for_command(command: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(command))
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "yolozu"
    return f"_{cleaned}_complete"


def _render_bash(*, command: str) -> str:
    fn = _func_name_for_command(command)
    top = " ".join(TOP_LEVEL_COMMANDS)
    lines = [
        f"# bash completion for {command}",
        f"{fn}() {{",
        '  local cur cmd',
        '  cur="${COMP_WORDS[COMP_CWORD]}"',
        '  if [[ "${COMP_CWORD}" -eq 1 ]]; then',
        f'    COMPREPLY=( $(compgen -W "{top}" -- "${{cur}}") )',
        "    return 0",
        "  fi",
        '  cmd="${COMP_WORDS[1]}"',
        '  case "${cmd}" in',
    ]
    for parent, children in NESTED_COMMANDS.items():
        child_text = " ".join(children)
        lines.extend(
            [
                f"    {parent})",
                f'      COMPREPLY=( $(compgen -W "{child_text}" -- "${{cur}}") )',
                "      return 0",
                "      ;;",
            ]
        )
    lines.extend(
        [
            "    *)",
            "      COMPREPLY=()",
            "      return 0",
            "      ;;",
            "  esac",
            "}",
            f"complete -F {fn} {command}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_zsh(*, command: str) -> str:
    fn = _func_name_for_command(command)
    top = " ".join(TOP_LEVEL_COMMANDS)
    lines = [
        f"#compdef {command}",
        f"# zsh completion for {command}",
        f"{fn}() {{",
        "  local -a top",
        f"  top=({top})",
        "  if (( CURRENT == 2 )); then",
        "    _describe 'command' top",
        "    return 0",
        "  fi",
        "  case \"$words[2]\" in",
    ]
    for parent, children in NESTED_COMMANDS.items():
        child_text = " ".join(children)
        lines.extend(
            [
                f"    {parent})",
                f"      _values 'subcommand' {child_text}",
                "      return 0",
                "      ;;",
            ]
        )
    lines.extend(
        [
            "  esac",
            "}",
            f"compdef {fn} {command}",
            "",
        ]
    )
    return "\n".join(lines)


def render_completion(*, shell: str, command: str = "yolozu") -> str:
    shell_name = str(shell).strip().lower()
    command_name = str(command).strip()
    if not command_name:
        command_name = "yolozu"
    if shell_name == "bash":
        return _render_bash(command=command_name)
    if shell_name == "zsh":
        return _render_zsh(command=command_name)
    raise ValueError(f"unsupported shell: {shell}")


def write_completion(*, shell: str, command: str, output: str) -> str:
    text = render_completion(shell=shell, command=command)
    out = str(output).strip()
    if not out or out == "-":
        return text
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)

