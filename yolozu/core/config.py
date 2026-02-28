"""Configuration loading, constraints validation, and minimal YAML parser.

Provides ``load_constraints``, ``load_symmetry_map``, and a self-contained
``simple_yaml_load`` fallback for environments where PyYAML is not installed.
"""

import json
from pathlib import Path

__all__ = [
    "DEFAULT_CONSTRAINTS",
    "default_runtime_config_path",
    "load_symmetry_map",
    "get_symmetry_spec",
    "load_constraints",
    "merge_defaults",
    "validate_constraints",
    "simple_yaml_load",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CONFIG_DIR = _REPO_ROOT / "configs" / "runtime"


DEFAULT_CONSTRAINTS = {
    "enabled": {
        "depth_prior": False,
        "table_plane": False,
        "upright": False,
        "template_gate": False,
    },
    "table_plane": {"n": [0.0, 0.0, 1.0], "d": 0.0},
    "depth_prior": {"default": {"min_z": None, "max_z": None}, "per_class": {}},
    "upright": {
        "default": {"roll_deg": [-180.0, 180.0], "pitch_deg": [-180.0, 180.0]},
        "per_class": {},
    },
    "template_gate": {"tau": 0.0},
}


def default_runtime_config_path(filename):
    modern = _RUNTIME_CONFIG_DIR / str(filename)
    if modern.exists():
        return modern
    legacy = _REPO_ROOT / str(filename)
    if legacy.exists():
        return legacy
    return modern


def load_symmetry_map(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("symmetry.json must contain an object")
    for key, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"symmetry entry for {key} must be an object")
        sym_type = spec.get("type", "none")
        if sym_type in ("C2", "C4", "Cn"):
            n = spec.get("n")
            if sym_type == "C2":
                n = 2
            elif sym_type == "C4":
                n = 4
            if sym_type in ("C2", "C4") or n is not None:
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"symmetry entry {key} has invalid n")
        axis = spec.get("axis")
        if axis is not None and (not isinstance(axis, list) or len(axis) != 3):
            raise ValueError(f"symmetry entry {key} has invalid axis")
    return data


def get_symmetry_spec(symmetry_map, class_key):
    if class_key in symmetry_map:
        return symmetry_map[class_key]
    if isinstance(class_key, int):
        return symmetry_map.get(str(class_key))
    if isinstance(class_key, str) and class_key.isdigit():
        return symmetry_map.get(int(class_key))
    return None


def load_constraints(path):
    text = Path(path).read_text()
    data = None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = simple_yaml_load(text)
    if data is None:
        data = {}
    merged = merge_defaults(DEFAULT_CONSTRAINTS, data)
    validate_constraints(merged)
    return merged


def merge_defaults(defaults, override):
    if not isinstance(override, dict):
        return defaults
    merged = {}
    for key, value in defaults.items():
        if key in override and isinstance(value, dict):
            merged[key] = merge_defaults(value, override[key])
        else:
            merged[key] = override.get(key, value)
    for key, value in override.items():
        if key not in merged:
            merged[key] = value
    return merged


def validate_constraints(cfg):
    enabled = cfg.get("enabled", {})
    for key in ("depth_prior", "table_plane", "upright", "template_gate"):
        if not isinstance(enabled.get(key), bool):
            raise ValueError(f"enabled.{key} must be boolean")

    plane = cfg.get("table_plane", {})
    n = plane.get("n", [])
    if not isinstance(n, list) or len(n) != 3:
        raise ValueError("table_plane.n must be length-3 list")
    d = plane.get("d")
    if not isinstance(d, (int, float)):
        raise ValueError("table_plane.d must be number")

    template_gate = cfg.get("template_gate", {})
    tau = template_gate.get("tau")
    if not isinstance(tau, (int, float)):
        raise ValueError("template_gate.tau must be number")


def simple_yaml_load(text):
    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.lstrip(" ")))

    def parse_value(value):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_value(item) for item in inner.split(",")]
        if value == "{}":
            return {}
        lower = value.lower()
        if lower in ("null", "none", "~"):
            return None
        if lower in ("true", "false"):
            return lower == "true"
        if value and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        try:
            if any(ch in value for ch in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value

    def parse_block(start_index, indent_level):
        out = {}
        i = start_index
        while i < len(lines):
            line_indent, content = lines[i]
            if line_indent < indent_level:
                break
            if line_indent > indent_level:
                raise ValueError("invalid indentation")
            key, sep, rest = content.partition(":")
            if not sep:
                raise ValueError(f"invalid line: {content}")
            key = key.strip()
            value = rest.strip()
            i += 1
            if value == "":
                if i < len(lines) and lines[i][0] > line_indent:
                    child_indent = lines[i][0]
                    child, i = parse_block(i, child_indent)
                    out[key] = child
                else:
                    out[key] = {}
            else:
                out[key] = parse_value(value)
        return out, i

    if not lines:
        return {}
    data, _ = parse_block(0, lines[0][0])
    return data
