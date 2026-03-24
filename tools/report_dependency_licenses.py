#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


_COPYLEFT_PATTERNS = [
    r"\bgnu\b",
    r"\bgpl\b",
    r"\bagpl\b",
    r"\blgpl\b",
    r"\bcopyleft\b",
]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_info() -> dict[str, Any]:
    try:
        sha = (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, PermissionError):
        sha = None
    try:
        dirty = subprocess.call(
            ["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        is_dirty = bool(dirty != 0)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, PermissionError):
        is_dirty = None
    return {"sha": sha, "dirty": is_dirty}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _parse_requirement_names(requirements_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in requirements_paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r", "--requirement")):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    inc = Path(parts[1].strip())
                    if not inc.is_absolute():
                        inc = REPO_ROOT / inc
                    names |= _parse_requirement_names([inc])
                continue
            if line.startswith(("--extra-index-url", "--index-url", "--find-links")):
                continue
            if line.startswith("--"):
                # Ignore other pip options.
                continue
            if "://" in line or line.startswith("git+"):
                # VCS/URL requirements are out of scope for this lightweight parser.
                continue

            # Extract distribution name up to the first version/operator delimiter.
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
            if not m:
                continue
            names.add(_normalize_name(m.group(1)))
    return names


def _copyleft_suspect(*, license_str: str | None, classifiers: list[str]) -> bool:
    text = " ".join([license_str or ""] + list(classifiers)).lower()
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in _COPYLEFT_PATTERNS)


@dataclass(frozen=True)
class _DistInfo:
    name: str
    version: str | None
    license: str | None
    classifiers: list[str]
    home_page: str | None
    direct: bool
    copyleft_suspect: bool
    license_unknown: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "classifiers": self.classifiers,
            "home_page": self.home_page,
            "direct": bool(self.direct),
            "copyleft_suspect": bool(self.copyleft_suspect),
            "license_unknown": bool(self.license_unknown),
        }


def _collect_distributions(*, direct_names: set[str]) -> list[_DistInfo]:
    out: list[_DistInfo] = []
    for dist in importlib_metadata.distributions():
        meta = dist.metadata
        name = meta.get("Name") or getattr(dist, "name", None) or ""
        if not name:
            continue
        license_str = meta.get("License")
        classifiers = list(meta.get_all("Classifier") or [])
        home_page = meta.get("Home-page") or meta.get("Project-URL") or None
        version = getattr(dist, "version", None)

        direct = _normalize_name(str(name)) in direct_names
        suspect = _copyleft_suspect(license_str=license_str, classifiers=classifiers)
        license_unknown = not bool((license_str or "").strip()) and not any("license" in c.lower() for c in classifiers)
        out.append(
            _DistInfo(
                name=str(name),
                version=str(version) if version is not None else None,
                license=str(license_str) if license_str is not None else None,
                classifiers=[str(c) for c in classifiers],
                home_page=str(home_page) if home_page is not None else None,
                direct=bool(direct),
                copyleft_suspect=bool(suspect),
                license_unknown=bool(license_unknown),
            )
        )
    out.sort(key=lambda d: _normalize_name(d.name))
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a best-effort dependency license report (not legal advice).")
    p.add_argument("--output", default="reports/dependency_licenses.json", help="Output JSON report path.")
    p.add_argument(
        "--requirements",
        action="append",
        default=None,
        help="Optional requirements file to mark direct dependencies (repeatable). Default: requirements*.txt at repo root.",
    )
    p.add_argument("--only-direct", action="store_true", help="Only include direct requirements (requires --requirements or defaults).")
    p.add_argument("--fail-on-copyleft", action="store_true", help="Exit non-zero if any copyleft-suspect license is detected.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    reqs_raw = args.requirements
    if not reqs_raw:
        req_paths = [
            REPO_ROOT / "requirements.txt",
            REPO_ROOT / "requirements-test.txt",
            REPO_ROOT / "requirements-dev.txt",
        ]
    else:
        req_paths = []
        for r in reqs_raw:
            p = Path(str(r))
            if not p.is_absolute():
                p = REPO_ROOT / p
            req_paths.append(p)

    direct_names = _parse_requirement_names(req_paths)
    dists = _collect_distributions(direct_names=direct_names)

    if bool(args.only_direct):
        dists = [d for d in dists if bool(d.direct)]

    copyleft = [d for d in dists if d.copyleft_suspect]
    unknown = [d for d in dists if d.license_unknown]
    direct_total = sum(1 for d in dists if d.direct)

    report = {
        "schema_version": 1,
        "timestamp": _now_utc(),
        "repo": {"root": str(REPO_ROOT), "git": _git_info()},
        "python": {"version": sys.version},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "requirements": {"paths": [str(p) for p in req_paths], "direct_names": sorted(direct_names)},
        "summary": {
            "packages_total": int(len(dists)),
            "packages_direct": int(direct_total),
            "copyleft_suspect": int(len(copyleft)),
            "license_unknown": int(len(unknown)),
        },
        "copyleft_suspects": [d.to_json() for d in copyleft],
        "packages": [d.to_json() for d in dists],
        "notes": [
            "This report is best-effort and not legal advice.",
            "It only inspects installed Python distributions; it does not audit CUDA/TensorRT/system libraries, datasets, or model weights.",
        ],
    }

    out_path = Path(str(args.output))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)

    if bool(args.fail_on_copyleft) and copyleft:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

