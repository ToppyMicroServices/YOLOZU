#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DENY_PATTERNS = [
    # Packages / repos commonly associated with GPL/AGPL baselines we don't want to vendor/require.
    r"\bultralytics\b",
    r"\byolov5\b",
    r"\byolov8\b",
    r"\byolov9\b",
    r"\byolov10\b",
    r"\byolo11\b",
    r"\byolo12\b",
    r"\byolo26\b",
]

# We still allow mentioning GPL/AGPL in docs to state constraints, but we forbid vendoring license texts.
DENY_FILE_TEXT_PATTERNS = [
    r"GNU GENERAL PUBLIC LICENSE",
    r"GNU AFFERO GENERAL PUBLIC LICENSE",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def check_license_file() -> None:
    lic = REPO_ROOT / "LICENSE"
    if not lic.exists():
        _fail("LICENSE missing at repo root")
    txt = _read_text(lic)
    if "Apache License" not in txt or "Version 2.0" not in txt:
        _fail("LICENSE does not look like Apache-2.0")


def check_no_git_submodules() -> None:
    gm = REPO_ROOT / ".gitmodules"
    if gm.exists():
        _fail(".gitmodules present (submodules can pull in incompatible licenses)")


def check_requirements_no_denylist() -> None:
    for name in ("requirements.txt", "requirements-test.txt", "requirements-dev.txt"):
        p = REPO_ROOT / name
        if not p.exists():
            continue
        txt = _read_text(p).lower()
        for pat in DENY_PATTERNS:
            if re.search(pat, txt, flags=re.IGNORECASE):
                _fail(f"denylisted token found in {name}: {pat}")


def check_fetch_script_is_official() -> None:
    p = REPO_ROOT / "tools" / "fetch_coco128.sh"
    txt = _read_text(p).lower()
    if "ultralytics" in txt or "github.com/ultralytics" in txt:
        _fail("tools/fetch_coco128.sh references ultralytics; expected official COCO hosting only")


def check_repo_text_no_license_texts() -> None:
    # Keep this cheap and targeted: scan *tracked* source/docs files for GPL/AGPL license blobs.
    # (We still allow 'GPL/AGPL' words in TODO/docs as policy statements.)
    allow_ext = {
        ".py",
        ".sh",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".ini",
        ".cfg",
        "",
    }

    tracked_paths: list[Path] = []
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT)
        for raw in out.decode("utf-8", errors="replace").splitlines():
            rel = raw.strip()
            if not rel:
                continue
            tracked_paths.append(REPO_ROOT / rel)
    except Exception:
        tracked_paths = []

    # Fallback for non-git environments (best-effort): scan the working tree.
    paths = tracked_paths if tracked_paths else [p for p in REPO_ROOT.rglob("*") if p.is_file()]

    for path in paths:
        if ".git" in path.parts or ".beads" in path.parts:
            continue
        if path.name == "LICENSE":
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in allow_ext:
            continue
        if not path.exists() or not path.is_file():
            continue

        txt = _read_text(path)
        for pat in DENY_FILE_TEXT_PATTERNS:
            if re.search(pat, txt, flags=re.IGNORECASE):
                _fail(f"found GPL/AGPL license text marker in {path.relative_to(REPO_ROOT)}: {pat}")


def check_model_zoo_registry_policy() -> None:
    zoo = REPO_ROOT / "yolozu" / "data" / "manifest" / "model_zoo.json"
    if not zoo.exists():
        return
    try:
        obj = json.loads(zoo.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"failed to parse model zoo registry JSON: {zoo}: {exc}")

    models = obj.get("models") or []
    if not isinstance(models, list):
        _fail(f"invalid model zoo registry: models must be a list: {zoo}")

    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip() or "<missing-id>"
        lic = str(item.get("license") or "").strip()
        sha = item.get("sha256")

        # Repository policy: shipped registry must be Apache-friendly only.
        if lic != "Apache-2.0":
            _fail(f"model zoo contains non-Apache license entry: id={model_id} license={lic!r} (expected Apache-2.0)")

        # Registry integrity: shipped entries must pin sha256 for reproducibility.
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha.strip() or ""):
            _fail(f"model zoo entry missing/invalid sha256: id={model_id} sha256={sha!r}")


def main() -> int:
    check_license_file()
    check_no_git_submodules()
    check_requirements_no_denylist()
    check_fetch_script_is_official()
    check_model_zoo_registry_policy()
    check_repo_text_no_license_texts()
    print("OK: license policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
