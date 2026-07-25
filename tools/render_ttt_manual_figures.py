#!/usr/bin/env python3
"""Render TTT evidence-boundary figures from a validated source artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"
MANUAL_FIGURES = REPO_ROOT / "manual" / "figures"
FIGURE_NAMES = (
    "ttt_method_results_summary.png",
    "ttt_compare_pipeline.png",
    "ttt_probe_example_panel.png",
)
MEASURED_RESOURCE_NAMES = (
    "checkpoint",
    "config",
    "dataset_manifest",
    "image_order",
    "baseline_predictions",
    "adapted_predictions",
)
SYNTHETIC_FORBIDDEN_KEY_TOKENS = {
    "accuracy",
    "changed",
    "delta",
    "improved",
    "improvement",
    "latency",
    "loss",
    "metric",
    "metrics",
    "quality",
    "runtime",
    "score",
    "throughput",
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_18 = _load_font(18)
FONT_22 = _load_font(22)
FONT_28 = _load_font(28)
FONT_40 = _load_font(40)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_items(value: Any, *, path: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from _walk_items(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, path=f"{path}[{index}]")


def _synthetic_key_is_measured(key: str) -> bool:
    normalized = key.strip().lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if any(token in SYNTHETIC_FORBIDDEN_KEY_TOKENS for token in tokens):
        return True
    return any(
        re.fullmatch(r"(?:ap|map|cocoap|cocomap)(?:\d+(?:_\d+)?)?", token)
        for token in tokens
    )


def _is_git_tracked(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _git_commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _validate_resource_binding(name: str, binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"measured provenance.{name} must be an object")
    declared_hash = str(binding.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
        raise ValueError(f"measured provenance.{name}.sha256 must be 64 lowercase hex")

    path_text = binding.get("path")
    url_text = binding.get("url")
    if bool(path_text) == bool(url_text):
        raise ValueError(
            f"measured provenance.{name} must set exactly one of path or url"
        )
    if path_text:
        path = Path(str(path_text))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"measured provenance.{name} path missing: {path}")
        if not _is_git_tracked(path):
            raise ValueError(
                f"measured provenance.{name} must reference a git-tracked file: {path}"
            )
        actual_hash = _sha256(path)
        if actual_hash != declared_hash:
            raise ValueError(
                f"measured provenance.{name} sha256 mismatch: "
                f"declared={declared_hash} actual={actual_hash}"
            )
        return {
            "path": path,
            "sha256": actual_hash,
            "verification": "local_git_tracked_hash_verified",
        }

    url = str(url_text)
    if not re.fullmatch(r"https://[^\s]+/releases/download/[^\s]+", url):
        raise ValueError(
            f"measured provenance.{name}.url must be an HTTPS release download URL"
        )
    return {
        "url": url,
        "sha256": declared_hash,
        "verification": "declared_not_fetched",
    }


def _validate_proxy_metric_names(value: Any, *, path: str = "$") -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_proxy_metric_names(child, path=f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if value.get("metric_backend") == "simple_map_proxy":
        for key, candidate in value.items():
            normalized = str(key).lower()
            if (
                re.fullmatch(
                    r"(?:map|coco_map|map50|map50_95|coco_map50|coco_map50_95)",
                    normalized,
                )
                and candidate is not None
            ):
                raise ValueError(
                    "simple_map_proxy values must use proxy_ap* names, never COCO mAP: "
                    f"{path}.{key}"
                )
    for key, child in value.items():
        _validate_proxy_metric_names(child, path=f"{path}.{key}")


def _validate_evidence_source(source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("TTT evidence source must be a JSON object")
    evidence_kind = str(source.get("evidence_kind") or "")
    if evidence_kind not in {"synthetic_fixture", "measured"}:
        raise ValueError("evidence_kind must be synthetic_fixture or measured")

    if evidence_kind == "synthetic_fixture":
        if source.get("promotion_eligible") is not False:
            raise ValueError("synthetic_fixture requires promotion_eligible=false")
        if source.get("efficacy") != "unavailable":
            raise ValueError("synthetic_fixture requires efficacy=unavailable")
        for item_path, key, value in _walk_items(source):
            if _synthetic_key_is_measured(key):
                raise ValueError(
                    f"synthetic_fixture forbids measured field {item_path}"
                )
            normalized = key.strip().lower()
            if (
                normalized in {"promotion_eligible", "promote", "promotion"}
                and value is not False
            ):
                raise ValueError(
                    f"synthetic_fixture forbids promotion claim at {item_path}"
                )
        return {"evidence_kind": evidence_kind, "resources": {}}

    if source.get("promotion_eligible") is not False:
        raise ValueError("measured evidence requires promotion_eligible=false")
    provenance = source.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("measured evidence requires provenance object")
    commit = str(provenance.get("commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("measured provenance.commit must be a 40-character git SHA")
    if not _git_commit_exists(commit):
        raise ValueError("measured provenance.commit does not resolve to a git commit")
    if not isinstance(provenance.get("seed"), int):
        raise ValueError("measured provenance.seed must be an integer")
    tool_versions = provenance.get("tool_versions")
    if not isinstance(tool_versions, dict) or not tool_versions:
        raise ValueError("measured provenance.tool_versions must be a non-empty object")
    if any(
        not str(key).strip() or not str(value).strip()
        for key, value in tool_versions.items()
    ):
        raise ValueError("measured provenance.tool_versions entries must be non-empty")

    resources = {
        name: _validate_resource_binding(name, provenance.get(name))
        for name in MEASURED_RESOURCE_NAMES
    }
    _validate_proxy_metric_names(source)
    return {"evidence_kind": evidence_kind, "resources": resources}


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word]).strip()
        if len(trial) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    width: int,
    line_gap: int = 8,
    fill=(20, 20, 20),
) -> int:
    x, y = xy
    for line in _wrap_text(text, width):
        draw.text((x, y), line, font=font, fill=fill)
        y = draw.textbbox((x, y), line, font=font)[3] + line_gap
    return y


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    body: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        box,
        radius=18,
        outline=(150, 160, 175),
        width=3,
        fill=(249, 250, 252),
    )
    draw.rounded_rectangle(
        (x0 + 12, y0 + 12, x1 - 12, y0 + 58),
        radius=12,
        fill=(232, 238, 246),
    )
    draw.text((x0 + 24, y0 + 20), title, font=FONT_28, fill=(20, 20, 20))
    _draw_wrapped(
        draw,
        (x0 + 36, y0 + 82),
        body,
        font=FONT_22,
        width=112,
    )


def _draw_summary_figure(source: dict[str, Any], out_path: Path) -> None:
    img = Image.new("RGB", (1800, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if source["evidence_kind"] == "synthetic_fixture":
        title = "TTT evidence status"
        intro = (
            "This checked-in source is a synthetic documentation fixture. It "
            "contains no measured quality, score, loss, latency, or improvement values."
        )
        panels = (
            (
                "Confirmed",
                "Rendering and workflow layout are reproducible from tracked source files.",
            ),
            (
                "Unknown / risk",
                "TTT efficacy is unavailable until compatible, hash-bound measured artifacts are attached.",
            ),
            (
                "Promotion",
                "Not eligible. Synthetic fixtures cannot promote a method, checkpoint, or default.",
            ),
        )
    else:
        provenance = source["provenance"]
        title = "TTT measured evidence binding"
        intro = (
            "Artifact identities and provenance were validated. Attachment alone "
            "does not establish efficacy or checkpoint promotion."
        )
        panels = (
            ("Commit", str(provenance["commit"])),
            ("Seed", str(provenance["seed"])),
            (
                "Conclusion",
                "Efficacy not established; promotion remains ineligible.",
            ),
        )
    draw.text((70, 45), title, font=FONT_40, fill=(20, 20, 20))
    _draw_wrapped(draw, (70, 105), intro, font=FONT_28, width=105, line_gap=10)
    y = 250
    for panel_title, body in panels:
        _panel(draw, (70, y, 1730, y + 160), title=panel_title, body=body)
        y += 190
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _draw_pipeline_figure(out_path: Path) -> None:
    img = Image.new("RGB", (1800, 700), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text(
        (70, 40),
        "TTT evidence workflow",
        font=FONT_40,
        fill=(20, 20, 20),
    )
    _draw_wrapped(
        draw,
        (70, 100),
        (
            "The shortest safe path validates prerequisites before execution and "
            "keeps diagnostic output separate from promotion evidence."
        ),
        font=FONT_28,
        width=108,
        line_gap=10,
    )
    boxes = (
        (
            "1. Bind",
            "Checkpoint, config, dataset manifest, ordered images, seed, commit, and tool versions.",
        ),
        (
            "2. Preflight",
            "Require full checkpoint compatibility and the selected method's runnable model path.",
        ),
        (
            "3. Compare",
            "Export baseline and adapted predictions; record every failed execution stage atomically.",
        ),
        (
            "4. Interpret",
            "Name proxy metrics as proxy AP, keep efficacy not established, and never auto-promote.",
        ),
    )
    x = 70
    for title, body in boxes:
        _panel(draw, (x, 245, x + 390, 565), title=title, body=body)
        x += 430
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _draw_probe_grid(
    source: dict[str, Any],
    out_path: Path,
    *,
    validation: dict[str, Any],
) -> None:
    img = Image.new("RGB", (1800, 700), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if source["evidence_kind"] == "synthetic_fixture":
        title = "TTT qualitative evidence prerequisite"
        intro = (
            "No prediction boxes are rendered from labels or ground truth. A "
            "measured panel requires hash-bound baseline and adapted predictions."
        )
        panels = (
            (
                "Current state",
                "Synthetic fixture only — efficacy unavailable, promotion ineligible, and no prediction fallback.",
            ),
            (
                "Change trigger",
                "Attach complete measured provenance and both prediction artifacts.",
            ),
        )
    else:
        resources = validation["resources"]
        title = "TTT prediction artifact binding"
        intro = (
            "Baseline and adapted predictions are mandatory. Release URL hashes "
            "remain declarations because this renderer does not fetch them."
        )
        panels = tuple(
            (
                name,
                (
                    f"{binding.get('path') or binding.get('url')} | "
                    f"{binding['verification']} | sha256={binding['sha256']}"
                ),
            )
            for name, binding in (
                ("baseline_predictions", resources["baseline_predictions"]),
                ("adapted_predictions", resources["adapted_predictions"]),
            )
        )
    draw.text((70, 45), title, font=FONT_40, fill=(20, 20, 20))
    _draw_wrapped(draw, (70, 110), intro, font=FONT_28, width=100, line_gap=10)
    y = 250
    for panel_title, body in panels:
        _panel(draw, (70, y, 1730, y + 175), title=panel_title, body=body)
        y += 205
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _restore_bundle(
    snapshots: dict[Path, bytes | None],
    pending: list[Path],
) -> None:
    for path in pending:
        path.unlink(missing_ok=True)
    for target, previous in snapshots.items():
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(previous)


def _render_and_publish(
    source: dict[str, Any],
    *,
    validation: dict[str, Any],
    docs_dir: Path,
    manual_dir: Path,
) -> None:
    targets = [
        *(docs_dir / name for name in FIGURE_NAMES),
        *(manual_dir / name for name in FIGURE_NAMES),
    ]
    snapshots = {
        target: target.read_bytes() if target.is_file() else None for target in targets
    }
    pending: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="ttt-figure-stage-", dir=REPO_ROOT) as td:
        stage = Path(td)
        rendered = [stage / name for name in FIGURE_NAMES]
        _draw_summary_figure(source, rendered[0])
        _draw_pipeline_figure(rendered[1])
        _draw_probe_grid(source, rendered[2], validation=validation)

        try:
            for target, rendered_path in zip(targets, rendered + rendered):
                target.parent.mkdir(parents=True, exist_ok=True)
                pending_path = target.with_name(f".{target.name}.pending")
                pending_path.unlink(missing_ok=True)
                shutil.copy2(rendered_path, pending_path)
                pending.append(pending_path)
            for pending_path, target in zip(pending, targets):
                os.replace(pending_path, target)
        except BaseException:
            _restore_bundle(snapshots, pending)
            raise


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render beginner-friendly TTT manual figures from validated evidence."
    )
    parser.add_argument(
        "--source-json",
        default="docs/assets/ttt_method_results_source.json",
        help="Validated TTT evidence source JSON.",
    )
    parser.add_argument(
        "--docs-assets-dir",
        default="docs/assets",
        help="Directory to receive generated docs asset PNGs.",
    )
    parser.add_argument(
        "--manual-figures-dir",
        default="manual/figures",
        help="Directory to receive synchronized manual PNGs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_path = Path(args.source_json)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validation = _validate_evidence_source(source)

    docs_dir = Path(args.docs_assets_dir)
    if not docs_dir.is_absolute():
        docs_dir = REPO_ROOT / docs_dir
    manual_dir = Path(args.manual_figures_dir)
    if not manual_dir.is_absolute():
        manual_dir = REPO_ROOT / manual_dir

    _render_and_publish(
        source,
        validation=validation,
        docs_dir=docs_dir,
        manual_dir=manual_dir,
    )
    for name in FIGURE_NAMES:
        print(docs_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
