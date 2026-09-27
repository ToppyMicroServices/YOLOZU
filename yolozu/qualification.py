"""Portable, deterministic release qualification packs.

The pack records normalized inputs, exact content fingerprints, evaluation
evidence, an explicit decision, and checksums.  It does not copy datasets,
predictions, or model weights into the output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from yolozu import __version__

BBoxFormat = Literal["cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"]

__all__ = [
    "QualificationError",
    "QualificationSpec",
    "QualificationResult",
    "PackVerificationResult",
    "load_qualification_spec",
    "qualify_release",
    "qualify_release_from_spec",
    "verify_qualification_pack",
    "diff_qualification_packs",
]

PACK_SCHEMA_VERSION = 1
PACK_FILES = (
    "request.json",
    "candidate_evaluation.json",
    "qualification.json",
)
Decision = Literal["pass", "hold", "fail"]
_SPEC_MAX_BYTES = 256 * 1024
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "predictions",
        "output_dir",
        "baseline_predictions",
        "evaluation",
        "thresholds",
        "force",
    }
)
_EVALUATION_KEYS = frozenset({"split", "bbox_format", "max_images", "dry_run"})
_THRESHOLD_KEYS = frozenset({"min_map50_95", "max_map50_95_drop"})


class QualificationError(ValueError):
    """The qualification request or pack is invalid."""


@dataclass(frozen=True)
class QualificationSpec:
    """Validated one-file request for the shared qualification engine."""

    path: Path
    dataset: Path
    predictions: Path
    output_dir: Path
    baseline_predictions: Path | None
    split: str | None
    bbox_format: BBoxFormat
    max_images: int | None
    dry_run: bool
    min_map50_95: float | None
    max_map50_95_drop: float | None
    force: bool
    semantic_digest: str

    def qualification_kwargs(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "predictions": self.predictions,
            "output_dir": self.output_dir,
            "baseline_predictions": self.baseline_predictions,
            "split": self.split,
            "bbox_format": self.bbox_format,
            "max_images": self.max_images,
            "dry_run": self.dry_run,
            "min_map50_95": self.min_map50_95,
            "max_map50_95_drop": self.max_map50_95_drop,
            "force": self.force,
        }


@dataclass(frozen=True)
class QualificationResult:
    pack_dir: Path
    decision: Decision
    qualification: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PACK_SCHEMA_VERSION,
            "ok": True,
            "pack_dir": str(self.pack_dir),
            "decision": self.decision,
            "passed": self.passed,
            "qualification": self.qualification,
        }


@dataclass(frozen=True)
class PackVerificationResult:
    pack_dir: Path
    ok: bool
    pack_digest: str | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PACK_SCHEMA_VERSION,
            "ok": self.ok,
            "pack_dir": str(self.pack_dir),
            "pack_digest": self.pack_digest,
            "errors": list(self.errors),
        }


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualificationError(f"expected a JSON object: {path}")
    return payload


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise QualificationError(f"{label} contains unknown keys: {', '.join(unknown)}")


def _spec_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise QualificationError(f"{label} must be an object")
    return dict(value)


def _spec_string(
    value: Any,
    *,
    label: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"{label} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise QualificationError(f"{label} must not contain control characters")
    return value


def _spec_path(base: Path, value: Any, *, label: str) -> Path:
    text = _spec_string(value, label=label)
    assert text is not None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_qualification_spec(path: str | Path) -> QualificationSpec:
    """Load a bounded, strict YAML qualification spec.

    Relative paths are resolved from the spec directory. Unknown fields and
    YAML values that cannot be represented as JSON are rejected so the
    semantic digest remains portable.
    """

    requested_path = Path(path).expanduser()
    if requested_path.is_symlink():
        raise QualificationError(f"qualification spec is not a regular file: {path}")
    spec_path = requested_path.resolve()
    if not spec_path.is_file():
        raise QualificationError(f"qualification spec is not a regular file: {path}")
    try:
        content = spec_path.read_bytes()
    except OSError as exc:
        raise QualificationError(f"could not read qualification spec: {exc}") from exc
    if len(content) > _SPEC_MAX_BYTES:
        raise QualificationError("qualification spec exceeds 256 KiB")
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QualificationError(f"could not parse qualification spec: {exc}") from exc
    document = _spec_mapping(payload, label="qualification spec")
    _strict_keys(document, allowed=_SPEC_KEYS, label="qualification spec")
    if document.get("schema_version") != 1:
        raise QualificationError("qualification spec schema_version must be 1")

    evaluation = _spec_mapping(document.get("evaluation", {}), label="evaluation")
    thresholds = _spec_mapping(document.get("thresholds", {}), label="thresholds")
    _strict_keys(evaluation, allowed=_EVALUATION_KEYS, label="evaluation")
    _strict_keys(thresholds, allowed=_THRESHOLD_KEYS, label="thresholds")

    split = _spec_string(
        evaluation.get("split"), label="evaluation.split", optional=True
    )
    bbox_format = evaluation.get("bbox_format", "cxcywh_norm")
    if bbox_format not in {"cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"}:
        raise QualificationError("evaluation.bbox_format is unsupported")
    max_images = evaluation.get("max_images")
    if max_images is not None and (
        isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or max_images <= 0
    ):
        raise QualificationError("evaluation.max_images must be a positive integer")
    dry_run = evaluation.get("dry_run", False)
    force = document.get("force", False)
    if not isinstance(dry_run, bool):
        raise QualificationError("evaluation.dry_run must be boolean")
    if not isinstance(force, bool):
        raise QualificationError("force must be boolean")

    baseline_value = document.get("baseline_predictions")
    baseline = (
        None
        if baseline_value is None
        else _spec_path(spec_path.parent, baseline_value, label="baseline_predictions")
    )
    minimum = _validate_threshold(
        "thresholds.min_map50_95", thresholds.get("min_map50_95")
    )
    maximum_drop = _validate_threshold(
        "thresholds.max_map50_95_drop", thresholds.get("max_map50_95_drop")
    )
    if maximum_drop is not None and baseline is None:
        raise QualificationError(
            "baseline_predictions is required with thresholds.max_map50_95_drop"
        )

    try:
        semantic_digest = _sha256_bytes(_canonical_bytes(document))
    except (TypeError, ValueError) as exc:
        raise QualificationError(
            "qualification spec must contain only JSON values"
        ) from exc
    return QualificationSpec(
        path=spec_path,
        dataset=_spec_path(spec_path.parent, document.get("dataset"), label="dataset"),
        predictions=_spec_path(
            spec_path.parent, document.get("predictions"), label="predictions"
        ),
        output_dir=_spec_path(
            spec_path.parent, document.get("output_dir"), label="output_dir"
        ),
        baseline_predictions=baseline,
        split=split,
        bbox_format=bbox_format,
        max_images=max_images,
        dry_run=dry_run,
        min_map50_95=minimum,
        max_map50_95_drop=maximum_drop,
        force=force,
        semantic_digest=semantic_digest,
    )


def _fingerprint_path(path: Path) -> dict[str, Any]:
    """Hash file content or a deterministic directory tree."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise QualificationError(f"input not found: {path}")
    if resolved.is_file():
        return {
            "kind": "file",
            "label": resolved.name,
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if not resolved.is_dir():
        raise QualificationError(f"input must be a file or directory: {path}")

    entries: list[dict[str, Any]] = []
    total_size = 0
    for candidate in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix()
        size = candidate.stat().st_size
        total_size += size
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": _sha256_file(candidate),
            }
        )
    return {
        "kind": "directory",
        "label": resolved.name,
        "sha256": _sha256_bytes(_canonical_bytes(entries)),
        "file_count": len(entries),
        "size_bytes": total_size,
    }


def _validate_threshold(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise QualificationError(f"{name} must be between 0 and 1")
    return normalized


def _portable_evaluation(
    result: Any, *, dataset_label: str, predictions_label: str
) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("timestamp", None)
    payload["dataset"] = dataset_label
    payload["predictions"] = predictions_label
    normalization = payload.get("normalization")
    if isinstance(normalization, dict) and normalization.get("classes") is not None:
        normalization["classes"] = Path(str(normalization["classes"])).name
    return payload


def _metric(payload: Mapping[str, Any], name: str) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _decision_payload(
    *,
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> tuple[Decision, list[dict[str, Any]]]:
    thresholds = request.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise QualificationError("request.thresholds must be an object")
    evaluation = request.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise QualificationError("request.evaluation must be an object")

    candidate_map = _metric(candidate, "map50_95")
    baseline_map = _metric(baseline or {}, "map50_95")
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "id": "candidate_metrics_available",
            "status": "pass" if candidate_map is not None else "unknown",
            "observed": candidate_map,
        }
    )

    minimum = thresholds.get("min_map50_95")
    if minimum is not None:
        checks.append(
            {
                "id": "min_map50_95",
                "status": (
                    "unknown"
                    if candidate_map is None
                    else ("pass" if candidate_map >= float(minimum) else "fail")
                ),
                "observed": candidate_map,
                "threshold": float(minimum),
            }
        )

    maximum_drop = thresholds.get("max_map50_95_drop")
    if maximum_drop is not None:
        drop = (
            None
            if candidate_map is None or baseline_map is None
            else baseline_map - candidate_map
        )
        checks.append(
            {
                "id": "max_map50_95_drop",
                "status": (
                    "unknown"
                    if drop is None
                    else ("pass" if drop <= float(maximum_drop) else "fail")
                ),
                "observed": drop,
                "threshold": float(maximum_drop),
                "baseline": baseline_map,
                "candidate": candidate_map,
            }
        )

    requested = [
        check for check in checks if check["id"] != "candidate_metrics_available"
    ]
    if bool(evaluation.get("dry_run")):
        return "hold", checks
    if not requested:
        return "hold", checks
    if any(check["status"] == "fail" for check in requested):
        return "fail", checks
    if candidate_map is None or any(
        check["status"] == "unknown" for check in requested
    ):
        return "hold", checks
    return "pass", checks


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    normalized = dict(request)
    normalized.pop("request_fingerprint", None)
    return _sha256_bytes(_canonical_bytes(normalized))


def _protocol_fingerprint(request: Mapping[str, Any]) -> str:
    protocol = request.get("protocol")
    if not isinstance(protocol, Mapping):
        raise QualificationError("request.protocol must be an object")
    return _sha256_bytes(_canonical_bytes(protocol))


def qualify_release(
    dataset: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    *,
    baseline_predictions: str | Path | None = None,
    split: str | None = None,
    bbox_format: BBoxFormat = "cxcywh_norm",
    max_images: int | None = None,
    dry_run: bool = False,
    min_map50_95: float | None = None,
    max_map50_95_drop: float | None = None,
    force: bool = False,
    _source_spec: Mapping[str, Any] | None = None,
) -> QualificationResult:
    """Create one self-checking release qualification pack atomically.

    At least one quality threshold and real metrics are required for ``pass``.
    Dry runs and unavailable metrics produce ``hold`` rather than a false pass.
    """

    from yolozu.api import evaluate_coco

    minimum = _validate_threshold("min_map50_95", min_map50_95)
    maximum_drop = _validate_threshold("max_map50_95_drop", max_map50_95_drop)
    if maximum_drop is not None and baseline_predictions is None:
        raise QualificationError(
            "baseline_predictions is required with max_map50_95_drop"
        )

    dataset_path = Path(dataset).expanduser().resolve()
    predictions_path = Path(predictions).expanduser().resolve()
    baseline_path = (
        Path(baseline_predictions).expanduser().resolve()
        if baseline_predictions is not None
        else None
    )
    target = Path(output_dir).expanduser().resolve()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    dataset_input = _fingerprint_path(dataset_path)
    candidate_input = _fingerprint_path(predictions_path)
    baseline_input = (
        _fingerprint_path(baseline_path) if baseline_path is not None else None
    )

    protocol = {
        "evaluator": "coco_detection",
        "evaluator_version": __version__,
        "dataset_sha256": dataset_input["sha256"],
        "split": split,
        "bbox_format": bbox_format,
        "max_images": max_images,
        "dry_run": bool(dry_run),
        "repair": False,
    }
    request: dict[str, Any] = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pack_format": "yolozu.release_qualification",
        "inputs": {
            "dataset": dataset_input,
            "candidate_predictions": candidate_input,
            "baseline_predictions": baseline_input,
        },
        "evaluation": {
            "split": split,
            "bbox_format": bbox_format,
            "max_images": max_images,
            "dry_run": bool(dry_run),
            "repair": False,
        },
        "thresholds": {
            "min_map50_95": minimum,
            "max_map50_95_drop": maximum_drop,
        },
        "protocol": protocol,
    }
    if _source_spec is not None:
        request["source_spec"] = dict(_source_spec)
    request["protocol_fingerprint"] = _protocol_fingerprint(request)
    request["request_fingerprint"] = _request_fingerprint(request)

    candidate_result = evaluate_coco(
        dataset_path,
        predictions_path,
        split=split,
        bbox_format=bbox_format,
        max_images=max_images,
        dry_run=dry_run,
    )
    candidate_payload = _portable_evaluation(
        candidate_result,
        dataset_label=str(dataset_input["label"]),
        predictions_label=str(candidate_input["label"]),
    )
    baseline_payload: dict[str, Any] | None = None
    if baseline_path is not None:
        baseline_result = evaluate_coco(
            dataset_path,
            baseline_path,
            split=split,
            bbox_format=bbox_format,
            max_images=max_images,
            dry_run=dry_run,
        )
        assert baseline_input is not None
        baseline_payload = _portable_evaluation(
            baseline_result,
            dataset_label=str(dataset_input["label"]),
            predictions_label=str(baseline_input["label"]),
        )

    decision, checks = _decision_payload(
        request=request,
        candidate=candidate_payload,
        baseline=baseline_payload,
    )
    qualification = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pack_format": "yolozu.release_qualification",
        "decision": decision,
        "passed": decision == "pass",
        "checks": checks,
        "request_fingerprint": request["request_fingerprint"],
        "protocol_fingerprint": request["protocol_fingerprint"],
        "evaluator_version": __version__,
        "limitations": [
            "Pack checksums detect accidental changes but do not provide publisher authenticity.",
            "The pack records content digests and labels; it does not embed source data or model weights.",
        ],
    }

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        _write_json(temp_dir / "request.json", request)
        _write_json(temp_dir / "candidate_evaluation.json", candidate_payload)
        if baseline_payload is not None:
            _write_json(temp_dir / "baseline_evaluation.json", baseline_payload)
        _write_json(temp_dir / "qualification.json", qualification)
        artifact_names = list(PACK_FILES)
        if baseline_payload is not None:
            artifact_names.append("baseline_evaluation.json")
        checksums = {
            "schema_version": PACK_SCHEMA_VERSION,
            "algorithm": "sha256",
            "files": {
                name: _sha256_file(temp_dir / name) for name in sorted(artifact_names)
            },
        }
        checksums["pack_digest"] = _sha256_bytes(_canonical_bytes(checksums["files"]))
        _write_json(temp_dir / "checksums.json", checksums)

        if target.exists():
            if not force:
                raise QualificationError(f"output directory already exists: {target}")
            existing = verify_qualification_pack(target)
            if not existing.ok:
                raise QualificationError(
                    "refusing to replace an existing directory that is not a valid qualification pack"
                )
            shutil.rmtree(target)
        os.replace(temp_dir, target)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return QualificationResult(
        pack_dir=target,
        decision=decision,
        qualification=qualification,
    )


def qualify_release_from_spec(path: str | Path) -> QualificationResult:
    """Create a release qualification pack from one strict YAML spec."""

    spec = load_qualification_spec(path)
    return qualify_release(
        **spec.qualification_kwargs(),
        _source_spec={
            "schema_version": 1,
            "label": spec.path.name,
            "semantic_sha256": spec.semantic_digest,
        },
    )


def verify_qualification_pack(pack_dir: str | Path) -> PackVerificationResult:
    """Verify checksums and semantic consistency for one pack."""

    pack = Path(pack_dir).expanduser().resolve()
    errors: list[str] = []
    pack_digest: str | None = None
    if not pack.is_dir():
        return PackVerificationResult(pack, False, None, ("pack directory not found",))

    try:
        checksums = _read_json(pack / "checksums.json")
        files = checksums.get("files")
        if checksums.get("algorithm") != "sha256" or not isinstance(files, dict):
            raise QualificationError("checksums.json has an unsupported structure")
        pack_digest_value = checksums.get("pack_digest")
        expected_pack_digest = _sha256_bytes(_canonical_bytes(files))
        if pack_digest_value != expected_pack_digest:
            errors.append("pack_digest mismatch")
        else:
            pack_digest = expected_pack_digest
        for name, expected in sorted(files.items()):
            if not isinstance(name, str) or Path(name).name != name:
                errors.append(f"unsafe checksum path: {name!r}")
                continue
            path = pack / name
            if path.is_symlink():
                errors.append(f"pack file must not be a symlink: {name}")
                continue
            if not path.is_file():
                errors.append(f"missing pack file: {name}")
                continue
            actual = _sha256_file(path)
            if expected != actual:
                errors.append(f"checksum mismatch: {name}")

        request = _read_json(pack / "request.json")
        candidate = _read_json(pack / "candidate_evaluation.json")
        qualification = _read_json(pack / "qualification.json")
        baseline_file = pack / "baseline_evaluation.json"
        baseline = _read_json(baseline_file) if baseline_file.is_file() else None
        inputs = request.get("inputs")
        if not isinstance(inputs, Mapping):
            errors.append("request.inputs must be an object")
            baseline_expected = False
        else:
            baseline_expected = inputs.get("baseline_predictions") is not None
        if baseline_expected != baseline_file.is_file():
            errors.append("baseline evaluation presence does not match request inputs")
        expected_files = set(PACK_FILES)
        if baseline_expected:
            expected_files.add("baseline_evaluation.json")
        if set(files) != expected_files:
            errors.append(
                "checksums file set does not match the pack interface contract"
            )
        actual_json_files = {
            path.name
            for path in pack.iterdir()
            if path.is_file()
            and path.suffix == ".json"
            and path.name != "checksums.json"
        }
        if actual_json_files != expected_files:
            errors.append(
                "pack JSON file set does not match the pack interface contract"
            )
        if request.get("request_fingerprint") != _request_fingerprint(request):
            errors.append("request_fingerprint mismatch")
        if request.get("protocol_fingerprint") != _protocol_fingerprint(request):
            errors.append("protocol_fingerprint mismatch")
        source_spec = request.get("source_spec")
        if source_spec is not None:
            if not isinstance(source_spec, Mapping):
                errors.append("request.source_spec must be an object")
            else:
                label = source_spec.get("label")
                digest = source_spec.get("semantic_sha256")
                if (
                    source_spec.get("schema_version") != 1
                    or not isinstance(label, str)
                    or Path(label).name != label
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    errors.append(
                        "request.source_spec has an invalid interface contract"
                    )
        expected_decision, expected_checks = _decision_payload(
            request=request,
            candidate=candidate,
            baseline=baseline,
        )
        if qualification.get("schema_version") != PACK_SCHEMA_VERSION:
            errors.append("qualification schema_version mismatch")
        if qualification.get("pack_format") != "yolozu.release_qualification":
            errors.append("qualification pack_format mismatch")
        if qualification.get("decision") != expected_decision:
            errors.append("qualification decision does not match embedded evidence")
        if qualification.get("passed") is not (expected_decision == "pass"):
            errors.append("qualification passed flag does not match its decision")
        if qualification.get("checks") != expected_checks:
            errors.append("qualification checks do not match embedded evidence")
        for key in ("request_fingerprint", "protocol_fingerprint"):
            if qualification.get(key) != request.get(key):
                errors.append(f"qualification {key} mismatch")
        evaluation_options = request.get("evaluation")
        if isinstance(evaluation_options, Mapping):
            for evidence_name, evidence in (
                ("candidate", candidate),
                ("baseline", baseline),
            ):
                if evidence is None:
                    continue
                if evidence.get("ok") is not True or evidence.get("status") != "ok":
                    errors.append(
                        f"{evidence_name} evaluation is not successful evidence"
                    )
                for key in ("dry_run", "bbox_format", "max_images"):
                    if evidence.get(key) != evaluation_options.get(key):
                        errors.append(f"{evidence_name} evaluation {key} mismatch")
    except QualificationError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"pack verification failed: {exc}")

    return PackVerificationResult(pack, not errors, pack_digest, tuple(errors))


def diff_qualification_packs(
    baseline_pack: str | Path,
    candidate_pack: str | Path,
) -> dict[str, Any]:
    """Compare verified packs only when their evaluation protocol matches."""

    baseline_verification = verify_qualification_pack(baseline_pack)
    candidate_verification = verify_qualification_pack(candidate_pack)
    errors = [
        *[f"baseline: {error}" for error in baseline_verification.errors],
        *[f"candidate: {error}" for error in candidate_verification.errors],
    ]
    if errors:
        return {
            "schema_version": PACK_SCHEMA_VERSION,
            "ok": False,
            "compatible": False,
            "errors": errors,
        }

    baseline_root = baseline_verification.pack_dir
    candidate_root = candidate_verification.pack_dir
    baseline_request = _read_json(baseline_root / "request.json")
    candidate_request = _read_json(candidate_root / "request.json")
    compatible = baseline_request.get("protocol_fingerprint") == candidate_request.get(
        "protocol_fingerprint"
    )
    if not compatible:
        return {
            "schema_version": PACK_SCHEMA_VERSION,
            "ok": True,
            "compatible": False,
            "reason": "protocol_fingerprint_mismatch",
            "baseline_protocol_fingerprint": baseline_request.get(
                "protocol_fingerprint"
            ),
            "candidate_protocol_fingerprint": candidate_request.get(
                "protocol_fingerprint"
            ),
        }

    baseline_evaluation = _read_json(baseline_root / "candidate_evaluation.json")
    candidate_evaluation = _read_json(candidate_root / "candidate_evaluation.json")
    metrics: dict[str, dict[str, float | None]] = {}
    for name in ("map50_95", "map50", "map75", "ar100"):
        baseline_value = _metric(baseline_evaluation, name)
        candidate_value = _metric(candidate_evaluation, name)
        delta = (
            None
            if baseline_value is None or candidate_value is None
            else candidate_value - baseline_value
        )
        metrics[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "ok": True,
        "compatible": True,
        "protocol_fingerprint": baseline_request.get("protocol_fingerprint"),
        "metrics": metrics,
        "baseline_pack_digest": baseline_verification.pack_digest,
        "candidate_pack_digest": candidate_verification.pack_digest,
    }
