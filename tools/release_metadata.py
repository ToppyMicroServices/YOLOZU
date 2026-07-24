"""Shared release metadata planning and validation helpers.

This module is intentionally import-only. Public release commands live in
``tools/release.py`` and ``tools/release_tag.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


RELEASE_METADATA_PATHS = (
    "yolozu/__init__.py",
    "CHANGELOG.md",
    "CITATION.cff",
    "tools/manifest.json",
    "yolozu/data/manifest/tools_manifest.json",
)
SOURCE_MANIFEST_PATH = "tools/manifest.json"
PACKAGED_MANIFEST_PATH = "yolozu/data/manifest/tools_manifest.json"
PINNED_PACKAGE_RE = re.compile(r"\byolozu==([0-9]+(?:\.[0-9]+){2,3})\b")


@dataclass(frozen=True)
class ReleaseMetadataPlan:
    """Validated, all-or-nothing release metadata update plan."""

    files: dict[str, str]
    original_files: dict[str, str]
    changed_paths: tuple[str, ...]
    validation_before: dict[str, Any]
    validation_after: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "changed_paths": list(self.changed_paths),
            "validation_before": self.validation_before,
            "validation_after": self.validation_after,
        }


def _read_metadata_files(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    return {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in RELEASE_METADATA_PATHS
    }


def _parse_package_version(text: str) -> str:
    matches = re.findall(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']\s*$', text)
    if len(matches) != 1:
        raise ValueError(
            "yolozu/__init__.py must contain exactly one __version__ assignment"
        )
    return str(matches[0])


def _parse_citation_scalar(text: str, key: str) -> str:
    matches = re.findall(
        rf"(?m)^{re.escape(key)}:\s*[\"']?([^\"'\n]+?)[\"']?\s*$",
        text,
    )
    if len(matches) != 1:
        raise ValueError(f"CITATION.cff must contain exactly one {key} field")
    return str(matches[0]).strip()


def _changelog_release_dates(text: str, version: str) -> list[str]:
    return re.findall(
        rf"(?m)^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        text,
    )


def _validate_date(value: str, *, source: str, errors: list[str]) -> None:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        errors.append(f"{source} has invalid release date {value!r}; expected YYYY-MM-DD")
        return
    if parsed.strftime("%Y-%m-%d") != value:
        errors.append(f"{source} has non-canonical release date {value!r}")


def _manifest_example_report(
    *,
    repo_root: Path,
    manifest_text: str,
    expected_version: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        errors.append(f"{SOURCE_MANIFEST_PATH} is not valid JSON: {exc}")
        return []
    if not isinstance(payload, dict):
        errors.append(f"{SOURCE_MANIFEST_PATH} must contain a JSON object")
        return []

    report: list[dict[str, Any]] = []
    tools = payload.get("tools")
    if not isinstance(tools, list):
        errors.append(f"{SOURCE_MANIFEST_PATH} must contain a tools array")
        return report

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = str(tool.get("id") or "<unknown>")
        examples = tool.get("examples")
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                continue
            command = str(example.get("command") or "")
            pinned_versions = sorted(set(PINNED_PACKAGE_RE.findall(command)))
            if not pinned_versions:
                continue
            policy = str(example.get("release_version_policy") or "").strip()
            evidence = str(example.get("release_version_evidence") or "").strip()
            item = {
                "tool_id": tool_id,
                "example_index": index,
                "policy": policy or "unclassified",
                "pinned_versions": pinned_versions,
                "evidence": evidence,
            }
            report.append(item)
            label = f"manifest example {tool_id}[{index}]"

            if policy not in {"current", "historical"}:
                errors.append(
                    f"{label} pins {', '.join(pinned_versions)} but must declare "
                    "release_version_policy as current or historical"
                )
                continue
            if policy == "current" and pinned_versions != [expected_version]:
                errors.append(
                    f"{label} is current-release-coupled but pins "
                    f"{', '.join(pinned_versions)} instead of {expected_version}"
                )
            if policy == "historical":
                if not evidence:
                    errors.append(
                        f"{label} is historical but has no release_version_evidence"
                    )
                    continue
                evidence_path = evidence.split("#", 1)[0]
                evidence_relative = Path(evidence_path)
                evidence_file = (repo_root / evidence_relative).resolve()
                if (
                    not evidence_path
                    or evidence_relative.is_absolute()
                    or ".." in evidence_relative.parts
                    or not evidence_file.is_relative_to(repo_root.resolve())
                    or not evidence_file.is_file()
                ):
                    errors.append(
                        f"{label} historical evidence must be an existing "
                        f"repo-relative file: {evidence_path}"
                    )
    return report


def validate_release_metadata_texts(
    *,
    repo_root: Path,
    files: dict[str, str],
    expected_version: str | None = None,
    expected_tag: str | None = None,
    tag_prefix: str = "v",
) -> dict[str, Any]:
    """Validate one synchronized release metadata snapshot without writing."""

    errors: list[str] = []
    package_version = ""
    citation_version = ""
    citation_date = ""
    changelog_date = ""

    try:
        package_version = _parse_package_version(files["yolozu/__init__.py"])
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))

    wanted_version = str(expected_version or package_version).strip()
    if not wanted_version:
        errors.append("expected release version is empty")
    elif package_version and package_version != wanted_version:
        errors.append(
            f"package version mismatch: expected {wanted_version}, "
            f"yolozu/__init__.py has {package_version}"
        )

    if expected_tag:
        tag = str(expected_tag).strip()
        prefix = str(tag_prefix)
        if prefix and not tag.startswith(prefix):
            errors.append(
                f"release tag {tag!r} does not start with configured prefix {prefix!r}"
            )
            tag_version = ""
        else:
            tag_version = tag[len(prefix) :] if prefix else tag
        if wanted_version and tag_version and tag_version != wanted_version:
            errors.append(
                f"release tag/version mismatch: tag={tag} resolves to "
                f"{tag_version}, expected={wanted_version}"
            )

    changelog_text = files.get("CHANGELOG.md", "")
    if wanted_version:
        dates = _changelog_release_dates(changelog_text, wanted_version)
        if not dates:
            errors.append(
                f"CHANGELOG.md missing release heading: "
                f"## [{wanted_version}] - YYYY-MM-DD"
            )
        elif len(dates) > 1:
            errors.append(
                f"CHANGELOG.md has duplicate release headings for {wanted_version}"
            )
        else:
            changelog_date = dates[0]
            _validate_date(changelog_date, source="CHANGELOG.md", errors=errors)

    try:
        citation_version = _parse_citation_scalar(files["CITATION.cff"], "version")
        citation_date = _parse_citation_scalar(
            files["CITATION.cff"], "date-released"
        )
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))
    if wanted_version and citation_version and citation_version != wanted_version:
        errors.append(
            f"CITATION.cff version mismatch: expected {wanted_version}, "
            f"found {citation_version}"
        )
    if citation_date:
        _validate_date(citation_date, source="CITATION.cff", errors=errors)
    if changelog_date and citation_date and changelog_date != citation_date:
        errors.append(
            "release date mismatch: "
            f"CHANGELOG.md has {changelog_date}, CITATION.cff has {citation_date}"
        )

    source_manifest = files.get(SOURCE_MANIFEST_PATH, "")
    packaged_manifest = files.get(PACKAGED_MANIFEST_PATH, "")
    manifests_identical = source_manifest == packaged_manifest
    if not manifests_identical:
        errors.append(
            f"{SOURCE_MANIFEST_PATH} and {PACKAGED_MANIFEST_PATH} are not byte-identical"
        )
    manifest_examples = _manifest_example_report(
        repo_root=Path(repo_root).resolve(),
        manifest_text=source_manifest,
        expected_version=wanted_version,
        errors=errors,
    )

    return {
        "ok": not errors,
        "expected_version": wanted_version,
        "expected_tag": str(expected_tag or ""),
        "tag_prefix": str(tag_prefix),
        "package_version": package_version,
        "changelog_date": changelog_date,
        "citation_version": citation_version,
        "citation_date_released": citation_date,
        "source_packaged_manifests_identical": manifests_identical,
        "manifest_examples": manifest_examples,
        "errors": errors,
    }


def validate_release_metadata(
    repo_root: Path,
    *,
    expected_version: str | None = None,
    expected_tag: str | None = None,
    tag_prefix: str = "v",
) -> dict[str, Any]:
    """Read and validate the repository's current release metadata."""

    root = Path(repo_root).resolve()
    try:
        files = _read_metadata_files(root)
    except OSError as exc:
        return {
            "ok": False,
            "expected_version": str(expected_version or ""),
            "expected_tag": str(expected_tag or ""),
            "tag_prefix": str(tag_prefix),
            "errors": [f"could not read release metadata: {exc}"],
        }
    return validate_release_metadata_texts(
        repo_root=root,
        files=files,
        expected_version=expected_version,
        expected_tag=expected_tag,
        tag_prefix=tag_prefix,
    )


def _replace_package_version(text: str, next_version: str) -> str:
    updated, count = re.subn(
        r'(?m)^(__version__\s*=\s*["\'])([^"\']+)(["\']\s*)$',
        rf"\g<1>{next_version}\g<3>",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(
            "yolozu/__init__.py must contain exactly one replaceable __version__ assignment"
        )
    return updated


def _replace_citation_scalar(text: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"(?m)^{re.escape(key)}:\s*.*$",
        f'{key}: "{value}"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"CITATION.cff must contain exactly one replaceable {key} field")
    return updated


def _insert_changelog_section(
    text: str,
    *,
    next_version: str,
    changelog_section: str,
) -> str:
    if _changelog_release_dates(text, next_version):
        raise ValueError(f"CHANGELOG.md already contains a release heading for {next_version}")
    marker = "## [Unreleased]"
    if text.count(marker) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one ## [Unreleased] marker")
    expected_prefix = f"## [{next_version}] - "
    if not changelog_section.startswith(expected_prefix):
        raise ValueError(
            f"generated changelog section must start with {expected_prefix}YYYY-MM-DD"
        )
    return text.replace(marker, f"{marker}\n\n{changelog_section.rstrip()}", 1)


def _replace_current_manifest_examples(
    manifest_text: str,
    *,
    current_version: str,
    next_version: str,
) -> str:
    payload = json.loads(manifest_text)
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        for example in tool.get("examples") or []:
            if not isinstance(example, dict):
                continue
            if str(example.get("release_version_policy") or "") != "current":
                continue
            for field in ("command", "description"):
                value = example.get(field)
                if not isinstance(value, str):
                    continue
                value = value.replace(current_version, next_version)
                value = value.replace(
                    current_version.replace(".", "_"),
                    next_version.replace(".", "_"),
                )
                example[field] = value
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def prepare_release_metadata(
    repo_root: Path,
    *,
    current_version: str,
    next_version: str,
    release_date: str,
    changelog_section: str,
) -> ReleaseMetadataPlan:
    """Build and validate the complete release metadata update before writing."""

    root = Path(repo_root).resolve()
    files = _read_metadata_files(root)
    before = validate_release_metadata_texts(
        repo_root=root,
        files=files,
        expected_version=current_version,
        expected_tag=f"v{current_version}",
    )
    if not before["ok"]:
        raise ValueError(
            "current release metadata is inconsistent: " + "; ".join(before["errors"])
        )

    planned = dict(files)
    planned["yolozu/__init__.py"] = _replace_package_version(
        planned["yolozu/__init__.py"],
        next_version,
    )
    planned["CHANGELOG.md"] = _insert_changelog_section(
        planned["CHANGELOG.md"],
        next_version=next_version,
        changelog_section=changelog_section,
    )
    citation = _replace_citation_scalar(
        planned["CITATION.cff"],
        "version",
        next_version,
    )
    planned["CITATION.cff"] = _replace_citation_scalar(
        citation,
        "date-released",
        release_date,
    )
    source_manifest = _replace_current_manifest_examples(
        planned[SOURCE_MANIFEST_PATH],
        current_version=current_version,
        next_version=next_version,
    )
    planned[SOURCE_MANIFEST_PATH] = source_manifest
    planned[PACKAGED_MANIFEST_PATH] = source_manifest

    after = validate_release_metadata_texts(
        repo_root=root,
        files=planned,
        expected_version=next_version,
        expected_tag=f"v{next_version}",
    )
    if not after["ok"]:
        raise ValueError(
            "planned release metadata is inconsistent: " + "; ".join(after["errors"])
        )

    changed_paths = tuple(
        relative
        for relative in RELEASE_METADATA_PATHS
        if planned[relative] != files[relative]
    )
    return ReleaseMetadataPlan(
        files=planned,
        original_files=files,
        changed_paths=changed_paths,
        validation_before=before,
        validation_after=after,
    )


def write_release_metadata_atomic(repo_root: Path, plan: ReleaseMetadataPlan) -> None:
    """Apply a validated multi-file plan with rollback on replacement failure."""

    if not plan.validation_after.get("ok"):
        raise ValueError("refusing to write an unvalidated release metadata plan")
    root = Path(repo_root).resolve()
    pending: dict[str, Path] = {}
    originals: dict[str, str] = {}
    replaced: list[str] = []
    try:
        for relative in RELEASE_METADATA_PATHS:
            current = (root / relative).read_text(encoding="utf-8")
            if current != plan.original_files.get(relative):
                raise RuntimeError(
                    f"release metadata changed after planning: {relative}; "
                    "re-run the release command"
                )

        for relative in plan.changed_paths:
            destination = root / relative
            originals[relative] = plan.original_files[relative]
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".release-metadata",
                dir=str(destination.parent),
            )
            temp_path = Path(temp_name)
            pending[relative] = temp_path
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(plan.files[relative])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temp_path, destination.stat().st_mode)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

        for relative in plan.changed_paths:
            os.replace(pending[relative], root / relative)
            replaced.append(relative)
        validation = validate_release_metadata(
            root,
            expected_version=str(plan.validation_after["expected_version"]),
            expected_tag=str(plan.validation_after["expected_tag"]),
            tag_prefix=str(plan.validation_after["tag_prefix"]),
        )
        if not validation["ok"]:
            raise RuntimeError(
                "written release metadata failed validation: "
                + "; ".join(validation["errors"])
            )
    except Exception as write_error:
        rollback_errors: list[str] = []
        for relative in reversed(replaced):
            destination = root / relative
            try:
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".release-metadata-rollback",
                    dir=str(destination.parent),
                )
                rollback_path = Path(temp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(originals[relative])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(rollback_path, destination.stat().st_mode)
                os.replace(rollback_path, destination)
            except Exception as rollback_error:
                rollback_errors.append(f"{relative}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                f"release metadata write failed ({write_error}); rollback also failed: "
                + "; ".join(rollback_errors)
            ) from write_error
        raise
    finally:
        for temp_path in pending.values():
            temp_path.unlink(missing_ok=True)
