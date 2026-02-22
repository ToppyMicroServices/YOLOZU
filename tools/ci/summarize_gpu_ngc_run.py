#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize gpu-ngc workflow run into deterministic DoD artifacts (JSON + Markdown)."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-attempt", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--workflow", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--check-runner-result", required=True)
    p.add_argument("--has-runner", required=True)
    p.add_argument("--probe-status", required=True)
    p.add_argument("--no-runner-result", required=True)
    p.add_argument("--gpu-job-result", required=True)
    p.add_argument("--artifact-dir", default=None, help="Directory containing downloaded gpu-ngc smoke artifacts.")
    p.add_argument("--out-run-info", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--out-summary-md", required=True)
    return p.parse_args(argv)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_summary(*, run_info: dict[str, Any], artifact_dir: Path | None) -> tuple[dict[str, Any], int]:
    has_runner = _boolish(str(run_info.get("has_runner", "")))
    probe_status = str(run_info.get("probe_status", ""))
    gpu_job_result = str(run_info.get("gpu_job_result", ""))
    findings: list[str] = []
    guidance: list[str] = []
    checks: dict[str, Any] = {}

    if not has_runner:
        status = "skip"
        findings.append("No idle self-hosted GPU runner was available.")
        if probe_status.startswith("http_403"):
            findings.append("Runner discovery API returned 403.")
            guidance.append("Set repository secret RUNNER_DISCOVERY_TOKEN (PAT with self-hosted runner visibility).")
        elif probe_status and probe_status != "ok":
            findings.append(f"Runner discovery probe status: {probe_status}")
    elif gpu_job_result != "success":
        if gpu_job_result == "skipped":
            status = "skip"
            findings.append("GPU smoke job was skipped after runner detection.")
            guidance.append("Verify NGC_API_KEY and runner labels if this should run.")
        else:
            status = "fail"
            findings.append(f"GPU smoke job result is {gpu_job_result}.")
    else:
        status = "pass"
        required_files = (
            "latency_trt.json",
            "pred_trt.json",
            "pred_onnxrt.json",
            "parity_trt_vs_onnxrt.log",
            "opencv_cuda_smoke.json",
        )
        if artifact_dir is None or not artifact_dir.exists():
            status = "fail"
            findings.append("GPU job succeeded but artifact directory is missing.")
        else:
            for name in required_files:
                checks[name] = bool((artifact_dir / name).is_file())
            missing = [name for name, ok in checks.items() if not ok]
            if missing:
                status = "fail"
                findings.append(f"Missing expected artifacts: {', '.join(sorted(missing))}")
            opencv_payload = _read_json(artifact_dir / "opencv_cuda_smoke.json")
            if isinstance(opencv_payload, dict):
                opencv_status = str(opencv_payload.get("status", "unknown"))
                checks["opencv_cuda_status"] = opencv_status
                if opencv_status == "error":
                    status = "fail"
                    findings.append("OpenCV CUDA smoke reported status=error.")
                elif opencv_status == "skipped":
                    findings.append("OpenCV CUDA smoke was skipped (CUDA backend unavailable in current OpenCV build).")
            else:
                findings.append("Could not parse opencv_cuda_smoke.json.")
                status = "fail"

    summary = {
        "schema_version": 1,
        "dod_status": status,
        "run_info": run_info,
        "checks": checks,
        "findings": findings,
        "guidance": guidance,
    }
    exit_code = 1 if status == "fail" else 0
    return summary, exit_code


def _to_markdown(summary: dict[str, Any]) -> str:
    run_info = summary.get("run_info", {})
    lines = [
        "# gpu-ngc DoD summary",
        "",
        f"- status: **{summary.get('dod_status', 'unknown')}**",
        f"- run_id: `{run_info.get('run_id', '')}`",
        f"- sha: `{run_info.get('sha', '')}`",
        f"- has_runner: `{run_info.get('has_runner', '')}`",
        f"- probe_status: `{run_info.get('probe_status', '')}`",
        f"- gpu_job_result: `{run_info.get('gpu_job_result', '')}`",
    ]
    findings = summary.get("findings") or []
    if findings:
        lines.append("")
        lines.append("## Findings")
        for item in findings:
            lines.append(f"- {item}")
    guidance = summary.get("guidance") or []
    if guidance:
        lines.append("")
        lines.append("## Guidance")
        for item in guidance:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_info = {
        "run_id": str(args.run_id),
        "run_attempt": str(args.run_attempt),
        "ref": str(args.ref),
        "sha": str(args.sha),
        "workflow": str(args.workflow),
        "job": str(args.job),
        "actor": str(args.actor),
        "check_runner_result": str(args.check_runner_result),
        "has_runner": str(args.has_runner),
        "probe_status": str(args.probe_status),
        "no_runner_result": str(args.no_runner_result),
        "gpu_job_result": str(args.gpu_job_result),
    }

    artifact_dir = None if not args.artifact_dir else Path(args.artifact_dir)
    summary, exit_code = _build_summary(run_info=run_info, artifact_dir=artifact_dir)

    out_run_info = Path(args.out_run_info)
    out_summary_json = Path(args.out_summary_json)
    out_summary_md = Path(args.out_summary_md)
    out_run_info.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_md.parent.mkdir(parents=True, exist_ok=True)

    out_run_info.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_summary_md.write_text(_to_markdown(summary), encoding="utf-8")
    print(out_summary_json)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
