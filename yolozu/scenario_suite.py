"""Synthetic scenario-suite benchmark report generator.

Builds per-scenario metrics (fps, mAP, recall, depth/pose error,
rejection rate) from a predefined set of scenario configurations.
"""

import json
import time

__all__ = ["SCENARIOS", "build_report", "main"]


SCENARIOS = [
    {"name": "symmetric_on"},
    {"name": "symmetric_off"},
    {"name": "tabletop_on"},
    {"name": "tabletop_off"},
    {"name": "extreme_depth"},
    {"name": "handheld_jitter_on"},
    {"name": "handheld_jitter_off"},
    {"name": "template_gate_on"},
    {"name": "template_gate_off"},
]


def _scenario_metrics(index):
    fps = 33.0 - index * 0.5
    return {
        "fps": max(5.0, fps),
        "map": round(0.4 + index * 0.01, 4),
        "recall": round(0.5 + index * 0.01, 4),
        "depth_error": round(0.2 + index * 0.005, 4),
        "pose_error": round(5.0 + index * 0.1, 4),
        "rejection_rate": round(0.1 + index * 0.01, 4),
    }


def _now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_report():
    scenarios = []
    summary = {
        "fps": 0.0,
        "map": 0.0,
        "recall": 0.0,
        "depth_error": 0.0,
        "pose_error": 0.0,
        "rejection_rate": 0.0,
    }
    for index, scenario in enumerate(SCENARIOS):
        metrics = _scenario_metrics(index)
        scenarios.append({"name": scenario["name"], "metrics": metrics})
        for key in summary:
            summary[key] += float(metrics.get(key, 0.0))
    if scenarios:
        for key in summary:
            summary[key] = float(summary[key]) / float(len(scenarios))
    return {
        "schema_version": 1,
        "timestamp": _now_utc_iso(),
        "summary": summary,
        "scenarios": scenarios,
    }


def main():
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
