import hashlib
import json
import tarfile
import unittest
from pathlib import Path, PurePosixPath


class TestArtifactResearchEvidence(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.summary_path = self.repo_root / "reports/artifact_research_evidence_2026-07-28.json"
        self.archive_path = self.repo_root / "reports/artifact_research_evidence_2026-07-28.tgz"
        self.report_path = self.repo_root / "reports/artifact_research_evidence_2026-07-28.md"

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_summary_records_three_deterministic_hold_repetitions(self):
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["kind"], "artifact_research_qualification")
        self.assertEqual(summary["qualification"]["repeats_completed"], 3)
        self.assertFalse(summary["qualification"]["production_promotion"])

        for lane in ("distillation", "hessian"):
            evidence = summary[lane]
            self.assertEqual(len(evidence["repetitions"]), 3)
            self.assertTrue(evidence["deterministic_prediction_output"])
            self.assertEqual(evidence["promotion_gate"]["decision"], "hold")
            hashes = {run["prediction_canonical_sha256"] for run in evidence["repetitions"]}
            self.assertEqual(len(hashes), 1)
            for run in evidence["repetitions"]:
                self.assertGreaterEqual(run["latency_overhead"]["total"], 0.0)

        for run in summary["hessian"]["repetitions"]:
            self.assertEqual(run["stop_reasons"], {"no_signal": 1280})
        self.assertEqual(summary["hessian"]["metric_delta"]["map50_95"], 0.0)

    def test_archive_is_safe_complete_and_hash_bound(self):
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn(self._sha256(self.summary_path), report)
        self.assertIn(self._sha256(self.archive_path), report)

        required = {
            "baseline_predictions.json",
            "baseline_eval.json",
            "qualification_summary.json",
        }
        for lane in ("distill", "hessian"):
            for repeat in range(1, 4):
                required.update(
                    {
                        f"{lane}_repeat_{repeat}.json",
                        f"{lane}_repeat_{repeat}_report.json",
                        f"{lane}_repeat_{repeat}_eval.json",
                    }
                )

        with tarfile.open(self.archive_path, "r:gz") as archive:
            members = archive.getmembers()
            relative_names = set()
            for member in members:
                path = PurePosixPath(member.name)
                self.assertFalse(path.is_absolute(), member.name)
                self.assertNotIn("..", path.parts, member.name)
                self.assertFalse(member.issym() or member.islnk(), member.name)
                if len(path.parts) > 1:
                    relative_names.add(PurePosixPath(*path.parts[1:]).as_posix())

        self.assertTrue(required.issubset(relative_names), sorted(required - relative_names))


if __name__ == "__main__":
    unittest.main()
