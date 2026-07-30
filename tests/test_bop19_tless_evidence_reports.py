import hashlib
import json
import math
import unittest
from pathlib import Path


class TestBOP19TLESSEvidenceReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.primary_path = (
            cls.repo_root / "reports" / "bop19_tless_pose_primary_2026-07-30.json"
        )
        cls.independent_path = (
            cls.repo_root
            / "reports"
            / "bop19_tless_pose_independent_2026-07-30.json"
        )
        cls.primary = json.loads(cls.primary_path.read_text(encoding="utf-8"))
        cls.independent = json.loads(cls.independent_path.read_text(encoding="utf-8"))

    def test_protocol_and_decision_boundaries_are_explicit(self) -> None:
        for role, payload in (
            ("primary", self.primary),
            ("independent", self.independent),
        ):
            with self.subTest(role=role):
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(payload["kind"], "bop19_tless_pose_qualification")
                self.assertEqual(payload["role"], role)
                self.assertFalse(payload["protocol"]["test_ground_truth_used_for_inference"])
                self.assertEqual(payload["protocol"]["translation_unit"], "millimetre")
                self.assertEqual(payload["decision"]["status"], "hold")
                self.assertEqual(payload["decision"]["efficacy"], "not_established")
                self.assertEqual(payload["dataset"]["target_entries"], 4904)
                self.assertEqual(payload["dataset"]["target_instances"], 6423)

    def test_three_seed_measurements_are_finite_and_not_overstated(self) -> None:
        self.assertEqual([row["seed"] for row in self.primary["runs"]], [11, 22, 33])
        for row in self.primary["runs"]:
            with self.subTest(seed=row["seed"]):
                official = row["official_bop19"]
                native = row["task_native"]
                self.assertTrue(math.isfinite(official["bop19_average_recall"]))
                self.assertGreaterEqual(official["bop19_average_recall"], 0.0)
                self.assertLessEqual(official["bop19_average_recall"], 1.0)
                self.assertTrue(math.isfinite(native["rotation_error_deg_mean"]))
                self.assertTrue(math.isfinite(native["translation_error_mm_mean"]))
                self.assertEqual(
                    native["matched_instances"] + native["unmatched_instances"],
                    native["target_instances"],
                )
                self.assertEqual(native["target_instances"], 6423)

    def test_independent_summary_reproduces_primary_metrics(self) -> None:
        primary_sha = hashlib.sha256(self.primary_path.read_bytes()).hexdigest()
        reproduction = self.independent["reproduction"]
        self.assertEqual(reproduction["source_summary_sha256"], primary_sha)
        self.assertTrue(reproduction["same_seed_metrics_within_1e-9"])
        for primary_run, independent_run in zip(
            self.primary["runs"], self.independent["runs"], strict=True
        ):
            self.assertEqual(primary_run["seed"], independent_run["seed"])
            self.assertEqual(
                primary_run["official_bop19"], independent_run["official_bop19"]
            )
            self.assertEqual(primary_run["task_native"], independent_run["task_native"])


if __name__ == "__main__":
    unittest.main()
