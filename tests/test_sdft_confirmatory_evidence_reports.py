import hashlib
import json
import unittest
from pathlib import Path


class TestSDFTConfirmatoryEvidenceReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.primary_path = (
            cls.repo_root / "reports" / "sdft_confirmatory_primary_2026-07-30.json"
        )
        cls.independent_path = (
            cls.repo_root
            / "reports"
            / "sdft_confirmatory_independent_2026-07-30.json"
        )
        cls.primary = json.loads(cls.primary_path.read_text(encoding="utf-8"))
        cls.independent = json.loads(cls.independent_path.read_text(encoding="utf-8"))

    def test_preregistered_three_seed_protocol_is_preserved(self) -> None:
        expected_runs = [
            ("naive", 44),
            ("sdft", 44),
            ("naive", 55),
            ("sdft", 55),
            ("naive", 66),
            ("sdft", 66),
        ]
        for role, payload in (
            ("primary", self.primary),
            ("independent", self.independent),
        ):
            with self.subTest(role=role):
                self.assertEqual(payload["role"], role)
                self.assertEqual(
                    [(row["method"], row["seed"]) for row in payload["runs"]],
                    expected_runs,
                )
                self.assertEqual(payload["decision"]["status"], "hold")
                self.assertEqual(payload["decision"]["efficacy"], "not_established")

    def test_nonzero_scores_do_not_override_failed_gate(self) -> None:
        assessment = self.primary["efficacy_assessment"]
        self.assertTrue(assessment["configured"])
        self.assertFalse(assessment["passed"])
        seed_results = assessment["seed_results"]
        self.assertEqual([row["seed"] for row in seed_results], [44, 55, 66])
        self.assertEqual([row["passed"] for row in seed_results], [True, True, False])
        for row in seed_results:
            with self.subTest(seed=row["seed"]):
                self.assertGreater(row["source_score"], 0.0)
                self.assertGreater(row["target_score"], 0.0)
        self.assertFalse(seed_results[-1]["checks"]["old_task_delta"])

    def test_independent_run_reproduces_direction_and_gate_outcome(self) -> None:
        primary_sha = hashlib.sha256(self.primary_path.read_bytes()).hexdigest()
        reproduction = self.independent["reproduction"]
        self.assertEqual(reproduction["source_summary_sha256"], primary_sha)
        self.assertTrue(reproduction["reproduced"])
        self.assertTrue(reproduction["direction_and_gate_outcome_match"])
        self.assertFalse(reproduction["efficacy_supported"])
        self.assertEqual(
            self.primary["efficacy_assessment"],
            self.independent["efficacy_assessment"],
        )
        self.assertEqual(self.primary["comparisons"], self.independent["comparisons"])


if __name__ == "__main__":
    unittest.main()
