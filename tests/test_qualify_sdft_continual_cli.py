import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestQualifySDFTContinualCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "tools" / "qualify_sdft_continual.py"
        spec = importlib.util.spec_from_file_location("qualify_sdft_continual", self.script)
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("three-seed", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)

    def test_refuses_existing_output_before_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.repo_root) as output_dir:
            proc = subprocess.run(
                [sys.executable, str(self.script), "--output-dir", output_dir],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing to replace existing output path", proc.stderr)

    def test_rejects_spec_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = Path(temp_dir) / "spec.json"
            spec.write_text(json.dumps({}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(self.script), "--spec", str(spec)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("spec must stay inside the repository", proc.stderr)

    def test_tracked_spec_has_three_seeds_and_real_coco(self) -> None:
        spec = json.loads(
            (self.repo_root / "configs/continual/sdft_coco128_blur_qualification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreaterEqual(len(spec["seeds"]), 3)
        self.assertEqual(spec["methods"], ["naive", "sdft"])
        self.assertEqual(spec["evaluation"]["backend"], "coco")
        self.assertTrue(spec["claim_boundary"]["independent_reproduction_required"])

    def test_nonzero_spec_preregisters_initial_training_and_efficacy_gates(self) -> None:
        spec = json.loads(
            (
                self.repo_root
                / "configs/continual/sdft_coco128_blur_nonzero_qualification.json"
            ).read_text(encoding="utf-8")
        )
        self.assertGreater(spec["initial_training"]["epochs"], 0)
        self.assertGreater(spec["initial_training"]["max_steps"], 0)
        self.assertGreater(spec["efficacy_gates"]["min_source_score"], 0.0)
        self.assertGreater(spec["efficacy_gates"]["min_target_score"], 0.0)
        self.assertTrue(spec["efficacy_gates"]["require_strict_old_task_improvement"])

    def test_confirmatory_spec_uses_unseen_seeds(self) -> None:
        spec = json.loads(
            (
                self.repo_root
                / "configs/continual/sdft_coco128_blur_confirmatory_qualification.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(spec["study_phase"], "confirmatory")
        self.assertFalse(set(spec["seeds"]) & set(spec["calibration_seeds_excluded"]))
        self.assertEqual(spec["seeds"], [44, 55, 66])

    def test_efficacy_assessment_requires_every_seed(self) -> None:
        gates = {
            "min_source_score": 0.01,
            "min_target_score": 0.01,
            "min_old_task_delta_sdft_minus_naive": 0.0,
            "min_new_task_delta_sdft_minus_naive": -0.001,
            "require_strict_old_task_improvement": True,
        }
        comparisons = [
            {
                "seed": seed,
                "final_old_task_delta_sdft_minus_naive": 0.001,
                "final_new_task_delta_sdft_minus_naive": 0.0,
            }
            for seed in (44, 55, 66)
        ]
        runs = [
            {
                "seed": seed,
                "method": "sdft",
                "matrix_values": [[0.02, 0.02], [0.02, 0.02]],
            }
            for seed in (44, 55, 66)
        ]
        passed = self.module._efficacy_assessment(
            comparisons=comparisons,
            runs=runs,
            gates=gates,
        )
        self.assertTrue(passed["passed"])
        comparisons[-1]["final_old_task_delta_sdft_minus_naive"] = 0.0
        failed = self.module._efficacy_assessment(
            comparisons=comparisons,
            runs=runs,
            gates=gates,
        )
        self.assertFalse(failed["passed"])

    def test_independent_reproduction_is_separate_from_efficacy(self) -> None:
        protocol = {
            "seeds": [44, 55, 66],
            "methods": ["naive", "sdft"],
            "train": {"max_steps": 20},
            "initial_training": {"epochs": 10},
            "evaluation": {"backend": "coco"},
            "promotion_gates": {"max_forgetting": 0.05},
        }
        assessment = {
            "passed": False,
            "gates": {"min_new_task_delta_sdft_minus_naive": -0.000001},
            "seed_results": [
                {
                    "seed": seed,
                    "old_task_delta_sdft_minus_naive": -0.1 if seed == 66 else 0.1,
                    "new_task_delta_sdft_minus_naive": 0.0,
                    "passed": seed != 66,
                }
                for seed in (44, 55, 66)
            ],
        }
        source = {"protocol": protocol, "efficacy_assessment": assessment}
        current = {"protocol": protocol, "efficacy_assessment": assessment}
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            result = self.module._independent_reproduction(
                source_summary=source,
                source_summary_path=source_path,
                current_summary=current,
            )
        self.assertTrue(result["reproduced"])
        self.assertFalse(result["efficacy_supported"])


if __name__ == "__main__":
    unittest.main()
