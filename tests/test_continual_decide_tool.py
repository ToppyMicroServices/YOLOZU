import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestContinualDecideTool(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "tools" / "continual_decide.py"
        self.wrapper = self.repo_root / "tools" / "yolozu.py"

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"continual_decide --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--eval-json", proc.stdout)
        self.assertIn("--max-forgetting", proc.stdout)
        self.assertIn("--ttt-active", proc.stdout)

    def test_wrapper_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.wrapper), "continual-decide", "--help"],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu continual-decide --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--eval-json", proc.stdout)

    def test_promote_when_hard_and_soft_gates_pass(self) -> None:
        eval_payload = {
            "schema_version": 2,
            "summary": {
                "avg_acc": 0.72,
                "forgetting": 0.03,
                "details": {"final": [0.68, 0.76]},
            },
            "tasks": [{"name": "task_a"}, {"name": "task_b"}],
            "matrix_values": [[0.70, 0.65], [0.50, 0.76]],
        }
        curation_payload = {
            "counts": {
                "samples_total": 100,
                "candidate_images": 10,
                "reviewed_labels": 12,
                "pseudo_labels_high_confidence": 24,
            }
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            td_path = Path(td)
            eval_json = self._write_json(td_path / "continual_eval.json", eval_payload)
            curation_json = self._write_json(td_path / "curation.json", curation_payload)
            out = td_path / "decision.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--eval-json",
                    str(eval_json),
                    "--curation-json",
                    str(curation_json),
                    "--min-new-task-score",
                    "0.70",
                    "--min-old-task-final",
                    "0.60",
                    "--min-reviewed-labels",
                    "10",
                    "--min-highconf-pseudo-labels",
                    "20",
                    "--min-total-curated-examples",
                    "30",
                    "--max-candidate-share",
                    "0.20",
                    "--output",
                    str(out),
                ],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"continual_decide promote case failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("decision"), "promote")
            self.assertEqual(payload.get("recommended_next"), "promote_candidate_checkpoint")

    def test_hold_when_forgetting_exceeds_threshold(self) -> None:
        eval_payload = {
            "schema_version": 2,
            "summary": {
                "avg_acc": 0.55,
                "forgetting": 0.21,
                "details": {"final": [0.40, 0.70]},
            },
            "tasks": [{"name": "task_a"}, {"name": "task_b"}],
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            td_path = Path(td)
            eval_json = self._write_json(td_path / "continual_eval.json", eval_payload)
            out = td_path / "decision.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--eval-json",
                    str(eval_json),
                    "--max-forgetting",
                    "0.05",
                    "--output",
                    str(out),
                ],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"continual_decide hold case failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("decision"), "hold")
            self.assertTrue(any(not gate.get("ok") for gate in payload.get("hard_gates") or []))

    def test_review_when_ttt_active_without_override(self) -> None:
        eval_payload = {
            "schema_version": 2,
            "summary": {
                "avg_acc": 0.75,
                "forgetting": 0.02,
                "details": {"final": [0.74, 0.78]},
            },
            "tasks": [{"name": "task_a"}, {"name": "task_b"}],
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            td_path = Path(td)
            eval_json = self._write_json(td_path / "continual_eval.json", eval_payload)
            out = td_path / "decision.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--eval-json",
                    str(eval_json),
                    "--ttt-active",
                    "--output",
                    str(out),
                ],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"continual_decide review case failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("decision"), "review")
            self.assertEqual(
                payload.get("recommended_next"),
                "review_ttt_scope_separately_before_checkpoint_promotion",
            )


if __name__ == "__main__":
    unittest.main()
