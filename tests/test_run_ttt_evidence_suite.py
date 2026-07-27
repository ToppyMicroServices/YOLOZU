import json
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

import tools.run_ttt_evidence_suite as suite


class TestRunTTTEvidenceSuite(TestCase):
    def test_requires_three_unique_seeds(self):
        args = suite._parse_args(
            [
                "--dataset",
                "data/a",
                "--shifted-dataset",
                "data/b",
                "--checkpoint",
                "base.pt",
                "--mim-checkpoint",
                "mim.pt",
                "--out",
                "reports/out",
                "--seeds",
                "1,2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least three"):
            suite._validate(args)

    def test_dry_run_builds_full_protocol_separated_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean = root / "clean"
            shifted = root / "shifted"
            clean.mkdir()
            shifted.mkdir()
            base = root / "base.pt"
            mim = root / "mim.pt"
            base.write_bytes(b"base")
            mim.write_bytes(b"mim")
            out = root / "out"
            completed = {
                "returncode": 0,
                "stdout": "plan.json\n",
                "stderr": "",
                "wall_seconds": 0.01,
            }
            with mock.patch.object(suite, "_run", return_value=completed) as runner:
                result = suite.main(
                    [
                        "--dataset",
                        str(clean),
                        "--shifted-dataset",
                        str(shifted),
                        "--checkpoint",
                        str(base),
                        "--mim-checkpoint",
                        str(mim),
                        "--out",
                        str(out),
                        "--seeds",
                        "11,22,33",
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(runner.call_count, 30)
            plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["state"], "not_executed")
            self.assertEqual(len(plan["matrix"]), 30)
            protocols = {
                item["method"]: item["protocol"] for item in plan["matrix"]
            }
            self.assertEqual(protocols["cotta"], "continual_stream")
            self.assertEqual(protocols["tent"], "sample_reset")
            for item in plan["matrix"]:
                command = item["command"]
                self.assertIn("--seed", command)
                self.assertIn("--dry-run", command)

    def test_help_lists_concise_inputs(self):
        with self.assertRaises(SystemExit) as caught:
            suite._parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    main()
