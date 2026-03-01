import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestPrepareTTTDomainShiftTargetScript(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        img_dir = dataset / "images" / "val"
        lbl_dir = dataset / "labels" / "val"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (32, 32), color=(120, 80, 40))
        img.save(img_dir / "000001.jpg")
        (lbl_dir / "000001.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (lbl_dir / "classes.json").write_text(json.dumps(["obj"]), encoding="utf-8")
        return dataset

    def test_recipe_is_deterministic_for_same_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "prepare_ttt_domain_shift_target.py"
        self.assertTrue(script.is_file(), f"missing script: {script}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out_a = root / "out_a"
            out_b = root / "out_b"

            cmd_base = [
                sys.executable,
                str(script),
                "--dataset-root",
                str(dataset),
                "--split",
                "val",
                "--corruption",
                "gaussian_noise",
                "--severity",
                "3",
                "--seed",
                "42",
                "--force",
            ]

            proc_a = subprocess.run(cmd_base + ["--out", str(out_a)], cwd=str(repo_root), capture_output=True, text=True, check=False)
            if proc_a.returncode != 0:
                self.fail(f"first run failed:\nstdout={proc_a.stdout}\nstderr={proc_a.stderr}")

            proc_b = subprocess.run(cmd_base + ["--out", str(out_b)], cwd=str(repo_root), capture_output=True, text=True, check=False)
            if proc_b.returncode != 0:
                self.fail(f"second run failed:\nstdout={proc_b.stdout}\nstderr={proc_b.stderr}")

            recipe_a = json.loads((out_a / "domain_shift_recipe.json").read_text(encoding="utf-8"))
            recipe_b = json.loads((out_b / "domain_shift_recipe.json").read_text(encoding="utf-8"))

            tgt_a = ((recipe_a.get("export_settings") or {}).get("domain_shift_target") or {})
            tgt_b = ((recipe_b.get("export_settings") or {}).get("domain_shift_target") or {})

            self.assertEqual(tgt_a.get("corruption"), "gaussian_noise")
            self.assertEqual(tgt_a.get("severity"), 3)
            self.assertEqual(tgt_a.get("seed"), 42)
            self.assertEqual(tgt_a.get("images_sha256"), tgt_b.get("images_sha256"))
            self.assertEqual(tgt_a.get("id"), tgt_b.get("id"))

    def test_script_supports_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "prepare_ttt_domain_shift_target.py"
        proc = subprocess.run([sys.executable, str(script), "--help"], cwd=str(repo_root), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.fail(f"--help failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("--corruption", combined)
        self.assertIn("--severity", combined)
        self.assertIn("--seed", combined)


if __name__ == "__main__":
    unittest.main()
