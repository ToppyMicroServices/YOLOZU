import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsDomainShiftRecipe(unittest.TestCase):
    def test_wrap_meta_includes_domain_shift_export_settings(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions.py"
        self.assertTrue(script.is_file(), f"missing script: {script}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            recipe = root / "domain_shift_recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "kind": "yolozu_domain_shift_recipe",
                        "version": 1,
                        "export_settings": {
                            "domain_shift_target": {
                                "id": "gaussian_blur_s2_seed0",
                                "corruption": "gaussian_blur",
                                "severity": 2,
                                "seed": 0,
                                "deterministic": True,
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            out = root / "pred.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--adapter",
                    "dummy",
                    "--dataset",
                    str(repo_root / "data" / "smoke"),
                    "--split",
                    "val",
                    "--max-images",
                    "1",
                    "--wrap",
                    "--domain-shift-recipe",
                    str(recipe),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            meta = payload.get("meta") or {}
            export_settings = meta.get("export_settings") or {}
            target = export_settings.get("domain_shift_target") or {}
            self.assertEqual(target.get("id"), "gaussian_blur_s2_seed0")
            self.assertEqual(target.get("corruption"), "gaussian_blur")
            self.assertEqual(target.get("severity"), 2)
            self.assertEqual(target.get("seed"), 0)

            recipe_meta = export_settings.get("domain_shift_recipe") or {}
            self.assertTrue(str(recipe_meta.get("path", "")).endswith("domain_shift_recipe.json"))
            self.assertEqual(len(str(recipe_meta.get("sha256", ""))), 64)


if __name__ == "__main__":
    unittest.main()
