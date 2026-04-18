import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCheckSegmentationParityTool(unittest.TestCase):
    def test_check_segmentation_parity_ok(self) -> None:
        np = None
        Image = None
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            self.skipTest("segmentation parity tool requires numpy and Pillow")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_segmentation_parity.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            ref_mask = root / "ref.png"
            cand_mask = root / "cand.png"
            Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(ref_mask)
            Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(cand_mask)
            ref_json = root / "ref.json"
            cand_json = root / "cand.json"
            out = root / "parity.json"
            ref_json.write_text(json.dumps([{"id": "sample1", "mask": str(ref_mask)}]), encoding="utf-8")
            cand_json.write_text(json.dumps([{"id": "sample1", "mask": str(cand_mask)}]), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, str(script), "--reference", str(ref_json), "--candidate", str(cand_json), "--output", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"check_segmentation_parity.py failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["images"], 1)


if __name__ == "__main__":
    unittest.main()
