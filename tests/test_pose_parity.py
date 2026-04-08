import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

from yolozu.eval.pose_parity import compare_pose_predictions


class TestPoseParity(TestCase):
    def test_compare_pose_predictions_reports_translation_delta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = root / "ref.json"
            cand = root / "cand.json"
            image_path = root / "sample.png"
            image_path.write_bytes(
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d4948445200000004000000040802000000269309290000001449444154789c636460f8cf800d3061151db41200cb43010f56344b120000000049454e44ae426082"
                )
            )
            base = {
                "image": str(image_path),
                "image_size": [4, 4],
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5},
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t_xyz": [0.0, 0.0, 1.0],
                    }
                ],
            }
            ref.write_text(json.dumps({"predictions": [base]}, indent=2), encoding="utf-8")
            cand_entry = json.loads(json.dumps(base))
            cand_entry["detections"][0]["t_xyz"] = [0.0, 0.0, 1.00005]
            cand.write_text(json.dumps({"predictions": [cand_entry]}, indent=2), encoding="utf-8")

            report = compare_pose_predictions(reference=ref, candidate=cand, trans_atol=1e-4)
            self.assertTrue(bool(report["ok"]))
            item = report["results"][0]
            self.assertEqual(item["counts"]["matched"], 1)
            match = item["matches"][0]
            self.assertAlmostEqual(float(match["trans_l2_diff"]), 0.00005, places=8)
            self.assertTrue(bool(match["trans_ok"]))


if __name__ == "__main__":
    main()
