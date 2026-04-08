import unittest
import json
import tempfile
from pathlib import Path

from yolozu.eval.keypoints_parity import compare_keypoints_predictions


class TestKeypointsParity(unittest.TestCase):
    def test_compare_keypoints_predictions_reports_kp_delta(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            reference.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "image": "images/val/sample.jpg",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.9,
                                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                        "keypoints": [
                                            {"x": 0.25, "y": 0.25, "v": 2},
                                            {"x": 0.75, "y": 0.25, "v": 2},
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "image": "images/val/sample.jpg",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.90001,
                                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                        "keypoints": [
                                            {"x": 0.25002, "y": 0.25, "v": 2},
                                            {"x": 0.75, "y": 0.25003, "v": 2},
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = compare_keypoints_predictions(reference=reference, candidate=candidate, kp_atol=1e-3)
            self.assertTrue(bool(report.get("ok")))
            match = ((report.get("results") or [])[0].get("matches") or [])[0]
            self.assertTrue(bool(match.get("keypoints_ok")))
            self.assertGreater(float(match.get("keypoints_max_abs_diff") or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
