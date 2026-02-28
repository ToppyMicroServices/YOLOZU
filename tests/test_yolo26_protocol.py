import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.eval_protocol import load_eval_protocol
from yolozu.map_targets import load_map_targets_doc


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestYOLO26Protocol(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_protocol_file_valid(self):
        protocol = load_eval_protocol("yolo26")
        self.assertEqual(protocol["id"], "yolo26")
        self.assertEqual(protocol["split"], "val2017")
        self.assertEqual(protocol["bbox_format"], "cxcywh_norm")
        self.assertEqual(protocol["metric_key"], "map50_95")
        self.assertEqual(protocol["imgsz"], 640)

    def test_doc_matches_protocol(self):
        protocol = load_eval_protocol("yolo26")
        doc = (self.repo_root / "docs" / "yolo26_eval_protocol.md").read_text()
        self.assertIn("Protocol file: `protocols/yolo26_eval.json`", doc)
        self.assertIn("Split: `val2017`", doc)
        self.assertIn("BBox format: `cxcywh_norm`", doc)
        self.assertIn("Metric key: `map50_95`", doc)
        self.assertEqual(protocol["split"], "val2017")
        self.assertEqual(protocol["bbox_format"], "cxcywh_norm")
        self.assertEqual(protocol["metric_key"], "map50_95")

    def test_targets_schema_valid(self):
        doc = load_map_targets_doc(self.repo_root / "baselines" / "yolo26_targets.json")
        targets = doc.get("targets") or {}
        for bucket in ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"):
            self.assertIn(bucket, targets)
        self.assertEqual(doc.get("metric_key"), "map50_95")
        self.assertEqual(doc.get("protocol_id"), "yolo26")

    def test_eval_suite_applies_protocol(self):
        mod = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite")
        args, protocol = mod._resolve_args(
            [
                "--protocol",
                "yolo26",
                "--predictions-glob",
                "reports/pred_yolo26*.json",
            ]
        )
        self.assertIsNotNone(protocol)
        self.assertEqual(protocol["id"], "yolo26")
        self.assertEqual(args.split, "val2017")
        self.assertEqual(args.bbox_format, "cxcywh_norm")

    def test_eval_coco_applies_protocol(self):
        mod = _load_module(self.repo_root / "tools" / "eval_coco.py", "eval_coco")
        args, protocol = mod._resolve_args(
            [
                "--protocol",
                "yolo26",
                "--predictions",
                "reports/predictions.json",
            ]
        )
        self.assertIsNotNone(protocol)
        self.assertEqual(protocol["id"], "yolo26")
        self.assertEqual(args.split, "val2017")
        self.assertEqual(args.bbox_format, "cxcywh_norm")

    # --- T8: e2e (no NMS) semantics tests ---

    def test_protocol_e2e_nms_field_yolo26(self):
        """yolo26 protocol declares e2e.nms = 'none'."""
        protocol = load_eval_protocol("yolo26")
        self.assertIn("e2e", protocol)
        self.assertEqual(protocol["e2e"]["nms"], "none")
        self.assertIn("no NMS", protocol.get("metric_label", ""))

    def test_protocol_e2e_nms_field_e2e_nms_free(self):
        """e2e_nms_free protocol declares e2e.nms = 'none'."""
        protocol = load_eval_protocol("e2e_nms_free")
        self.assertIn("e2e", protocol)
        self.assertEqual(protocol["e2e"]["nms"], "none")

    def test_protocol_e2e_nms_field_nms_applied(self):
        """nms_applied protocol declares e2e.nms = 'applied'."""
        protocol = load_eval_protocol("nms_applied")
        self.assertIn("e2e", protocol)
        self.assertEqual(protocol["e2e"]["nms"], "applied")

    def test_all_protocols_have_e2e_block(self):
        """Every registered protocol must declare an e2e block with nms field."""
        for proto_id in ("yolo26", "e2e_nms_free", "nms_applied"):
            with self.subTest(proto_id=proto_id):
                protocol = load_eval_protocol(proto_id)
                self.assertIn("e2e", protocol, f"{proto_id} missing e2e block")
                self.assertIn("nms", protocol["e2e"], f"{proto_id} missing e2e.nms")
                self.assertIn(
                    protocol["e2e"]["nms"],
                    ("none", "applied"),
                    f"{proto_id} has invalid e2e.nms value",
                )

    def test_eval_coco_all_protocols_resolve(self):
        """All three --protocol choices resolve correctly via eval_coco."""
        mod = _load_module(self.repo_root / "tools" / "eval_coco.py", "eval_coco")
        for proto_id in ("yolo26", "e2e_nms_free", "nms_applied"):
            with self.subTest(proto_id=proto_id):
                args, protocol = mod._resolve_args(
                    ["--protocol", proto_id, "--predictions", "reports/predictions.json"]
                )
                self.assertIsNotNone(protocol)
                self.assertEqual(protocol["id"], proto_id)


if __name__ == "__main__":
    unittest.main()

