from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from yolozu.datasets.coco import COCO_80_CLASSES
from yolozu.predictions import validate_predictions_payload


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _load_tool(repo_root: Path):
    path = repo_root / "tools" / "generate_runtime_parity_case_study.py"
    spec = importlib.util.spec_from_file_location("generate_runtime_parity_case_study", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRuntimeParityCaseStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.artifact_root = (
            cls.repo_root / "docs" / "assets" / "case_studies" / "maskrcnn_eager_torchscript"
        )
        cls.tool = _load_tool(cls.repo_root)

    def test_help_does_not_require_optional_torch_runtime(self):
        script = self.repo_root / "tools" / "generate_runtime_parity_case_study.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allow-download", result.stdout)
        self.assertIn("--baseline-dir", result.stdout)
        self.assertIn("implicit baseline", result.stdout)
        self.assertIn("stable COCO evaluation lane", result.stdout)

    def test_category_mapping_removes_coco_metadata_gaps(self):
        valid_ids = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        )
        categories = ["N/A"] * 91
        categories[0] = "__background__"
        for class_id, category_id in enumerate(valid_ids):
            categories[category_id] = COCO_80_CLASSES[class_id]

        mapping = self.tool._category_id_map(categories)

        self.assertEqual(len(mapping), 80)
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[13], 11)
        self.assertEqual(mapping[90], 79)
        self.assertNotIn(12, mapping)

    def test_smoke_dataset_declares_standard_coco80_mapping(self):
        classes_path = self.repo_root / "data" / "smoke" / "labels" / "val" / "classes.json"
        payload = json.loads(classes_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["names"], list(COCO_80_CLASSES))
        self.assertEqual(payload["category_id_to_class_id"]["13"], 11)
        self.assertEqual(payload["class_to_category_id"]["11"], 13)
        self.assertEqual(payload["category_id_to_class_id"]["90"], 79)
        mapping = self.tool._dataset_class_mapping(
            dataset_root=self.repo_root / "data" / "smoke",
            split="val",
        )
        self.assertEqual(mapping["class_count"], 80)
        self.assertEqual(len(mapping["semantic_sha256"]), 64)

    def test_dataset_class_mapping_rejects_wrong_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            labels = dataset_root / "labels" / "val"
            labels.mkdir(parents=True)
            source = self.repo_root / "data" / "smoke" / "labels" / "val" / "classes.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["names"][0], payload["names"][1] = payload["names"][1], payload["names"][0]
            (labels / "classes.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "class order"):
                self.tool._dataset_class_mapping(dataset_root=dataset_root, split="val")

    def test_official_checkpoint_full_sha_is_always_enforced(self):
        fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: {})
        fake_weights = types.SimpleNamespace(
            COCO_V1=types.SimpleNamespace(
                url="https://example.invalid/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
                meta={"categories": []},
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "weights.pth"
            weights.write_bytes(b"not the official checkpoint")
            with self.assertRaisesRegex(SystemExit, "not the pinned official COCO_V1"):
                self.tool._official_weights(
                    torch=fake_torch,
                    weights_enum=fake_weights,
                    local_weights=str(weights),
                    allow_download=False,
                    expected_sha256=None,
                )

    def test_metric_tolerance_must_be_finite(self):
        args = self.tool._parse_args(["--metric-atol", "inf"])
        with self.assertRaisesRegex(SystemExit, "finite"):
            self.tool._validate_args(args)

    def test_reproduction_environment_ignores_platform_patch_and_local_suffix(self):
        baseline = {
            "python": "3.12.13",
            "platform": {
                "system": "Darwin",
                "release": "25.5.0",
                "machine": "arm64",
                "processor": "arm",
            },
            "packages": {
                "yolozu": "0.1.dev1",
                "torch": "2.10.0+cpu",
                "torchvision": "0.25.0+cpu",
                "Pillow": "12.2.0",
                "numpy": "2.4.4",
                "pycocotools": "2.0.11",
            },
            "torch": {
                "device": "cpu",
                "threads": 1,
                "deterministic_algorithms": True,
            },
        }
        candidate = json.loads(json.dumps(baseline))
        candidate["python"] = "3.12.7"
        candidate["platform"] = {
            "system": "Linux",
            "release": "6.8.0",
            "machine": "x86_64",
            "processor": "",
        }
        candidate["packages"]["yolozu"] = None
        candidate["packages"]["torch"] = "2.10.0+cu128"

        self.assertEqual(
            self.tool._environment_identity(candidate),
            self.tool._environment_identity(baseline),
        )
        candidate["packages"]["numpy"] = "2.4.5"
        self.assertNotEqual(
            self.tool._environment_identity(candidate),
            self.tool._environment_identity(baseline),
        )

    def test_portable_commands_quote_paths_as_single_shell_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset with spaces;literal"
            output = root / "output with spaces;literal"
            weights = root / "weights with spaces;literal.pth"
            args = self.tool._parse_args(
                [
                    "--dataset",
                    str(dataset),
                    "--weights",
                    str(weights),
                    "--output-dir",
                    str(output),
                ]
            )
            commands = self.tool._portable_commands(
                args=args,
                split="val split",
                output_dir=output,
                predictions_eager=output / "eager predictions.json",
                predictions_scripted=output / "scripted predictions.json",
            )

            generate = shlex.split(commands["generate"])
            self.assertEqual(generate[generate.index("--dataset") + 1], str(dataset.resolve()))
            self.assertEqual(generate[generate.index("--weights") + 1], str(weights.resolve()))
            self.assertEqual(generate[generate.index("--output-dir") + 1], str(output.resolve()))
            evaluate = shlex.split(commands["evaluate"][0])
            self.assertEqual(evaluate[evaluate.index("--split") + 1], "val split")
            self.assertEqual(
                evaluate[evaluate.index("--predictions") + 1],
                str((output / "eager predictions.json").resolve()),
            )

            logical = self.tool._logical_command(
                [
                    "/tmp/python with spaces",
                    str(self.repo_root / "tools" / "eval_coco.py"),
                    "--output",
                    str(output / "staging result.json"),
                ],
                staging_dir=output,
                published_output_dir=root / "published with spaces;literal",
            )
            logical_args = shlex.split(logical)
            self.assertEqual(logical_args[0], "tools/eval_coco.py")
            self.assertEqual(
                logical_args[-1],
                str((root / "published with spaces;literal" / "staging result.json").resolve()),
            )

    def test_artifact_index_includes_reproduction_only_for_compared_run(self):
        self.assertNotIn(
            "reproduction_check",
            self.tool._artifact_index(include_reproduction=False),
        )
        self.assertEqual(
            self.tool._artifact_index(include_reproduction=True)["reproduction_check"],
            "reproduction_check.json",
        )

    def test_prediction_conversion_uses_contiguous_class_ids_and_normalized_boxes(self):
        output = {
            "boxes": _FakeTensor([[10.0, 20.0, 50.0, 60.0]]),
            "labels": _FakeTensor([13]),
            "scores": _FakeTensor([0.75]),
        }
        entries = self.tool._to_predictions(
            outputs=[output],
            selected=[{"image": "data/smoke/images/val/sample.jpg"}],
            sizes=[{"width": 100, "height": 80}],
            category_map={13: 11},
            score_threshold=0.5,
            max_detections=20,
        )

        detection = entries[0]["detections"][0]
        self.assertEqual(entries[0]["schema_version"], 2)
        self.assertEqual(detection["class_id"], 11)
        self.assertAlmostEqual(detection["bbox"]["cx"], 0.3)
        self.assertAlmostEqual(detection["bbox"]["cy"], 0.5)
        self.assertAlmostEqual(detection["bbox"]["w"], 0.4)
        self.assertAlmostEqual(detection["bbox"]["h"], 0.5)

    def test_failed_generation_preserves_existing_output_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bundle"
            output_dir.mkdir()
            sentinel = output_dir / "summary.json"
            sentinel.write_text("existing evidence\n", encoding="utf-8")
            with mock.patch.object(self.tool, "_selected_inputs", side_effect=RuntimeError("stop after staging")):
                with self.assertRaisesRegex(RuntimeError, "stop after staging"):
                    self.tool.main(
                        [
                            "--dataset",
                            "data/smoke",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing evidence\n")

    def test_existing_output_is_used_as_implicit_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bundle"
            output_dir.mkdir()
            sentinel = output_dir / "summary.json"
            sentinel.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                self.tool,
                "_generate_case_study",
                return_value=0,
            ) as generate:
                result = self.tool.main(
                    [
                        "--dataset",
                        "data/smoke",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(generate.call_args.kwargs["baseline_dir"], output_dir.resolve())
            self.assertEqual(
                generate.call_args.kwargs["baseline_mode"],
                "implicit_existing_output",
            )

    def test_explicit_baseline_may_equal_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bundle"
            output_dir.mkdir()
            with mock.patch.object(
                self.tool,
                "_generate_case_study",
                return_value=0,
            ) as generate:
                result = self.tool.main(
                    [
                        "--dataset",
                        "data/smoke",
                        "--output-dir",
                        str(output_dir),
                        "--baseline-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(generate.call_args.kwargs["baseline_dir"], output_dir.resolve())
            self.assertEqual(generate.call_args.kwargs["baseline_mode"], "explicit")

    def test_publish_rejects_symlink_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging = root / "staging"
            target = root / "target"
            link = root / "output-link"
            staging.mkdir()
            target.mkdir()
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(SystemExit, "must not be a symlink"):
                self.tool._publish_staged_artifacts(staging_dir=staging, output_dir=link)

    def test_failed_reproduction_does_not_publish_staged_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging = root / "staging"
            output = root / "output"
            staging.mkdir()
            output.mkdir()
            (staging / "summary.json").write_text("candidate\n", encoding="utf-8")
            sentinel = output / "summary.json"
            sentinel.write_text("existing\n", encoding="utf-8")

            result = self.tool._publish_after_checks(
                staging_dir=staging,
                output_dir=output,
                parity_ok=True,
                reproduction_ok=False,
            )

            self.assertEqual(result, 2)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(
                (staging / "summary.json").read_text(encoding="utf-8"),
                "candidate\n",
            )

    def test_baseline_check_rejects_protocol_drift_and_bad_checksums(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            shutil.copytree(self.artifact_root, baseline)
            shutil.copytree(self.artifact_root, candidate)

            candidate_summary_path = candidate / "summary.json"
            candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
            candidate_summary["source"]["clean_before_run"] = True
            candidate_summary_path.write_text(
                json.dumps(candidate_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            matching = self.tool._baseline_check(
                baseline_dir=baseline,
                candidate_dir=candidate,
                metric_atol=1e-8,
            )
            self.assertTrue(matching["ok"], matching["checks"])
            self.assertTrue(matching["candidate"]["source_clean_before_run"])

            candidate_protocol_path = candidate / "protocol.json"
            candidate_protocol = json.loads(candidate_protocol_path.read_text(encoding="utf-8"))
            candidate_protocol["fixed_conditions"]["export_filter"]["score_threshold"] = 0.75
            candidate_protocol_path.write_text(
                json.dumps(candidate_protocol, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            drifted = self.tool._baseline_check(
                baseline_dir=baseline,
                candidate_dir=candidate,
                metric_atol=1e-8,
            )
            checks = {item["name"]: item["ok"] for item in drifted["checks"]}
            self.assertFalse(drifted["ok"])
            self.assertFalse(checks["protocol_sha256"])

            baseline_protocol_path = baseline / "protocol.json"
            baseline_protocol_path.write_text("{}\n", encoding="utf-8")
            tampered = self.tool._baseline_check(
                baseline_dir=baseline,
                candidate_dir=candidate,
                metric_atol=1e-8,
            )
            checks = {item["name"]: item["ok"] for item in tampered["checks"]}
            self.assertFalse(checks["baseline_checksums_valid"])

    def test_baseline_checksum_manifest_requires_every_core_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline"
            shutil.copytree(self.artifact_root, baseline)
            missing = "comparison.svg"
            (baseline / missing).unlink()
            checksum_path = baseline / "checksums.sha256"
            retained = [
                line
                for line in checksum_path.read_text(encoding="utf-8").splitlines()
                if not line.endswith(f"  {missing}")
            ]
            checksum_path.write_text("\n".join(retained) + "\n", encoding="utf-8")

            result = self.tool._verify_checksum_manifest(baseline)

            self.assertFalse(result["ok"])
            self.assertIn(f"{missing}: missing checksum entry", result["errors"])

    def test_committed_predictions_and_reports_are_real_and_consistent(self):
        eager = json.loads((self.artifact_root / "predictions_eager.json").read_text(encoding="utf-8"))
        scripted = json.loads(
            (self.artifact_root / "predictions_torchscript.json").read_text(encoding="utf-8")
        )
        validate_predictions_payload(eager, strict=True)
        validate_predictions_payload(scripted, strict=True)
        self.assertTrue(eager["meta"]["extra"]["runtime_executed"])
        self.assertTrue(scripted["meta"]["extra"]["runtime_executed"])
        self.assertEqual(len(eager["predictions"]), 2)
        self.assertEqual(len(scripted["predictions"]), 2)

        summary = json.loads((self.artifact_root / "summary.json").read_text(encoding="utf-8"))
        parity = json.loads((self.artifact_root / "parity.json").read_text(encoding="utf-8"))
        protocol = json.loads((self.artifact_root / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["case_study_id"], self.tool.CASE_STUDY_ID)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["runtime_execution"]["eager"])
        self.assertTrue(summary["runtime_execution"]["torchscript"])
        self.assertTrue(summary["parity"]["ok"])
        self.assertTrue(parity["ok"])
        self.assertIsNone(protocol["canonical_protocol_preset"])
        self.assertEqual(protocol["evaluation_lane"], "tools/eval_coco.py")
        self.assertEqual(
            protocol["fixed_conditions"]["dataset_class_mapping"]["class_names"],
            list(COCO_80_CLASSES),
        )
        self.assertEqual(
            summary["results"]["eager"]["metrics"],
            summary["results"]["torchscript"]["metrics"],
        )
        self.assertGreater(summary["results"]["eager"]["detections"], 0)
        self.assertEqual(summary["weights"]["sha256"], self.tool.OFFICIAL_WEIGHTS_SHA256)
        for relative, expected in summary["source"]["file_sha256"].items():
            source_path = self.repo_root / relative
            self.assertTrue(source_path.is_file(), relative)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), expected, relative)

    def test_public_counts_and_metrics_match_committed_evidence(self):
        summary = json.loads((self.artifact_root / "summary.json").read_text(encoding="utf-8"))
        eager_eval = json.loads(
            (self.artifact_root / "eval_eager.json").read_text(encoding="utf-8")
        )
        scripted_eval = json.loads(
            (self.artifact_root / "eval_torchscript.json").read_text(encoding="utf-8")
        )
        eager_predictions = json.loads(
            (self.artifact_root / "predictions_eager.json").read_text(encoding="utf-8")
        )
        scripted_predictions = json.loads(
            (self.artifact_root / "predictions_torchscript.json").read_text(encoding="utf-8")
        )
        eager_count = sum(
            len(entry["detections"]) for entry in eager_predictions["predictions"]
        )
        scripted_count = sum(
            len(entry["detections"]) for entry in scripted_predictions["predictions"]
        )
        self.assertEqual(summary["results"]["eager"]["detections"], eager_count)
        self.assertEqual(summary["results"]["torchscript"]["detections"], scripted_count)
        self.assertEqual(summary["results"]["eager"]["metrics"], eager_eval["metrics"])
        self.assertEqual(
            summary["results"]["torchscript"]["metrics"],
            scripted_eval["metrics"],
        )

        doc = (
            self.repo_root / "docs" / "case_studies" / "maskrcnn_eager_torchscript.md"
        ).read_text(encoding="utf-8")
        manual = (
            self.repo_root / "manual" / "chapters" / "09_parity_bench_protocols.tex"
        ).read_text(encoding="utf-8")
        self.assertIn(f"Each retained {eager_count} detections.", doc)
        self.assertIn(f"retained {eager_count} detections in each path", manual)
        labels = {
            "map50_95": "mAP 50–95",
            "map50": "mAP 50",
            "map75": "mAP 75",
            "ar100": "AR 100",
        }
        for key, label in labels.items():
            eager_value = summary["results"]["eager"]["metrics"][key]
            scripted_value = summary["results"]["torchscript"]["metrics"][key]
            delta = summary["metric_deltas_torchscript_minus_eager"][key]
            delta_display = "0" if delta == 0 else f"{delta:.10f}"
            self.assertIn(
                f"| {label} | {eager_value:.10f} | {scripted_value:.10f} | {delta_display} |",
                doc,
            )

    def test_checksums_svg_and_public_paths_are_valid(self):
        checksum_path = self.artifact_root / "checksums.sha256"
        checksummed: set[str] = set()
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split("  ", 1)
            checksummed.add(filename)
            target = self.artifact_root / filename
            self.assertTrue(target.is_file(), filename)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest, filename)
        expected = set(self.tool.ARTIFACT_NAMES) - {"checksums.sha256"}
        self.assertEqual(checksummed, expected)

        svg = self.artifact_root / "comparison.svg"
        root = ET.fromstring(svg.read_text(encoding="utf-8"))
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("PyTorch eager", svg.read_text(encoding="utf-8"))
        self.assertIn("TorchScript", svg.read_text(encoding="utf-8"))

        for path in self.artifact_root.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, path.name)
            self.assertNotIn("/private/tmp/", text, path.name)
            self.assertNotIn("/home/", text, path.name)

        reproduction = json.loads(
            (self.artifact_root / "reproduction_check.json").read_text(encoding="utf-8")
        )
        self.assertEqual(reproduction["schema_version"], 2)
        self.assertTrue(reproduction["ok"])
        self.assertTrue(reproduction["baseline_checksums"]["ok"])
        self.assertTrue(reproduction["candidate"]["source_clean_before_run"])
        self.assertEqual(
            reproduction["candidate"]["source_sha256"],
            reproduction["baseline"]["source_sha256"],
        )
        self.assertEqual(
            reproduction["candidate"]["environment_sha256"],
            reproduction["baseline"]["environment_sha256"],
        )
        self.assertEqual(len(reproduction["candidate"]["summary_sha256"]), 64)

    def test_smoke_asset_generator_help_and_fixture_mapping(self):
        script = self.repo_root / "tools" / "generate_smoke_assets.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("standard COCO80 classes", result.stdout)
        self.assertIn("mapping", result.stdout)
        classes = json.loads(
            (self.repo_root / "data" / "smoke" / "labels" / "val" / "classes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(classes["names"], list(COCO_80_CLASSES))
        self.assertEqual(classes["class_to_category_id"]["79"], 90)

    def test_case_study_document_links_all_required_evidence(self):
        doc = (self.repo_root / "docs" / "case_studies" / "maskrcnn_eager_torchscript.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "predictions_eager.json",
            "predictions_torchscript.json",
            "eval_eager.json",
            "eval_torchscript.json",
            "parity.json",
            "protocol.json",
            "environment.json",
            "comparison.svg",
            "checksums.sha256",
            "reproduction_check.json",
        ):
            self.assertIn(required, doc)
        self.assertIn("not a model-quality benchmark", doc)
        self.assertIn("stable `eval-coco` lane", doc)


if __name__ == "__main__":
    unittest.main()
