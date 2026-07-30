import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image


class TestBopPosePipelineSafety(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.download = self.repo_root / "tools/download_bop_dataset.py"
        self.prepare = self.repo_root / "tools/prepare_bop_yolozu.py"
        self.summarize = self.repo_root / "tools/summarize_bop_tless_qualification.py"

    def _run(self, command):
        return subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def _make_bop_fixture(self, root: Path) -> Path:
        bop_root = root / "bop"
        scene = bop_root / "train_primesense/000001"
        (scene / "rgb").mkdir(parents=True)
        (scene / "depth").mkdir()
        (scene / "mask_visib").mkdir()
        (bop_root / "download_manifest.json").write_text(
            json.dumps({"dataset": "tless", "complete": True}),
            encoding="utf-8",
        )
        models = bop_root / "models_cad"
        models.mkdir(parents=True)
        (models / "obj_000001.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "100 0 0",
                    "0 100 0",
                    "",
                ]
            ),
            encoding="ascii",
        )
        (models / "models_info.json").write_text(
            json.dumps({"1": {"symmetries_discrete": [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]}}),
            encoding="utf-8",
        )
        Image.new("RGB", (20, 10), color=(100, 120, 140)).save(scene / "rgb/000000.png")
        Image.new("I;16", (20, 10), color=500).save(scene / "depth/000000.png")
        mask = Image.new("L", (20, 10), color=0)
        for x in range(2, 14):
            for y in range(1, 10):
                mask.putpixel((x, y), 255)
        mask.save(scene / "mask_visib/000000_000000.png")
        (scene / "scene_gt.json").write_text(
            json.dumps(
                {
                    "0": [
                        {
                            "obj_id": 1,
                            "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            "cam_t_m2c": [10, 20, 500],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (scene / "scene_camera.json").write_text(
            json.dumps({"0": {"cam_K": [100, 0, 10, 0, 100, 5, 0, 0, 1], "depth_scale": 1.0}}),
            encoding="utf-8",
        )
        (scene / "scene_gt_info.json").write_text(
            json.dumps({"0": [{"bbox_visib": [2, 1, 12, 9], "visib_fract": 1.0}]}),
            encoding="utf-8",
        )
        return bop_root

    def test_shell_entrypoints_support_help(self):
        for script in (
            "deploy/runpod/bootstrap_bop_tless_train_primesense.sh",
            "deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh",
        ):
            proc = self._run(["bash", script, "--help"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Usage:", proc.stdout)
        run_script = (
            self.repo_root / "deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn('rm -rf "${DATASET_OUT}"', run_script)
        self.assertIn('train_checkpoint "${baseline_dir}" 0 "${seed}"', run_script)
        self.assertIn("checkpoint_sha256", run_script)
        self.assertIn("add_mean", run_script)
        self.assertIn("adds_mean", run_script)

    def test_downloader_rejects_archive_path_escape_before_network(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            proc = self._run(
                [
                    sys.executable,
                    str(self.download),
                    "--dataset",
                    "tless",
                    "--archives",
                    "../escape.zip",
                    "--out",
                    str(Path(temp_dir) / "out"),
                ]
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("archive must be a plain .zip filename", proc.stderr)

    def test_tless_default_download_includes_cad_models(self):
        spec = importlib.util.spec_from_file_location("download_bop_dataset_defaults_test", self.download)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        self.assertEqual(
            module.DATASET_DEFAULT_ARCHIVES["tless"],
            ["tless_base.zip", "tless_models.zip", "tless_train_primesense.zip"],
        )

    def test_downloader_rejects_unsafe_zip_member(self):
        spec = importlib.util.spec_from_file_location("download_bop_dataset_test", self.download)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "unsafe zip member path"):
                module._extract_zip(archive, root / "out", force=False)
            self.assertFalse((root / "escape.txt").exists())

    def test_cad_keypoint_visibility_uses_bop_visible_mask(self):
        spec = importlib.util.spec_from_file_location("prepare_bop_visibility_test", self.prepare)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            mask_path = Path(temp_dir) / "mask.png"
            Image.new("L", (10, 10), color=0).save(mask_path)
            kwargs = {
                "anchors": [[0.0, 0.0, 0.0]],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [0.0, 0.0, 1.0],
                "intrinsics": [10.0, 0.0, 5.0, 0.0, 10.0, 5.0, 0.0, 0.0, 1.0],
                "width": 10,
                "height": 10,
                "count": 1,
                "visible_mask_path": mask_path,
            }
            self.assertEqual(module._project_cad_anchors(**kwargs)[0][2], 1)

            visible = Image.new("L", (10, 10), color=0)
            visible.putpixel((5, 5), 255)
            visible.save(mask_path)
            self.assertEqual(module._project_cad_anchors(**kwargs)[0][2], 2)

    def test_downloader_rejects_symlink_cache_archive(self):
        spec = importlib.util.spec_from_file_location("download_bop_dataset_symlink_test", self.download)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            target = root / "target.zip"
            target.write_bytes(b"not-a-zip")
            linked = root / "linked.zip"
            linked.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "refusing symlink archive cache path"):
                module._download("https://example.invalid/linked.zip", linked, force=False)

    def test_cached_archive_writes_hash_and_license_manifest(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "tless_base.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("tless/models/readme.txt", "fixture")
            out = root / "out"
            proc = self._run(
                [
                    sys.executable,
                    str(self.download),
                    "--dataset",
                    "tless",
                    "--archives",
                    "tless_base.zip",
                    "--out",
                    str(out),
                    "--cache",
                    str(cache),
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / "download_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "bop_download_manifest")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["dataset"], "tless")
            self.assertEqual(manifest["license"]["spdx"], "CC-BY-4.0")
            self.assertEqual(len(manifest["archives"][0]["sha256"]), 64)
            self.assertGreater(manifest["archives"][0]["bytes"], 0)
            self.assertEqual(manifest["archives"][0]["extraction_status"], "complete_extracted")

    def test_converter_replaces_only_owned_output(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            bop_root = self._make_bop_fixture(root)
            output = root / "converted"
            base = [
                sys.executable,
                str(self.prepare),
                "--bop-root",
                str(bop_root),
                "--split",
                "train_primesense",
                "--out",
                str(output),
                "--max-images",
                "1",
                "--cad-keypoints",
                "2",
            ]

            first = self._run(base)
            self.assertEqual(first.returncode, 0, first.stderr)
            marker = output / ".yolozu_bop_output.json"
            self.assertTrue(marker.is_file())
            report = json.loads((output / "conversion_reports/train2017.json").read_text(encoding="utf-8"))
            self.assertEqual(report["converted_images"], 1)
            self.assertEqual(report["translation_scale_to_meters"], 0.001)
            self.assertEqual(report["dataset_license"]["spdx"], "CC-BY-4.0")
            self.assertEqual(report["cad_points_unit"], "meters")
            self.assertEqual(report["cad_keypoints"], 2)
            self.assertEqual(report["cad_models"]["1"]["points"], 3)
            self.assertEqual(len(report["cad_models"]["1"]["points_sha256"]), 64)
            self.assertEqual(len(report["cad_models"]["1"]["source_sha256"]), 64)
            sidecar = json.loads((output / "labels/train2017/000001_000000.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["cad_points_unit"], "meters")
            self.assertEqual(len(sidecar["cad_points"]), 1)
            self.assertTrue(Path(sidecar["depth_path"]).is_file())
            self.assertTrue(Path(sidecar["mask_path"][0]).is_file())
            self.assertEqual(sidecar["mask_classes"], [0])
            cad_points = json.loads(Path(sidecar["cad_points"][0]).read_text(encoding="utf-8"))
            self.assertEqual(cad_points[1], [0.1, 0.0, 0.0])
            self.assertIn("symmetries_discrete", sidecar["bop_symmetry"][0])
            label_fields = (output / "labels/train2017/000001_000000.txt").read_text(
                encoding="utf-8"
            ).split()
            self.assertEqual(len(label_fields), 11)
            self.assertEqual([int(value) for value in label_fields[7::3]], [2, 0])
            descriptor = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(descriptor["keypoints_meta"]["num_keypoints"], 2)
            summary = json.loads((output / "prepare_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["checks"]["strict_ground_truth"])
            self.assertFalse(summary["label_provenance"]["model_inference_used"])
            self.assertEqual(
                summary["annotation_counts"],
                {
                    "images_with_depth": 1,
                    "instances_with_mask": 1,
                    "instances_with_pose6d": 1,
                    "visible_projected_keypoints": 1,
                },
            )

            refused = self._run(base)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("output directory already exists", refused.stderr)

            replaced = self._run([*base, "--overwrite"])
            self.assertEqual(replaced.returncode, 0, replaced.stderr)

            appended = self._run(
                [
                    *base,
                    "--out-split",
                    "val2017",
                    "--partition-modulus",
                    "2",
                    "--partition-remainder",
                    "0",
                    "--partition-mode",
                    "include",
                    "--append-owned",
                ]
            )
            self.assertEqual(appended.returncode, 0, appended.stderr)
            descriptor = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(set(descriptor["splits"]), {"train2017", "val2017"})

            marker.write_text('{"kind":"not-owned","schema_version":1}\n', encoding="utf-8")
            unowned = self._run([*base, "--overwrite"])
            self.assertNotEqual(unowned.returncode, 0)
            self.assertIn("refusing to delete unowned output directory", unowned.stderr)

    def test_converter_refuses_unowned_empty_and_source_nested_outputs(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            bop_root = self._make_bop_fixture(root)
            empty = root / "empty"
            empty.mkdir()
            base = [
                sys.executable,
                str(self.prepare),
                "--bop-root",
                str(bop_root),
                "--split",
                "train_primesense",
            ]

            empty_result = self._run([*base, "--out", str(empty)])
            self.assertNotEqual(empty_result.returncode, 0)
            self.assertIn("output directory already exists", empty_result.stderr)
            self.assertFalse((empty / ".yolozu_bop_output.json").exists())

            nested = bop_root / "converted"
            nested_result = self._run([*base, "--out", str(nested)])
            self.assertNotEqual(nested_result.returncode, 0)
            self.assertIn("refusing output inside the BOP source", nested_result.stderr)
            self.assertFalse(nested.exists())

    def test_manifest_declares_bop_tools_as_research(self):
        manifest = json.loads((self.repo_root / "tools/manifest.json").read_text(encoding="utf-8"))
        tools = {item["id"]: item for item in manifest["tools"]}
        for tool_id in (
            "download_bop_dataset",
            "prepare_bop_yolozu",
            "summarize_bop_tless_qualification",
        ):
            self.assertIn(tool_id, tools)
            self.assertEqual(tools[tool_id]["maturity"], "research")
            self.assertTrue(tools[tool_id]["effects"]["writes"])
        self.assertTrue(tools["download_bop_dataset"]["requires"]["network"])
        self.assertEqual(
            tools["summarize_bop_tless_qualification"]["contracts"]["produces"],
            ["bop_tless_qualification_json"],
        )

        protocol = (self.repo_root / "docs/bop_tless_protocol.md").read_text(encoding="utf-8")
        self.assertIn("human 3D skeleton pose is unsupported", protocol)
        self.assertIn("diagnostic frame holdout", protocol)

    def test_qualification_summary_preserves_hold_decision(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            root = Path(temp_dir)
            run_base = root / "runs"
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "prepare_summary.json").write_text(
                json.dumps(
                    {
                        "checks": {"strict_ground_truth": True},
                        "label_provenance": {"model_inference_used": False},
                        "annotation_counts": {"instances_with_pose6d": 1},
                    }
                ),
                encoding="utf-8",
            )
            download_manifest = root / "download_manifest.json"
            download_manifest.write_text(
                json.dumps(
                    {
                        "archives": [{"name": "tless_train_primesense.zip", "sha256": "a" * 64}],
                        "license": {"spdx": "CC-BY-4.0"},
                    }
                ),
                encoding="utf-8",
            )
            common = {
                "seed": 11,
                "bbox_metrics": {"map50_95": 0.0},
                "task_native_metrics": {"pose6d": {"match_rate": 0.0}},
                "thresholds": {"iou": 0.5},
                "checkpoint_sha256": "b" * 64,
                "config": "config.json",
                "config_sha256": "c" * 64,
                "model_implementation_license": "Apache-2.0",
                "runtime_seconds": 1,
                "train_resource": {"peak_rss_bytes": 100},
            }
            for kind, epochs in (("baseline", 0), ("trained", 1)):
                run = run_base / kind
                run.mkdir(parents=True)
                payload = {**common, "run_kind": kind, "epochs": epochs}
                (run / "run_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "qualification_summary.json"
            proc = self._run(
                [
                    sys.executable,
                    str(self.summarize),
                    "--run-base",
                    str(run_base),
                    "--dataset",
                    str(dataset),
                    "--download-manifest",
                    str(download_manifest),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["decision"], "hold")
            self.assertEqual(summary["efficacy"], "not_established")
            self.assertFalse(summary["promotion_eligible"])
            self.assertIn("fewer_than_three_seeds", summary["qualification_reasons"])


if __name__ == "__main__":
    unittest.main()
