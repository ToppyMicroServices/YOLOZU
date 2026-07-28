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
        bop_root = root / "tless"
        scene = bop_root / "train_primesense/000001"
        (scene / "rgb").mkdir(parents=True)
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
            json.dumps({"0": [{"bbox_visib": [2, 1, 10, 6], "visib_fract": 1.0}]}),
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
            self.assertEqual(report["cad_models"]["1"]["points"], 3)
            self.assertEqual(len(report["cad_models"]["1"]["points_sha256"]), 64)
            self.assertEqual(len(report["cad_models"]["1"]["source_sha256"]), 64)
            sidecar = json.loads((output / "labels/train2017/000001_000000.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["cad_points_unit"], "meters")
            self.assertEqual(len(sidecar["cad_points"]), 1)
            cad_points = json.loads(Path(sidecar["cad_points"][0]).read_text(encoding="utf-8"))
            self.assertEqual(cad_points[1], [0.1, 0.0, 0.0])
            self.assertIn("symmetries_discrete", sidecar["bop_symmetry"][0])

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
        for tool_id in ("download_bop_dataset", "prepare_bop_yolozu"):
            self.assertIn(tool_id, tools)
            self.assertEqual(tools[tool_id]["maturity"], "research")
            self.assertTrue(tools[tool_id]["effects"]["writes"])
        self.assertTrue(tools["download_bop_dataset"]["requires"]["network"])

        protocol = (self.repo_root / "docs/bop_tless_protocol.md").read_text(encoding="utf-8")
        self.assertIn("human 3D skeleton pose is unsupported", protocol)
        self.assertIn("diagnostic frame holdout", protocol)


if __name__ == "__main__":
    unittest.main()
