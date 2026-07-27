import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMakeSubsetDataset(unittest.TestCase):
    def _make_dummy_dataset(self, root: Path) -> Path:
        dataset_root = root / "dataset"
        images = dataset_root / "images" / "val2017"
        labels = dataset_root / "labels" / "val2017"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            (images / f"{i:06d}.jpg").write_bytes(b"")
            if i % 2 == 0:
                (labels / f"{i:06d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return dataset_root

    def test_make_subset_dataset_is_deterministic(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)

            out1 = root / "out1"
            out2 = root / "out2"
            cmd = [
                sys.executable,
                str(script),
                "--dataset",
                str(dataset_root),
                "--split",
                "val2017",
                "--n",
                "5",
                "--seed",
                "123",
                "--strategy",
                "hash",
            ]

            proc1 = subprocess.run(cmd + ["--out", str(out1)], cwd=str(repo_root), capture_output=True, text=True)
            self.assertEqual(proc1.returncode, 0, proc1.stdout + proc1.stderr)

            proc2 = subprocess.run(cmd + ["--out", str(out2)], cwd=str(repo_root), capture_output=True, text=True)
            self.assertEqual(proc2.returncode, 0, proc2.stdout + proc2.stderr)

            list1 = (out1 / "subset_images.txt").read_text(encoding="utf-8")
            list2 = (out2 / "subset_images.txt").read_text(encoding="utf-8")
            self.assertEqual(list1, list2)

            payload = json.loads((out1 / "subset.json").read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "yolozu_subset_dataset")
            images = payload.get("images") or []
            self.assertEqual(len(images), 5)
            sha = payload.get("images_sha256")
            self.assertIsInstance(sha, str)
            self.assertTrue(sha)

            out_images = list((out1 / "images" / "val2017").glob("*.jpg"))
            self.assertEqual(len(out_images), 5)
            self.assertTrue((out1 / ".yolozu_subset_output.json").is_file())
            self.assertGreater(int((payload.get("artifacts") or {}).get("files") or 0), 0)

    def test_preserves_metadata_sidecars_and_keypoint_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "val"
            labels = dataset_root / "labels" / "val"
            depth = dataset_root / "depth" / "val"
            masks = dataset_root / "masks" / "val"
            for directory in (images, labels, depth, masks):
                directory.mkdir(parents=True, exist_ok=True)
            (images / "sample.jpg").write_bytes(b"image")
            (labels / "sample.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.4 0.4 2\n",
                encoding="utf-8",
            )
            (depth / "sample.npy").write_bytes(b"depth")
            (masks / "sample.png").write_bytes(b"mask")
            (labels / "sample.json").write_text(
                json.dumps(
                    {
                        "depth_path": "depth/val/sample.npy",
                        "mask_path": "masks/val/sample.png",
                        "K_gt": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "R_gt": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                        "t_gt": [[0, 0, 1]],
                    }
                ),
                encoding="utf-8",
            )
            (labels / "classes.json").write_text(
                json.dumps({"class_names": ["thing"], "keypoint_names": ["kp"]}),
                encoding="utf-8",
            )
            (dataset_root / "dataset.json").write_text(
                json.dumps({"task": "multi", "label_provenance": {"depth": "fixture"}}),
                encoding="utf-8",
            )

            output = root / "subset"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val",
                    "--n",
                    "1",
                    "--strategy",
                    "first",
                    "--copy",
                    "--out",
                    str(output),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual((output / "depth" / "val" / "sample.npy").read_bytes(), b"depth")
            self.assertEqual((output / "masks" / "val" / "sample.png").read_bytes(), b"mask")
            self.assertTrue((output / "labels" / "val" / "classes.json").is_file())
            metadata = json.loads((output / "labels" / "val" / "sample.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata.get("depth_path"), "depth/val/sample.npy")
            self.assertEqual(metadata.get("mask_path"), "masks/val/sample.png")
            payload = json.loads((output / "subset.json").read_text(encoding="utf-8"))
            hashes = (payload.get("artifacts") or {}).get("sha256") or {}
            self.assertIn("depth/val/sample.npy", hashes)
            self.assertIn("masks/val/sample.png", hashes)
            self.assertIn("dataset.json", payload.get("source_metadata_sha256") or {})

    def test_overwrite_refuses_unowned_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)
            output = root / "unowned"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--n",
                    "1",
                    "--out",
                    str(output),
                    "--overwrite",
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unowned", proc.stdout + proc.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_external_sidecar_list_uses_distinct_owned_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)
            external = root / "external"
            external.mkdir()
            mask_a = external / "a.png"
            mask_b = external / "b.png"
            mask_a.write_bytes(b"mask-a")
            mask_b.write_bytes(b"mask-b")
            (dataset_root / "labels" / "val2017" / "000000.json").write_text(
                json.dumps({"mask_path": [str(mask_a), str(mask_b)]}),
                encoding="utf-8",
            )

            output = root / "subset"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--n",
                    "1",
                    "--strategy",
                    "first",
                    "--copy",
                    "--out",
                    str(output),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metadata = json.loads((output / "labels" / "val2017" / "000000.json").read_text(encoding="utf-8"))
            paths = metadata.get("mask_path") or []
            self.assertEqual(len(paths), 2)
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual((output / paths[0]).read_bytes(), b"mask-a")
            self.assertEqual((output / paths[1]).read_bytes(), b"mask-b")

    def test_missing_referenced_sidecar_fails(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)
            (dataset_root / "labels" / "val2017" / "000000.json").write_text(
                json.dumps({"depth_path": "depth/val2017/missing.npy"}),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--n",
                    "1",
                    "--strategy",
                    "first",
                    "--out",
                    str(root / "subset"),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("referenced sidecar does not exist", proc.stdout + proc.stderr)

    def test_refuses_symlink_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)
            target = root / "target"
            target.mkdir()
            output = root / "output-link"
            output.symlink_to(target, target_is_directory=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--n",
                    "1",
                    "--out",
                    str(output),
                    "--overwrite",
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("symlink output", proc.stdout + proc.stderr)
            self.assertTrue(target.is_dir())

    def test_owned_output_can_be_replaced(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._make_dummy_dataset(root)
            output = root / "subset"
            base_command = [
                sys.executable,
                str(script),
                "--dataset",
                str(dataset_root),
                "--split",
                "val2017",
                "--n",
                "1",
                "--strategy",
                "first",
                "--out",
                str(output),
            ]
            first = subprocess.run(base_command, cwd=str(repo_root), capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            stale = output / "stale.txt"
            stale.write_text("stale", encoding="utf-8")
            second = subprocess.run(
                base_command + ["--overwrite"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertFalse(stale.exists())
            self.assertTrue((output / ".yolozu_subset_output.json").is_file())

    def test_prepare_output_refuses_repository_root(self):
        from tools.make_subset_dataset import _prepare_output

        repo_root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(SystemExit, "protected output directory"):
            _prepare_output(repo_root, overwrite=True)

    def test_tracked_real_multitask_subset_preserves_training_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"
        dataset_root = repo_root / "data" / "real_multitask_fewshot"
        self.assertTrue(dataset_root.is_dir(), "missing tracked real multitask fixture")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            output = Path(td) / "subset"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val",
                    "--n",
                    "2",
                    "--strategy",
                    "first",
                    "--copy",
                    "--out",
                    str(output),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            selected = ["000000468577", "000000512194"]
            for stem in selected:
                for relative in (
                    Path("images") / "val" / f"{stem}.jpg",
                    Path("labels") / "val" / f"{stem}.txt",
                    Path("masks") / "val" / f"{stem}_inst.png",
                    Path("depth") / "val" / f"{stem}_depth.npy",
                ):
                    self.assertEqual((dataset_root / relative).read_bytes(), (output / relative).read_bytes())
                source_meta = json.loads((dataset_root / "labels" / "val" / f"{stem}.json").read_text(encoding="utf-8"))
                subset_meta = json.loads((output / "labels" / "val" / f"{stem}.json").read_text(encoding="utf-8"))
                self.assertEqual(source_meta, subset_meta)

            from rtdetr_pose.dataset import build_manifest

            manifest = build_manifest(output, split="val")
            records = manifest.get("images") or []
            self.assertEqual(len(records), 2)
            self.assertEqual(sum(len(record.get("labels") or []) for record in records), 3)
            self.assertEqual(
                sum(len(label.get("keypoints") or []) for record in records for label in record.get("labels") or []),
                18,
            )
            self.assertTrue(all(record.get("mask_path") for record in records))
            self.assertTrue(all(record.get("depth_path") for record in records))
            self.assertTrue(all(record.get("R_gt") and record.get("t_gt") and record.get("K_gt") for record in records))
            keypoints_meta = manifest.get("keypoints_meta") or {}
            self.assertEqual(int(keypoints_meta.get("num_keypoints") or 0), 6)

    def test_refuses_source_output_overlap_before_writing(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "make_subset_dataset.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            dataset_root = self._make_dummy_dataset(Path(td))
            sentinel = dataset_root / "images" / "val2017" / "000000.jpg"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--n",
                    "1",
                    "--out",
                    str(dataset_root),
                    "--overwrite",
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("must not overlap", proc.stdout + proc.stderr)
            self.assertTrue(sentinel.is_file())


if __name__ == "__main__":
    unittest.main()
