import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from yolozu.contracts.synthgen import (
    SYNTHGEN_COORDINATE_SYSTEM,
    normalize_synthgen_sample,
    validate_synthgen_sample,
)
from yolozu.data.synthgen_shard_dataset import SynthGenShardDataset, collate_synthgen_batch
from yolozu.data.synthgen_stream_dataset import SynthGenStreamDataset


def _canonical_scene_spec(object_count: int) -> dict:
    identity = np.eye(4, dtype=np.float32).tolist()
    return {
        "coordinate_system": SYNTHGEN_COORDINATE_SYSTEM,
        "camera": {"world_to_camera": identity},
        "objects": [{"object_to_world": identity} for _ in range(object_count)],
    }


def _inline_3d_sample() -> dict:
    sample = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth_ndc": np.zeros((8, 8), dtype=np.float32),
        "inst_id": np.zeros((8, 8), dtype=np.uint32),
        "sem_id": np.zeros((8, 8), dtype=np.uint16),
        "bbox2d_visible": np.zeros((1, 4), dtype=np.float32),
        "kpts2d": np.zeros((1, 4, 3), dtype=np.float32),
        "kpts3d_object": np.zeros((1, 4, 3), dtype=np.float32),
        "pose_obj2cam": np.eye(4, dtype=np.float32),
        "prompt": "x",
        "scene_spec": _canonical_scene_spec(1),
        "schema_id": "animal_v1",
        "schema_version": "1",
        "asset_ids": ["a"],
        "inst_map": {"0": 0},
    }
    sample["kpts2d"][..., 2] = 2
    return sample


def _write_sample_assets(root: Path, *, stem: str, schema_id: str) -> dict:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 32
    image[..., 1] = 64
    image[..., 2] = 128
    Image.fromarray(image).save(root / f"{stem}_image.png")

    depth = np.linspace(0.0, 1.0, num=16 * 16, dtype=np.float32).reshape(16, 16)
    np.save(root / f"{stem}_depth.npy", depth)

    inst_id = np.zeros((16, 16), dtype=np.uint32)
    inst_id[:, 8:] = 1
    np.save(root / f"{stem}_inst.npy", inst_id)

    sem_id = np.zeros((16, 16), dtype=np.uint16)
    sem_id[4:12, 4:12] = 2
    np.save(root / f"{stem}_sem.npy", sem_id)

    kpts2d = np.array(
        [
            [[4.0, 4.0, 2.0], [6.0, 6.0, 2.0], [8.0, 8.0, 1.0]],
            [[10.0, 10.0, 2.0], [12.0, 12.0, 2.0], [14.0, 14.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.save(root / f"{stem}_kpts.npy", kpts2d)

    bbox2d_visible = np.array([[8.0, 0.0, 16.0, 16.0], [-1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
    np.save(root / f"{stem}_bbox.npy", bbox2d_visible)

    kpts3d_object = np.zeros((2, 3, 3), dtype=np.float32)
    np.save(root / f"{stem}_kpts3d.npy", kpts3d_object)

    pose_obj2cam = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    np.save(root / f"{stem}_pose.npy", pose_obj2cam)

    scene_spec = _canonical_scene_spec(2)
    inst_map = {"0": 0, "1": 1}

    return {
        "sample_id": stem,
        "image": f"{stem}_image.png",
        "depth_ndc": f"{stem}_depth.npy",
        "inst_id": f"{stem}_inst.npy",
        "sem_id": f"{stem}_sem.npy",
        "bbox2d_visible": f"{stem}_bbox.npy",
        "kpts2d": f"{stem}_kpts.npy",
        "kpts3d_object": f"{stem}_kpts3d.npy",
        "pose_obj2cam": f"{stem}_pose.npy",
        "prompt": "synthetic prompt",
        "scene_spec": json.dumps(scene_spec),
        "schema_id": schema_id,
        "schema_version": "1",
        "asset_ids": ["asset_a", "asset_b"],
        "inst_map": json.dumps(inst_map),
    }


class TestSynthGenContract(unittest.TestCase):
    def test_validate_and_normalize_inline_sample(self):
        sample = _inline_3d_sample()

        result = validate_synthgen_sample(sample)
        self.assertTrue(result.ok, msg=f"errors: {result.errors}")

        normalized = normalize_synthgen_sample(sample)
        self.assertIsInstance(normalized["scene_spec"], dict)
        self.assertIsInstance(normalized["inst_map"], dict)
        self.assertEqual(normalized["depth_ndc"].dtype, np.float32)
        self.assertEqual(normalized["inst_id"].dtype, np.uint32)
        self.assertEqual(normalized["pose_obj2cam"].shape, (4, 4))

    def test_validate_rejects_reflected_pose(self):
        sample = _inline_3d_sample()
        sample["pose_obj2cam"][0, 0] = -1.0

        result = validate_synthgen_sample(sample)

        self.assertFalse(result.ok)
        self.assertTrue(any("det(R)=+1" in error for error in result.errors))

    def test_validate_rejects_scaled_pose(self):
        sample = _inline_3d_sample()
        sample["pose_obj2cam"][1, 1] = 0.5

        result = validate_synthgen_sample(sample)

        self.assertFalse(result.ok)
        self.assertTrue(any("without scale or shear" in error for error in result.errors))

    def test_validate_rejects_pose_composition_mismatch(self):
        sample = _inline_3d_sample()
        sample["scene_spec"]["objects"][0]["object_to_world"][0][3] = 1.0

        result = validate_synthgen_sample(sample)

        self.assertFalse(result.ok)
        self.assertTrue(
            any("world_to_camera @ object_to_world" in error for error in result.errors)
        )

    def test_validate_rejects_3d_labels_without_coordinate_system(self):
        sample = _inline_3d_sample()
        del sample["scene_spec"]["coordinate_system"]

        result = validate_synthgen_sample(sample)

        self.assertFalse(result.ok)
        self.assertTrue(any("coordinate_system" in error for error in result.errors))

    def test_validate_rejects_depth_range(self):
        sample = {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "depth_ndc": np.full((8, 8), 1.5, dtype=np.float32),
            "inst_id": np.zeros((8, 8), dtype=np.uint32),
            "sem_id": np.zeros((8, 8), dtype=np.uint16),
            "kpts2d": np.zeros((1, 2, 3), dtype=np.float32),
            "prompt": "x",
            "scene_spec": "{\"seed\": 1}",
            "schema_id": "mechanical_v1",
            "schema_version": "1",
            "asset_ids": ["a"],
            "inst_map": "{\"0\": 0}",
        }
        result = validate_synthgen_sample(sample)
        self.assertFalse(result.ok)
        self.assertTrue(any("depth_ndc: expected range [0,1]" in e for e in result.errors))

    def test_shard_dataset_schema_filter_and_collate(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            shards = root / "shards"
            shards.mkdir(parents=True, exist_ok=True)

            animal = _write_sample_assets(root, stem="animal_0", schema_id="animal_v1")
            mechanical = _write_sample_assets(root, stem="mech_0", schema_id="mechanical_v1")
            (shards / "train_000.jsonl").write_text(
                json.dumps(animal) + "\n" + json.dumps(mechanical) + "\n",
                encoding="utf-8",
            )

            ds_all = SynthGenShardDataset(root)
            self.assertEqual(len(ds_all), 2)
            ds_animal = SynthGenShardDataset(root, schema_id="animal_v1")
            self.assertEqual(len(ds_animal), 1)

            sample = ds_animal[0]
            self.assertEqual(sample["schema_id"], "animal_v1")
            self.assertEqual(sample["image"].dtype, np.uint8)
            self.assertIsInstance(sample["scene_spec"], dict)
            self.assertIsInstance(sample["inst_map"], dict)
            self.assertEqual(sample["bbox2d_visible"].shape, (2, 4))
            self.assertEqual(sample["kpts3d_object"].shape, (2, 3, 3))
            self.assertEqual(sample["pose_obj2cam"].shape, (2, 4, 4))

            batch = collate_synthgen_batch([sample, sample], pad_keypoints=True)
            self.assertIn("kpts2d", batch)
            self.assertIn("kpts2d_mask", batch)
            self.assertEqual(batch["kpts2d"].shape[0], 2)
            self.assertEqual(batch["kpts2d_mask"].shape[0], 2)

    def test_collate_pads_all_instance_labels(self):
        sample_a = {
            "bbox2d_visible": np.zeros((1, 4), dtype=np.float32),
            "kpts2d": np.ones((1, 3, 3), dtype=np.float32),
            "kpts3d_object": np.zeros((1, 3, 3), dtype=np.float32),
            "pose_obj2cam": np.ones((1, 4, 4), dtype=np.float32),
        }
        sample_b = {
            "bbox2d_visible": np.ones((3, 4), dtype=np.float32),
            "kpts2d": np.ones((3, 3, 3), dtype=np.float32),
            "kpts3d_object": np.ones((3, 3, 3), dtype=np.float32),
            "pose_obj2cam": np.ones((3, 4, 4), dtype=np.float32),
        }

        batch = collate_synthgen_batch([sample_a, sample_b], pad_instance_labels=True)

        self.assertEqual(batch["bbox2d_visible"].shape, (2, 3, 4))
        self.assertEqual(batch["kpts2d"].shape, (2, 3, 3, 3))
        self.assertEqual(batch["kpts3d_object"].shape, (2, 3, 3, 3))
        self.assertEqual(batch["pose_obj2cam"].shape, (2, 3, 4, 4))
        self.assertEqual(batch["instance_mask"].tolist(), [[True, False, False], [True, True, True]])
        self.assertTrue(np.all(batch["bbox2d_visible"][0, 1:] == -1.0))
        self.assertTrue(np.all(np.isnan(batch["kpts3d_object"][0, 1:])))
        self.assertTrue(np.all(batch["pose_obj2cam"][0, 1:] == 0.0))

    def test_validate_rejects_misaligned_optional_instance_labels(self):
        sample = {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "depth_ndc": np.zeros((8, 8), dtype=np.float32),
            "inst_id": np.zeros((8, 8), dtype=np.uint32),
            "sem_id": np.zeros((8, 8), dtype=np.uint16),
            "bbox2d_visible": np.zeros((2, 4), dtype=np.float32),
            "kpts2d": np.zeros((1, 4, 3), dtype=np.float32),
            "kpts3d_object": np.zeros((1, 3, 3), dtype=np.float32),
            "pose_obj2cam": np.eye(4, dtype=np.float32)[None, ...],
            "prompt": "x",
            "scene_spec": _canonical_scene_spec(1),
            "schema_id": "animal_v1",
            "schema_version": "1",
            "asset_ids": ["a"],
            "inst_map": "{\"0\": 0}",
        }

        result = validate_synthgen_sample(sample)

        self.assertFalse(result.ok)
        self.assertTrue(any("bbox2d_visible: expected N_inst=1" in error for error in result.errors))
        self.assertTrue(any("kpts3d_object: expected K=4" in error for error in result.errors))

    def test_stream_dataset_filters_schema(self):
        sample_a = {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "depth_ndc": np.zeros((8, 8), dtype=np.float32),
            "inst_id": np.zeros((8, 8), dtype=np.uint32),
            "sem_id": np.zeros((8, 8), dtype=np.uint16),
            "kpts2d": np.zeros((1, 2, 3), dtype=np.float32),
            "prompt": "a",
            "scene_spec": "{\"seed\": 1}",
            "schema_id": "animal_v1",
            "schema_version": "1",
            "asset_ids": ["asset_a"],
            "inst_map": "{\"0\": 0}",
        }
        sample_b = dict(sample_a)
        sample_b["schema_id"] = "mechanical_v1"

        dataset = SynthGenStreamDataset([sample_a, sample_b], schema_id="animal_v1")
        rows = list(dataset)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema_id"], "animal_v1")

    def test_validate_synthgen_contract_tool_accepts_path_based_shard(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_synthgen_contract.py"
        self.assertTrue(script.exists())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "shards").mkdir(parents=True, exist_ok=True)
            rec = _write_sample_assets(root, stem="animal_0", schema_id="animal_v1")
            shard = root / "shards" / "train_000.jsonl"
            shard.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, str(script), "--input", str(shard), "--max-samples", "1"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"validate_synthgen_contract.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("OK:", proc.stdout + proc.stderr)

    def test_smoke_fixture_contract_and_schema_filter(self):
        repo_root = Path(__file__).resolve().parents[1]
        smoke_root = repo_root / "data" / "smoke" / "synthgen_minishard"
        self.assertTrue((smoke_root / "shards" / "train_000.jsonl").exists())

        ds_all = SynthGenShardDataset(smoke_root)
        self.assertGreaterEqual(len(ds_all), 2)
        schema_ids = {str(ds_all[i].get("schema_id")) for i in range(len(ds_all))}
        self.assertIn("animal_v1", schema_ids)
        self.assertIn("mechanical_v1", schema_ids)

        ds_animal = SynthGenShardDataset(smoke_root, schema_id="animal_v1")
        ds_mech = SynthGenShardDataset(smoke_root, schema_id="mechanical_v1")
        self.assertGreaterEqual(len(ds_animal), 1)
        self.assertGreaterEqual(len(ds_mech), 1)

    def test_smoke_synthgen_tool_runs_end_to_end(self):
        repo_root = Path(__file__).resolve().parents[1]
        smoke_root = repo_root / "data" / "smoke" / "synthgen_minishard"
        script = repo_root / "tools" / "smoke_synthgen.py"
        self.assertTrue(script.exists())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            output_dir = Path(td) / "reports"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(smoke_root),
                    "--predictions",
                    str(smoke_root / "predictions_synthgen_smoke.json"),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"smoke_synthgen.py failed:\n{proc.stdout}\n{proc.stderr}")

            overlay = output_dir / "smoke_synthgen_overlay.png"
            report = output_dir / "smoke_synthgen_eval.json"
            summary = output_dir / "smoke_synthgen_summary.json"
            self.assertTrue(overlay.exists(), "missing synthgen overlay artifact")
            self.assertTrue(report.exists(), "missing synthgen eval report")
            self.assertTrue(summary.exists(), "missing synthgen summary")
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "ok")

    def test_smoke_synthgen_help_exposes_fresh_handoff_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, str(repo_root / "tools/smoke_synthgen.py"), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--synthgen-repo", proc.stdout)
        self.assertIn("--synthgen-python", proc.stdout)
        self.assertIn("--backend", proc.stdout)
        self.assertIn("--mode", proc.stdout)

    def test_synthgen_json_schema_file_exists_and_required_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "schemas" / "synthgen_sample.schema.json"
        self.assertTrue(schema_path.exists(), "missing schemas/synthgen_sample.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required = set(schema.get("required") or [])
        for field in (
            "image",
            "depth_ndc",
            "inst_id",
            "sem_id",
            "kpts2d",
            "prompt",
            "scene_spec",
            "schema_id",
            "schema_version",
            "asset_ids",
            "inst_map",
        ):
            self.assertIn(field, required, f"required field missing in schema: {field}")


if __name__ == "__main__":
    unittest.main()
