import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from yolozu.contracts.synthgen import normalize_synthgen_sample, validate_synthgen_sample
from yolozu.data.synthgen_shard_dataset import SynthGenShardDataset, collate_synthgen_batch
from yolozu.data.synthgen_stream_dataset import SynthGenStreamDataset


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

    scene_spec = {"lighting": "studio", "seed": 7}
    inst_map = {"0": 0, "1": 1}

    return {
        "sample_id": stem,
        "image": f"{stem}_image.png",
        "depth_ndc": f"{stem}_depth.npy",
        "inst_id": f"{stem}_inst.npy",
        "sem_id": f"{stem}_sem.npy",
        "kpts2d": f"{stem}_kpts.npy",
        "prompt": "synthetic prompt",
        "scene_spec": json.dumps(scene_spec),
        "schema_id": schema_id,
        "schema_version": "1",
        "asset_ids": ["asset_a", "asset_b"],
        "inst_map": json.dumps(inst_map),
    }


class TestSynthGenContract(unittest.TestCase):
    def test_validate_and_normalize_inline_sample(self):
        sample = {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "depth_ndc": np.zeros((8, 8), dtype=np.float32),
            "inst_id": np.zeros((8, 8), dtype=np.uint32),
            "sem_id": np.zeros((8, 8), dtype=np.uint16),
            "kpts2d": np.zeros((1, 4, 3), dtype=np.float32),
            "prompt": "x",
            "scene_spec": "{\"seed\": 1}",
            "schema_id": "animal_v1",
            "schema_version": "1",
            "asset_ids": ["a"],
            "inst_map": "{\"0\": 0}",
        }
        sample["kpts2d"][..., 2] = 2

        result = validate_synthgen_sample(sample)
        self.assertTrue(result.ok, msg=f"errors: {result.errors}")

        normalized = normalize_synthgen_sample(sample)
        self.assertIsInstance(normalized["scene_spec"], dict)
        self.assertIsInstance(normalized["inst_map"], dict)
        self.assertEqual(normalized["depth_ndc"].dtype, np.float32)
        self.assertEqual(normalized["inst_id"].dtype, np.uint32)

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

            batch = collate_synthgen_batch([sample, sample], pad_keypoints=True)
            self.assertIn("kpts2d", batch)
            self.assertIn("kpts2d_mask", batch)
            self.assertEqual(batch["kpts2d"].shape[0], 2)
            self.assertEqual(batch["kpts2d_mask"].shape[0], 2)

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


if __name__ == "__main__":
    unittest.main()
