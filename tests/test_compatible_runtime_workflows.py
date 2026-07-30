import unittest
from pathlib import Path

import yaml


class TestCompatibleRuntimeWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.workflow_paths = (
            cls.repo_root / ".github" / "workflows" / "gpu_practical_suite_machine.yml",
            cls.repo_root
            / ".github"
            / "workflows"
            / "external_runtime_qualification_machine.yml",
        )

    def test_workflows_install_headless_runtime_libraries(self) -> None:
        for path in self.workflow_paths:
            with self.subTest(workflow=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("libxcb1", text)
                self.assertIn("libgl1", text)
                self.assertIn("libglib2.0-0", text)

    def test_tao_requires_terminal_success_and_checkpoint(self) -> None:
        for path in self.workflow_paths:
            with self.subTest(workflow=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn(
                    'records[-1].get("message") == "Train finished successfully."',
                    text,
                )
                self.assertIn('record.get("status") == "FAILURE"', text)
                self.assertIn('path.suffix in {".pth", ".pt"}', text)
                self.assertIn("TAO reported SUCCESS without a checkpoint", text)
                self.assertIn("docker image inspect", text)
                self.assertIn("runtime_evidence.json", text)
                self.assertIn('"image_digest": os.environ["TAO_IMAGE_DIGEST"]', text)

    def test_tao_feature_levels_match_backbone_outputs(self) -> None:
        spec_path = (
            self.repo_root
            / "configs"
            / "examples"
            / "finetune_external"
            / "tao_dino_runtime_smoke.yaml"
        )
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        self.assertEqual(spec["model"]["num_feature_levels"], 4)

    def test_open_source_qualification_fails_closed_on_evidence_gaps(self) -> None:
        script = (
            self.repo_root / "scripts" / "run_external_runtime_gpu_qualification.sh"
        ).read_text(encoding="utf-8")
        for field in (
            "all_training_executed",
            "all_checkpoints_recorded",
            "all_resource_usage_recorded",
            "all_handoff_contracts_validated",
            "qualification_passed",
        ):
            self.assertIn(field, script)
        self.assertIn('"interface contract" in output_type', script)
        self.assertIn('output_type == "evaluation_report"', script)
        self.assertIn('output_type == "parity_report"', script)
        self.assertIn('raise SystemExit(1)', script)

    def test_runtime_abi_is_re_pinned_and_probed_after_editable_installs(self) -> None:
        script = (
            self.repo_root / "scripts" / "run_external_runtime_gpu_qualification.sh"
        ).read_text(encoding="utf-8")
        editable_install = script.rindex(
            'python3 -m pip install --disable-pip-version-check -e "${MMSEG_ROOT}"'
        )
        final_numpy_pin = script.rindex('"numpy==1.26.4"')
        self.assertGreater(final_numpy_pin, editable_install)
        self.assertIn('"regex==2024.11.6"', script)
        self.assertIn("import xtcocotools._mask", script)
        self.assertIn("torch.from_numpy", script)

    def test_openmmlab_smoke_configs_disable_worker_and_weight_fetches(self) -> None:
        config_root = (
            self.repo_root / "configs" / "examples" / "finetune_external"
        )
        for name in (
            "mmdetection_finetune_smoke.py",
            "mmpose_finetune_smoke.py",
            "mmseg_finetune_smoke.py",
        ):
            with self.subTest(config=name):
                text = (config_root / name).read_text(encoding="utf-8")
                self.assertIn("persistent_workers=False", text)
        for name in (
            "mmpose_finetune_smoke.py",
            "mmseg_finetune_smoke.py",
        ):
            with self.subTest(scratch_config=name):
                text = (config_root / name).read_text(encoding="utf-8")
                self.assertIn("load_from = None", text)
                self.assertIn("model = dict(backbone=dict(init_cfg=None))", text)

    def test_yolox_config_preserves_callable_preprocess_method(self) -> None:
        path = (
            self.repo_root
            / "configs"
            / "examples"
            / "finetune_external"
            / "yolox_s_finetune_smoke.py"
        )
        text = path.read_text(encoding="utf-8")
        self.assertNotIn('self.preprocess = "letterbox"', text)
        self.assertNotIn('self.optimizer = "SGD"', text)

    def test_mmpose_validation_uses_fixture_ground_truth_boxes(self) -> None:
        path = (
            self.repo_root
            / "configs"
            / "examples"
            / "finetune_external"
            / "mmpose_finetune_smoke.py"
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("bbox_file=None", text)
        self.assertIn(
            'ann_file=f"{dataset_root}/annotations/person_keypoints_{split}.json"',
            text,
        )


if __name__ == "__main__":
    unittest.main()
