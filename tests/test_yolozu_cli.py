import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestYOLOZUCLI(unittest.TestCase):
    def test_help_lists_continual_commands(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("continual-train", proc.stdout)
        self.assertIn("continual-eval", proc.stdout)
        self.assertIn("long-tail-recipe", proc.stdout)

    def test_legacy_wrapper_forwards_supported_package_commands(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        help_proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if help_proc.returncode != 0:
            self.fail(f"tools/yolozu.py --help failed:\n{help_proc.stdout}\n{help_proc.stderr}")
        for cmd in ("eval-coco", "benchmark", "parity", "validate", "train"):
            self.assertIn(cmd, help_proc.stdout)

        forwarded_proc = subprocess.run(
            [sys.executable, str(script), "eval-coco", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if forwarded_proc.returncode != 0:
            self.fail(f"tools/yolozu.py eval-coco --help failed:\n{forwarded_proc.stdout}\n{forwarded_proc.stderr}")
        self.assertIn("COCOeval", forwarded_proc.stdout)
        self.assertIn("--predictions", forwarded_proc.stdout)

    def test_guide_prints_beginner_routes_and_json(self):
        repo_root = Path(__file__).resolve().parents[1]

        text_proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "guide", "--goal", "evaluate"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if text_proc.returncode != 0:
            self.fail(f"yolozu guide failed:\n{text_proc.stdout}\n{text_proc.stderr}")
        self.assertIn("YOLOZU guide", text_proc.stdout)
        self.assertIn("validate predictions", text_proc.stdout)
        self.assertIn("eval-coco", text_proc.stdout)
        self.assertIn("docs/predictions_schema.md", text_proc.stdout)

        first_proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "guide", "--goal", "first-run"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if first_proc.returncode != 0:
            self.fail(f"yolozu guide --goal first-run failed:\n{first_proc.stdout}\n{first_proc.stderr}")
        self.assertIn("demo instance-seg", first_proc.stdout)
        self.assertIn("overlays/overlay_img_0000.png", first_proc.stdout)

        json_proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "guide", "--goal", "debug", "--json"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if json_proc.returncode != 0:
            self.fail(f"yolozu guide --json failed:\n{json_proc.stdout}\n{json_proc.stderr}")
        payload = json.loads(json_proc.stdout)
        self.assertEqual(payload["kind"], "yolozu_guide")
        self.assertEqual(payload["goal"], "debug")
        self.assertIn("doctor", " ".join(payload["routes"]["debug"]["commands"]))
        for route in payload["routes"].values():
            for doc in route["docs"]:
                self.assertTrue((repo_root / doc).is_file(), f"guide references missing doc: {doc}")

    def test_legacy_wrapper_forwards_guide(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "guide", "--goal", "first-run"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"tools/yolozu.py guide failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("demo instance-seg", proc.stdout)
        self.assertIn("doctor --proof", proc.stdout)

    def test_doctor_proof_writes_artifacts_and_compares_metrics(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            doctor_out = root / "doctor.json"
            proof_dir = root / "doctor_proof"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "doctor",
                    "--proof",
                    "--output",
                    str(doctor_out),
                    "--proof-dir",
                    str(proof_dir),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu doctor --proof failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(doctor_out.is_file())
            payload = json.loads(doctor_out.read_text(encoding="utf-8"))
            proof = payload.get("proof") or {}
            self.assertEqual(proof.get("status"), "pass")
            self.assertEqual(proof.get("warnings"), [])
            self.assertEqual((proof.get("observed_metrics") or {}).get("map50"), 1.0)
            self.assertEqual((proof.get("observed_metrics") or {}).get("map50_95"), 1.0)
            artifacts = proof.get("artifacts") or {}
            for key in ("dataset", "predictions", "eval_report", "proof_report"):
                self.assertTrue(Path(str(artifacts.get(key))).exists(), f"missing proof artifact: {key}")

    def test_demo_overview_points_to_visible_png_quickstart(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out_path = Path(td) / "overview.json"
            proc = subprocess.run(
                [sys.executable, str(script), "demo", "overview", "--output", str(out_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"tools/yolozu.py demo overview failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("visible quickstart", proc.stdout)
            self.assertIn("demo instance-seg", proc.stdout)
            self.assertIn("overlays/overlay_img_0000.png", proc.stdout)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("visible_quickstart", payload)

    def test_packaged_quickstart_settings_are_listed(self):
        repo_root = Path(__file__).resolve().parents[1]

        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "resources", "list"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu resources list failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("configs/quickstart/instance_seg_demo.yaml", proc.stdout)
        self.assertIn("configs/quickstart/predict_images_dummy.yaml", proc.stdout)

    def test_train_help_lists_external_backends(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "train", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu train --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--external-backend", proc.stdout)
        self.assertIn("yolox", proc.stdout)
        self.assertIn("--dataset-root", proc.stdout)
        self.assertIn("--records-json", proc.stdout)
        self.assertIn("--val-records-json", proc.stdout)
        self.assertIn("detectron2", proc.stdout)
        self.assertIn("mmdetection", proc.stdout)
        self.assertIn("mmpose", proc.stdout)
        self.assertIn("mmseg", proc.stdout)
        self.assertIn("tao", proc.stdout)
        self.assertIn("ultralytics", proc.stdout)
        self.assertIn("hf-detr", proc.stdout)

    def test_train_orchestrate_help_lists_core_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "train-orchestrate", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu train-orchestrate --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--spec", proc.stdout)
        self.assertIn("--execute", proc.stdout)
        self.assertIn("--registry-out", proc.stdout)

    def test_dataset_training_help_lists_preflight_and_keypoints_lane(self):
        repo_root = Path(__file__).resolve().parents[1]
        commands = [
            ["doctor", "train-dataset", "--help"],
            ["doctor", "import", "--help"],
            ["import", "dataset", "--help"],
            ["migrate", "dataset", "--help"],
        ]
        for args in commands:
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", *args],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("coco-keypoints", proc.stdout)
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "doctor", "train-dataset", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertIn("--records-json", proc.stdout)
        self.assertIn("--val-records-json", proc.stdout)

    def test_completion_help_lists_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "completion", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"completion --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--shell", proc.stdout)
        self.assertIn("--command", proc.stdout)
        self.assertIn("--output", proc.stdout)
        self.assertIn("-s", proc.stdout)
        self.assertIn("-c", proc.stdout)
        self.assertIn("-o", proc.stdout)

    def test_completion_bash_stdout_contains_complete_directive(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "completion", "--shell", "bash", "--command", "yolozu"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"completion bash failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("complete -F", proc.stdout)
        self.assertIn(" yolozu", proc.stdout)

    def test_completion_zsh_stdout_contains_compdef(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "completion", "--shell", "zsh", "--command", "yolozu"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"completion zsh failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("#compdef yolozu", proc.stdout)
        self.assertIn("compdef", proc.stdout)

    def test_doctor_writes_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "doctor.json"
            proc = subprocess.run(
                [sys.executable, str(script), "dr", "-o", str(out_path.relative_to(repo_root))],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertIn(proc.returncode, (0, 1), f"unexpected doctor exit code: {proc.returncode}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text())
            self.assertIn("timestamp", payload)
            self.assertIn("gpu", payload)
            self.assertIn("env", payload)
            self.assertIn("runtime_capabilities", payload)
            self.assertIn("drift_hints", payload)
            self.assertIsInstance(payload.get("drift_hints"), list)
            runtime = payload.get("runtime_capabilities") or {}
            torch_runtime = runtime.get("torch") or {}
            ort_runtime = runtime.get("onnxruntime") or {}
            self.assertIn("mps_available", torch_runtime)
            self.assertIn("mps_built", torch_runtime)
            self.assertIn("coreml_provider", ort_runtime)
            links = payload.get("guidance_links") or {}
            self.assertIn("backend_parity", links)
            self.assertIn("onnx_parity", links)
            if proc.returncode == 1:
                errors = payload.get("errors") or []
                self.assertTrue(errors, "doctor exit code 1 should include errors")

    def test_doctor_explain_prints_human_summary_and_preserves_json_output(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "doctor.json"
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "doctor", "--explain", "--output", str(out_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertIn(proc.returncode, (0, 1), f"unexpected doctor --explain exit code: {proc.returncode}")
            self.assertIn("YOLOZU doctor explanation", proc.stdout)
            self.assertIn("Status:", proc.stdout)
            self.assertIn("Recommended next commands:", proc.stdout)
            self.assertIn("yolozu guide --goal first-run", proc.stdout)
            self.assertTrue(out_path.is_file(), "doctor --explain should still write JSON to --output")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "yolozu_doctor")

        json_proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "doctor", "--output", "-"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertIn(json_proc.returncode, (0, 1), f"unexpected doctor JSON exit code: {json_proc.returncode}")
        payload = json.loads(json_proc.stdout)
        self.assertEqual(payload.get("kind"), "yolozu_doctor")
        self.assertNotIn("YOLOZU doctor explanation", json_proc.stdout)

    def test_support_ultralytics_detr_alias_forwards(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "sud", "ls", "-j"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu sud ls -j failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        layers = payload.get("layers") or {}
        self.assertIn("trainer_runner", layers)

    def test_long_tail_recipe_help_lists_pytorch_options(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "long-tail-recipe", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"long-tail-recipe --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--metric-plugin", proc.stdout)
        self.assertIn("--lr-scheduler", proc.stdout)
        self.assertIn("torch_cross_entropy", proc.stdout)
        self.assertIn("torch_reduce_on_plateau", proc.stdout)

    def test_export_rejects_invalid_topk(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        dataset = repo_root / "data" / "smoke"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "export",
                "--backend",
                "dummy",
                "--dataset",
                str(dataset),
                "--split",
                "val",
                "--topk",
                "0",
                "--output",
                "reports/_tmp_invalid_topk.json",
                "--force",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--topk must be >= 1", proc.stderr)

    def test_export_rejects_negative_max_images(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        dataset = repo_root / "data" / "smoke"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "export",
                "--backend",
                "dummy",
                "--dataset",
                str(dataset),
                "--split",
                "val",
                "--max-images",
                "-1",
                "--output",
                "reports/_tmp_invalid_max_images.json",
                "--force",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--max-images must be >= 0", proc.stderr)

    def test_export_dummy_injects_run_meta(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            # Dummy adapter doesn't open images; an empty file is fine.
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            out_path = root / "preds.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "dummy",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu export failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("predictions", payload)
            meta = payload.get("meta") or {}
            self.assertIn("run", meta)
            run = meta.get("run") or {}
            self.assertIn("config_hash", run)
            self.assertIn("git", run)
            self.assertIn("env", run)

    def test_export_executorch_dry_run_succeeds(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = repo_root / "data" / "smoke"
            model_path = root / "dummy.pte"
            out_path = root / "preds.json"
            model_path.write_bytes(b"dummy")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "executorch",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val",
                    "--model",
                    str(model_path),
                    "--dry-run",
                    "--output",
                    str(out_path),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu export --backend executorch failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("predictions", payload)
            meta = payload.get("meta", {})
            self.assertEqual(meta.get("adapter"), "executorch")
            self.assertEqual((meta.get("extra") or {}).get("exporter"), "executorch")

    def test_export_executorch_runtime_output_decode_succeeds(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = repo_root / "data" / "smoke"
            model_path = root / "dummy.pte"
            runtime_output = root / "executorch_runtime_outputs.json"
            run_dir = root / "run"
            model_path.write_bytes(b"dummy")
            runtime_output.write_text(
                json.dumps({"000000000009.jpg": [[0.1, 0.2, 0.5, 0.7, 0.9, 3]]}),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "executorch",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "val",
                    "--max-images",
                    "1",
                    "--model",
                    str(model_path),
                    "--runtime-output-json",
                    str(runtime_output),
                    "--boxes-scale",
                    "norm",
                    "--run-dir",
                    str(run_dir),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu export --backend executorch runtime decode failed:\n{proc.stdout}\n{proc.stderr}")
            out_path = Path(proc.stdout.strip().splitlines()[-1])
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            entries = payload.get("predictions") or []
            self.assertEqual(entries[0]["detections"][0]["class_id"], 3)
            extra = payload.get("meta", {}).get("extra", {})
            self.assertEqual(extra.get("runtime_decode", {}).get("contract"), "combined_xyxy_score_class")
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            config_fp = run_config.get("config_fingerprint", {})
            self.assertEqual(config_fp.get("runtime_output_json"), str(runtime_output))
            self.assertIsInstance(config_fp.get("runtime_output_json_sha256"), str)
            self.assertEqual(config_fp.get("boxes_scale"), "norm")

    def test_export_rejects_torch_only_flags_on_yolox(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        dataset = repo_root / "data" / "smoke"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "export",
                "--backend",
                "yolox",
                "--dataset",
                str(dataset),
                "--split",
                "val",
                "--tta",
                "--torch-compile",
                "--infer-batch-size",
                "2",
                "--output",
                "reports/_tmp_invalid_torch_only_flags.json",
                "--force",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--tta/--ttt/--lora-* are only supported", proc.stderr)
        self.assertIn("--torch-compile*", proc.stderr)

    def test_export_opencv_dnn_dry_run_respects_max_images(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            (images / "000001.jpg").write_bytes(b"")
            (images / "000002.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (labels / "000002.txt").write_text("0 0.4 0.4 0.2 0.2\n")

            onnx_path = root / "dummy.onnx"
            onnx_path.write_bytes(b"dummy")
            out_path = root / "preds.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "opencv-dnn",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--onnx",
                    str(onnx_path),
                    "--max-images",
                    "1",
                    "--dry-run",
                    "--output",
                    str(out_path),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu export --backend opencv-dnn --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            preds = payload.get("predictions") or []
            self.assertEqual(len(preds), 1)

    def test_export_cache_reuses_by_fingerprint(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")

            cache_dir = root / "cache"

            cmd = [
                sys.executable,
                str(script),
                "export",
                "--backend",
                "dummy",
                "--dataset",
                str(dataset_root),
                "--split",
                "train2017",
                "--max-images",
                "1",
                "--cache",
                "--cache-dir",
                str(cache_dir),
            ]
            proc1 = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc1.returncode != 0:
                self.fail(f"yolozu export --cache failed:\n{proc1.stdout}\n{proc1.stderr}")
            out_path = Path(proc1.stdout.strip().splitlines()[-1])
            self.assertTrue(out_path.is_file())

            payload1 = json.loads(out_path.read_text())
            ts1 = payload1.get("meta", {}).get("run", {}).get("timestamp")
            self.assertIsInstance(ts1, str)

            run_cfg = out_path.parent / "run_config.json"
            self.assertTrue(run_cfg.is_file())
            cfg_payload = json.loads(run_cfg.read_text())
            self.assertEqual(cfg_payload.get("config_hash"), payload1.get("meta", {}).get("run", {}).get("config_hash"))

            proc2 = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc2.returncode != 0:
                self.fail(f"yolozu export --cache (2nd run) failed:\n{proc2.stdout}\n{proc2.stderr}")
            out_path2 = Path(proc2.stdout.strip().splitlines()[-1])
            self.assertEqual(out_path2, out_path)

            payload2 = json.loads(out_path2.read_text())
            ts2 = payload2.get("meta", {}).get("run", {}).get("timestamp")
            self.assertEqual(ts2, ts1)

    def test_sweep_wrapper_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        proc = subprocess.run(
            [sys.executable, str(script), "sweep", "--config", "docs/hpo_sweep_example.json", "--dry-run", "--max-runs", "1"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"yolozu sweep failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("python3 tools/mock_train.py", proc.stdout)

    def test_predict_images_dummy_writes_overlays_and_html(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        pil_image = None
        try:
            from PIL import Image as pil_image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"PIL not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            input_dir = root / "images"
            input_dir.mkdir(parents=True, exist_ok=True)
            img_path = input_dir / "a.jpg"
            self.assertIsNotNone(pil_image)
            pil_image.new("RGB", (16, 16), color=(0, 0, 0)).save(img_path)

            out_path = root / "preds.json"
            overlays_dir = root / "overlays"
            html_path = root / "report.html"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "predict-images",
                    "--backend",
                    "dummy",
                    "--input-dir",
                    str(input_dir),
                    "--max-images",
                    "1",
                    "--output",
                    str(out_path),
                    "--overlays-dir",
                    str(overlays_dir),
                    "--html",
                    str(html_path),
                    "--progress",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu predict-images failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text())
            preds = payload.get("predictions") or []
            self.assertEqual(len(preds), 1)
            self.assertEqual(Path(preds[0]["image"]), img_path)

            self.assertTrue(html_path.is_file())
            self.assertTrue(overlays_dir.is_dir())
            overlays = list(overlays_dir.glob("*.png"))
            self.assertTrue(overlays, "expected at least one overlay image")
            self.assertIn("overlays_dir:", proc.stdout)
            self.assertIn("first_overlay:", proc.stdout)
            self.assertIn(str(overlays[0]), proc.stdout)
            self.assertIn("predict-images", proc.stderr)
            self.assertIn("render overlays", proc.stderr)

    def test_eval_instance_seg_demo_writes_html_and_overlays(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        try:
            import numpy as _  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        dataset_root = repo_root / "examples" / "instance_seg_demo" / "dataset"
        preds_path = repo_root / "examples" / "instance_seg_demo" / "predictions" / "instance_seg_predictions.json"
        pred_root = repo_root / "examples" / "instance_seg_demo" / "predictions"
        classes_path = repo_root / "examples" / "instance_seg_demo" / "classes.txt"
        if not (dataset_root.is_dir() and preds_path.is_file()):
            self.skipTest("instance_seg_demo missing")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_json = root / "instance_seg_eval.json"
            out_html = root / "instance_seg_eval.html"
            overlays_dir = root / "overlays"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "eval-instance-seg",
                    "--dataset",
                    str(dataset_root.relative_to(repo_root)),
                    "--split",
                    "val2017",
                    "--predictions",
                    str(preds_path.relative_to(repo_root)),
                    "--pred-root",
                    str(pred_root.relative_to(repo_root)),
                    "--classes",
                    str(classes_path.relative_to(repo_root)),
                    "--output",
                    str(out_json.relative_to(repo_root)),
                    "--html",
                    str(out_html.relative_to(repo_root)),
                    "--overlays-dir",
                    str(overlays_dir.relative_to(repo_root)),
                    "--max-overlays",
                    "1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu eval-instance-seg failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_json.is_file())
            report = json.loads(out_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("map50")), 1.0, places=6)

            self.assertTrue(out_html.is_file())
            self.assertTrue(overlays_dir.is_dir())
            overlays = list(overlays_dir.glob("*.png"))
            self.assertTrue(overlays, "expected at least one overlay image")

    def test_calibrate_auto_detects_seg_task(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_seg.json"
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "instances": [
                                    {"class_id": 0, "score": 0.7, "mask": "masks/a.png"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_preds = root / "pred_seg_calibrated.json"
            out_report = root / "calib_report_seg.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "calibrate",
                    "--method",
                    "fracal",
                    "--task",
                    "auto",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(preds_in),
                    "--output",
                    str(out_preds),
                    "--output-report",
                    str(out_report),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu calibrate --task auto (seg) failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_preds.is_file())
            self.assertTrue(out_report.is_file())
            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report.get("task"), "seg")
            self.assertEqual((report.get("calibration") or {}).get("method"), "fracal")

    def test_calibrate_rejects_invalid_stats_in_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_bbox.json"
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "detections": [
                                    {"class_id": 0, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bad_stats = root / "bad_stats.json"
            bad_stats.write_text(json.dumps({"schema_version": 1, "method": "fracal"}), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "calibrate",
                    "--method",
                    "fracal",
                    "--task",
                    "bbox",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(preds_in),
                    "--stats-in",
                    str(bad_stats),
                    "--output",
                    str(root / "pred_bbox_calibrated.json"),
                    "--output-report",
                    str(root / "calib_report_bbox.json"),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("invalid stats file", proc.stderr)

    def test_calibrate_pose_preserves_keypoints_and_sets_task(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_pose.json"
            keypoints = [[0.1, 0.2, 0.9], [0.4, 0.5, 0.8]]
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.8,
                                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                                        "keypoints": keypoints,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_preds = root / "pred_pose_calibrated.json"
            out_report = root / "calib_report_pose.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "calibrate",
                    "--method",
                    "fracal",
                    "--task",
                    "pose",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(preds_in),
                    "--output",
                    str(out_preds),
                    "--output-report",
                    str(out_report),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu calibrate --task pose failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_preds.read_text(encoding="utf-8"))
            det = payload["predictions"][0]["detections"][0]
            self.assertEqual(det.get("keypoints"), keypoints)

            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report.get("task"), "pose")

    def test_calibrate_auto_detects_pose_task(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_pose_auto.json"
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.8,
                                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                                        "keypoints": [[0.1, 0.2, 0.9]],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_report = root / "calib_report_pose_auto.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "calibrate",
                    "--method",
                    "fracal",
                    "--task",
                    "auto",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(preds_in),
                    "--output",
                    str(root / "pred_pose_auto_calibrated.json"),
                    "--output-report",
                    str(out_report),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu calibrate --task auto (pose) failed:\n{proc.stdout}\n{proc.stderr}")

            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report.get("task"), "pose")

    def test_calibrate_supports_la_and_norcal_methods(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_bbox.json"
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "detections": [
                                    {"class_id": 0, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            for method in ("la", "norcal"):
                out_preds = root / f"pred_bbox_{method}.json"
                out_report = root / f"calib_report_{method}.json"
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "calibrate",
                        "--method",
                        method,
                        "--task",
                        "bbox",
                        "--dataset",
                        str(dataset_root),
                        "--split",
                        "train2017",
                        "--predictions",
                        str(preds_in),
                        "--output",
                        str(out_preds),
                        "--output-report",
                        str(out_report),
                        "--force",
                    ],
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                )
                if proc.returncode != 0:
                    self.fail(f"yolozu calibrate --method {method} failed:\n{proc.stdout}\n{proc.stderr}")

                report = json.loads(out_report.read_text(encoding="utf-8"))
                self.assertEqual(report.get("method"), method)
                self.assertEqual((report.get("calibration") or {}).get("method"), method)

    def test_calibrate_supports_temperature_method(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "train2017"
            labels = dataset_root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            preds_in = root / "pred_bbox.json"
            preds_in.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "predictions": [
                            {
                                "image": "000001.jpg",
                                "detections": [
                                    {"class_id": 0, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_report = root / "calib_report_temperature.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "calibrate",
                    "--method",
                    "temperature",
                    "--temperature",
                    "1.5",
                    "--task",
                    "bbox",
                    "--dataset",
                    str(dataset_root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(preds_in),
                    "--output",
                    str(root / "pred_bbox_temperature.json"),
                    "--output-report",
                    str(out_report),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu calibrate --method temperature failed:\n{proc.stdout}\n{proc.stderr}")

            report = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(report.get("method"), "temperature")
            self.assertEqual((report.get("calibration") or {}).get("method"), "temperature")

    def test_list_models_with_custom_registry(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-model",
                                "summary": "toy",
                                "family": "test",
                                "source": {"type": "official_url", "url": "file:///tmp/toy.bin"},
                                "version": "v1",
                                "license": "Apache-2.0",
                                "sha256": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "list",
                    "models",
                    "--registry",
                    str(registry_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu list models failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("toy-model", proc.stdout)

    def test_fetch_requires_license_and_uses_cache(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"abc123")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-model",
                                "summary": "toy",
                                "family": "test",
                                "source": {"type": "official_url", "url": src.resolve().as_uri()},
                                "version": "v1",
                                "license": "Apache-2.0",
                                "sha256": sha,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "models"
            cache_dir = root / "cache"

            no_accept = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(no_accept.returncode, 0)
            self.assertIn("--accept-license", no_accept.stderr)

            yes_accept = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if yes_accept.returncode != 0:
                self.fail(f"yolozu fetch failed:\n{yes_accept.stdout}\n{yes_accept.stderr}")
            fetched = out_dir / "toy-model" / "weights.bin"
            meta = out_dir / "toy-model" / "meta.json"
            self.assertTrue(fetched.is_file())
            self.assertTrue(meta.is_file())
            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("sha256"), sha)
            self.assertEqual(payload.get("source"), "official_url")

            src.unlink()
            out_dir_2 = root / "models2"
            cached_run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir_2),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if cached_run.returncode != 0:
                self.fail(f"yolozu fetch (cache) failed:\n{cached_run.stdout}\n{cached_run.stderr}")
            self.assertTrue((out_dir_2 / "toy-model" / "weights.bin").is_file())

    def test_fetch_requires_sha256_unless_allow_unsafe(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"abc123")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()

            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-model",
                                "summary": "toy",
                                "family": "test",
                                "source": {"type": "official_url", "url": src.resolve().as_uri()},
                                "version": "v1",
                                "license": "Apache-2.0",
                                "sha256": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "models"
            cache_dir = root / "cache"

            no_sha = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(no_sha.returncode, 0)
            self.assertIn("--allow-unsafe", no_sha.stderr)

            yes_sha = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                    "--allow-unsafe",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if yes_sha.returncode != 0:
                self.fail(f"yolozu fetch --allow-unsafe failed:\n{yes_sha.stdout}\n{yes_sha.stderr}")
            meta = out_dir / "toy-model" / "meta.json"
            self.assertTrue(meta.is_file())
            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("sha256"), sha)

    def test_fetch_requires_allow_non_apache_for_copyleft_license(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"abc123")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()

            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-model",
                                "summary": "toy",
                                "family": "test",
                                "source": {"type": "official_url", "url": src.resolve().as_uri()},
                                "version": "v1",
                                "license": "AGPL-3.0",
                                "sha256": sha,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "models"
            cache_dir = root / "cache"

            blocked = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("--allow-non-apache", blocked.stderr)

            allowed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "fetch",
                    "toy-model",
                    "--registry",
                    str(registry_path),
                    "--out",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--accept-license",
                    "--allow-non-apache",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if allowed.returncode != 0:
                self.fail(f"yolozu fetch --allow-non-apache failed:\n{allowed.stdout}\n{allowed.stderr}")


if __name__ == "__main__":
    unittest.main()
