import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class TestExternalInferenceTemplatesSmoke(unittest.TestCase):
    def test_cpp_stub_emits_strict_valid_predictions_when_compiler_available(self):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++ compiler available")

        repo_root = Path(__file__).resolve().parents[1]
        src = repo_root / "examples" / "infer_cpp" / "src" / "main_stub.cpp"
        inc = repo_root / "examples" / "infer_cpp" / "src"
        self.assertTrue(src.is_file(), "missing C++ stub source")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            bin_path = root / "yolozu_infer_stub"
            out = root / "predictions.json"

            proc = subprocess.run(
                [compiler, "-std=c++17", f"-I{inc}", str(src), "-o", str(bin_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"C++ stub compile failed:\n{proc.stdout}\n{proc.stderr}")

            proc2 = subprocess.run(
                [str(bin_path), "--help"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc2.returncode, 0, msg=f"C++ --help failed:\n{proc2.stdout}\n{proc2.stderr}")

            proc2b = subprocess.run(
                [str(bin_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc2b.returncode, 2, msg=f"C++ missing-args behavior changed:\n{proc2b.stdout}\n{proc2b.stderr}")

            proc2 = subprocess.run(
                [str(bin_path), "--image", "/abs/path.jpg", "--output", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc2.returncode != 0:
                self.fail(f"C++ stub run failed:\n{proc2.stdout}\n{proc2.stderr}")

            proc3 = subprocess.run(
                [sys.executable, str(repo_root / "tools" / "validate_predictions.py"), str(out), "--strict"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc3.returncode != 0:
                self.fail(f"validate_predictions.py failed:\n{proc3.stdout}\n{proc3.stderr}")

    def test_rust_stub_emits_strict_valid_predictions_when_cargo_available(self):
        if shutil.which("cargo") is None:
            self.skipTest("cargo not available")

        repo_root = Path(__file__).resolve().parents[1]
        proj = repo_root / "examples" / "infer_rust" / "Cargo.toml"
        self.assertTrue(proj.is_file(), "missing Rust stub Cargo.toml")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "predictions.json"

            proc = subprocess.run(
                ["cargo", "build", "--release", "--manifest-path", str(proj), "-q"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"cargo build failed:\n{proc.stdout}\n{proc.stderr}")

            bin_path = repo_root / "examples" / "infer_rust" / "target" / "release" / "yolozu_infer_rust"
            if not bin_path.exists():
                self.fail(f"missing built binary: {bin_path}")

            proc2 = subprocess.run(
                [str(bin_path), "--help"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc2.returncode, 2, msg=f"rust --help behavior changed:\n{proc2.stdout}\n{proc2.stderr}")

            proc2b = subprocess.run(
                [str(bin_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc2b.returncode, 2, msg=f"rust missing-args behavior changed:\n{proc2b.stdout}\n{proc2b.stderr}")

            proc2 = subprocess.run(
                [str(bin_path), "--out", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc2.returncode != 0:
                self.fail(f"rust stub run failed:\n{proc2.stdout}\n{proc2.stderr}")

            proc3 = subprocess.run(
                [sys.executable, str(repo_root / "tools" / "validate_predictions.py"), str(out), "--strict"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc3.returncode != 0:
                self.fail(f"validate_predictions.py failed:\n{proc3.stdout}\n{proc3.stderr}")

            proc4 = subprocess.run(
                [
                    str(bin_path),
                    "--mode",
                    "onnxrt",
                    "--onnx",
                    "/tmp/model.onnx",
                    "--out",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc4.returncode, 2, msg=f"rust onnxrt fallback behavior changed:\n{proc4.stdout}\n{proc4.stderr}")
            self.assertIn("without the 'onnxruntime' feature", proc4.stderr)

    def test_rust_template_declares_optional_onnxruntime_feature(self):
        repo_root = Path(__file__).resolve().parents[1]
        cargo_toml = repo_root / "examples" / "infer_rust" / "Cargo.toml"
        self.assertTrue(cargo_toml.is_file(), f"missing Rust template Cargo.toml: {cargo_toml}")

        parsed = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
        features = parsed.get("features") or {}

        self.assertIn("onnxruntime", features, "expected onnxruntime feature in Rust template")
        self.assertEqual(features.get("default"), [], "Rust template default features should stay empty")

    def test_external_inference_docs_pin_production_lane_handoff(self):
        repo_root = Path(__file__).resolve().parents[1]
        docs = [
            repo_root / "docs" / "external_inference.md",
            repo_root / "examples" / "infer_cpp" / "README.md",
            repo_root / "examples" / "infer_rust" / "README.md",
        ]

        for path in docs:
            with self.subTest(path=path.relative_to(repo_root)):
                text = path.read_text(encoding="utf-8")
                self.assertIn("Production lane interface contract", text)
                self.assertIn("predictions interface contract", text)
                self.assertIn("Error behavior:", text)
                self.assertIn("python3 tools/validate_predictions.py /path/to/predictions.json --strict", text)
                self.assertIn("yolozu eval-coco", text)
                self.assertIn(
                    "python3 tools/check_predictions_parity.py "
                    "--reference reports/pred_torch.json "
                    "--candidate /path/to/predictions.json "
                    "> reports/external_parity.json",
                    text,
                )
                self.assertIn("reports/external_parity.json", text)

    def test_rust_onnxruntime_mode_declares_decode_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        src = (repo_root / "examples" / "infer_rust" / "src" / "main.rs").read_text(encoding="utf-8")
        readme = (repo_root / "examples" / "infer_rust" / "README.md").read_text(encoding="utf-8")

        self.assertIn("--combined-format", src)
        self.assertIn("xyxy_score_class", src)
        self.assertIn("decoded_detections_per_image", src)
        self.assertIn("xyxy_score_class", readme)
        self.assertNotIn("emits empty detections with backend metadata", readme)

    def test_rust_onnxruntime_embedded_decode_filters_scales_and_preserves_images(self):
        try:
            import numpy  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable for embedded decode test: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        src = (repo_root / "examples" / "infer_rust" / "src" / "main.rs").read_text(encoding="utf-8")
        start_marker = 'let py = r#"\n'
        end_marker = '\n"#;'
        start = src.index(start_marker) + len(start_marker)
        end = src.index(end_marker, start)
        embedded_py = src[start:end]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            fake_runtime = root / "onnxruntime.py"
            fake_runtime.write_text(
                "\n".join(
                    [
                        "class _Input:",
                        "    name = 'input'",
                        "",
                        "class InferenceSession:",
                        "    def __init__(self, path, providers=None):",
                        "        self.path = path",
                        "        self.providers = providers or []",
                        "    def get_inputs(self):",
                        "        return [_Input()]",
                        "    def run(self, _unused, inputs):",
                        "        import numpy as np",
                        "        assert 'input' in inputs",
                        "        return [np.array([",
                        "            [0.0, 0.0, 32.0, 16.0, 0.90, 3.0],",
                        "            [16.0, 16.0, 32.0, 64.0, 0.80, 2.0],",
                        "            [4.0, 4.0, 12.0, 12.0, 0.20, 4.0],",
                        "        ], dtype=np.float32)]",
                    ]
                ),
                encoding="utf-8",
            )
            images = ['images/val/quote"image.jpg', r"images\val\backslash.jpg"]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    embedded_py,
                    "model.onnx",
                    "1,3,64,64",
                    json.dumps(images),
                    "xyxy_score_class",
                    "abs",
                    "64x64",
                    "0.5",
                    "1",
                ],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"embedded Rust ONNXRuntime decode failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(proc.stdout)
            self.assertEqual([item["image"] for item in payload["predictions"]], images)
            self.assertEqual(payload["meta"]["backend"], "onnxruntime-rust")
            extra = payload["meta"]["extra"]
            self.assertEqual(extra["combined_format"], "xyxy_score_class")
            self.assertEqual(extra["boxes_scale"], "abs")
            self.assertEqual(extra["input_size"], [64, 64])
            self.assertEqual(extra["output_shapes"], [[3, 6]])
            self.assertEqual(extra["decoded_detections_per_image"], 1)

            detections = payload["predictions"][0]["detections"]
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["class_id"], 3)
            self.assertAlmostEqual(detections[0]["score"], 0.9, places=5)
            self.assertAlmostEqual(detections[0]["bbox"]["cx"], 0.25, places=6)
            self.assertAlmostEqual(detections[0]["bbox"]["cy"], 0.125, places=6)
            self.assertAlmostEqual(detections[0]["bbox"]["w"], 0.5, places=6)
            self.assertAlmostEqual(detections[0]["bbox"]["h"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
