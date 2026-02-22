import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
