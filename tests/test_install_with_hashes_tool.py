import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "install_with_hashes.py"
SPEC = importlib.util.spec_from_file_location("install_with_hashes", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
install_with_hashes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_with_hashes)


class InstallWithHashesToolTests(unittest.TestCase):
    def test_write_hash_locked_requirements_from_wheels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheel = root / "demo_pkg-1.2.3-py3-none-any.whl"
            wheel.write_bytes(b"fake-wheel")
            out = root / "requirements.lock"

            lines = install_with_hashes._write_hash_locked_requirements([wheel], out)

            digest = hashlib.sha256(b"fake-wheel").hexdigest()
            self.assertEqual(lines, [f"demo-pkg==1.2.3 --hash=sha256:{digest}"])
            self.assertEqual(out.read_text(encoding="utf-8"), f"demo-pkg==1.2.3 --hash=sha256:{digest}\n")

    def test_parser_rejects_empty_action(self):
        with self.assertRaises(SystemExit):
            install_with_hashes.main([])

    def test_parser_supports_index_urls(self):
        parser = install_with_hashes.build_parser()
        args = parser.parse_args(
            [
                "--requirements",
                "requirements-locks/requirements-runtime.lock",
                "--index-url",
                "https://example.com/simple",
                "--extra-index-url",
                "https://download.example.com/simple",
                "--extra-index-url",
                "https://mirror.example.com/simple",
            ]
        )
        self.assertEqual(args.index_url, "https://example.com/simple")
        self.assertEqual(
            args.extra_index_url,
            [
                "https://download.example.com/simple",
                "https://mirror.example.com/simple",
            ],
        )

    def test_install_hash_locked_requirements_uses_ignore_installed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            req = root / "req.lock"
            req.write_text("demo==1.0\n", encoding="utf-8")
            fake_wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
            fake_wheel.write_bytes(b"wheel")

            with patch.object(install_with_hashes, "_download_exact_requirements", return_value=[fake_wheel]):
                with patch.object(install_with_hashes, "_run") as run_mock:
                    install_with_hashes._install_hash_locked_requirements("python3", [req], wheelhouse)

            cmd = run_mock.call_args[0][0]
            self.assertIn("--ignore-installed", cmd)

    def test_install_local_wheel_uses_ignore_installed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            fake_wheel = wheelhouse / "demo_pkg-1.2.3-py3-none-any.whl"
            fake_wheel.write_bytes(b"fake-wheel")

            with patch.object(install_with_hashes, "_build_local_wheel", return_value=fake_wheel):
                with patch.object(install_with_hashes, "_run") as run_mock:
                    install_with_hashes._install_local_wheel("python3", root, wheelhouse)

            cmd = run_mock.call_args[0][0]
            self.assertIn("--ignore-installed", cmd)


if __name__ == "__main__":
    unittest.main()
