import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
