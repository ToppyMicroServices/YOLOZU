import hashlib
import importlib.resources
import unittest

from yolozu.inference.export_orchestrator import config_sha256


class TestExportConfigFingerprint(unittest.TestCase):
    def test_builtin_config_hashes_effective_packaged_content(self):
        content = (
            importlib.resources.files("rtdetr_pose")
            .joinpath("configs/base.json")
            .read_bytes()
        )
        self.assertEqual(
            config_sha256("builtin:base"),
            hashlib.sha256(content).hexdigest(),
        )
        self.assertEqual(
            config_sha256("pkg:base"),
            hashlib.sha256(content).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
