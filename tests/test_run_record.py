import unittest
from pathlib import Path
from unittest.mock import patch


class TestRunRecord(unittest.TestCase):
    def test_build_run_record_outside_git(self):
        from yolozu.run_record import build_run_record, validate_run_record_contract

        tmp_root = Path("/tmp") / "yolozu_run_record_test_no_git"
        # Intentionally do not create a .git dir.
        record = build_run_record(
            repo_root=tmp_root,
            argv=["--foo", "bar"],
            args={"x": 1, "image_size": 64},
            dataset_root="/data",
        )

        self.assertIn("versions", record)
        self.assertIn("git", record)
        self.assertEqual(record["argv"], ["--foo", "bar"])
        self.assertEqual(record["args"]["x"], 1)
        self.assertEqual(record["dataset_root"], "/data")
        self.assertIn("schema_version", record)
        self.assertIn("dependency_lock", record)
        self.assertIn("preprocess", record)
        self.assertIn("command", record)
        self.assertIn("runtime", record)
        self.assertIn("hardware", record)
        self.assertIsInstance(record["git"], dict)
        validate_run_record_contract(record, require_git_sha=False)

    def test_validate_run_record_contract_rejects_missing_preprocess(self):
        from yolozu.run_record import build_run_record, validate_run_record_contract

        record = build_run_record(repo_root=Path("."), argv=["--foo"], args={"x": 1}, dataset_root="/data")
        with self.assertRaises(ValueError):
            validate_run_record_contract(record, require_git_sha=False)

    def test_git_info_no_crash(self):
        from yolozu.run_record import git_info

        info = git_info("/tmp")
        self.assertIsInstance(info, dict)

    def test_safe_version_handles_os_error(self):
        from yolozu.run_record import _safe_version

        with patch("builtins.__import__", side_effect=OSError("boom")):
            self.assertIsNone(_safe_version("torch"))

    def test_accelerator_info_handles_runtime_probe_errors(self):
        from yolozu.run_record import accelerator_info

        class FakeCuda:
            @staticmethod
            def is_available():
                raise RuntimeError("boom")

        class FakeCudnn:
            @staticmethod
            def version():
                return 1

            @staticmethod
            def is_available():
                return False

        class FakeMps:
            @staticmethod
            def is_available():
                return False

        class FakeBackends:
            cudnn = FakeCudnn()
            mps = FakeMps()

        class FakeVersion:
            cuda = None

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()
            version = FakeVersion()

        with patch("builtins.__import__", return_value=FakeTorch()):
            info = accelerator_info()
        self.assertTrue(info["torch_available"])
        self.assertFalse(info["cuda"]["available"])


if __name__ == "__main__":
    unittest.main()
