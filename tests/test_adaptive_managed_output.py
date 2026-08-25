from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

from yolozu.adaptive.managed_output import (
    ManagedOutputError,
    ManagedOutputLimits,
    ManagedOutputTransaction,
    recover_managed_output,
)


LIMITS = ManagedOutputLimits(
    max_files=8,
    max_file_bytes=1024 * 1024,
    max_total_bytes=4 * 1024 * 1024,
)


def _publish(root: Path, destination: str, files: dict[str, bytes], *, force: bool = False) -> None:
    with ManagedOutputTransaction(
        root=root,
        destination=destination,
        declared_paths=files,
        limits=LIMITS,
        force=force,
    ) as transaction:
        for name, data in files.items():
            transaction.write_bytes(name, data)
        transaction.commit()


class _InjectedFailure(RuntimeError):
    pass


class ManagedOutputTransactionTests(TestCase):
    def test_publish_builds_exact_manifest_and_reports_scoped_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("predictions.json", "masks/0001.bin"),
                limits=LIMITS,
            ) as transaction:
                transaction.write_bytes("masks/0001.bin", b"mask")
                transaction.write_bytes("predictions.json", b"{}\n")
                capabilities = transaction.commit()
                self.assertTrue(transaction.published)
                self.assertTrue(capabilities.directory_relative_no_follow)
                self.assertTrue(capabilities.same_filesystem_rename_probed)
                self.assertEqual(capabilities.platform, "posix")
                self.assertIn(
                    capabilities.power_loss_durability,
                    {"best_effort", "unsupported"},
                )

            output = root / "result"
            manifest = json.loads((output / "checksums.json").read_bytes())
            self.assertEqual(
                manifest["expected_paths"],
                ["masks/0001.bin", "predictions.json"],
            )
            self.assertEqual(manifest["file_count"], 2)
            self.assertNotIn("checksums.json", manifest["expected_paths"])
            self.assertEqual(
                manifest["files"][0]["sha256"],
                hashlib.sha256(b"mask").hexdigest(),
            )
            self.assertEqual(
                sorted(path.relative_to(output).as_posix() for path in output.rglob("*")),
                ["checksums.json", "masks", "masks/0001.bin", "predictions.json"],
            )
            self.assertFalse(any("stage" in item.name or "backup" in item.name for item in root.iterdir()))

    def test_bounds_paths_and_declared_membership_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = ("../escape", "/absolute", "nested\\windows")
            for path in bad:
                with self.subTest(path=path), self.assertRaises(ManagedOutputError):
                    ManagedOutputTransaction(
                        root=root,
                        destination="result",
                        declared_paths=(path,),
                        limits=LIMITS,
                    )
            with self.assertRaisesRegex(ManagedOutputError, "checksums.json is implicit"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("checksums.json",),
                    limits=LIMITS,
                )
            with ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("one.bin",),
                limits=ManagedOutputLimits(max_files=2, max_file_bytes=3, max_total_bytes=1024),
            ) as transaction:
                with self.assertRaisesRegex(ManagedOutputError, "undeclared_output"):
                    transaction.write_bytes("two.bin", b"x")
                with self.assertRaisesRegex(ManagedOutputError, "file_limit_exceeded"):
                    transaction.write_bytes("one.bin", b"1234")
                transaction.write_bytes("one.bin", b"123")
                with self.assertRaisesRegex(ManagedOutputError, "duplicate_output"):
                    transaction.write_bytes("one.bin", b"123")

    def test_incomplete_transaction_aborts_without_recursive_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("a/data.bin", "b/data.bin"),
                limits=LIMITS,
            ) as transaction:
                transaction.write_bytes("a/data.bin", b"a")
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_destination_requires_force_and_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old.bin": b"old"})
            with self.assertRaisesRegex(ManagedOutputError, "destination_exists"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                )
            (root / "result" / "unknown.bin").write_bytes(b"unknown")
            with self.assertRaisesRegex(ManagedOutputError, "tree_membership_mismatch"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )
            self.assertTrue((root / "result" / "old.bin").exists())
            self.assertTrue((root / "result" / "unknown.bin").exists())

    def test_force_replaces_only_exact_validated_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old/one.bin": b"old", "old/two.bin": b"two"})
            _publish(root, "result", {"new.bin": b"new"}, force=True)
            self.assertEqual((root / "result" / "new.bin").read_bytes(), b"new")
            self.assertFalse((root / "result" / "old").exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["result"],
            )

    def test_symlink_hardlink_special_and_self_entry_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"data.bin": b"data"})
            data = root / "result" / "data.bin"
            data.unlink()
            data.symlink_to(root / "outside")
            with self.assertRaisesRegex(ManagedOutputError, "unsafe_tree_entry"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"data.bin": b"data"})
            os.link(root / "result" / "data.bin", root / "outside-link")
            with self.assertRaisesRegex(ManagedOutputError, "hardlinked"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )

        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _publish(root, "result", {"data.bin": b"data"})
                (root / "result" / "data.bin").unlink()
                os.mkfifo(root / "result" / "data.bin")
                with self.assertRaisesRegex(ManagedOutputError, "unsafe_tree_entry"):
                    ManagedOutputTransaction(
                        root=root,
                        destination="result",
                        declared_paths=("new.bin",),
                        limits=LIMITS,
                        force=True,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            output.mkdir()
            manifest = {
                "schema_version": 1,
                "files": [
                    {
                        "path": "checksums.json",
                        "size_bytes": 0,
                        "sha256": "0" * 64,
                    }
                ],
                "expected_paths": ["checksums.json"],
                "file_count": 1,
                "total_bytes": 0,
            }
            (output / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ManagedOutputError, "manifest_self_entry"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )

    def test_reordered_and_mismatched_manifests_and_cross_device_state_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"a.bin": b"a", "b.bin": b"b"})
            manifest_path = root / "result" / "checksums.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["files"].reverse()
            manifest["expected_paths"].reverse()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ManagedOutputError, "byte-sorted"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"a.bin": b"a"})
            manifest_path = root / "result" / "checksums.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["files"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ManagedOutputError, "checksum_mismatch"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_fstat = os.fstat

            def changed_device(descriptor: int) -> os.stat_result:
                result = real_fstat(descriptor)
                if stat.S_ISDIR(result.st_mode) and descriptor != transaction._guard.parent_fd:
                    values = list(result)
                    values[2] = result.st_dev + 1
                    return os.stat_result(values)
                return result

            transaction = ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("data.bin",),
                limits=LIMITS,
            )
            transaction.write_bytes("data.bin", b"data")
            with mock.patch(
                "yolozu.adaptive.managed_output.os.fstat", side_effect=changed_device
            ):
                with self.assertRaisesRegex(ManagedOutputError, "cross_filesystem"):
                    transaction.commit()
            transaction.abort()
            transaction.close()

    def test_fault_before_new_visibility_restores_old_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old.bin": b"old"})

            def inject(step: str) -> None:
                if step == "before_rename_stage_to_destination":
                    raise _InjectedFailure(step)

            with self.assertRaises(_InjectedFailure):
                with ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                    fault_hook=inject,
                ) as transaction:
                    transaction.write_bytes("new.bin", b"new")
                    transaction.commit()
            self.assertEqual((root / "result" / "old.bin").read_bytes(), b"old")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["result"])

    def test_interrupted_after_visibility_recovers_committed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old.bin": b"old"})

            def inject(step: str) -> None:
                if step == "after_rename_stage_to_destination":
                    raise _InjectedFailure(step)

            transaction = ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("new.bin",),
                limits=LIMITS,
                force=True,
                fault_hook=inject,
            )
            transaction.write_bytes("new.bin", b"new")
            with self.assertRaises(_InjectedFailure):
                transaction.commit()
            transaction.close()

            self.assertEqual((root / "result" / "new.bin").read_bytes(), b"new")
            self.assertTrue(any("backup" in path.name for path in root.iterdir()))
            self.assertEqual(
                recover_managed_output(root=root, destination="result", limits=LIMITS),
                "committed",
            )
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["result"])
            self.assertEqual(
                recover_managed_output(root=root, destination="result", limits=LIMITS),
                "no_recovery_needed",
            )

    def test_recovery_rolls_back_known_old_moved_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old.bin": b"old"})

            def inject(step: str) -> None:
                if step == "after_rename_stage_to_destination":
                    raise _InjectedFailure(step)

            transaction = ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("new.bin",),
                limits=LIMITS,
                force=True,
                fault_hook=inject,
            )
            transaction.write_bytes("new.bin", b"new")
            with self.assertRaises(_InjectedFailure):
                transaction.commit()
            transaction.close()
            marker_path = root / ".result.yolozu-output-transaction.json"
            marker = json.loads(marker_path.read_bytes())
            os.rename(root / "result", root / marker["stage_name"])

            self.assertEqual(
                recover_managed_output(root=root, destination="result", limits=LIMITS),
                "rolled_back",
            )
            self.assertEqual((root / "result" / "old.bin").read_bytes(), b"old")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["result"])

    def test_parent_and_destination_swap_hooks_fail_without_mutating_substitute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old.bin": b"old"})

            def inject(step: str) -> None:
                if step == "before_rename_old_to_backup":
                    os.rename(root / "result", root / "original")
                    (root / "result").mkdir()
                    (root / "result" / "substitute").write_bytes(b"keep")

            with self.assertRaisesRegex(ManagedOutputError, "identity_changed"):
                with ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("new.bin",),
                    limits=LIMITS,
                    force=True,
                    fault_hook=inject,
                ) as transaction:
                    transaction.write_bytes("new.bin", b"new")
                    transaction.commit()
            self.assertEqual((root / "result" / "substitute").read_bytes(), b"keep")
            self.assertEqual((root / "original" / "old.bin").read_bytes(), b"old")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            transaction = ManagedOutputTransaction(
                root=root,
                destination="parent/result",
                declared_paths=("new.bin",),
                limits=LIMITS,
            )
            transaction.write_bytes("new.bin", b"new")
            os.rename(parent, root / "original-parent")
            parent.mkdir()
            with self.assertRaisesRegex(ManagedOutputError, "parent_changed"):
                transaction.commit()
            transaction.close()
            self.assertEqual(list(parent.iterdir()), [])

    def test_collision_and_ambiguous_recovery_retain_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / ".result.yolozu-output-transaction.json"
            marker.write_bytes(b"not a valid marker")
            with self.assertRaisesRegex(ManagedOutputError, "recovery_required"):
                ManagedOutputTransaction(
                    root=root,
                    destination="result",
                    declared_paths=("data.bin",),
                    limits=LIMITS,
                )
            with self.assertRaisesRegex(ManagedOutputError, "manual_recovery_required"):
                recover_managed_output(root=root, destination="result", limits=LIMITS)
            self.assertEqual(marker.read_bytes(), b"not a valid marker")

    def test_fault_injection_at_each_observed_io_boundary_keeps_a_complete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _publish(root, "result", {"old/data.bin": b"old"})
            trace: list[str] = []
            with ManagedOutputTransaction(
                root=root,
                destination="result",
                declared_paths=("new/data.bin",),
                limits=LIMITS,
                force=True,
                fault_hook=trace.append,
            ) as transaction:
                transaction.write_bytes("new/data.bin", b"new")
                transaction.commit()
        def boundary_key(step: str) -> str:
            for prefix in (
                "before_cleanup_tree:.result.backup.",
                "after_cleanup_tree:.result.backup.",
            ):
                if step.startswith(prefix):
                    return prefix
            return step

        boundaries = tuple(
            dict.fromkeys(
                boundary_key(step)
                for step in trace
                if step.startswith(("before_", "after_"))
            )
        )
        self.assertGreaterEqual(len(boundaries), 30)

        for target in boundaries:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sentinel = root / "caller-owned.txt"
                sentinel.write_bytes(b"keep")
                _publish(root, "result", {"old/data.bin": b"old"})
                fired = False

                def inject(step: str) -> None:
                    nonlocal fired
                    if not fired and boundary_key(step) == target:
                        fired = True
                        raise _InjectedFailure(target)

                try:
                    with ManagedOutputTransaction(
                        root=root,
                        destination="result",
                        declared_paths=("new/data.bin",),
                        limits=LIMITS,
                        force=True,
                        fault_hook=inject,
                    ) as transaction:
                        transaction.write_bytes("new/data.bin", b"new")
                        transaction.commit()
                except _InjectedFailure:
                    self.assertTrue(fired)
                self.assertTrue(fired)
                old_visible = (root / "result" / "old" / "data.bin").is_file()
                new_visible = (root / "result" / "new" / "data.bin").is_file()
                self.assertTrue(old_visible or new_visible)
                self.assertEqual(sentinel.read_bytes(), b"keep")
                marker = root / ".result.yolozu-output-transaction.json"
                if marker.exists():
                    try:
                        recover_managed_output(
                            root=root,
                            destination="result",
                            limits=LIMITS,
                        )
                    except ManagedOutputError as exc:
                        self.assertEqual(exc.code, "manual_recovery_required")
                    old_visible = (root / "result" / "old" / "data.bin").is_file()
                    new_visible = (root / "result" / "new" / "data.bin").is_file()
                    self.assertTrue(old_visible or new_visible)
                    self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_import_has_no_filesystem_effects_or_heavy_runtime_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code = """
import json, os, sys
before = sorted(os.listdir('.'))
import yolozu.adaptive.managed_output
after = sorted(os.listdir('.'))
heavy = [name for name in ('torch','onnxruntime','tensorrt','cv2','coremltools') if name in sys.modules]
print(json.dumps({'before': before, 'after': after, 'heavy': heavy}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                },
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["before"], result["after"])
            self.assertEqual(result["heavy"], [])


if __name__ == "__main__":
    main()
