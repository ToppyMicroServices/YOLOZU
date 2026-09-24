from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolozu.adaptive import checkpoint_preparation as preparation


class TestTorchvisionCheckpointPreparation(unittest.TestCase):
    def _prepare(self, root: Path, *, accepted: bool = True):
        source_bytes = b"exact-source-checkpoint"
        converted_bytes = b"deterministic-safetensors"
        source = root / "source.pth"
        source.write_bytes(source_bytes)
        patches = (
            patch.object(
                preparation,
                "MASKRCNN_CHECKPOINT_SIZE_BYTES",
                len(source_bytes),
            ),
            patch.object(
                preparation,
                "MASKRCNN_CHECKPOINT_SHA256",
                hashlib.sha256(source_bytes).hexdigest(),
            ),
            patch.object(
                preparation,
                "MASKRCNN_SAFETENSORS_SIZE_BYTES",
                len(converted_bytes),
            ),
            patch.object(
                preparation,
                "MASKRCNN_SAFETENSORS_SHA256",
                hashlib.sha256(converted_bytes).hexdigest(),
            ),
            patch.object(
                preparation,
                "_convert_verified_checkpoint",
                side_effect=lambda _source, destination: destination.write_bytes(
                    converted_bytes
                ),
            ),
        )
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        return preparation.prepare_torchvision_maskrcnn_checkpoint(
            checkpoint_path=source,
            artifact_root=root / "cache",
            accept_upstream_terms=accepted,
        )

    def test_prepares_exact_artifact_and_privacy_safe_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self._prepare(root)

            self.assertFalse(result.artifact_reused)
            self.assertEqual(result.artifact_path.read_bytes(), b"deterministic-safetensors")
            provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
            self.assertTrue(provenance["source"]["provided_by_user"])
            self.assertFalse(provenance["source"]["downloaded_by_yolozu"])
            self.assertFalse(provenance["source"]["local_path_recorded"])
            self.assertTrue(provenance["upstream_terms"]["acknowledged_by_user"])
            self.assertTrue(provenance["conversion"]["performed_by_this_invocation"])
            self.assertEqual(
                provenance["licensing"]["checkpoint_license_expression"],
                "NOASSERTION",
            )
            self.assertFalse(
                provenance["licensing"]["converted_artifact_redistributed_by_yolozu"]
            )
            self.assertNotIn(str(root), result.provenance_path.read_text(encoding="utf-8"))

    def test_exact_existing_artifact_is_reused_without_rewriting_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._prepare(root)
            first_provenance = first.provenance_path.read_bytes()
            second = self._prepare(root)

            self.assertTrue(second.artifact_reused)
            self.assertEqual(second.provenance_path.read_bytes(), first_provenance)
            self.assertEqual(second.provenance["prepared_at"], first.provenance["prepared_at"])
            self.assertTrue(second.provenance["conversion"]["performed_by_this_invocation"])

    def test_exact_artifact_without_provenance_records_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._prepare(root)
            first.provenance_path.unlink()

            second = self._prepare(root)
            self.assertTrue(second.artifact_reused)
            self.assertFalse(
                second.provenance["conversion"]["performed_by_this_invocation"]
            )

    def test_acknowledgement_is_required_before_reading_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(PermissionError, "acknowledgement"):
                preparation.prepare_torchvision_maskrcnn_checkpoint(
                    checkpoint_path=root / "absent.pth",
                    artifact_root=root / "cache",
                    accept_upstream_terms=False,
                )
            self.assertFalse((root / "cache").exists())

    def test_existing_mismatched_artifact_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._prepare(root)
            first.artifact_path.write_bytes(b"unexpected-existing-bytes")

            with self.assertRaisesRegex(ValueError, "artifact (size|sha256) mismatch"):
                self._prepare(root)
            self.assertEqual(
                first.artifact_path.read_bytes(),
                b"unexpected-existing-bytes",
            )

    def test_source_digest_mismatch_creates_no_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pth"
            source.write_bytes(b"wrong")
            with patch.object(
                preparation,
                "MASKRCNN_CHECKPOINT_SIZE_BYTES",
                len(b"wrong"),
            ), self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                preparation.prepare_torchvision_maskrcnn_checkpoint(
                    checkpoint_path=source,
                    artifact_root=root / "cache",
                    accept_upstream_terms=True,
                )
            self.assertFalse((root / "cache").exists())


if __name__ == "__main__":
    unittest.main()
