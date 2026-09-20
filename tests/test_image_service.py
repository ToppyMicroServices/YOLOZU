from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from yolozu.integrations.image_service import (
    ImageService,
    ImageServiceError,
)
from yolozu.integrations.mcp_cli import _server_options


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def _options(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "transport": "streamable-http",
        "surface": "image-service",
        "host": "127.0.0.1",
        "port": 8000,
        "http_path": "/mcp",
        "public_url": None,
        "auth_token_env": "YOLOZU_MCP_AUTH_TOKEN",
        "tenant_id": "local",
        "retention_seconds": 86_400,
    }
    values.update(overrides)
    return Namespace(**values)


class TestImageService(unittest.TestCase):
    def _wait(self, service: ImageService, job_id: str) -> dict:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = service.get_job(job_id=job_id)
            if result["job"]["status"] not in {"queued", "running"}:
                return result
            time.sleep(0.01)
        self.fail("image service job did not finish")

    def test_capabilities_are_read_only_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = ImageService(workspace=root)
            result = service.capabilities()
            self.assertTrue(result["ok"])
            self.assertEqual(
                result["service"]["selection_policy"],
                "qualified_registered_pipeline_or_abstain",
            )
            self.assertFalse((root / "runs").exists())

    def test_put_asset_returns_identity_without_path(self) -> None:
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = ImageService(workspace=root, tenant_id="tenant-a")
            result = service.put_asset(
                content_base64=base64.b64encode(data).decode("ascii"),
                media_type="image/png",
            )
            asset = result["asset"]
            self.assertRegex(asset["asset_id"], r"^asset_[0-9a-f]{24}$")
            self.assertEqual(asset["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual((asset["width"], asset["height"]), (8, 6))
            self.assertNotIn(str(root), json.dumps(result))
            stored = list((root / "runs" / "mcp_image_service").rglob("source.png"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].read_bytes(), data)
            self.assertEqual(stored[0].stat().st_mode & 0o777, 0o600)

    def test_put_asset_rejects_mime_mismatch_and_invalid_base64(self) -> None:
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            service = ImageService(workspace=td)
            with self.assertRaises(ImageServiceError) as mismatch:
                service.put_asset(
                    content_base64=base64.b64encode(data).decode("ascii"),
                    media_type="image/jpeg",
                )
            self.assertEqual(mismatch.exception.code, "media_type_mismatch")
            with self.assertRaises(ImageServiceError) as invalid:
                service.put_asset(content_base64="not base64", media_type="image/png")
            self.assertEqual(invalid.exception.code, "invalid_asset")

    def test_asset_ids_do_not_cross_tenant_roots(self) -> None:
        data = _png_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        with tempfile.TemporaryDirectory() as td:
            first = ImageService(workspace=td, tenant_id="tenant-a")
            second = ImageService(workspace=td, tenant_id="tenant-b")
            asset_id = first.put_asset(
                content_base64=encoded,
                media_type="image/png",
            )["asset"]["asset_id"]
            with self.assertRaises(ImageServiceError) as missing:
                second.submit_job(asset_id=asset_id, fixed_classes=["cat"])
            self.assertEqual(missing.exception.code, "asset_not_found")

    def test_submit_abstains_without_executing_fallback(self) -> None:
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            service = ImageService(workspace=td)
            asset_id = service.put_asset(
                content_base64=base64.b64encode(data).decode("ascii"),
                media_type="image/png",
            )["asset"]["asset_id"]
            decision = {
                "schema_version": 1,
                "decision_id": "d" * 64,
                "status": "abstained",
                "reason_codes": ["no_qualified_candidate"],
            }
            with (
                patch(
                    "yolozu.integrations.image_service.recommend_image_pipeline",
                    return_value={"decision": decision},
                ),
                patch(
                    "yolozu.integrations.image_service.process_images"
                ) as process,
            ):
                queued = service.submit_job(
                    asset_id=asset_id,
                    fixed_classes=["cat"],
                    execute=True,
                )
                result = self._wait(service, queued["job"]["job_id"])
            process.assert_not_called()
            public = result["job"]["result"]
            self.assertEqual(public["outcome"], "abstained")
            self.assertEqual(
                public["decision"]["reason_codes"],
                ["no_qualified_candidate"],
            )
            self.assertNotIn(str(td), json.dumps(result))

    def test_selected_job_defaults_to_preflight_only(self) -> None:
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            service = ImageService(workspace=td)
            asset_id = service.put_asset(
                content_base64=base64.b64encode(data).decode("ascii"),
                media_type="image/png",
            )["asset"]["asset_id"]
            decision = {
                "schema_version": 1,
                "decision_id": "e" * 64,
                "status": "selected",
                "reason_codes": [],
                "selected_bundle": {"bundle_id": "fixture"},
            }
            with (
                patch(
                    "yolozu.integrations.image_service.recommend_image_pipeline",
                    return_value={"decision": decision},
                ),
                patch(
                    "yolozu.integrations.image_service.process_images",
                    return_value={"ok": True, "exit_code": 0, "executed": False},
                ) as process,
            ):
                queued = service.submit_job(
                    asset_id=asset_id,
                    fixed_classes=["cat"],
                )
                result = self._wait(service, queued["job"]["job_id"])
            self.assertEqual(result["job"]["result"]["outcome"], "ready")
            self.assertFalse(result["job"]["result"]["executed"])
            self.assertTrue(process.call_args.kwargs["dry_run"])

    def test_expired_asset_and_output_cleanup_stays_in_tenant_root(self) -> None:
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = ImageService(
                workspace=root,
                tenant_id="tenant-a",
                retention_seconds=300,
            )
            asset_id = service.put_asset(
                content_base64=base64.b64encode(data).decode("ascii"),
                media_type="image/png",
            )["asset"]["asset_id"]
            asset_dir = service.assets_root / asset_id
            output_dir = service.outputs_root / ("run_" + "a" * 24)
            output_dir.mkdir()
            outside = root / "outside"
            outside.mkdir()
            old = time.time() - 301
            os.utime(asset_dir, (old, old))
            os.utime(output_dir, (old, old))
            result = service.purge_expired()
            self.assertEqual(
                result,
                {"assets": 1, "outputs": 1, "jobs": 0},
            )
            self.assertFalse(asset_dir.exists())
            self.assertFalse(output_dir.exists())
            self.assertTrue(outside.exists())

    def test_tenant_asset_and_active_job_capacity_fail_closed(self) -> None:
        data = _png_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        with tempfile.TemporaryDirectory() as td:
            service = ImageService(workspace=td)
            with patch(
                "yolozu.integrations.image_service.MAX_ASSETS_PER_TENANT",
                1,
            ):
                asset_id = service.put_asset(
                    content_base64=encoded,
                    media_type="image/png",
                )["asset"]["asset_id"]
                with self.assertRaises(ImageServiceError) as full:
                    service.put_asset(
                        content_base64=encoded,
                        media_type="image/png",
                    )
            self.assertEqual(full.exception.code, "asset_capacity_reached")
            with patch(
                "yolozu.integrations.image_service.MAX_ACTIVE_JOBS",
                0,
            ):
                with self.assertRaises(ImageServiceError) as queue_full:
                    service.submit_job(
                        asset_id=asset_id,
                        fixed_classes=["cat"],
                    )
            self.assertEqual(queue_full.exception.code, "job_capacity_reached")


class TestMcpServerOptions(unittest.TestCase):
    def test_loopback_http_is_allowed_without_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            options, token = _server_options(_options())
        self.assertIsNone(token)
        self.assertEqual(options["surface"], "image-service")

    def test_public_bind_requires_narrow_surface_https_and_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TLS proxy"):
                _server_options(_options(host="0.0.0.0"))
            with self.assertRaisesRegex(ValueError, "image-service"):
                _server_options(
                    _options(
                        host="0.0.0.0",
                        surface="full",
                        public_url="https://vision.example/mcp",
                    )
                )
        with patch.dict(
            os.environ,
            {"YOLOZU_MCP_AUTH_TOKEN": "x" * 32},
            clear=True,
        ):
            options, token = _server_options(
                _options(
                    host="0.0.0.0",
                    public_url="https://vision.example/mcp",
                )
            )
        self.assertEqual(token, "x" * 32)
        self.assertEqual(options["public_url"], "https://vision.example/mcp")

    def test_public_url_rejects_plain_http(self) -> None:
        with patch.dict(
            os.environ,
            {"YOLOZU_MCP_AUTH_TOKEN": "x" * 32},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                _server_options(
                    _options(
                        host="0.0.0.0",
                        public_url="http://vision.example/mcp",
                    )
                )

    def test_public_url_on_loopback_still_requires_narrow_authenticated_surface(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "bearer token"):
                _server_options(
                    _options(public_url="https://vision.example/mcp")
                )
        with patch.dict(
            os.environ,
            {"YOLOZU_MCP_AUTH_TOKEN": "x" * 32},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "image-service"):
                _server_options(
                    _options(
                        surface="full",
                        public_url="https://vision.example/mcp",
                    )
                )
            with self.assertRaisesRegex(ValueError, "must match"):
                _server_options(
                    _options(public_url="https://vision.example/other")
                )

    def test_token_rejects_control_characters(self) -> None:
        with patch.dict(
            os.environ,
            {"YOLOZU_MCP_AUTH_TOKEN": "x" * 31 + "\n"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "control"):
                _server_options(_options())

    @unittest.skipUnless(
        importlib.util.find_spec("mcp") is not None,
        "optional mcp dependency is not installed",
    )
    def test_direct_server_boundary_rejects_external_full_surface(self) -> None:
        from yolozu.integrations.mcp_server import _http_server_boundary

        with self.assertRaisesRegex(ValueError, "image-service"):
            _http_server_boundary(
                surface="full",
                host="127.0.0.1",
                port=8000,
                streamable_http_path="/mcp",
                auth_token="x" * 32,
                public_url="https://vision.example/mcp",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("mcp") is not None,
        "optional mcp dependency is not installed",
    )
    def test_direct_server_boundary_formats_ipv6_host(self) -> None:
        from yolozu.integrations.mcp_server import _http_server_boundary

        hosts, origins = _http_server_boundary(
            surface="image-service",
            host="::1",
            port=8000,
            streamable_http_path="/mcp",
            auth_token=None,
            public_url=None,
        )
        self.assertEqual(hosts, ["[::1]:8000"])
        self.assertEqual(origins, [])


if __name__ == "__main__":
    unittest.main()
