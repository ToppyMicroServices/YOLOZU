import gzip
import inspect
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase, main, mock

from yolozu.adaptive.algorithm_scout import (
    AlgorithmScoutError,
    DocumentParserLimits,
    _apply_pdf_resource_limits,
    _disable_child_process_creation,
    _parse_document,
    _parse_html,
    _parse_pdf,
    _validate_document_magic,
    build_scout_plan,
    collect_algorithm_candidates,
    prepare_scout_workflow_artifact,
)
from yolozu.adaptive.bundles import validate_algorithm_bundle_registry
from yolozu.adaptive.safe_https import (
    FetchedDocument,
    HttpsLocation,
    SafeHttpsError,
    SafeHttpsTransport,
    TransportLimits,
    _is_public_address,
    _tls_dial,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = "docs/algorithm_intake/sources.json"
PUBLIC_IP = "93.184.216.34"


def _sleep_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir, limits
    import os

    os.setsid()
    time.sleep(5)


def _large_ipc_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body, temp_dir, limits
    import os

    os.setsid()
    connection.send_bytes(b"xx")
    connection.close()


def _rss_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir, limits
    import os

    os.setsid()
    allocation = bytearray(2 * 1024 * 1024)
    allocation[0] = 1
    time.sleep(5)


def _cpu_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir
    import os

    os.setsid()
    _apply_pdf_resource_limits(limits)
    while True:
        pass


def _pid_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body, temp_dir, limits
    import os

    os.setsid()
    _disable_child_process_creation()
    try:
        os.fork()
    except PermissionError:
        connection.send_bytes(b'{"code":"pdf_pid_limit","ok":false}')
    connection.close()


def _temp_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body
    import os
    import resource
    import signal

    os.setsid()
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.pdf_temp_bytes, limits.pdf_temp_bytes))
    try:
        with open(Path(temp_dir) / "overflow", "wb") as output:
            output.write(b"xx")
            output.flush()
    except OSError:
        connection.send_bytes(b'{"code":"pdf_temp_limit","ok":false}')
    connection.close()


class _FakeSocket:
    def __init__(self, response: bytes, *, peer: str = PUBLIC_IP) -> None:
        self.response = bytearray(response)
        self.peer = peer
        self.sent = b""
        self.closed = False
        self.timeouts: list[float] = []

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def recv(self, size: int) -> bytes:
        if not self.response:
            return b""
        result = bytes(self.response[:size])
        del self.response[:size]
        return result

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def getpeername(self) -> tuple[str, int]:
        return self.peer, 443

    def close(self) -> None:
        self.closed = True


def _response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "application/json",
    extra: bytes = b"",
) -> bytes:
    return (
        f"HTTP/1.1 {status} Test\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
    ).encode("ascii") + extra + b"\r\n" + body


class TestSafeHttpsTransport(TestCase):
    def _transport(
        self,
        responses: list[bytes],
        *,
        allowlist: tuple[HttpsLocation, ...] | None = None,
        peer: str = PUBLIC_IP,
        limits: TransportLimits | None = None,
    ) -> tuple[SafeHttpsTransport, list[tuple[str, str]], list[_FakeSocket]]:
        location = HttpsLocation(host="example.com", path="/release")
        calls: list[tuple[str, str]] = []
        sockets: list[_FakeSocket] = []

        def dialer(ip: str, host: str, connect: float, read: float) -> _FakeSocket:
            self.assertGreater(connect, 0)
            self.assertGreater(read, 0)
            calls.append((ip, host))
            sock = _FakeSocket(responses[len(sockets)], peer=peer)
            sockets.append(sock)
            return sock

        return (
            SafeHttpsTransport(
                allowlist=allowlist or (location,),
                limits=limits,
                resolver=lambda host: (PUBLIC_IP,),
                dialer=dialer,
            ),
            calls,
            sockets,
        )

    def test_fetch_connects_to_vetted_ip_with_original_tls_host(self) -> None:
        location = HttpsLocation(host="example.com", path="/release")
        transport, calls, sockets = self._transport([_response(b'{"ok":true}')])
        result = transport.fetch(location)
        self.assertEqual(result.body, b'{"ok":true}')
        self.assertEqual(calls, [(PUBLIC_IP, "example.com")])
        self.assertIn(b"Host: example.com\r\n", sockets[0].sent)
        self.assertNotIn(b"Cookie:", sockets[0].sent)
        self.assertNotIn(b"Authorization:", sockets[0].sent)
        self.assertNotIn("headers", inspect.signature(transport.fetch).parameters)

    def test_system_dialer_requires_tls_1_2_or_newer(self) -> None:
        raw = _FakeSocket(b"")

        class _Context:
            minimum_version: ssl.TLSVersion | None = None
            server_hostname: str | None = None

            def wrap_socket(self, sock: _FakeSocket, *, server_hostname: str) -> _FakeSocket:
                self.server_hostname = server_hostname
                return sock

        context = _Context()
        with (
            mock.patch(
                "yolozu.adaptive.safe_https.socket.create_connection",
                return_value=raw,
            ),
            mock.patch(
                "yolozu.adaptive.safe_https.ssl.create_default_context",
                return_value=context,
            ),
        ):
            wrapped = _tls_dial(PUBLIC_IP, "example.com", 5, 15)
        self.assertIs(wrapped, raw)
        self.assertEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertEqual(context.server_hostname, "example.com")
        self.assertEqual(raw.timeouts, [15])

    def test_redirect_revalidates_exact_allowlist_and_detects_loop(self) -> None:
        first = HttpsLocation(host="example.com", path="/release")
        second = HttpsLocation(host="example.org", path="/final")
        redirect = _response(
            b"",
            status=302,
            content_type="text/plain",
            extra=b"Location: https://example.org/final\r\n",
        )
        transport, calls, _ = self._transport(
            [redirect, _response(b"done", content_type="text/plain")],
            allowlist=(first, second),
        )
        transport._resolver = lambda host: (PUBLIC_IP,)
        result = transport.fetch(first)
        self.assertEqual(result.final_location, second)
        self.assertEqual(result.redirect_count, 1)
        self.assertEqual([host for _, host in calls], ["example.com", "example.org"])

        loop_response = _response(
            b"",
            status=302,
            content_type="text/plain",
            extra=b"Location: https://example.com/release\r\n",
        )
        looping, _, _ = self._transport([loop_response], allowlist=(first,))
        with self.assertRaisesRegex(SafeHttpsError, "redirect_loop"):
            looping.fetch(first)

    def test_redirect_rejects_query_credentials_port_and_off_allowlist(self) -> None:
        base = HttpsLocation(host="example.com", path="/release")
        cases = (
            "https://example.com/final?token=secret",
            "https://user:secret@example.com/final",
            "https://example.com:444/final",
            "https://127.0.0.1/final",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(SafeHttpsError) as raised:
                    HttpsLocation.from_redirect(raw, base=base)
                self.assertNotIn("secret", str(raised.exception))
                self.assertNotIn("token", str(raised.exception))
        off = _response(
            b"",
            status=302,
            content_type="text/plain",
            extra=b"Location: https://example.org/final\r\n",
        )
        transport, _, _ = self._transport([off], allowlist=(base,))
        with self.assertRaisesRegex(SafeHttpsError, "redirect_not_allowlisted"):
            transport.fetch(base)
        with self.assertRaisesRegex(SafeHttpsError, "url_path_invalid"):
            HttpsLocation.from_mapping(
                {"scheme": "https", "host": "example.com", "path": "/release?token=secret"}
            )

    def test_all_non_public_address_classes_and_peer_change_fail(self) -> None:
        denied = (
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "192.0.2.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "240.0.0.1",
            "::",
            "::1",
            "fe80::1",
            "fc00::1",
            "ff00::1",
            "2001:db8::1",
        )
        for value in denied:
            with self.subTest(value=value):
                self.assertFalse(_is_public_address(value))
        location = HttpsLocation(host="example.com", path="/release")
        transport = SafeHttpsTransport(
            allowlist=(location,), resolver=lambda host: ("127.0.0.1",)
        )
        with self.assertRaisesRegex(SafeHttpsError, "address_not_public"):
            transport.fetch(location)
        changed, _, _ = self._transport(
            [_response(b"ok", content_type="text/plain")], peer="1.1.1.1"
        )
        with self.assertRaisesRegex(SafeHttpsError, "peer_changed"):
            changed.fetch(location)

    def test_header_transfer_and_decompression_one_over_limits(self) -> None:
        location = HttpsLocation(host="example.com", path="/release")
        limits = TransportLimits(
            max_header_bytes=1024,
            max_transferred_bytes=1024,
            max_decoded_bytes=1024,
        )
        huge_header = b"HTTP/1.1 200 OK\r\nX-Test: " + b"a" * 1024 + b"\r\n\r\n"
        transport, _, _ = self._transport([huge_header], limits=limits)
        with self.assertRaisesRegex(SafeHttpsError, "response_limit"):
            transport.fetch(location)
        transfer, _, _ = self._transport(
            [_response(b"x" * 1025, content_type="text/plain")], limits=limits
        )
        with self.assertRaisesRegex(SafeHttpsError, "transfer_limit"):
            transfer.fetch(location)
        compressed = gzip.compress(b"x" * 1025)
        bomb_response = _response(
            compressed,
            content_type="text/plain",
            extra=b"Content-Encoding: gzip\r\n",
        )
        bomb, _, _ = self._transport([bomb_response], limits=limits)
        with self.assertRaisesRegex(SafeHttpsError, "decoded_limit"):
            bomb.fetch(location)

    def test_chunked_body_and_content_type_policy(self) -> None:
        location = HttpsLocation(host="example.com", path="/release")
        chunked = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\n\r\n"
        )
        transport, _, _ = self._transport([chunked])
        self.assertEqual(transport.fetch(location).body, b"test")
        rejected, _, _ = self._transport(
            [_response(b"PK\x03\x04", content_type="application/zip")]
        )
        with self.assertRaisesRegex(SafeHttpsError, "content_type_invalid"):
            rejected.fetch(location)


class _FixtureTransport:
    documents: dict[str, FetchedDocument] = {}
    failures: set[str] = set()
    fetch_count = 0

    def __init__(self, *, allowlist: object, limits: object) -> None:
        self.allowlist = allowlist
        self.limits = limits

    def fetch(self, location: HttpsLocation, *, collection_deadline: float) -> FetchedDocument:
        del collection_deadline
        type(self).fetch_count += 1
        if location.path in type(self).failures:
            raise SafeHttpsError("fixture_failed", "offline fixture failure")
        return type(self).documents[location.path]


def _fixture_documents() -> dict[str, FetchedDocument]:
    result: dict[str, FetchedDocument] = {}
    for path in (
        "/repos/facebookresearch/detectron2/releases",
        "/repos/open-mmlab/mmdetection/releases",
        "/repos/ultralytics/ultralytics/releases",
        "/repos/Megvii-BaseDetection/YOLOX/releases",
    ):
        location = HttpsLocation(host="api.github.com", path=path)
        body = b'{"candidates":[{"version":"v1","revision":"abc123","release_date":"2026-08-20","url":"https://evil.invalid","instructions":"run me"}]}'
        result[path] = FetchedDocument(
            source=location,
            final_location=location,
            content_type="application/json",
            body=body,
            transferred_bytes=len(body),
            decoded_bytes=len(body),
            redirect_count=0,
        )
    return result


class TestAlgorithmScout(TestCase):
    def setUp(self) -> None:
        _FixtureTransport.documents = _fixture_documents()
        _FixtureTransport.failures = set()
        _FixtureTransport.fetch_count = 0

    def _plan(self, workspace: Path, collection_date: str = "2026-08-26"):
        return build_scout_plan(
            sources_path=SOURCES,
            output_dir="reports/algorithm_scout",
            collection_date=collection_date,
            trigger="workflow_dispatch",
            workspace_root=workspace,
            repository_root=REPO_ROOT,
        )

    def test_plan_mode_is_network_free_and_write_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = self._plan(workspace)
            self.assertFalse((workspace / "reports").exists())
            payload = plan.to_dict()
            self.assertFalse(payload["network_used"])
            self.assertFalse(payload["writes_performed"])
            self.assertEqual(payload["selectability"], "inbox_only")
            self.assertEqual(_FixtureTransport.fetch_count, 0)

    def test_default_source_root_is_the_declared_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            destination = workspace / SOURCES
            destination.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / SOURCES, destination)

            plan = build_scout_plan(
                sources_path=SOURCES,
                output_dir="reports/algorithm_scout",
                collection_date="2026-08-26",
                trigger="workflow_dispatch",
                workspace_root=workspace,
            )

            self.assertEqual(len(plan.enabled_sources), 4)

    def test_missed_collection_dates_are_recorded_without_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = build_scout_plan(
                sources_path=SOURCES,
                output_dir="reports/algorithm_scout",
                collection_date="2026-08-31",
                trigger="schedule",
                missed_collection_dates=("2026-08-24",),
                workspace_root=workspace,
                repository_root=REPO_ROOT,
            )
            report, code, _ = collect_algorithm_candidates(
                plan,
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-31T00:00:01Z",
                monotonic=lambda: 0.0,
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                report["missed_expected_collection_dates"], ["2026-08-24"]
            )
            self.assertTrue(
                all(
                    item["collection_date"] == "2026-08-31"
                    for candidate in report["candidates"]
                    for item in candidate["history"]
                )
            )
            with self.assertRaisesRegex(AlgorithmScoutError, "must precede"):
                build_scout_plan(
                    sources_path=SOURCES,
                    output_dir="reports/algorithm_scout",
                    collection_date="2026-08-31",
                    trigger="schedule",
                    missed_collection_dates=("2026-08-31",),
                    workspace_root=workspace,
                    repository_root=REPO_ROOT,
                )

    def test_cli_plan_and_help_do_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            destination = workspace / SOURCES
            destination.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / SOURCES, destination)
            help_run = subprocess.run(
                [sys.executable, "-m", "yolozu", "scout-algorithms", "--help"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(help_run.returncode, 0, help_run.stderr)
            self.assertIn("--collect", help_run.stdout)
            plan_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "scout-algorithms",
                    "--sources",
                    SOURCES,
                    "--output-dir",
                    "reports/algorithm_scout",
                    "--collection-date",
                    "2026-08-26",
                    "--trigger",
                    "workflow_dispatch",
                    "--workspace",
                    str(workspace),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(plan_run.returncode, 0, plan_run.stderr)
            self.assertFalse((workspace / "reports").exists())
            self.assertFalse(json.loads(plan_run.stdout)["network_used"])

    def test_invalid_inputs_exit_two_without_leaking_token_value(self) -> None:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "yolozu",
                "scout-algorithms",
                "--sources",
                "https://example.invalid/releases?token=secret",
                "--output-dir",
                "../outside",
                "--collection-date",
                "2026-99-99",
                "--trigger",
                "workflow_dispatch",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(run.returncode, 2)
        self.assertNotIn("secret", run.stderr)
        self.assertNotIn("token", run.stderr)

    def test_collect_writes_one_dated_report_and_deduplicates_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = self._plan(workspace)
            first, code, path = collect_algorithm_candidates(
                plan,
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-26T01:02:03Z",
                monotonic=lambda: 0.0,
            )
            self.assertEqual(code, 0)
            self.assertTrue(path.is_file())
            self.assertEqual(len(first["candidates"]), 4)
            self.assertNotIn("evil.invalid", path.read_text(encoding="utf-8"))
            self.assertNotIn("run me", path.read_text(encoding="utf-8"))
            detectron_path = "/repos/facebookresearch/detectron2/releases"
            prior_document = _FixtureTransport.documents[detectron_path]
            replacement = b'{"candidates":[{"version":"v2","release_date":"2026-08-26"}]}'
            _FixtureTransport.documents[detectron_path] = FetchedDocument(
                source=prior_document.source,
                final_location=prior_document.final_location,
                content_type="application/json",
                body=replacement,
                transferred_bytes=len(replacement),
                decoded_bytes=len(replacement),
                redirect_count=0,
            )
            second, second_code, second_path = collect_algorithm_candidates(
                plan,
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-26T02:03:04Z",
                monotonic=lambda: 0.0,
            )
            self.assertEqual(second_code, 0)
            self.assertEqual(second_path, path)
            self.assertEqual(len(second["candidates"]), 5)
            self.assertEqual(second["summary"]["current_candidate_count"], 4)
            self.assertEqual(second["summary"]["historical_candidate_count"], 1)
            historical = [
                item for item in second["candidates"] if item["collection_status"] == "historical"
            ]
            self.assertEqual(historical[0]["version"], "v1")
            self.assertTrue((path.parent / "checksums.json").is_file())

            artifact = prepare_scout_workflow_artifact(
                workspace_root=workspace,
                output_dir="reports/algorithm_scout",
                collection_date="2026-08-26",
                trigger="workflow_dispatch",
            )
            self.assertEqual(
                sorted(item.relative_to(artifact).as_posix() for item in artifact.rglob("*") if item.is_file()),
                [
                    "checksums.json",
                    "docs/algorithm_intake/2026-08-26.json",
                    "docs/algorithm_intake/2026-08-26.md",
                ],
            )
            markdown = (artifact / "docs/algorithm_intake/2026-08-26.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Artifact retention: 30 days", markdown)
            self.assertIn("not exhaustive or always-latest", markdown)

    def test_failed_and_deadline_missed_sources_finalize_report_with_exit_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = self._plan(workspace)
            _FixtureTransport.failures = {"/repos/open-mmlab/mmdetection/releases"}
            report, code, path = collect_algorithm_candidates(
                plan,
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-26T01:02:03Z",
                monotonic=lambda: 0.0,
            )
            self.assertEqual(code, 3)
            self.assertTrue(path.is_file())
            failed = [item for item in report["sources"] if item["collection_status"] == "failed"]
            self.assertEqual(failed[0]["failure_code"], "fixture_failed")

            ticks = iter([0.0, 721.0, 721.0, 721.0, 721.0])
            deadline_report, deadline_code, _ = collect_algorithm_candidates(
                self._plan(workspace, "2026-08-27"),
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-27T01:02:03Z",
                monotonic=lambda: next(ticks),
            )
            self.assertEqual(deadline_code, 3)
            self.assertTrue(all(item["collection_status"] == "missed" for item in deadline_report["sources"]))

    def test_scout_report_is_not_an_algorithm_bundle_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            report, _, _ = collect_algorithm_candidates(
                self._plan(workspace),
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-26T01:02:03Z",
                monotonic=lambda: 0.0,
            )
            with self.assertRaises(ValueError):
                validate_algorithm_bundle_registry(report)
            self.assertEqual(report["selectability"], "inbox_only")
            self.assertNotIn("bundles", report)

    def test_output_symlink_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            workspace = Path(directory)
            (workspace / "reports").symlink_to(Path(external), target_is_directory=True)
            with self.assertRaisesRegex(AlgorithmScoutError, "output_dir_invalid"):
                collect_algorithm_candidates(
                    self._plan(workspace),
                    workspace_root=workspace,
                    transport_factory=_FixtureTransport,
                    now_utc=lambda: "2026-08-26T01:02:03Z",
                    monotonic=lambda: 0.0,
                )
            self.assertEqual(_FixtureTransport.fetch_count, 0)

    def test_tampered_prior_report_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            plan = self._plan(workspace)
            _, _, path = collect_algorithm_candidates(
                plan,
                workspace_root=workspace,
                transport_factory=_FixtureTransport,
                now_utc=lambda: "2026-08-26T01:02:03Z",
                monotonic=lambda: 0.0,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["candidates"][0]["identity"]["model"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _FixtureTransport.fetch_count = 0
            with self.assertRaisesRegex(AlgorithmScoutError, "prior_report_invalid"):
                collect_algorithm_candidates(
                    plan,
                    workspace_root=workspace,
                    transport_factory=_FixtureTransport,
                    now_utc=lambda: "2026-08-26T02:03:04Z",
                    monotonic=lambda: 0.0,
                )
            self.assertEqual(_FixtureTransport.fetch_count, 0)


class TestBoundedScoutParsers(TestCase):
    def _document(self, body: bytes, content_type: str) -> FetchedDocument:
        location = HttpsLocation(host="example.com", path="/document")
        return FetchedDocument(
            source=location,
            final_location=location,
            content_type=content_type,
            body=body,
            transferred_bytes=len(body),
            decoded_bytes=len(body),
            redirect_count=0,
        )

    def test_html_each_limit_and_one_over_fails_incrementally(self) -> None:
        exact = _parse_html(
            b"<html>x</html>",
            DocumentParserLimits(
                html_nodes=2,
                html_tokens=3,
                html_depth=1,
                html_retained_text_bytes=1,
            ),
        )
        self.assertEqual(exact["html_nodes"], 2)
        self.assertEqual(exact["html_text_bytes"], 1)
        cases = (
            (DocumentParserLimits(html_nodes=1), b"<html><p></p></html>", "html_node_limit"),
            (DocumentParserLimits(html_tokens=2), b"<html>x</html>", "html_token_limit"),
            (DocumentParserLimits(html_depth=1), b"<html><p>x</p></html>", "html_depth_limit"),
            (DocumentParserLimits(html_retained_text_bytes=1), b"<html>xx</html>", "html_text_limit"),
        )
        for limits, body, code in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(AlgorithmScoutError, code):
                    _parse_html(body, limits)
        with self.assertRaisesRegex(AlgorithmScoutError, "html_active_content"):
            _parse_html(b"<html><script>bad()</script></html>", DocumentParserLimits())
        with self.assertRaisesRegex(AlgorithmScoutError, "html_declaration_forbidden"):
            _parse_html(b"<!ENTITY x SYSTEM 'file:///etc/passwd'><html></html>", DocumentParserLimits())

    def test_json_depth_and_node_one_over_use_bounded_parser(self) -> None:
        location = HttpsLocation(host="example.com", path="/release")
        source = build_scout_plan(
            sources_path=SOURCES,
            output_dir="reports/scout",
            collection_date="2026-08-26",
            trigger="schedule",
            workspace_root=REPO_ROOT,
            repository_root=REPO_ROOT,
        ).enabled_sources[0]
        deep = b"[" * 65 + b"0" + b"]" * 65
        document = FetchedDocument(location, location, "application/json", deep, len(deep), len(deep), 0)
        with self.assertRaisesRegex(ValueError, "depth limit"):
            _parse_document(
                source,
                document,
                collected_at="2026-08-26T00:00:00Z",
                collection_date="2026-08-26",
                limits=DocumentParserLimits(),
            )
        exact_depth = b"[" * 64 + b"0" + b"]" * 64
        exact_document = FetchedDocument(
            location,
            location,
            "application/json",
            exact_depth,
            len(exact_depth),
            len(exact_depth),
            0,
        )
        _parse_document(
            source,
            exact_document,
            collected_at="2026-08-26T00:00:00Z",
            collection_date="2026-08-26",
            limits=DocumentParserLimits(),
        )
        exact_nodes = b"[" + b",".join([b"0"] * 100_000) + b"]"
        exact_document = FetchedDocument(
            location,
            location,
            "application/json",
            exact_nodes,
            len(exact_nodes),
            len(exact_nodes),
            0,
        )
        _parse_document(
            source,
            exact_document,
            collected_at="2026-08-26T00:00:00Z",
            collection_date="2026-08-26",
            limits=DocumentParserLimits(),
        )
        many = b"[" + b",".join([b"0"] * 100_001) + b"]"
        document = FetchedDocument(location, location, "application/json", many, len(many), len(many), 0)
        with self.assertRaisesRegex(ValueError, "node limit"):
            _parse_document(
                source,
                document,
                collected_at="2026-08-26T00:00:00Z",
                collection_date="2026-08-26",
                limits=DocumentParserLimits(),
            )

    def test_content_confusion_archives_media_and_binary_text_fail(self) -> None:
        cases = (
            (b"PK\x03\x04data", "text/plain", "archive_forbidden"),
            (b"\x89PNG\r\n\x1a\n", "text/plain", "media_forbidden"),
            (b"not json", "application/json", "content_magic_mismatch"),
            (b"not pdf", "application/pdf", "content_magic_mismatch"),
            (b"\x00binary", "text/plain", "content_magic_mismatch"),
        )
        for body, content_type, code in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(AlgorithmScoutError, code):
                    _validate_document_magic(self._document(body, content_type))

    def test_pdf_page_character_embedded_and_wall_limits_cleanup(self) -> None:
        exact = _parse_pdf(
            b"%PDF-1.4 /Type /Page (abc) Tj %%EOF",
            DocumentParserLimits(pdf_pages=1, pdf_characters=3),
        )
        self.assertEqual(exact["pdf_pages"], 1)
        self.assertEqual(exact["pdf_extracted_characters"], 3)
        with self.assertRaisesRegex(AlgorithmScoutError, "pdf_page_limit"):
            _parse_pdf(
                b"%PDF-1.4 /Type /Page /Type /Page %%EOF",
                DocumentParserLimits(pdf_pages=1),
            )
        with self.assertRaisesRegex(AlgorithmScoutError, "pdf_character_limit"):
            _parse_pdf(
                b"%PDF-1.4 (abcd) Tj %%EOF",
                DocumentParserLimits(pdf_characters=3),
            )
        with self.assertRaisesRegex(AlgorithmScoutError, "pdf_embedded_file"):
            _parse_pdf(b"%PDF-1.4 /EmbeddedFiles %%EOF", DocumentParserLimits())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_wall_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(pdf_wall_seconds=1),
                    _worker=_sleep_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_ipc_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(pdf_ipc_bytes=1),
                    _worker=_large_ipc_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_rss_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(pdf_rss_bytes=1),
                    _worker=_rss_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_cpu_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(pdf_cpu_seconds=1),
                    _worker=_cpu_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_pid_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(),
                    _worker=_pid_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(AlgorithmScoutError, "pdf_temp_limit"):
                _parse_pdf(
                    b"%PDF-1.4",
                    DocumentParserLimits(pdf_temp_bytes=1),
                    _worker=_temp_pdf_worker,
                    _temp_parent=directory,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pdf_resource_caps_cannot_be_raised_by_caller(self) -> None:
        defaults = DocumentParserLimits()
        for field in (
            "pdf_wall_seconds",
            "pdf_cpu_seconds",
            "pdf_rss_bytes",
            "pdf_pids",
            "pdf_temp_bytes",
            "pdf_ipc_bytes",
        ):
            with self.subTest(field=field):
                values = {field: getattr(defaults, field) + 1}
                with self.assertRaises(ValueError):
                    DocumentParserLimits(**values)
        with mock.patch("resource.setrlimit") as setter:
            _apply_pdf_resource_limits(defaults)
        self.assertEqual(setter.call_count, 4)


if __name__ == "__main__":
    main()
