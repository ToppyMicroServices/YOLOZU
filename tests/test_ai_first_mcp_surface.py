import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.integrations.ai_surface import (
    ai_surface_sets,
    discover_manifest_tools,
    generate_config,
    list_manifest_tools,
    review_config,
)


class TestAiFirstMcpSurface(unittest.TestCase):
    def test_packaged_manifest_discovery_is_sorted_and_filterable(self):
        ids = list_manifest_tools(ids_only=True)
        self.assertGreater(len(ids), 100)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(
            list_manifest_tools(guaranteed=True, ids_only=True),
            [
                "doctor",
                "generate_config",
                "review_config",
                "validate_predictions",
            ],
        )
        supported = list_manifest_tools(supported=True, ids_only=True)
        self.assertEqual(len(supported), 26)
        self.assertIn("eval_coco", supported)
        self.assertIn("validate_predictions", supported)
        self.assertIn("recommend_image_pipeline", supported)
        supported_records = list_manifest_tools(supported=True)
        self.assertEqual(
            {item["id"] for item in supported_records},
            set(supported),
        )
        for item in supported_records:
            with self.subTest(tool=item["id"]):
                self.assertTrue(item["summary"].strip())
                self.assertIsInstance(item["input_schema"], dict)
                self.assertEqual(item["input_schema"].get("type"), "object")
                self.assertIn("properties", item["input_schema"])
                self.assertIn("maturity_source", item)
                self.assertIn("tags_source", item)
                self.assertIn("mcp_live", item["surface_tiers"])
        filtered = list_manifest_tools(
            maturity="stable",
            tag="validation",
        )
        self.assertGreater(len(filtered), 0)
        self.assertTrue(
            all(
                item["maturity"] == "stable"
                and "validation" in item["tags"]
                for item in filtered
            )
        )

    def test_maturity_filter_reports_unclassified_live_tools(self):
        discovery = discover_manifest_tools(
            supported=True,
            maturity="stable",
        )
        rows = discovery["tools"]
        diagnostics = discovery["filter_diagnostics"]

        self.assertGreater(len(rows), 0)
        self.assertTrue(
            all(item["maturity"] == "stable" for item in rows)
        )
        self.assertGreater(
            diagnostics["excluded_unclassified_maturity"],
            0,
        )
        self.assertIn("explicit metadata only", diagnostics["semantics"])

        all_live = {
            item["id"]: item
            for item in list_manifest_tools(supported=True)
        }
        self.assertIsNone(all_live["doctor"]["maturity"])
        self.assertEqual(
            all_live["doctor"]["maturity_source"],
            "unclassified",
        )
        self.assertNotEqual(
            all_live["doctor"]["maturity"],
            "stable",
            "guaranteed_ai_safe must not be inferred as maturity=stable",
        )

    def test_ai_surface_sets_distinguish_public_guarantees(self):
        surfaces = ai_surface_sets()
        self.assertIn("mcp_live", surfaces)
        self.assertIn("guaranteed_ai_safe", surfaces)
        self.assertIn("config_review", surfaces)
        self.assertIn("actions_public", surfaces)
        self.assertNotIn(
            "generate_config",
            surfaces["actions_public"]["tool_ids"],
        )

    def test_manifest_schema_requires_ai_surface_ssot(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (
                repo_root
                / "docs"
                / "schemas"
                / "tools_manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn("ai_surfaces", schema["required"])

    def test_manifest_override_drives_filters_and_surface_sets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "manifest.json"
            surface = {
                "availability": "test",
                "tool_ids": ["custom_tool"],
            }
            path.write_text(
                json.dumps(
                    {
                        "ai_surfaces": {
                            "mcp_live": surface,
                            "guaranteed_ai_safe": surface,
                            "config_review": surface,
                            "actions_public": surface,
                        },
                        "tools": [
                            {
                                "id": "custom_tool",
                                "maturity": "stable",
                                "tags": ["custom"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                self.assertEqual(
                    list_manifest_tools(
                        manifest_path="manifest.json",
                        guaranteed=True,
                        ids_only=True,
                    ),
                    ["custom_tool"],
                )
                self.assertEqual(
                    ai_surface_sets("manifest.json")["mcp_live"]["tool_ids"],
                    ["custom_tool"],
                )
            finally:
                os.chdir(previous)

    def test_python_manifest_override_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                for unsafe in ("../outside.json", str(outside)):
                    with self.subTest(path=unsafe):
                        with self.assertRaisesRegex(
                            ValueError,
                            "workspace|traversal",
                        ):
                            ai_surface_sets(unsafe)
            finally:
                os.chdir(previous)

    def test_ai_surface_import_and_help_do_not_create_job_storage(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            imported = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from yolozu.integrations.ai_surface import "
                        "list_manifest_tools; list_manifest_tools("
                        "guaranteed=True, ids_only=True)"
                    ),
                ],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, msg=imported.stderr)
            helped = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools" / "run_mcp_server.py"),
                    "--help",
                ],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(helped.returncode, 0, msg=helped.stderr)
            self.assertFalse((workspace / "runs").exists())

    def test_generate_config_shape_is_stable(self):
        cfg = generate_config()
        self.assertEqual(cfg.get("schema_version"), 1)
        self.assertIn("goal", cfg)
        self.assertIn("tool", cfg)
        self.assertIn("arguments", cfg)
        self.assertIn("safety", cfg)
        self.assertIsInstance(cfg.get("recommended_sequence"), list)
        self.assertGreaterEqual(len(cfg.get("recommended_sequence") or []), 1)

    def test_review_config_accepts_safe_config(self):
        cfg = generate_config()
        out = review_config(cfg, workspace_root=".")
        self.assertTrue(bool(out.get("ok")))
        self.assertEqual(out.get("schema_version"), 1)
        self.assertIn("issues", out)
        self.assertIn("warnings", out)

    def test_review_config_rejects_workspace_escape(self):
        cfg = generate_config(output="/tmp/escape.json")
        out = review_config(cfg, workspace_root=".")
        self.assertFalse(bool(out.get("ok")))
        issues = out.get("issues") or []
        codes = {str(item.get("code")) for item in issues if isinstance(item, dict)}
        self.assertIn("unsafe_output_path", codes)

    def test_review_config_rejects_normalized_relative_workspace_escape(self):
        cfg = generate_config(output="reports/../../outside.json")
        out = review_config(cfg, workspace_root=".")
        self.assertFalse(bool(out.get("ok")))
        codes = {
            str(item.get("code"))
            for item in out.get("issues") or []
            if isinstance(item, dict)
        }
        self.assertIn("unsafe_output_path", codes)

    def test_run_mcp_server_help_and_samples(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_mcp_server.py"
        self.assertTrue(script.is_file())

        help_proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(help_proc.returncode, 0, msg=f"--help failed:\n{help_proc.stdout}\n{help_proc.stderr}")
        self.assertIn("--print-tools", help_proc.stdout)
        self.assertIn("--guaranteed", help_proc.stdout)
        self.assertIn("--supported", help_proc.stdout)
        self.assertIn("--maturity", help_proc.stdout)
        self.assertIn("--tag", help_proc.stdout)
        self.assertIn("--ids-only", help_proc.stdout)
        self.assertIn("--sample-generate-config", help_proc.stdout)
        self.assertIn("--sample-review-config", help_proc.stdout)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            compact_proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--print-tools",
                    "--guaranteed",
                    "--ids-only",
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                compact_proc.returncode,
                0,
                msg=f"compact discovery failed:\n{compact_proc.stdout}\n{compact_proc.stderr}",
            )
            compact = json.loads(compact_proc.stdout)
            self.assertTrue(compact["filters"]["ids_only"])
            self.assertEqual(
                compact["selected_tool_ids"],
                [
                    "doctor",
                    "generate_config",
                    "review_config",
                    "validate_predictions",
                ],
            )
            self.assertEqual(
                compact["manifest_tools"],
                compact["selected_tool_ids"],
            )
            self.assertEqual(compact["surface_counts"]["mcp_live"], 26)
            self.assertNotIn("surfaces", compact)
            self.assertLess(len(compact_proc.stdout.encode("utf-8")), 1_500)
            self.assertEqual(compact_proc.stdout.count("\n"), 1)

            filtered_proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--print-tools",
                    "--supported",
                    "--ids-only",
                    "--maturity",
                    "stable",
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(filtered_proc.returncode, 0)
            filtered_doc = json.loads(filtered_proc.stdout)
            expected_stable = {
                item["id"]
                for item in list_manifest_tools(supported=True)
                if item["maturity"] == "stable"
            }
            self.assertEqual(
                set(filtered_doc["selected_tool_ids"]),
                expected_stable,
            )
            diagnostics = filtered_doc["filter_diagnostics"]
            self.assertGreater(
                diagnostics["excluded_unclassified_maturity"],
                0,
            )
            self.assertIn(
                "explicit metadata only",
                diagnostics["semantics"],
            )

            override_path = root / "manifest.json"
            surface = {
                "availability": "test",
                "tool_ids": ["custom_tool"],
            }
            override_path.write_text(
                json.dumps(
                    {
                        "ai_surfaces": {
                            "mcp_live": surface,
                            "guaranteed_ai_safe": surface,
                            "config_review": surface,
                            "actions_public": surface,
                        },
                        "tools": [{"id": "custom_tool"}],
                    }
                ),
                encoding="utf-8",
            )
            override_proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(override_path),
                    "--print-tools",
                    "--ids-only",
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(override_proc.returncode, 0)
            override_doc = json.loads(override_proc.stdout)
            self.assertEqual(
                override_doc["supported_mcp_tools"],
                ["custom_tool"],
            )
            self.assertEqual(override_doc["selected_tool_ids"], ["custom_tool"])
            self.assertEqual(
                override_doc["surface_counts"]["mcp_live"],
                1,
            )
            self.assertEqual(
                override_doc["manifest_tools"],
                ["custom_tool"],
            )

            with tempfile.TemporaryDirectory() as outside_td:
                outside_manifest = Path(outside_td) / "manifest.json"
                outside_manifest.write_text("{}", encoding="utf-8")
                unsafe_manifest = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--manifest",
                        str(outside_manifest),
                        "--print-tools",
                    ],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(unsafe_manifest.returncode, 2)
                self.assertEqual(
                    json.loads(unsafe_manifest.stderr)["error"]["code"],
                    "invalid_manifest",
                )

            cfg_path = root / "ai_generate_config.json"
            gen_proc = subprocess.run(
                [sys.executable, str(script), "--sample-generate-config"],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(gen_proc.returncode, 0, msg=f"generate sample failed:\n{gen_proc.stdout}\n{gen_proc.stderr}")
            cfg_path.write_text(gen_proc.stdout, encoding="utf-8")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg.get("schema_version"), 1)

            review_proc = subprocess.run(
                [sys.executable, str(script), "--sample-review-config", str(cfg_path)],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(review_proc.returncode, 0, msg=f"review sample failed:\n{review_proc.stdout}\n{review_proc.stderr}")
            review_doc = json.loads(review_proc.stdout)
            self.assertIn("ok", review_doc)
            self.assertIn("issues", review_doc)

            rejected_cfg = root / "rejected_config.json"
            rejected_cfg.write_text(
                json.dumps(generate_config(output="/tmp/yolozu-outside.json")),
                encoding="utf-8",
            )
            rejected_review = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--sample-review-config",
                    str(rejected_cfg),
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(rejected_review.returncode, 1)
            self.assertFalse(json.loads(rejected_review.stdout)["ok"])

            with tempfile.TemporaryDirectory() as outside_td:
                outside_config = Path(outside_td) / "config.json"
                outside_config.write_text("{}", encoding="utf-8")
                unsafe_review = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--sample-review-config",
                        str(outside_config),
                    ],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(unsafe_review.returncode, 2)
                self.assertEqual(
                    json.loads(unsafe_review.stderr)["error"]["code"],
                    "unsafe_or_invalid_config",
                )

    def test_run_actions_api_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_actions_api.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"run_actions_api --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--host", proc.stdout)
        self.assertIn("--port", proc.stdout)
        self.assertIn("--workers", proc.stdout)


if __name__ == "__main__":
    unittest.main()
