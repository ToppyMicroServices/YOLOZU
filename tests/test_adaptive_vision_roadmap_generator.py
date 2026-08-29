import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path


def _load_generator():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "generate_adaptive_vision_roadmap.py"
    spec = importlib.util.spec_from_file_location("generate_adaptive_vision_roadmap", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAdaptiveVisionRoadmapGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.source = self.repo_root / "yolozu" / "data" / "manifest" / "adaptive_vision_roadmap.json"
        self.output = self.repo_root / "reports" / "adaptive_vision_roadmap.md"
        self.script = self.repo_root / "tools" / "generate_adaptive_vision_roadmap.py"

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_generated_report_is_current(self) -> None:
        proc = self._run("--check", "--json")
        if proc.returncode != 0:
            self.fail(f"roadmap report drifted:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["drifted"])
        self.assertEqual(payload["phases"], 5)
        self.assertEqual(payload["root_issue_id"], "YOLOZU-ll2.81")

    def test_projection_keeps_current_and_future_boundaries_separate(self) -> None:
        payload = json.loads(self.source.read_text(encoding="utf-8"))
        boundary = payload["product_boundary"]
        self.assertEqual(payload["kind"], "adaptive_vision_roadmap_projection")
        self.assertEqual(boundary["public_availability"], "future_experimental_work")
        self.assertEqual(boundary["target_maturity"], "Experimental")
        self.assertIn("prediction validation and evaluation", boundary["current_stable_lane"])
        self.assertIn("otherwise abstain", boundary["recommendation_boundary"])
        self.assertIn("does not embed an LLM", boundary["natural_language_boundary"])
        self.assertIn("no implicit download", boundary["execution_boundary"])
        self.assertIn("official hardware", payload["guardrails"][-1])
        self.assertIn("renewed community demand", payload["phases"][4]["deliverable"])
        self.assertIn("optional community evidence", payload["phases"][3]["deliverable"])
        self.assertEqual(payload["scope"]["initial_tasks"], ["object_detection", "instance_segmentation"])
        self.assertEqual([phase["issue_id"] for phase in payload["phases"]], [f"YOLOZU-ll2.81.{n}" for n in range(1, 6)])
        self.assertNotIn("owner", payload)
        self.assertNotIn("assignee", payload)

    def test_report_states_that_it_is_not_qualification_evidence(self) -> None:
        report = self.output.read_text(encoding="utf-8")
        self.assertIn("This is a roadmap projection, not qualification evidence.", report)
        self.assertIn("current Stable lane remains prediction validation and evaluation", report)
        self.assertIn("Discovery never changes the selectable channel by itself.", report)
        self.assertIn("The phase table is a scope projection, not live task status", report)
        self.assertIn("Retain contract-only streaming, tracking, and OCR lanes", report)
        self.assertNotIn("universally best", report.lower())
        self.assertNotIn("always latest", report.lower())

    def test_schema_copy_and_packaged_projection_are_present(self) -> None:
        canonical = self.repo_root / "docs" / "schemas" / "adaptive_vision_roadmap.schema.json"
        packaged = self.repo_root / "yolozu" / "data" / "schemas" / "adaptive_vision_roadmap.schema.json"
        self.assertEqual(canonical.read_bytes(), packaged.read_bytes())
        resource = files("yolozu.data").joinpath("manifest").joinpath("adaptive_vision_roadmap.json")
        resource_payload = json.loads(resource.read_text(encoding="utf-8"))
        self.assertEqual(resource_payload["snapshot"]["root_issue_id"], "YOLOZU-ll2.81")

    def test_public_docs_link_to_the_report_and_projection(self) -> None:
        expected = {
            "README.md": "reports/adaptive_vision_roadmap.md",
            "Readme_jp.md": "reports/adaptive_vision_roadmap.md",
            "docs/README.md": "../reports/adaptive_vision_roadmap.md",
            "docs/roadmap.md": "../reports/adaptive_vision_roadmap.md",
            "docs/production_readiness.md": "../reports/adaptive_vision_roadmap.md",
        }
        for rel, link in expected.items():
            with self.subTest(path=rel):
                text = (self.repo_root / rel).read_text(encoding="utf-8")
                self.assertIn(link, text)
                self.assertIn("adaptive_vision_roadmap.json", text)

    def test_check_mode_detects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.repo_root / "reports") as temp_dir:
            output = Path(temp_dir) / "roadmap.md"
            output.write_text("# stale\n", encoding="utf-8")
            proc = self._run("--output", str(output), "--check", "--json")
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(json.loads(proc.stdout)["drifted"])

    def test_duplicate_keys_fail_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.repo_root / "reports") as temp_dir:
            source = Path(temp_dir) / "bad.json"
            source.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
            proc = self._run("--source", str(source), "--check", "--json")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("duplicate JSON key", json.loads(proc.stdout)["error"])

    def test_validator_rejects_a_public_availability_overclaim(self) -> None:
        generator = _load_generator()
        payload = json.loads(self.source.read_text(encoding="utf-8"))
        payload["product_boundary"]["public_availability"] = "available"
        with self.assertRaisesRegex(ValueError, "future_experimental_work"):
            generator.validate_projection(payload)

    def test_schema_order_constraints_match_the_semantic_validator(self) -> None:
        generator = _load_generator()
        schema = json.loads(
            (self.repo_root / "docs" / "schemas" / "adaptive_vision_roadmap.schema.json").read_text(
                encoding="utf-8"
            )
        )
        properties = schema["properties"]
        self.assertEqual(
            [item["const"] for item in properties["continuous_intake"]["prefixItems"]],
            generator.INTAKE_FLOW,
        )
        self.assertFalse(properties["continuous_intake"]["items"])

        expected_scope = {
            "initial_inputs": ["single_image", "bounded_directory"],
            "initial_tasks": ["object_detection", "instance_segmentation"],
            "later_bounded_lanes": ["local_stream", "session_tracking", "static_image_ocr"],
        }
        for key, expected in expected_scope.items():
            with self.subTest(scope=key):
                field = properties["scope"]["properties"][key]
                self.assertEqual([item["const"] for item in field["prefixItems"]], expected)
                self.assertFalse(field["items"])

        phase_prefixes = properties["phases"]["prefixItems"]
        self.assertFalse(properties["phases"]["items"])
        self.assertEqual(len(phase_prefixes), 5)
        for index, phase_schema in enumerate(phase_prefixes, start=1):
            exact = phase_schema["allOf"][1]["properties"]
            self.assertEqual(exact["order"]["const"], index)
            self.assertEqual(exact["issue_id"]["const"], f"YOLOZU-ll2.81.{index}")

        payload = json.loads(self.source.read_text(encoding="utf-8"))
        payload["continuous_intake"][0], payload["continuous_intake"][1] = (
            payload["continuous_intake"][1],
            payload["continuous_intake"][0],
        )
        with self.assertRaisesRegex(ValueError, "governed ordered flow"):
            generator.validate_projection(payload)

    def test_validator_enforces_published_schema_string_limits(self) -> None:
        generator = _load_generator()
        schema = json.loads(
            (self.repo_root / "docs" / "schemas" / "adaptive_vision_roadmap.schema.json").read_text(
                encoding="utf-8"
            )
        )
        properties = schema["properties"]
        cases = [
            (
                "guardrails item",
                properties["guardrails"]["items"]["maxLength"],
                lambda value, text: value["guardrails"].__setitem__(0, text),
            ),
            (
                "out_of_scope item",
                properties["out_of_scope"]["items"]["maxLength"],
                lambda value, text: value["out_of_scope"].__setitem__(0, text),
            ),
            (
                "phase title",
                schema["$defs"]["phase"]["properties"]["title"]["maxLength"],
                lambda value, text: value["phases"][0].__setitem__("title", text),
            ),
            (
                "phase deliverable",
                schema["$defs"]["phase"]["properties"]["deliverable"]["maxLength"],
                lambda value, text: value["phases"][0].__setitem__("deliverable", text),
            ),
            (
                "current stable lane",
                properties["product_boundary"]["properties"]["current_stable_lane"]["maxLength"],
                lambda value, text: value["product_boundary"].__setitem__("current_stable_lane", text),
            ),
            (
                "execution boundary",
                properties["product_boundary"]["properties"]["execution_boundary"]["maxLength"],
                lambda value, text: value["product_boundary"].__setitem__("execution_boundary", text),
            ),
            (
                "natural language boundary",
                properties["product_boundary"]["properties"]["natural_language_boundary"]["maxLength"],
                lambda value, text: value["product_boundary"].__setitem__("natural_language_boundary", text),
            ),
            (
                "recommendation boundary",
                properties["product_boundary"]["properties"]["recommendation_boundary"]["maxLength"],
                lambda value, text: value["product_boundary"].__setitem__("recommendation_boundary", text),
            ),
            (
                "selection factor",
                properties["scope"]["properties"]["selection_factors"]["items"]["maxLength"],
                lambda value, text: value["scope"]["selection_factors"].__setitem__(0, text),
            ),
            (
                "live task state",
                properties["source_of_truth"]["properties"]["live_task_state"]["maxLength"],
                lambda value, text: value["source_of_truth"].__setitem__("live_task_state", text),
            ),
        ]
        base = json.loads(self.source.read_text(encoding="utf-8"))
        for label, maximum, mutate in cases:
            with self.subTest(field=label):
                payload = copy.deepcopy(base)
                mutate(payload, "x" * (maximum + 1))
                with self.assertRaises(ValueError):
                    generator.validate_projection(payload)

    def test_external_paths_and_source_output_alias_fail_without_path_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            external = Path(temp_dir) / "private-roadmap.json"
            external.write_text(self.source.read_text(encoding="utf-8"), encoding="utf-8")
            proc = self._run("--source", str(external), "--check", "--json")
        self.assertEqual(proc.returncode, 2)
        payload = json.loads(proc.stdout)
        self.assertIn("must stay within the repository", payload["error"])
        self.assertNotIn(temp_dir, proc.stdout + proc.stderr)

        alias = self._run("--source", str(self.source), "--output", str(self.source), "--check", "--json")
        self.assertEqual(alias.returncode, 2)
        self.assertIn("source and output must be different", json.loads(alias.stdout)["error"])

        with tempfile.TemporaryDirectory(dir=self.repo_root / "reports") as temp_dir:
            hard_link = Path(temp_dir) / "source-hard-link.json"
            os.link(self.source, hard_link)
            hard_link_alias = self._run(
                "--source",
                str(self.source),
                "--output",
                str(hard_link),
                "--check",
                "--json",
            )
        self.assertEqual(hard_link_alias.returncode, 2)
        self.assertIn("must not reference the same file", json.loads(hard_link_alias.stdout)["error"])

    def test_help_is_side_effect_free(self) -> None:
        before = self.output.read_bytes()
        proc = self._run("--help")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--check", proc.stdout)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
