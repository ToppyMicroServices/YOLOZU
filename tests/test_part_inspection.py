import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yolozu.eval.part_inspection import compare_inspection, compare_inspection_files


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/eval_part_inspection.py"


def detection(width=0.05, *, cid=0, score=0.9):
    return {
        "class_id": cid,
        "score": score,
        "bbox": {"cx": 0.5, "cy": 0.5, "w": width, "h": 0.1},
    }


def inputs():
    protocol = {
        "schema_version": 1,
        "protocol_id": "synthetic-v1",
        "split": "test",
        "models": {
            "seat": {
                "model_id": "fixture-only",
                "model_sha256": "0" * 64,
                "inference_settings": {"nms": False},
            }
        },
        "class_map": {"0": "scratch", "1": "clean"},
        "score_threshold": 0.5,
        "sampling": "independent_parts",
        "evidence_kind": "synthetic",
    }
    rules = {
        "schema_version": 1,
        "ruleset_id": "synthetic",
        "version": "1",
        "precedence": "first_match",
        "regions": {
            "seat": {
                "adapter": "detections",
                "defect_class_ids": [0],
                "bad_view_threshold": 1,
            }
        },
        "rules": [
            {
                "rule_id": "size",
                "region_id": "seat",
                "class_id": 0,
                "priority": 0,
                "min_count": 1,
                "min_width": 10,
                "unit": "px",
                "source": "synthetic test, not a manufacturing specification",
            }
        ],
        "calibrations": {},
    }
    mapping = {
        "schema_version": 1,
        "label_source": "synthetic labels",
        "parts": [],
        "views": [],
    }
    predictions = []
    for pid, label in (
        ("ok-small", "OK"),
        ("nok-small", "NOK"),
        ("nok-large", "NOK"),
        ("ok-empty", "OK"),
        ("missing-nok", "NOK"),
        ("missing-ok", "OK"),
    ):
        mapping["parts"].append(
            {
                "part_id": pid,
                "label": label,
                "split": "test",
                "expected_views": [
                    {"view_id": v, "region_id": "seat"} for v in ("a", "b")
                ],
            }
        )
        for vid in ("a", "b"):
            image = f"{pid}-{vid}.png"
            mapping["views"].append(
                {
                    "image": image,
                    "part_id": pid,
                    "view_id": vid,
                    "region_id": "seat",
                    "width_px": 100,
                    "height_px": 100,
                }
            )
            if pid.startswith("missing-") and vid == "b":
                continue
            dets = (
                []
                if pid == "ok-empty"
                else [detection(0.2 if pid == "nok-large" else 0.05)]
            )
            predictions.append(
                {"schema_version": 2, "image": image, "detections": dets}
            )
    return {
        "predictions": predictions,
        "mapping": mapping,
        "rules": rules,
        "protocol": protocol,
    }


class TestPartInspection(unittest.TestCase):
    def test_harmful_filter_and_full_cohort_accounting(self):
        data = inputs()
        original = copy.deepcopy(data)
        report = compare_inspection(**data)
        self.assertEqual(data, original)
        self.assertFalse(report["production_qualified"])
        self.assertEqual(report["outcome"], "comparison_only")
        self.assertEqual(
            report["arms"]["baseline"]["counts"],
            {"TP": 2, "FN": 0, "FP": 1, "TN": 1, "abstained_OK": 1, "abstained_NOK": 1},
        )
        self.assertEqual(
            report["arms"]["rules"]["counts"],
            {"TP": 1, "FN": 1, "FP": 0, "TN": 2, "abstained_OK": 1, "abstained_NOK": 1},
        )
        paired = report["paired"]
        self.assertEqual(paired["repaired_false_rejects"], 1)
        self.assertEqual(paired["introduced_false_accepts"], 1)
        self.assertEqual(paired["repaired_false_accepts"], 0)
        self.assertEqual(paired["introduced_false_rejects"], 0)
        self.assertEqual(paired["common_decided_support"], 4)
        self.assertEqual(sum(paired["transitions"].values()), 6)
        metrics = report["arms"]["rules"]["metrics"]
        for key, num, den in (
            ("nok_recall", 1, 2),
            ("false_reject_rate", 0, 2),
            ("decision_coverage", 4, 6),
            ("nok_detection_yield", 1, 3),
            ("correct_accept_yield", 2, 3),
        ):
            self.assertEqual(
                (metrics[key]["numerator"], metrics[key]["denominator"]), (num, den)
            )
        self.assertGreater(
            report["arms"]["baseline"]["metrics"]["nok_recall"]["wilson_95"][0], 0
        )
        self.assertEqual(report["image_metrics"]["status"], "not_computed")

    def test_permutations_preserve_decisions_metrics_and_inputs(self):
        data = inputs()
        expected = compare_inspection(**data)
        randomizer = random.Random(17)
        for _ in range(12):
            changed = copy.deepcopy(data)
            randomizer.shuffle(changed["predictions"])
            randomizer.shuffle(changed["mapping"]["parts"])
            randomizer.shuffle(changed["mapping"]["views"])
            for part in changed["mapping"]["parts"]:
                randomizer.shuffle(part["expected_views"])
            for pred in changed["predictions"]:
                randomizer.shuffle(pred["detections"])
            report = compare_inspection(**changed)
            for key in ("arms", "paired", "parts"):
                self.assertEqual(report[key], expected[key])
        self.assertEqual(compare_inspection(**data), expected)

    def test_exact_score_and_dimension_boundaries(self):
        for width, score, expected in (
            (0.1, 0.5, True),
            (math.nextafter(0.1, 0), 0.5, False),
            (0.1, math.nextafter(0.5, 0), False),
        ):
            with self.subTest(width=width, score=score):
                data = inputs()
                data["predictions"][0]["detections"] = [detection(width, score=score)]
                report = compare_inspection(**data)
                part = next(p for p in report["parts"] if p["part_id"] == "ok-small")
                trace = next(v for v in part["views"] if v["view_id"] == "a")
                self.assertEqual(trace["rules_bad"], expected)

    def test_min_count_is_per_frame_not_across_views(self):
        data = inputs()
        data["rules"]["rules"][0]["min_count"] = 2
        report = compare_inspection(**data)
        part = next(p for p in report["parts"] if p["part_id"] == "nok-large")
        self.assertEqual(part["rules"], "OK")  # one candidate in each of two views
        pred = next(p for p in data["predictions"] if p["image"] == "nok-large-a.png")
        pred["detections"].append(detection(0.2))
        part = next(
            p
            for p in compare_inspection(**data)["parts"]
            if p["part_id"] == "nok-large"
        )
        self.assertEqual(part["rules"], "NOK")

    def test_mm_dimensions_and_anisotropic_calibration(self):
        data = inputs()
        data["rules"]["rules"][0].update(unit="mm", min_width=1, min_height=2)
        data["rules"]["calibrations"] = {
            "c1": {"unit": "mm_per_pixel", "x": 0.1, "y": 0.2, "source": "synthetic"}
        }
        for view in data["mapping"]["views"]:
            view["calibration_id"] = "c1"
        part = next(
            p
            for p in compare_inspection(**data)["parts"]
            if p["part_id"] == "nok-large"
        )
        self.assertEqual(part["rules"], "NOK")
        data["rules"]["calibrations"]["c1"]["y"] = 0.1
        self.assertEqual(compare_inspection(**data)["arms"]["rules"]["counts"]["TP"], 0)

    def test_first_match_precedence_does_not_fall_through_dimensions(self):
        data = inputs()
        second = {
            **data["rules"]["rules"][0],
            "rule_id": "later",
            "priority": 1,
            "min_width": 0,
        }
        data["rules"]["rules"].insert(0, second)
        report = compare_inspection(**data)
        part = next(p for p in report["parts"] if p["part_id"] == "ok-small")
        self.assertEqual(part["rules"], "OK")
        self.assertEqual(part["views"][0]["classes"][0]["shadowed_rule_ids"], ["later"])
        data["rules"]["rules"].reverse()
        self.assertEqual(compare_inspection(**data)["parts"], report["parts"])
        data["rules"]["rules"] = []
        report = compare_inspection(**data)
        self.assertEqual(report["arms"]["rules"], report["arms"]["baseline"])

    def test_classification_adapter_and_region_aggregation(self):
        data = inputs()
        data["rules"]["regions"]["seat"].update(
            adapter="classification", bad_view_threshold=2
        )
        data["rules"]["rules"] = []
        for view in data["mapping"]["views"]:
            del view["width_px"], view["height_px"]
        for pred in data["predictions"]:
            bad = bool(pred.pop("detections")) and pred["image"].endswith("-a.png")
            pred["classification"] = {"class_id": 0 if bad else 1, "score": 0.9}
        report = compare_inspection(**data)
        self.assertEqual(report["arms"]["baseline"]["counts"]["TP"], 2)
        self.assertEqual(report["arms"]["rules"]["counts"]["TP"], 0)
        next(p for p in data["predictions"] if p["image"] == "nok-large-b.png")[
            "classification"
        ]["class_id"] = 0
        self.assertEqual(compare_inspection(**data)["arms"]["rules"]["counts"]["TP"], 1)

    def test_mixed_region_adapters(self):
        data = inputs()
        data["protocol"]["models"]["shaft"] = copy.deepcopy(
            data["protocol"]["models"]["seat"]
        )
        data["rules"]["regions"]["shaft"] = {
            "adapter": "classification",
            "defect_class_ids": [0],
            "bad_view_threshold": 1,
        }
        for part in data["mapping"]["parts"]:
            part["expected_views"].append({"view_id": "shaft", "region_id": "shaft"})
            image = part["part_id"] + "-shaft.png"
            data["mapping"]["views"].append(
                {
                    "image": image,
                    "part_id": part["part_id"],
                    "view_id": "shaft",
                    "region_id": "shaft",
                }
            )
            data["predictions"].append(
                {"image": image, "classification": {"class_id": 1, "score": 0.8}}
            )
        self.assertEqual(compare_inspection(**data)["arms"]["rules"]["counts"]["TP"], 1)

    def test_no_positive_no_negative_and_dependent_samples(self):
        for label in ("OK", "NOK"):
            data = inputs()
            for part in data["mapping"]["parts"]:
                part["label"] = label
            report = compare_inspection(**data)
            absent = "nok_recall" if label == "OK" else "false_reject_rate"
            self.assertIsNone(report["arms"]["rules"]["metrics"][absent]["value"])
            self.assertIsNone(report["arms"]["rules"]["metrics"][absent]["wilson_95"])
        data = inputs()
        data["protocol"]["sampling"] = "dependent_parts"
        report = compare_inspection(**data)
        self.assertIsNone(report["sampling"]["interval_method"])
        self.assertTrue(
            all(
                v["wilson_95"] is None
                for v in report["arms"]["rules"]["metrics"].values()
            )
        )

    def test_all_abstained_and_missing_mapping(self):
        for missing in ("predictions", "views"):
            data = inputs()
            data["predictions"] = []
            if missing == "views":
                data["mapping"]["views"] = []
            report = compare_inspection(**data)
            self.assertEqual(report["paired"]["common_decided_support"], 0)
            self.assertEqual(
                report["arms"]["rules"]["metrics"]["decision_coverage"]["value"], 0
            )
            self.assertIsNone(report["arms"]["rules"]["metrics"]["nok_recall"]["value"])
            self.assertEqual(
                report["arms"]["rules"]["metrics"]["nok_detection_yield"]["value"], 0
            )

    def test_invalid_input_matrix_fails_before_scoring(self):
        mutations = {
            "duplicate prediction": lambda d: d["predictions"].append(
                copy.deepcopy(d["predictions"][0])
            ),
            "duplicate image mapping": lambda d: d["mapping"]["views"].append(
                copy.deepcopy(d["mapping"]["views"][0])
            ),
            "duplicate view": lambda d: d["mapping"]["views"].append(
                {**d["mapping"]["views"][0], "image": "other.png"}
            ),
            "cross-split part": lambda d: d["mapping"]["parts"].append(
                {**d["mapping"]["parts"][0], "split": "train"}
            ),
            "conflicting label": lambda d: d["mapping"]["parts"].append(
                {**d["mapping"]["parts"][0], "label": "NOK"}
            ),
            "duplicate expected view": lambda d: d["mapping"]["parts"][0][
                "expected_views"
            ].append({"view_id": "a", "region_id": "seat"}),
            "unknown prediction image": lambda d: d["predictions"][0].update(
                image="not-mapped.png"
            ),
            "unknown part": lambda d: d["mapping"]["views"][0].update(
                part_id="unknown"
            ),
            "unknown region": lambda d: d["mapping"]["views"][0].update(
                region_id="unknown"
            ),
            "view outside expected set": lambda d: d["mapping"]["views"][0].update(
                view_id="extra"
            ),
            "unsupported rule field": lambda d: d["rules"]["rules"][0].update(
                action="ignore"
            ),
            "bad units": lambda d: d["rules"]["rules"][0].update(unit="inch"),
            "missing units": lambda d: d["rules"]["rules"][0].pop("unit"),
            "missing calibration": lambda d: d["rules"]["rules"][0].update(unit="mm"),
            "unknown calibration": lambda d: d["mapping"]["views"][0].update(
                calibration_id="unknown"
            ),
            "bad geometry": lambda d: d["mapping"]["views"][0].update(width_px=0),
            "impossible view threshold": lambda d: d["rules"]["regions"]["seat"].update(
                bad_view_threshold=3
            ),
            "bad count": lambda d: d["rules"]["rules"][0].update(min_count=0),
            "ambiguous priority": lambda d: d["rules"]["rules"].append(
                {**d["rules"]["rules"][0], "rule_id": "overlap"}
            ),
            "duplicate rule id": lambda d: d["rules"]["rules"].append(
                {**d["rules"]["rules"][0], "priority": 2}
            ),
            "invalid label": lambda d: d["mapping"]["parts"][0].update(label="unknown"),
            "unknown adapter": lambda d: d["rules"]["regions"]["seat"].update(
                adapter="magic"
            ),
            "unknown class": lambda d: d["predictions"][0]["detections"][0].update(
                class_id=99
            ),
            "bool class": lambda d: d["predictions"][0]["detections"][0].update(
                class_id=False
            ),
            "nonfinite score": lambda d: d["predictions"][0]["detections"][0].update(
                score=float("nan")
            ),
            "out-of-range score": lambda d: d["predictions"][0]["detections"][0].update(
                score=2
            ),
            "bad bbox": lambda d: d["predictions"][0]["detections"][0]["bbox"].update(
                w=-1
            ),
            "missing detections": lambda d: d["predictions"][0].pop("detections"),
            "null detections": lambda d: d["predictions"][0].update(detections=None),
            "malformed entry": lambda d: d["predictions"].append(None),
            "missing model digest": lambda d: d["protocol"]["models"]["seat"].pop(
                "model_sha256"
            ),
            "empty cohort": lambda d: d["protocol"].update(split="absent"),
            "unsupported version": lambda d: d["mapping"].update(schema_version=2),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                data = inputs()
                mutate(data)
                with self.assertRaises(ValueError):
                    compare_inspection(**data)

    def test_wrapped_predictions_and_non_evaluation_parts(self):
        data = inputs()
        data["predictions"] = {
            "schema_version": 1,
            "predictions": data["predictions"],
            "meta": {"origin": "synthetic"},
        }
        data["mapping"]["parts"][0]["split"] = "train"
        report = compare_inspection(**data)
        self.assertEqual(report["arms"]["baseline"]["support"]["total"], 5)
        self.assertNotIn("ok-small", [p["part_id"] for p in report["parts"]])

    def test_calibration_overflow_fails_even_on_empty_predictions(self):
        data = inputs()
        data["rules"]["rules"][0]["unit"] = "mm"
        for scale in (1e308, 5e-324):
            data["rules"]["calibrations"] = {
                "bad": {
                    "unit": "mm_per_pixel",
                    "x": scale,
                    "y": scale,
                    "source": "invalid synthetic",
                }
            }
            for view in data["mapping"]["views"]:
                view["calibration_id"] = "bad"
            data["predictions"] = []
            with self.assertRaisesRegex(ValueError, "overflow or underflow"):
                compare_inspection(**data)

    def test_all_possible_filtering_transitions_and_monotonicity(self):
        data = inputs()
        randomizer = random.Random(59)
        for _ in range(60):
            data["rules"]["rules"][0].update(
                min_width=randomizer.randint(0, 50), min_count=randomizer.randint(1, 3)
            )
            data["rules"]["regions"]["seat"]["bad_view_threshold"] = randomizer.randint(
                1, 2
            )
            for pred in data["predictions"]:
                pred["detections"] = [
                    detection(randomizer.uniform(0.001, 1), score=randomizer.random())
                    for _ in range(randomizer.randint(0, 4))
                ]
            report = compare_inspection(**data)
            for part in report["parts"]:
                if part["baseline"] == "OK":
                    self.assertEqual(part["rules"], "OK")
                self.assertEqual(
                    part["baseline"] == "ABSTAIN", part["rules"] == "ABSTAIN"
                )
            for arm in ("baseline", "rules"):
                self.assertEqual(sum(report["arms"][arm]["counts"].values()), 6)
        # OK->NOK and one-arm-only abstentions are impossible in this lane.
        # The reachable transitions are covered by the labeled fixture below.
        data = inputs()
        data["rules"]["rules"] = []
        report = compare_inspection(**data)
        self.assertEqual(report["paired"]["transitions"]["OK:NOK->NOK"], 1)
        for pred in data["predictions"]:
            pred["detections"] = []
        report = compare_inspection(**data)
        self.assertEqual(report["paired"]["transitions"]["NOK:OK->OK"], 2)


class TestPartInspectionFiles(unittest.TestCase):
    def write_inputs(self, directory):
        paths = {}
        for name, value in inputs().items():
            path = Path(directory) / f"{name}.json"
            path.write_text(json.dumps(value, indent=2) + "\n")
            paths[name] = path
        return paths

    def run_cli(self, paths, output):
        args = [arg for k, v in paths.items() for arg in (f"--{k}", str(v))]
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args, "--output", str(output)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ROOT,
        )

    def test_cli_help(self):
        for flag in ("-h", "--help"):
            result = subprocess.run(
                [sys.executable, str(SCRIPT), flag],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--mapping", result.stdout)

    def test_committed_example(self):
        base = ROOT / "docs/examples/part_inspection"
        paths = {
            key: base / f"{key}.json"
            for key in ("predictions", "mapping", "rules", "protocol")
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "report.json"
            result = self.run_cli(paths, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text())
            self.assertEqual(report["evidence_kind"], "synthetic")
            self.assertEqual(
                report["arms"]["rules"]["counts"],
                {
                    "TP": 1,
                    "FP": 0,
                    "FN": 1,
                    "TN": 1,
                    "abstained_OK": 0,
                    "abstained_NOK": 0,
                },
            )

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO")
    def test_fifo_input_fails_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            fifo = Path(directory) / "input.fifo"
            os.mkfifo(fifo)
            paths["predictions"] = fifo
            result = self.run_cli(paths, Path(directory) / "result.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("regular file", result.stderr)

    def test_concurrent_publish_has_one_winner_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            args = [arg for k, v in paths.items() for arg in (f"--{k}", str(v))]
            output = Path(directory) / "shared.json"
            processes = []
            try:
                for _ in range(3):
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(SCRIPT),
                                *args,
                                "--output",
                                str(output),
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                    )
                results = [process.communicate(timeout=15) for process in processes]
                self.assertEqual(
                    sorted(p.returncode for p in processes), [0, 2, 2], results
                )
                self.assertEqual(
                    json.loads(output.read_text())["kind"],
                    "yolozu_part_inspection_comparison",
                )
                self.assertEqual(list(Path(directory).glob(".part-inspection-*")), [])
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def test_cli_bytes_hashes_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            before = {k: p.read_bytes() for k, p in paths.items()}
            out = Path(directory) / "report.json"
            result = self.run_cli(paths, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text())
            self.assertEqual(
                report["provenance"]["input_sha256"],
                {k: hashlib.sha256(v).hexdigest() for k, v in before.items()},
            )
            saved = out.read_bytes()
            self.assertEqual(self.run_cli(paths, out).returncode, 2)
            self.assertEqual(out.read_bytes(), saved)
            for path in paths.values():
                self.assertEqual(self.run_cli(paths, path).returncode, 2)
            self.assertEqual({k: p.read_bytes() for k, p in paths.items()}, before)
            self.assertEqual(list(Path(directory).glob(".part-inspection-*")), [])
            second = Path(directory) / "report2.json"
            self.assertEqual(self.run_cli(paths, second).returncode, 0)
            self.assertEqual(second.read_bytes(), saved)

    def test_hardlink_symlink_outputs_preserve_input(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            original = paths["predictions"].read_bytes()
            for kind in ("hard", "soft"):
                output = Path(directory) / kind
                if kind == "hard":
                    os.link(paths["predictions"], output)
                else:
                    output.symlink_to(paths["predictions"])
                self.assertEqual(self.run_cli(paths, output).returncode, 2)
            self.assertEqual(paths["predictions"].read_bytes(), original)

    def test_malformed_json_and_size_limit_do_not_write_report(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            with patch("yolozu.eval.part_inspection.MAX_INPUT_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    compare_inspection_files(**paths)
            for invalid in (
                '{"parts": [], "parts": []}',
                '{"predictions": [NaN]}',
                "{",
            ):
                paths["predictions"].write_text(invalid)
                out = Path(directory) / "absent" / "report.json"
                result = self.run_cli(paths, out)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(out.parent.exists())

    def test_input_changed_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            before = paths["predictions"].stat()
            changed = SimpleNamespace(
                st_size=before.st_size,
                st_mtime_ns=before.st_mtime_ns + 1,
                st_ctime_ns=before.st_ctime_ns,
            )
            with patch(
                "yolozu.eval.part_inspection.os.fstat", side_effect=[before, changed]
            ):
                with self.assertRaisesRegex(ValueError, "changed while reading"):
                    compare_inspection_files(**paths)

    def test_output_failure_cleans_temporary_file(self):
        from tools.eval_part_inspection import main

        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(directory)
            args = [arg for k, v in paths.items() for arg in (f"--{k}", str(v))]
            out = Path(directory) / "result.json"
            with patch(
                "tools.eval_part_inspection.os.link",
                side_effect=OSError("simulated failure"),
            ):
                self.assertEqual(main([*args, "--output", str(out)]), 2)
            self.assertFalse(out.exists())
            self.assertEqual(list(Path(directory).glob(".part-inspection-*")), [])


if __name__ == "__main__":
    unittest.main()
