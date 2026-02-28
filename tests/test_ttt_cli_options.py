import argparse
import unittest

from yolozu.tta.cli_options import (
    add_ttt_arguments,
    build_ttt_cli_args,
    build_ttt_config_from_args,
    build_ttt_settings_from_args,
)


def _parse_ttt_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    add_ttt_arguments(parser, include_enable_flag=True)
    return parser.parse_args(argv)


class TestTTTCLIOptions(unittest.TestCase):
    def test_parser_accepts_task_specific_presets(self):
        ns_pose = _parse_ttt_args(["--ttt", "--ttt-preset", "pose_safe"])
        self.assertEqual(ns_pose.ttt_preset, "pose_safe")
        ns_pose_mim = _parse_ttt_args(["--ttt", "--ttt-preset", "pose_mim"])
        self.assertEqual(ns_pose_mim.ttt_preset, "pose_mim")

    def test_builders_wire_sdft_and_aux_fields(self):
        ns = _parse_ttt_args(
            [
                "--ttt",
                "--ttt-sdft-task",
                "depth",
                "--ttt-aux-pose-weight",
                "0.2",
                "--ttt-aux-keypoints-weight",
                "0.3",
                "--ttt-aux-depth-weight",
                "0.4",
                "--ttt-aux-seg-weight",
                "0.5",
                "--ttt-aux-temperature",
                "1.3",
            ]
        )
        cfg = build_ttt_config_from_args(ns)
        self.assertEqual(cfg.sdft_task, "depth")
        self.assertEqual(cfg.aux_pose_weight, 0.2)
        self.assertEqual(cfg.aux_keypoints_weight, 0.3)
        self.assertEqual(cfg.aux_depth_weight, 0.4)
        self.assertEqual(cfg.aux_seg_weight, 0.5)
        self.assertEqual(cfg.aux_temperature, 1.3)

        settings = build_ttt_settings_from_args(ns)
        self.assertEqual(settings["sdft_task"], "depth")
        self.assertEqual(settings["aux"]["pose_weight"], 0.2)
        self.assertEqual(settings["aux"]["keypoints_weight"], 0.3)
        self.assertEqual(settings["aux"]["depth_weight"], 0.4)
        self.assertEqual(settings["aux"]["seg_weight"], 0.5)
        self.assertEqual(settings["aux"]["temperature"], 1.3)
        self.assertIsNone(settings["include"])
        self.assertIsNone(settings["exclude"])

    def test_build_ttt_cli_args_serializes_new_flags(self):
        ns = _parse_ttt_args(
            [
                "--ttt",
                "--ttt-sdft-task",
                "seg",
                "--ttt-aux-seg-weight",
                "0.75",
                "--ttt-aux-temperature",
                "0.9",
            ]
        )
        cmd = build_ttt_cli_args(ns, include_enable_flag=True)
        self.assertIn("--ttt", cmd)
        self.assertIn("--ttt-sdft-task", cmd)
        self.assertIn("seg", cmd)
        self.assertIn("--ttt-aux-seg-weight", cmd)
        self.assertIn("0.75", cmd)
        self.assertIn("--ttt-aux-temperature", cmd)
        self.assertIn("0.9", cmd)


if __name__ == "__main__":
    unittest.main()
