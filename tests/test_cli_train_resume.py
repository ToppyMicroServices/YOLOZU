import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from yolozu import cli_commands, cli_entry


class TestCLITrainResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.config = self.root / "train.json"
        self.config.write_text("{}\n", encoding="utf-8")

    def test_reference_resume_from_is_forwarded_once(self):
        checkpoint = self.root / "saved bundle.pt"
        for flag_args in (["--resume-from", str(checkpoint)], [f"--resume-from={checkpoint}"]):
            with self.subTest(flag_args=flag_args), patch.object(cli_entry, "_cmd_train", return_value=0) as train:
                code = cli_entry.main(["train", str(self.config), *flag_args, "--epochs", "2"])
                self.assertEqual(code, 0)
                train.assert_called_once_with(
                    self.config, extra_args=["--epochs", "2", "--resume-from", str(checkpoint)]
                )

    def test_reference_missing_checkpoint_is_not_discarded(self):
        missing = self.root / "missing.pt"
        with patch.object(cli_entry, "_cmd_train", side_effect=SystemExit("checkpoint not found")) as train:
            with self.assertRaisesRegex(SystemExit, "checkpoint not found"):
                cli_entry.main(["train", str(self.config), "--resume-from", str(missing)])
        train.assert_called_once_with(self.config, extra_args=["--resume-from", str(missing)])

    def test_reference_resume_boolean_is_not_resume_from_abbreviation(self):
        with patch.object(cli_entry, "_cmd_train", return_value=0) as train:
            code = cli_entry.main(["train", str(self.config), "--resume", "--epochs", "2"])
        self.assertEqual(code, 0)
        train.assert_called_once_with(self.config, extra_args=["--epochs", "2", "--resume"])

    def test_reference_trainer_flags_are_not_abbreviated_by_outer_parser(self):
        with patch.object(cli_entry, "_cmd_train", return_value=0) as train:
            code = cli_entry.main(["train", str(self.config), "--resume-fr", "some.pt"])
        self.assertEqual(code, 0)
        train.assert_called_once_with(self.config, extra_args=["--resume-fr", "some.pt"])

    def test_external_resume_from_keeps_backend_and_other_flags(self):
        checkpoint = self.root / "external.pt"
        for backend in ("yolox", "detectron2", "mmdetection", "mmpose", "mmseg", "tao", "ultralytics", "hf-detr"):
            with self.subTest(backend=backend), patch.object(cli_entry, "_cmd_train_external", return_value=0) as train:
                code = cli_entry.main([
                    "train", str(self.config), "--external-backend", backend,
                    "--resume-from", str(checkpoint), "--dry-run", "--dataset", "data/smoke",
                ])
                self.assertEqual(code, 0)
                parsed = train.call_args.args[0]
                self.assertEqual(parsed.external_backend, backend)
                self.assertEqual(parsed.config, str(self.config))
                self.assertEqual(
                    train.call_args.kwargs["extra_args"],
                    ["--dry-run", "--dataset", "data/smoke", "--resume-from", str(checkpoint)],
                )

    def test_public_tao_task_reaches_external_helper_once(self):
        for flag_args in (["--tao-task", "segformer"], ["--tao-task=segformer"]):
            with self.subTest(flag_args=flag_args), patch.object(
                cli_commands.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as launch:
                code = cli_entry.main([
                    "train", str(self.config), "--external-backend", "tao",
                    *flag_args, "--dry-run", "--dataset", "data/smoke",
                ])
                self.assertEqual(code, 0)
                command = launch.call_args.args[0]
                self.assertEqual(command[2], "train-tao")
                self.assertEqual(command.count("--tao-task"), 1)
                self.assertEqual(command[command.index("--tao-task") + 1], "segformer")
                self.assertIn("--dry-run", command)
                self.assertEqual(command[command.index("--dataset") + 1], "data/smoke")

    def test_default_tao_task_is_not_overridden(self):
        with patch.object(
            cli_commands.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as launch:
            self.assertEqual(
                cli_entry.main(["train", str(self.config), "--external-backend", "tao", "--dry-run"]),
                0,
            )
        self.assertNotIn("--tao-task", launch.call_args.args[0])

    def test_tao_task_is_not_sent_to_other_external_helpers(self):
        with patch.object(
            cli_commands.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as launch:
            self.assertEqual(
                cli_entry.main([
                    "train", str(self.config), "--external-backend", "yolox",
                    "--tao-task", "segformer", "--dry-run",
                ]),
                0,
            )
        self.assertNotIn("--tao-task", launch.call_args.args[0])

    def test_reference_resume_flags_remain_explicit_when_both_are_supplied(self):
        checkpoint = self.root / "override.pt"
        with patch.object(cli_entry, "_cmd_train", return_value=0) as train:
            cli_entry.main(["train", str(self.config), "--resume", "--resume-from", str(checkpoint)])
        train.assert_called_once_with(
            self.config, extra_args=["--resume-from", str(checkpoint), "--resume"]
        )

    def test_external_resume_boolean_is_rejected_before_launch(self):
        error = io.StringIO()
        with redirect_stderr(error), patch.object(cli_entry, "_cmd_train_external") as train:
            with self.assertRaises(SystemExit) as stopped:
                cli_entry.main(["train", str(self.config), "--external-backend", "yolox", "--resume"])
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("use --resume-from for external training", error.getvalue())
        train.assert_not_called()

    def test_train_help_exposes_both_resume_flags(self):
        for help_flag in ("-h", "--help"):
            output = io.StringIO()
            with self.subTest(help_flag=help_flag), redirect_stdout(output), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    cli_entry.main(["train", help_flag])
                self.assertEqual(stopped.exception.code, 0)
                self.assertIn("--resume-from", output.getvalue())
                self.assertIn("--resume", output.getvalue())
                self.assertIn("reference trainer", output.getvalue())


if __name__ == "__main__":
    unittest.main()
