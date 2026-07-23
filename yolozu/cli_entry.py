"""Parser and dispatch entrypoint for the YOLOZU CLI."""

from __future__ import annotations

from .cli_commands import (
    _cmd_train_import_preview,
    _cmd_train,
    _cmd_train_external,
    _cmd_train_orchestrate,
    _cmd_test,
    _cmd_doctor_import,
    _cmd_doctor_train_dataset,
    _cmd_doctor,
    _cmd_registry_list,
    _cmd_registry_run,
    _cmd_registry_show,
    _cmd_registry_validate,
    _cmd_list_models,
    _cmd_fetch_model,
    _cmd_export,
    _cmd_export_dataset,
    _cmd_predict_images,
    _cmd_eval_coco,
    _cmd_calibrate,
    _cmd_eval_long_tail,
    _cmd_long_tail_recipe,
    _cmd_benchmark,
    _cmd_parity,
    _cmd_predictions,
    _cmd_validate,
    _cmd_eval_instance_seg,
    _cmd_onnxrt_export,
    _cmd_onnxrt_quantize,
    _cmd_resources,
    _cmd_migrate,
    _cmd_import,
)
from .cli_demo import handle_demo_command
from .cli_completion import write_completion
from yolozu.eval.benchmark_flags import (
    BATCH_HELP,
    HALF_HELP,
    LATENCY_SOURCE_HELP,
    NMS_HELP,
    PARITY_REFERENCE_HELP,
    STRICT_HELP,
)
from yolozu.inference.export_orchestrator import parse_common_export_args
from yolozu.predictions.bbox_formats import SUPPORTED_PREDICTION_BBOX_FORMATS
import argparse
import json
from pathlib import Path
import sys

_PACKAGE = sys.modules.get(__package__)
__version__ = str(getattr(_PACKAGE, "__version__", "unknown"))
BENCHMARK_PARITY_REFERENCE_BACKENDS = ("auto", "torch", "onnx", "engine", "torchscript", "openvino")


GUIDE_ROUTES: dict[str, dict[str, object]] = {
    "first-run": {
        "title": "First run: install, check, visible PNG demo",
        "use_when": "You are new to YOLOZU and want a safe smoke path with files you can inspect.",
        "commands": [
            "python3 -m pip install -U yolozu",
            "yolozu doctor --proof",
            "yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress",
        ],
        "outputs": [
            "reports/doctor.json",
            "reports/doctor_proof/proof_report.json",
            "reports/quickstart_instance_seg/instance_seg_demo_report.json",
            "reports/quickstart_instance_seg/overlays/overlay_img_0000.png",
        ],
        "docs": ["docs/install.md", "docs/README.md", "configs/quickstart/instance_seg_demo.yaml"],
    },
    "evaluate": {
        "title": "Evaluate existing predictions",
        "use_when": "You already have a predictions.json artifact and a YOLO-style dataset.",
        "commands": [
            "yolozu validate predictions reports/predictions.json --strict",
            "yolozu eval-coco --dataset data/smoke --predictions reports/predictions.json --dry-run --output reports/eval.json",
        ],
        "outputs": ["reports/eval.json"],
        "docs": ["docs/predictions_schema.md", "docs/external_inference.md"],
    },
    "export": {
        "title": "Export predictions from images or a runtime",
        "use_when": "You need YOLOZU predictions.json before evaluation.",
        "commands": [
            "yolozu predict-images --backend dummy --input-dir data/smoke/images/val --output reports/predictions.json --overlays-dir reports/predict_overlays --html reports/predict_images.html --progress",
            "yolozu validate predictions reports/predictions.json --strict",
        ],
        "outputs": ["reports/predictions.json", "reports/predict_images.html", "reports/predict_overlays/000000_<image>.png"],
        "docs": ["docs/training_inference_export.md", "docs/predictions_schema.md", "configs/quickstart/predict_images_dummy.yaml"],
    },
    "debug": {
        "title": "Debug environment or dataset issues",
        "use_when": "Install, imports, dataset layout, or runtime checks are failing.",
        "commands": [
            "yolozu doctor --explain",
            "yolozu doctor --proof",
            "yolozu doctor import --dataset-from auto --dataset data/smoke --output -",
            "yolozu validate dataset data/smoke --mode warn",
        ],
        "outputs": ["stdout diagnostics", "reports/doctor.json when --output is a path"],
        "docs": ["docs/install.md", "docs/support.md"],
    },
}


def _guide_payload(goal: str) -> dict[str, object]:
    routes = GUIDE_ROUTES if goal == "all" else {goal: GUIDE_ROUTES[goal]}
    return {
        "kind": "yolozu_guide",
        "schema_version": 1,
        "goal": goal,
        "routes": routes,
        "tip": "Start with first-run if you are unsure; use --json for automation.",
    }


def _render_guide_text(goal: str) -> str:
    payload = _guide_payload(goal)
    lines = [
        "YOLOZU guide",
        "Start with `yolozu guide --goal first-run` if you are unsure.",
        "",
    ]
    routes = payload["routes"]
    assert isinstance(routes, dict)
    for route_id, route in routes.items():
        assert isinstance(route, dict)
        lines.append(f"[{route_id}] {route['title']}")
        lines.append(f"Use when: {route['use_when']}")
        lines.append("Commands:")
        for command in route["commands"]:
            lines.append(f"  {command}")
        lines.append("Expected output:")
        for output in route["outputs"]:
            lines.append(f"  {output}")
        lines.append("Read next:")
        for doc in route["docs"]:
            lines.append(f"  {doc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yolozu",
        epilog=(
            "© 2026 ToppyMicroServices OÜ\n"
            "Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia\n"
            "Registry code: 16551297\n"
            "Contact: develop@toppymicros.com"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    guide = sub.add_parser("guide", help="Show beginner-friendly routes and copy-paste commands.")
    guide.add_argument(
        "--goal",
        choices=tuple(["all", *GUIDE_ROUTES.keys()]),
        default="all",
        help="Route to show (default: all).",
    )
    guide.add_argument("--json", action="store_true", help="Emit machine-readable guide JSON.")

    doctor = sub.add_parser(
        "doctor",
        aliases=["dr"],
        help="Check the environment. Use --explain for beginner-friendly next actions.",
    )
    doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path (use - for stdout).")
    doctor.add_argument("--explain", action="store_true", help="Print a human-readable summary and next commands.")
    doctor.add_argument(
        "--proof",
        action="store_true",
        help="Run a CPU-only toy dataset + known predictions validation/eval proof.",
    )
    doctor.add_argument(
        "--proof-dir",
        default="reports/doctor_proof",
        help="Directory for doctor proof artifacts (default: reports/doctor_proof).",
    )
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=False)
    doctor_imp = doctor_sub.add_parser("import", help="Summarize dataset/config import resolution (宣伝用).")
    doctor_imp.add_argument("--output", default="-", help="Output JSON path (use - for stdout).")
    doctor_imp.add_argument(
        "--dataset-from",
        choices=("auto", "ultralytics", "coco", "coco-keypoints", "coco-instances", "segmentation"),
        default=None,
        help="Optional dataset import adapter to summarize.",
    )
    doctor_imp.add_argument("--dataset", default=None, help="(dataset-from auto|ultralytics|coco|segmentation) dataset root or descriptor path.")
    doctor_imp.add_argument(
        "--config-from",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        default=None,
        help="Optional config import adapter to summarize.",
    )
    doctor_imp.add_argument("--data", default=None, help="(dataset-from ultralytics) data.yaml path.")
    doctor_imp.add_argument("--args", default=None, help="(config-from ultralytics) args.yaml path.")
    doctor_imp.add_argument("--task", choices=("detect", "segment", "pose"), default=None, help="(dataset-from ultralytics) Task override.")
    doctor_imp.add_argument("--split", default=None, help="Split name (e.g. val2017/val/train).")
    doctor_imp.add_argument("--max-images", type=int, default=200, help="Cap number of samples loaded for summary (default: 200).")
    doctor_imp.add_argument("--instances", default=None, help="(dataset-from coco-instances) instances_*.json path.")
    doctor_imp.add_argument("--images-dir", default=None, help="(dataset-from coco-instances) images directory for this split.")
    doctor_imp.add_argument("--include-crowd", action="store_true", help="(dataset-from coco-instances) Include iscrowd annotations.")
    doctor_imp.add_argument("--config", default=None, help="(config-from mmdet/yolox/detectron2) config file path.")

    doctor_train = doctor_sub.add_parser(
        "train-dataset",
        help="Check whether a dataset is ready for the RT-DETR reference trainer.",
    )
    doctor_train.add_argument("--output", default="-", help="Output JSON path (use - for stdout).")
    doctor_train.add_argument(
        "--from",
        dest="dataset_from",
        choices=(
            "auto",
            "ultralytics",
            "coco",
            "coco-keypoints",
            "coco-instances",
            "segmentation",
            "classification",
            "obb",
            "depth",
            "pose6d",
        ),
        default="auto",
        help="Dataset source family to check (default: auto).",
    )
    doctor_train.add_argument("--dataset", default=None, help="Dataset root, data.yaml, or dataset.json path.")
    doctor_train.add_argument("--split", default=None, help="Split name (for example train, val, val2017).")
    doctor_train.add_argument("--records-json", default=None, help="Training records JSON to check instead of scanning --dataset.")
    doctor_train.add_argument("--val-records-json", default=None, help="Optional validation records JSON to check.")
    doctor_train.add_argument("--instances", default=None, help="COCO instances/person_keypoints JSON path for explicit COCO checks.")
    doctor_train.add_argument("--images-dir", default=None, help="COCO images directory for explicit COCO checks.")
    doctor_train.add_argument("--include-crowd", action="store_true", help="Include COCO iscrowd annotations.")
    doctor_train.add_argument("--max-images", type=int, default=200, help="Cap records inspected for summary (default: 200).")

    list_p = sub.add_parser("list", help="List registries and built-in catalogs.")
    list_sub = list_p.add_subparsers(dest="list_command", required=True)
    list_models = list_sub.add_parser("models", help="List fetchable model IDs.")
    list_models.add_argument("--registry", default=None, help="Optional registry JSON path (default: packaged model zoo).")
    list_models.add_argument("--json", action="store_true", help="Emit JSON.")

    fetch = sub.add_parser("fetch", help="Download a model artifact from the built-in (or custom) model registry.")
    fetch.add_argument("model_id", help="Model id from `yolozu list models`.")
    fetch.add_argument("--out", default="models", help="Output root directory (default: models).")
    fetch.add_argument("--cache-dir", default=None, help="Cache directory (default: ~/.cache/yolozu/models).")
    fetch.add_argument("--registry", default=None, help="Optional registry JSON path override.")
    fetch.add_argument("--accept-license", action="store_true", help="Required acknowledgment to download the selected model.")
    fetch.add_argument("--allow-unsafe", action="store_true", help="Allow fetching models without sha256 in the registry.")
    fetch.add_argument("--allow-non-apache", action="store_true", help="Allow fetching models with non-Apache-friendly licenses.")
    fetch.add_argument("--retries", type=int, default=3, help="Download retry count (default: 3).")
    fetch.add_argument("--timeout", type=float, default=60.0, help="Download timeout in seconds (default: 60).")
    fetch.add_argument("--force", action="store_true", help="Re-download to cache and overwrite output artifact.")

    export = sub.add_parser("export", help="Export predictions.json artifacts across the supported backend lanes.")
    parse_common_export_args(export)

    export_dataset = sub.add_parser("export-dataset", help="Export a YOLOZU dataset into YOLO, COCO, KITTI, or segmentation layout.")
    export_dataset.add_argument(
        "to_format",
        choices=("yolo", "kitti", "coco", "segmentation"),
        help="Target dataset layout.",
    )
    export_dataset.add_argument("--dataset", required=True, help="YOLOZU dataset root or dataset.json wrapper.")
    export_dataset.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    export_dataset.add_argument("--out-dir", required=True, help="Output directory for the exported dataset layout.")
    export_dataset.add_argument("--image-mode", choices=("copy", "symlink"), default="copy", help="How to materialize exported assets (default: copy).")
    export_dataset.add_argument("--force", action="store_true", help="Overwrite the output directory if it exists.")

    predict = sub.add_parser("predict-images", help="Run folder inference and write predictions JSON + overlays + HTML.")
    parse_common_export_args(predict)
    predict.add_argument("-i", "--input-dir", required=True, help="Folder containing images.")
    predict.add_argument("--glob", action="append", default=None, help="Glob pattern under --input-dir (repeatable).")
    predict.add_argument("-v", "--overlays-dir", default="reports/overlays", help="Directory to write overlay images.")
    predict.add_argument("-H", "--html", default="reports/predict_images.html", help="Optional HTML report output path.")
    predict.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    predict.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show stderr progress bars (default: auto on interactive terminals).",
    )

    eval_coco = sub.add_parser("eval-coco", help="Evaluate detections with COCOeval (optional extra: yolozu[coco]).")
    eval_coco.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eval_coco.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    eval_coco.add_argument("--predictions", required=True, help="Predictions JSON path.")
    eval_coco.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
        help="How to interpret detection bbox fields (default: cxcywh_norm).",
    )
    eval_coco.add_argument("--dry-run", action="store_true", help="Skip COCOeval; only validate/convert predictions.")
    eval_coco.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    eval_coco.add_argument("--classes", default=None, help="Optional labels/<split>/classes.json for class-id normalization.")
    eval_coco.add_argument(
        "--assume-class-id-is-category-id",
        action="store_true",
        help="Treat class_id in predictions as COCO category_id when --classes is set.",
    )
    eval_coco.add_argument("--output", default="reports/coco_eval.json", help="Output report path.")

    calibrate = sub.add_parser("calibrate", help="Apply post-hoc FRACAL calibration to bbox or instance-seg predictions JSON.")
    calibrate.add_argument("--method", choices=("fracal", "la", "norcal", "temperature"), default="fracal", help="Calibration method (default: fracal).")
    calibrate.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    calibrate.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    calibrate.add_argument("--task", choices=("auto", "bbox", "seg", "pose"), default="auto", help="Prediction task type (default: auto).")
    calibrate.add_argument("--predictions", required=True, help="Input predictions JSON path.")
    calibrate.add_argument("--output", default="reports/predictions_calibrated.json", help="Output calibrated predictions JSON path.")
    calibrate.add_argument("--output-report", default="reports/calibration_fracal_report.json", help="Output calibration report JSON path.")
    calibrate.add_argument("--stats-in", default=None, help="Optional precomputed FRACAL stats JSON path (class_counts).")
    calibrate.add_argument("--stats-out", default=None, help="Optional output path to write computed FRACAL stats JSON.")
    calibrate.add_argument("--max-images", type=int, default=None, help="Optional cap for calibration/eval subset size.")
    calibrate.add_argument("--alpha", type=float, default=0.5, help="FRACAL class-frequency exponent (default: 0.5).")
    calibrate.add_argument("--strength", type=float, default=1.0, help="Blend ratio [0,1] between original and FRACAL scores (default: 1.0).")
    calibrate.add_argument("--tau", type=float, default=1.0, help="(method=la) logit adjustment coefficient tau (default: 1.0).")
    calibrate.add_argument("--gamma", type=float, default=1.0, help="(method=norcal) frequency exponent gamma (default: 1.0).")
    calibrate.add_argument("--temperature", type=float, default=1.0, help="(method=temperature) global temperature T (>0, default: 1.0).")
    calibrate.add_argument("--fit-temperature", action="store_true", help="(method=temperature, bbox/pose) fit T on validation subset by minimizing binary NLL.")
    calibrate.add_argument("--temperature-grid", default="0.5,0.75,1.0,1.25,1.5,2.0", help="(method=temperature) candidate T values for --fit-temperature.")
    calibrate.add_argument("--fit-iou", type=float, default=0.5, help="(method=temperature) IoU threshold for positive matching in T fitting.")
    calibrate.add_argument("--fit-max-detections", type=int, default=100, help="(method=temperature) max detections/image used when fitting T.")
    calibrate.add_argument("--min-score", type=float, default=None, help="Optional post-clamp minimum score.")
    calibrate.add_argument("--max-score", type=float, default=None, help="Optional post-clamp maximum score.")
    calibrate.add_argument("--allow-rgb-masks", action="store_true", help="(task=seg) Treat RGB masks as valid by using channel-0 as foreground.")
    calibrate.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")

    eval_lt = sub.add_parser("eval-long-tail", help="Evaluate long-tail detection metrics in one standardized report.")
    eval_lt.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eval_lt.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    eval_lt.add_argument("--predictions", required=True, help="Predictions JSON path.")
    eval_lt.add_argument("--output", default="reports/long_tail_eval.json", help="Output long-tail report JSON path.")
    eval_lt.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    eval_lt.add_argument("--max-detections", type=int, default=100, help="Max detections per image for AR/calibration matching.")
    eval_lt.add_argument("--head-fraction", type=float, default=0.33, help="Top class fraction assigned to head bin.")
    eval_lt.add_argument("--medium-fraction", type=float, default=0.67, help="Top class fraction assigned up to medium bin.")
    eval_lt.add_argument("--calibration-bins", type=int, default=10, help="Bin count for calibration metrics (ECE/confidence bias).")
    eval_lt.add_argument("--calibration-iou", type=float, default=0.5, help="IoU threshold for calibration correctness matching.")

    lt_recipe = sub.add_parser("long-tail-recipe", help="Generate a decoupled long-tail training recipe with plugin-style rebalance config.")
    lt_recipe.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    lt_recipe.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    lt_recipe.add_argument("--output", default="reports/long_tail_recipe.json", help="Output recipe JSON path.")
    lt_recipe.add_argument("--max-images", type=int, default=None, help="Optional cap for recipe stat scan.")
    lt_recipe.add_argument("--seed", type=int, default=0, help="Seed recorded in recipe for reproducibility.")
    lt_recipe.add_argument("--stage1-epochs", type=int, default=90, help="Representation learning stage epochs.")
    lt_recipe.add_argument("--stage2-epochs", type=int, default=30, help="Classifier re-training stage epochs.")
    lt_recipe.add_argument("--rebalance-sampler", choices=("none", "class_balanced"), default="class_balanced", help="Sampler plugin selection.")
    lt_recipe.add_argument(
        "--loss-plugin",
        choices=("none", "focal", "ldam", "torch_cross_entropy", "torch_nll_loss", "torch_bce_with_logits"),
        default="focal",
        help="Loss plugin selection.",
    )
    lt_recipe.add_argument(
        "--metric-plugin",
        choices=("none", "torch_top1_accuracy", "torch_top5_accuracy", "torch_cross_entropy"),
        default="torch_top1_accuracy",
        help="Validation metric plugin selection.",
    )
    lt_recipe.add_argument(
        "--lr-scheduler",
        choices=("none", "torch_step_lr", "torch_cosine_annealing_lr", "torch_reduce_on_plateau", "torch_one_cycle_lr"),
        default="none",
        help="Learning-rate scheduler selection.",
    )
    lt_recipe.add_argument("--logit-adjustment-tau", type=float, default=1.0, help="Logit adjustment strength (0 disables).")
    lt_recipe.add_argument("--lort-tau", type=float, default=0.0, help="Frequency-free logits retargeting strength (0 disables).")
    lt_recipe.add_argument("--class-balanced-beta", type=float, default=0.999, help="Effective-number beta for class-balanced weights.")
    lt_recipe.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (recipe parameter).")
    lt_recipe.add_argument("--ldam-margin", type=float, default=0.5, help="LDAM margin (recipe parameter).")
    lt_recipe.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    bench = sub.add_parser(
        "benchmark",
        help="Ultralytics-parity benchmark entrypoint with real, artifact-backed, and explicit skipped results.",
    )
    bench.add_argument("-m", "--model", required=True, help="Model/weights path recorded in the benchmark report.")
    bench.add_argument("--torch-model", default=None, help="Optional torch backend model/depth-artifact override.")
    bench.add_argument("--onnx-model", default=None, help="Optional ONNX backend model/depth-artifact override.")
    bench.add_argument("--engine-model", default=None, help="Optional TensorRT engine/depth-artifact override.")
    bench.add_argument("--torchscript-model", default=None, help="Optional TorchScript backend model/artifact override.")
    bench.add_argument("--openvino-model", default=None, help="Optional OpenVINO IR model/artifact override.")
    bench.add_argument("-d", "--data", required=True, help="Dataset root or data.yaml path recorded in the benchmark report.")
    bench.add_argument("--depth-mask", default=None, help="Optional valid-pixel mask used for task=depth artifact evaluation.")
    bench.add_argument(
        "--depth-align",
        choices=("none", "median_scale"),
        default="median_scale",
        help="Depth artifact alignment mode for task=depth benchmark eval/parity (default: median_scale).",
    )
    bench.add_argument("--depth-parity-mae-atol", type=float, default=0.02, help="Depth parity MAE threshold (default: 0.02).")
    bench.add_argument("--depth-parity-rmse-atol", type=float, default=0.03, help="Depth parity RMSE threshold (default: 0.03).")
    bench.add_argument(
        "--segmentation-parity-mismatch-atol",
        type=float,
        default=0.0,
        help="Segmentation parity mismatch-rate tolerance (default: 0.0, exact mask match).",
    )
    bench.add_argument(
        "--parity-reference-backend",
        choices=BENCHMARK_PARITY_REFERENCE_BACKENDS,
        default="auto",
        help=PARITY_REFERENCE_HELP,
    )
    bench.add_argument("--keypoints-parity-iou-thresh", type=float, default=0.99, help="Keypoints parity IoU threshold (default: 0.99).")
    bench.add_argument("--keypoints-parity-score-atol", type=float, default=1e-4, help="Keypoints parity score tolerance (default: 1e-4).")
    bench.add_argument("--keypoints-parity-bbox-atol", type=float, default=1e-4, help="Keypoints parity bbox tolerance (default: 1e-4).")
    bench.add_argument("--keypoints-parity-kp-atol", type=float, default=1e-4, help="Keypoints parity keypoint tolerance in normalized coords (default: 1e-4).")
    bench.add_argument("--pose-parity-rot-deg-atol", type=float, default=1e-3, help="6DoF parity rotation threshold in degrees (default: 1e-3).")
    bench.add_argument("--pose-parity-trans-atol", type=float, default=1e-4, help="6DoF parity translation L2 threshold in meters (default: 1e-4).")
    bench.add_argument("--pose-parity-depth-atol", type=float, default=1e-4, help="6DoF parity depth threshold in meters (default: 1e-4).")
    bench.add_argument("-i", "--imgsz", type=int, default=640, help="Input image size (default: 640).")
    bench.add_argument("--half", action=argparse.BooleanOptionalAction, default=False, help=HALF_HELP)
    bench.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False, help="Record INT8 intent.")
    bench.add_argument("--device", default="cpu", help="Target device string (default: cpu).")
    bench.add_argument("--verbose", action="store_true", help="Print per-format status lines.")
    bench.add_argument("-f", "--format", default="all", help="Comma-separated Phase-1 formats or all.")
    bench.add_argument(
        "--task",
        default="detect",
        choices=(
            "6dof",
            "classification",
            "classify",
            "cls",
            "depth",
            "detect",
            "detection",
            "keypoints",
            "obb",
            "pose",
            "pose-6d",
            "pose6d",
            "pose_6d",
            "seg",
            "segmentation",
        ),
        help="Benchmark task label. Canonical tasks: detect, segmentation, classification, obb, keypoints, depth, pose6d.",
    )
    bench.add_argument("--split", default=None, help="Dataset split label.")
    bench.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Optional eval protocol passed through to eval_suite and torch exporter.",
    )
    bench.add_argument("--max-images", type=int, default=None, help="Optional max image count recorded in the report.")
    bench.add_argument("--dry-run", action="store_true", help="Validate wiring and planned artifacts without timing runs.")
    bench.add_argument(
        "--strict",
        action="store_true",
        help=STRICT_HELP,
    )
    bench.add_argument("--repro-policy", choices=("strict", "relaxed", "off"), default="relaxed")
    bench.add_argument("--runtime-lock", default="none", help="Runtime lock label recorded in run_meta.")
    bench.add_argument("--run-id", default=None, help="Optional run id (default: UTC timestamp).")
    bench.add_argument("-o", "--output", default="reports/benchmark_report.json", help="Benchmark report JSON path.")
    bench.add_argument("--history", default=None, help="Optional JSONL history file path.")
    bench.add_argument("--predictions-output", default=None, help="Optional file/dir/template for predictions artifacts.")
    bench.add_argument("--eval-output", default=None, help="Optional file/dir/template for eval artifacts.")
    bench.add_argument("--parity-output", default=None, help="Optional file/dir/template for parity artifacts.")
    bench.add_argument("--batch", type=int, default=1, help=BATCH_HELP)
    bench.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False, help="Record dynamic-shape intent.")
    bench.add_argument("--nms", action=argparse.BooleanOptionalAction, default=False, help=NMS_HELP)
    bench.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=False, help="Record ONNX simplify intent.")
    bench.add_argument("--opset", type=int, default=17, help="Record ONNX opset (default: 17).")
    bench.add_argument("--workspace", type=float, default=4.0, help="Record TensorRT workspace in GiB (default: 4).")
    bench.add_argument("--fraction", type=float, default=1.0, help="Record dataset fraction knob (default: 1.0).")
    bench.add_argument(
        "--latency-source",
        choices=("auto", "synthetic_step", "dataset_pass_wall_time", "artifact_eval"),
        default="auto",
        help=LATENCY_SOURCE_HELP,
    )
    bench.add_argument("--iterations", type=int, default=50, help="Synthetic latency iterations (default: 50).")
    bench.add_argument("--warmup", type=int, default=5, help="Synthetic latency warmup iterations (default: 5).")
    bench.add_argument("--sleep-s", type=float, default=0.0, help="Synthetic latency sleep per step (default: 0).")

    parity = sub.add_parser("parity", help="Compare two predictions JSON artifacts for backend parity.")
    parity.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    parity.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime).")
    parity.add_argument(
        "--bbox-format",
        choices=SUPPORTED_PREDICTION_BBOX_FORMATS,
        default="auto",
        help="BBox format stored in both JSONs, or auto to use bbox.format/keys (default: auto).",
    )
    parity.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold for a match.")
    parity.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    parity.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox differences.")
    parity.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    parity.add_argument("--image-size", default=None, help="Optional fixed image size (N or W,H) to skip image reads.")

    predictions = sub.add_parser("predictions", help="Predictions artifact utilities.")
    predictions_sub = predictions.add_subparsers(dest="predictions_command", required=True)
    pred_migrate = predictions_sub.add_parser("migrate", help="Migrate predictions entry schema versions (v1 -> v2).")
    pred_migrate.add_argument("--input", required=True, help="Input predictions JSON path.")
    pred_migrate.add_argument("--output", required=True, help="Output predictions JSON path.")
    pred_migrate.add_argument(
        "--from",
        dest="from_version",
        choices=("v1",),
        required=True,
        help="Source predictions entry schema version.",
    )
    pred_migrate.add_argument(
        "--to",
        dest="to_version",
        choices=("v2",),
        required=True,
        help="Target predictions entry schema version.",
    )
    pred_migrate.add_argument(
        "--strict-source",
        action="store_true",
        help="Fail when input entries do not match --from version constraints.",
    )
    pred_migrate.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    validate = sub.add_parser("validate", help="Validate artifacts (predictions JSON, instance-seg predictions).")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    val_pred = validate_sub.add_parser("predictions", help="Validate predictions JSON (detections+bbox schema).")
    val_pred.add_argument("path", type=str, help="Path to predictions JSON (list or wrapper).")
    val_pred.add_argument("--strict", action="store_true", help="Strict validation (types, required keys).")

    val_seg = validate_sub.add_parser("seg", help="Validate semantic segmentation predictions JSON (id->mask mapping).")
    val_seg.add_argument("path", type=str, help="Path to segmentation predictions JSON.")

    val_is = validate_sub.add_parser("instance-seg", help="Validate instance-segmentation predictions JSON (PNG masks).")
    val_is.add_argument("path", type=str, help="Path to instance-seg predictions JSON.")

    val_ds = validate_sub.add_parser("dataset", help="Validate a YOLO-format dataset layout + labels.")
    val_ds.add_argument("dataset", type=str, help="YOLO-format dataset root (contains images/ + labels/).")
    val_ds.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    val_ds.add_argument(
        "--label-format",
        choices=("detect", "segment"),
        default=None,
        help="How to parse label txt files (default: detect). Use segment for YOLO polygon labels.",
    )
    val_ds.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images checked.")
    val_ds.add_argument("--strict", action="store_true", help="Strict bbox checks (range + inside-image).")
    val_ds.add_argument("--mode", choices=("fail", "warn"), default="fail", help="fail=exit nonzero on errors; warn=always exit 0.")
    val_ds.add_argument("--no-check-images", action="store_true", help="Skip image existence/size checks.")

    eis = sub.add_parser(
        "eval-instance-seg",
        help="Evaluate instance segmentation predictions (mask mAP over PNG masks).",
    )
    eis.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eis.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    eis.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    eis.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    eis.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    eis.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    eis.add_argument("--html", default=None, help="Optional HTML report path.")
    eis.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    eis.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    eis.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    eis.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    eis.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    eis.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")
    eis.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    eis.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    eis.add_argument("--diag-iou", type=float, default=0.5, help="IoU threshold used for per-image diagnostics/overlay selection (default: 0.5).")
    eis.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in the report/meta and HTML (default: 100).")
    eis.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )

    onnxrt = sub.add_parser("onnxrt", help="ONNXRuntime utilities (optional extra: yolozu[onnxrt]).")
    onnxrt_sub = onnxrt.add_subparsers(dest="onnxrt_command", required=True)
    onnxrt_export = onnxrt_sub.add_parser("export", help="Run ONNXRuntime inference and export predictions JSON.")
    onnxrt_export.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    onnxrt_export.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    onnxrt_export.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    onnxrt_export.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    onnxrt_export.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    onnxrt_export.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    onnxrt_export.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    onnxrt_export.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    onnxrt_export.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    onnxrt_export.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    onnxrt_export.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    onnxrt_export.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    onnxrt_export.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics", "yolo_runtime"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    onnxrt_export.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    onnxrt_export.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] wrt input_size (default: norm).",
    )
    onnxrt_export.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    onnxrt_export.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    onnxrt_export.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    onnxrt_export.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS when decoding raw output.",
    )
    onnxrt_export.add_argument("--imgsz", type=int, default=640, help="Input image size (square, default: 640).")
    onnxrt_export.add_argument("--output", default="reports/predictions_onnxrt.json", help="Where to write predictions JSON.")
    onnxrt_export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    onnxrt_export.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    onnxrt_export.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")

    onnxrt_quant = onnxrt_sub.add_parser("quantize", help="Quantize an ONNX model using ONNXRuntime (dynamic).")
    onnxrt_quant.add_argument("--onnx", required=True, help="Input ONNX model path.")
    onnxrt_quant.add_argument("--output", required=True, help="Output ONNX model path.")
    onnxrt_quant.add_argument(
        "--weight-type",
        choices=("qint8", "quint8"),
        default="qint8",
        help="Weight quantization type (default: qint8).",
    )
    onnxrt_quant.add_argument("--per-channel", action="store_true", help="Quantize weights per channel.")
    onnxrt_quant.add_argument("--reduce-range", action="store_true", help="Use 7-bit quantization for weights.")
    onnxrt_quant.add_argument("--op-types", default=None, help="Comma-separated operator types to quantize (default: all supported).")
    onnxrt_quant.add_argument("--use-external-data-format", action="store_true", help="Write weights as external data (>2GB models).")

    resources_p = sub.add_parser("resources", help="Access packaged configs/schemas/protocols.")
    resources_sub = resources_p.add_subparsers(dest="resources_command", required=True)
    resources_sub.add_parser("list", help="List packaged resource paths.")
    cat = resources_sub.add_parser("cat", help="Print a packaged resource to stdout.")
    cat.add_argument("path", type=str, help="Resource path under yolozu/data (e.g., schemas/predictions.schema.json).")
    copy = resources_sub.add_parser("copy", help="Copy a packaged resource to a file path.")
    copy.add_argument("path", type=str, help="Resource path under yolozu/data.")
    copy.add_argument("--output", required=True, help="Output file path.")
    copy.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    migrate = sub.add_parser("migrate", help="Migration helpers (dataset/config/predictions).")
    migrate_sub = migrate.add_subparsers(dest="migrate_command", required=True)

    mig_dataset = migrate_sub.add_parser("dataset", help="Generate dataset.json wrapper for external dataset layouts.")
    mig_dataset.add_argument(
                            "--from",
                            dest="from_format",
                            choices=("auto", "ultralytics", "coco", "coco-keypoints", "segmentation"),
        required=True,
        help="Source ecosystem.",
    )
    mig_dataset.add_argument("--dataset", default=None, help="(auto) Dataset root or descriptor path to inspect.")
    mig_dataset.add_argument("--data", default=None, help="(Ultralytics) data.yaml path (preferred).")
    mig_dataset.add_argument("--args", default=None, help="(Ultralytics) args.yaml (optional; used for task/data inference).")
    mig_dataset.add_argument(
        "--split",
        default=None,
        help="Split name (Ultralytics/segmentation: inferred from layout; COCO: annotations split, default: val2017).",
    )
    mig_dataset.add_argument(
        "--task",
        choices=("detect", "segment", "pose"),
        default=None,
        help="(Ultralytics) Override task inference (segment enables polygon label parsing).",
    )
    mig_dataset.add_argument("--coco-root", default=None, help="(COCO) Root containing images/ and annotations/.")
    mig_dataset.add_argument("--instances-json", default=None, help="(COCO) Override instances JSON path.")
    mig_dataset.add_argument(
        "--mode",
        choices=("manifest", "symlink", "copy"),
        default="manifest",
        help="(COCO) Image handling: manifest=do not copy; symlink/copy into output/images/<split>.",
    )
    mig_dataset.add_argument("--include-crowd", action="store_true", help="(COCO) Include iscrowd annotations.")
    mig_dataset.add_argument("--output", required=True, help="Output directory or dataset.json file path.")
    mig_dataset.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    mig_preds = migrate_sub.add_parser("predictions", help="Convert external prediction outputs into YOLOZU predictions.json.")
    mig_preds.add_argument(
        "--from",
        dest="from_format",
        choices=("coco-results",),
        required=True,
        help="Source prediction format.",
    )
    mig_preds.add_argument("--results", required=True, help="COCO results JSON path (list of detections).")
    mig_preds.add_argument("--instances", required=True, help="COCO instances JSON path (for image_id mapping + sizes).")
    mig_preds.add_argument("--output", required=True, help="Output predictions.json path.")
    mig_preds.add_argument("--score-threshold", type=float, default=0.0, help="Minimum score to keep (default: 0.0).")
    mig_preds.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    mig_seg = migrate_sub.add_parser("seg-dataset", help="Generate semantic segmentation dataset descriptor JSON.")
    mig_seg.add_argument(
        "--from",
        dest="from_format",
        choices=("voc", "cityscapes", "ade20k"),
        required=True,
        help="Source dataset type.",
    )
    mig_seg.add_argument("--root", required=True, help="Dataset root path.")
    mig_seg.add_argument("--split", default="val", help="Split name (train|val|test, dataset-specific aliases allowed).")
    mig_seg.add_argument("--output", required=True, help="Output descriptor JSON path.")
    mig_seg.add_argument("--path-type", choices=("absolute", "relative"), default="absolute", help="Emit absolute or relative paths.")
    mig_seg.add_argument("--mode", choices=("manifest", "symlink", "copy"), default="manifest", help="Descriptor mode hint.")
    mig_seg.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    mig_seg.add_argument("--year", default=None, help="(VOC) Optional year selector (e.g. 2012).")
    mig_seg.add_argument(
        "--masks-dirname",
        default="SegmentationClass",
        help="(VOC) Mask directory name under VOC year root (default: SegmentationClass).",
    )
    mig_seg.add_argument(
        "--label-type",
        choices=("labelTrainIds", "labelIds"),
        default="labelTrainIds",
        help="(Cityscapes) Mask suffix type (default: labelTrainIds).",
    )

    imp = sub.add_parser("import", help="Import adapters (read-only projection into canonical schema).")
    imp_sub = imp.add_subparsers(dest="import_command", required=True)

    imp_dataset = imp_sub.add_parser("dataset", help="Generate a read-only dataset wrapper for external layouts.")
    imp_dataset.add_argument(
        "--from",
        dest="from_format",
        choices=("auto", "ultralytics", "coco", "coco-keypoints", "coco-instances", "segmentation"),
        required=True,
        help="Source ecosystem.",
    )
    imp_dataset.add_argument("--dataset", default=None, help="(auto|ultralytics|coco|segmentation) Dataset root or descriptor path.")
    imp_dataset.add_argument("--output", required=True, help="Output directory (wrapper) or dataset.json file path.")
    imp_dataset.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    imp_dataset.add_argument("--split", default=None, help="Split name (COCO default: val2017; Ultralytics: from data.yaml).")

    imp_dataset.add_argument("--data", default=None, help="(Ultralytics) data.yaml path (preferred).")
    imp_dataset.add_argument("--args", default=None, help="(Ultralytics) args.yaml (optional; used for task/data inference).")
    imp_dataset.add_argument("--task", choices=("detect", "segment", "pose"), default=None, help="(Ultralytics) Task override.")

    imp_dataset.add_argument("--instances", default=None, help="(COCO) instances_*.json path.")
    imp_dataset.add_argument("--images-dir", default=None, help="(COCO) Images directory for this split.")
    imp_dataset.add_argument("--include-crowd", action="store_true", help="(COCO) Include iscrowd annotations.")

    imp_cfg = imp_sub.add_parser("config", help="Project external configs into canonical TrainConfig (major keys only).")
    imp_cfg.add_argument(
        "--from",
        dest="from_format",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        required=True,
        help="Source ecosystem.",
    )
    imp_cfg.add_argument("--args", default=None, help="(Ultralytics) args.yaml path.")
    imp_cfg.add_argument("--config", default=None, help="(MMDet/YOLOX/Detectron2) config file path.")
    imp_cfg.add_argument("--output", required=True, help="Output path (file or directory).")
    imp_cfg.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    train_p = sub.add_parser(
        "train",
        help=(
            "Train with the RT-DETR pose reference trainer by default, or use "
            "--external-backend yolox|detectron2|mmdetection|mmpose|mmseg|tao|ultralytics|hf-detr for external training lanes."
        ),
        epilog=(
            "Reference trainer options after the config path are forwarded to rtdetr_pose.train_minimal. "
            "Common dataset inputs include --dataset-root, --records-json, --extra-records-json, "
            "and --val-records-json."
        ),
    )
    train_p.add_argument(
        "config",
        nargs="?",
        type=str,
        help=(
            "Reference train config YAML/JSON. When --external-backend is selected, "
            "this becomes the backend-specific model/config handle "
            "(YOLOX exp file, Detectron2/MM-family/TAO config path, Ultralytics model path/id, or HF model id)."
        ),
    )
    train_p.add_argument(
        "--import",
        dest="import_from",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        default=None,
        help="Optional shorthand: resolve external config into canonical TrainConfig before training.",
    )
    train_p.add_argument("--data", default=None, help="(train --import ultralytics) data.yaml path for dataset preview.")
    train_p.add_argument("--cfg", default=None, help="(train --import) external framework config/args path.")
    train_p.add_argument(
        "--resolved-config-out",
        default="reports/train_config_resolved_import.json",
        help="Output path for canonical TrainConfig resolved by train --import.",
    )
    train_p.add_argument(
        "--force-import-overwrite",
        action="store_true",
        help="Overwrite --resolved-config-out if it already exists.",
    )
    train_p.add_argument(
        "--external-backend",
        choices=("yolox", "detectron2", "mmdetection", "mmpose", "mmseg", "tao", "ultralytics", "hf-detr"),
        default=None,
        help=(
            "Optional repo-side external training lane. Use backend-specific flags after "
            "--external-backend; they are forwarded to tools/support_external_training.py."
        ),
    )
    train_p.add_argument(
        "--resume-from",
        default=None,
        help="Forwarded checkpoint path for external fine-tune/resume flows.",
    )
    train_p.add_argument(
        "--tao-task",
        default=None,
        help="Forwarded TAO task name when --external-backend tao is active.",
    )

    train_orch = sub.add_parser(
        "train-orchestrate",
        help="Plan or execute a small multi-backend training batch from one orchestration spec.",
    )
    train_orch.add_argument("--spec", required=True, help="JSON orchestration spec with experiments[].")
    train_orch.add_argument(
        "--output",
        default="reports/training_orchestration_report.json",
        help="Output report JSON path.",
    )
    train_orch.add_argument("--registry-out", default=None, help="Optional JSONL registry file to append executed training runs.")
    train_orch.add_argument("--execute", action="store_true", help="Run the planned commands.")
    train_orch.add_argument("--dry-run", action="store_true", help="Append --dry-run when missing.")
    train_orch.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failing execution.")

    test_p = sub.add_parser("test", help="Run scenario suite (dummy/precomputed adapters are CPU-only).")
    test_p.add_argument("config", type=str, help="Path to test config YAML/JSON (test_setting.yaml).")
    demo = sub.add_parser("demo", help="Run small self-contained demos (CPU-friendly).")
    demo.add_argument(
        "--coco-instances-json",
        default=None,
        help="(demo suite) COCO instances_*.json path to enable the polygon-mask instance-seg demo.",
    )
    demo.add_argument(
        "--coco-images-dir",
        default=None,
        help="(demo suite) COCO images dir (joined with image.file_name) for the polygon-mask instance-seg demo.",
    )
    demo_sub = demo.add_subparsers(dest="demo_command", required=False)

    demo_ov = demo_sub.add_parser("overview", help="Write a demo coverage overview report (tasks/dependencies/commands).")
    demo_ov.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: demo_output/overview/<utc>/demo_overview_report.json).",
    )

    demo_is = demo_sub.add_parser("instance-seg", help="Instance-seg eval demo (numpy + Pillow).")
    demo_is.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/instance_seg/<utc>).")
    demo_is.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_is.add_argument("--num-images", type=int, default=8, help="Number of images (default: 8).")
    demo_is.add_argument("--image-size", type=int, default=96, help="Square image size (default: 96).")
    demo_is.add_argument("--max-instances", type=int, default=2, help="Max instances per image (default: 2).")
    demo_is.add_argument(
        "--background",
        choices=("synthetic", "coco128", "coco-instances", "yolo-bbox"),
        default="coco-instances",
        help=(
            "Background source: synthetic shapes, COCO128 (bbox-derived), COCO instances polygons, "
            "or a YOLO-style bbox dataset (default: coco-instances)."
        ),
    )
    demo_is.add_argument(
        "--coco-instances-json",
        default=None,
        help=(
            "(background=coco-instances) Path to COCO instances_*.json (polygon segmentations). "
            "If omitted, defaults to data/coco/annotations/instances_val2017.json."
        ),
    )
    demo_is.add_argument(
        "--coco-images-dir",
        default=None,
        help=(
            "(background=coco-instances) Root images dir for the COCO split (joined with image.file_name). "
            "If omitted, defaults to data/coco/images/val2017."
        ),
    )
    demo_is.add_argument(
        "--yolo-root",
        default=None,
        help=(
            "(background=yolo-bbox) YOLO-style dataset root containing images/<split> and labels/<split> "
            "(labels are YOLO bbox or YOLO-seg polygon rows)."
        ),
    )
    demo_is.add_argument(
        "--yolo-split",
        default="val",
        help="(background=yolo-bbox) Split folder name under images/ and labels/ (default: val).",
    )
    demo_is.add_argument(
        "--inference",
        choices=("none", "auto", "torchvision"),
        default=None,
        help=(
            "(background=coco-instances) Instance-seg inference backend. "
            "Default: auto when background=coco-instances, otherwise none."
        ),
    )
    demo_is.add_argument(
        "--device",
        default="auto",
        help="(inference) Torch device (cpu|cuda|mps|auto) (default: auto).",
    )
    demo_is.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="(inference) Score threshold for predicted instances (default: 0.5).",
    )
    demo_is.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show stderr progress bars (default: auto on interactive terminals).",
    )

    demo_ist = demo_sub.add_parser(
        "instance-seg-tta",
        help="Real-image instance-seg TTA compare demo (Mask R-CNN raw vs hflip TTA on corrupted COCO images).",
    )
    demo_ist.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/instance_seg_tta/<utc>).")
    demo_ist.add_argument("--seed", type=int, default=0, help="Random seed for image sampling/corruption (default: 0).")
    demo_ist.add_argument("--num-images", type=int, default=8, help="Max candidate images to scan before selecting the best case (default: 8).")
    demo_ist.add_argument(
        "--coco-instances-json",
        default=None,
        help="Path to COCO instances_*.json (default: data/coco/annotations/instances_val2017.json).",
    )
    demo_ist.add_argument(
        "--coco-images-dir",
        default=None,
        help="Root COCO images dir (default: data/coco/images/val2017).",
    )
    demo_ist.add_argument("--device", default="auto", help="Torch device (cpu|cuda|mps|auto) (default: auto).")
    demo_ist.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for predicted instances (default: 0.25).",
    )
    demo_ist.add_argument("--max-instances", type=int, default=8, help="Max predicted instances per pass (default: 8).")
    demo_ist.add_argument(
        "--corruption",
        choices=("gaussian_blur", "gaussian_noise", "brightness", "contrast", "jpeg"),
        default="brightness",
        help="Corruption preset applied before inference (default: brightness).",
    )
    demo_ist.add_argument("--severity", type=int, default=5, help="Corruption severity 1..5 (default: 5).")
    demo_ist.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Optional fixed COCO image_id. If omitted, the demo scans up to --num-images and selects the clearest improvement.",
    )

    demo_cl = demo_sub.add_parser("continual", help="Toy continual-learning demo (requires torch; CPU OK).")
    demo_cl.add_argument("--output", default=None, help="Output JSON path or dir (default: runs/yolozu_demos/continual/...).")
    demo_cl.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_cl.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_cl.add_argument(
        "--practical",
        action="store_true",
        help=(
            "Run a more practical vision demo (MNIST rotation shift + ResNet18 backbone) "
            "with CPU-friendly fast defaults. Equivalent to setting --problem mnist_rotate with smaller steps/sizes."
        ),
    )
    demo_cl.add_argument(
        "--fast",
        action="store_true",
        help="Reduce steps and sample counts to keep CPU runs short (applies to toy2d and mnist_rotate).",
    )
    demo_cl.add_argument(
        "--problem",
        default="toy2d",
        choices=("toy2d", "mnist_rotate"),
        help="Continual-learning problem: toy2d or MNIST rotation shift (default: toy2d).",
    )
    demo_cl.add_argument(
        "--data-dir",
        default=str(Path("data") / "torchvision"),
        help="(problem=mnist_rotate) Torchvision dataset root dir (default: data/torchvision).",
    )
    demo_cl.add_argument("--method", default="ewc_replay", choices=("naive", "ewc", "replay", "ewc_replay"))
    demo_cl.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=("naive", "ewc", "replay", "ewc_replay"),
        help="Run multiple methods and write a suite report.",
    )
    demo_cl.add_argument(
        "--compare",
        action="store_true",
        help="Convenience flag: run all methods (naive/ewc/replay/ewc_replay) and write a suite report.",
    )
    demo_cl.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a markdown summary table next to the JSON output (suite or single).",
    )
    demo_cl.add_argument("--steps-a", type=int, default=200, help="Training steps on domain A (default: 200).")
    demo_cl.add_argument("--steps-b", type=int, default=200, help="Training steps on domain B (default: 200).")
    demo_cl.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    demo_cl.add_argument("--hidden", type=int, default=32, help="Hidden units (default: 32).")
    demo_cl.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2).")
    demo_cl.add_argument("--corr", type=float, default=2.0, help="Spurious correlation magnitude (default: 2.0).")
    demo_cl.add_argument("--noise", type=float, default=0.6, help="Feature noise std (default: 0.6).")
    demo_cl.add_argument("--n-train", type=int, default=4096, help="Train samples per domain (default: 4096).")
    demo_cl.add_argument("--n-eval", type=int, default=1024, help="Eval samples per domain (default: 1024).")
    demo_cl.add_argument("--ewc-lambda", type=float, default=20.0, help="EWC penalty weight (default: 20.0).")
    demo_cl.add_argument("--fisher-batches", type=int, default=64, help="Batches for Fisher estimate (default: 64).")
    demo_cl.add_argument("--replay-capacity", type=int, default=512, help="Replay buffer capacity (default: 512).")
    demo_cl.add_argument("--replay-k", type=int, default=64, help="Replay samples per step (default: 64).")

    demo_kp = demo_sub.add_parser("keypoints", help="Keypoints inference demo (torchvision Keypoint R-CNN).")
    demo_kp.add_argument("--image", default=None, help="Input image path (default: a bundled smoke image if present).")
    demo_kp.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/keypoints/<utc>).")
    demo_kp.add_argument("--device", default="auto", help="Torch device (cpu|cuda|mps|auto) (default: auto).")
    demo_kp.add_argument("--score-threshold", type=float, default=0.7, help="Min score to keep persons (default: 0.7).")
    demo_kp.add_argument("--max-persons", type=int, default=3, help="Max persons to render (default: 3).")

    demo_pose = demo_sub.add_parser("pose", help="6D pose demo (chessboard or ArUco + OpenCV solvePnP).")
    demo_pose.add_argument("--image", default=None, help="Input image path (default: a generated chessboard sample).")
    demo_pose.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/pose/<utc>).")
    demo_pose.add_argument(
        "--backend",
        choices=("chessboard", "aruco", "densefusion"),
        default="chessboard",
        help="Pose backend (default: chessboard).",
    )
    demo_pose.add_argument("--pattern-cols", type=int, default=None, help="Chessboard inner corners (cols).")
    demo_pose.add_argument("--pattern-rows", type=int, default=None, help="Chessboard inner corners (rows).")
    demo_pose.add_argument("--square-size", type=float, default=0.04, help="Chessboard square size in meters (default: 0.04).")
    demo_pose.add_argument("--aruco-dict", default="DICT_4X4_50", help="ArUco dictionary name (default: DICT_4X4_50).")
    demo_pose.add_argument("--aruco-id", type=int, default=23, help="ArUco marker id (default: 23).")
    demo_pose.add_argument("--marker-length", type=float, default=0.05, help="ArUco marker length in meters (default: 0.05).")
    demo_pose.add_argument("--densefusion-root", default=None, help="DenseFusion repo root (default: demo_output/pose/_densefusion).")
    demo_pose.add_argument("--densefusion-object", default="ape", help="LineMOD object for DenseFusion demo (default: ape).")
    demo_pose.add_argument(
        "--densefusion-auto-download",
        action="store_true",
        help="Auto-download DenseFusion assets (large) (default: enabled).",
    )
    demo_pose.add_argument(
        "--no-densefusion-auto-download",
        dest="densefusion_auto_download",
        action="store_false",
        help="Disable auto-download for DenseFusion assets.",
    )
    demo_pose.set_defaults(densefusion_auto_download=True)
    demo_pose.add_argument("--densefusion-model", default=None, help="DenseFusion pose model checkpoint path.")
    demo_pose.add_argument("--densefusion-refine-model", default=None, help="DenseFusion refiner checkpoint path.")
    demo_pose.add_argument("--camera-fx", type=float, default=None, help="Camera fx (default: inferred from image).")
    demo_pose.add_argument("--camera-fy", type=float, default=None, help="Camera fy (default: inferred from image).")
    demo_pose.add_argument("--camera-cx", type=float, default=None, help="Camera cx (default: image center).")
    demo_pose.add_argument("--camera-cy", type=float, default=None, help="Camera cy (default: image center).")
    demo_pose.add_argument(
        "--sample-source",
        choices=("auto", "download", "synthetic"),
        default="auto",
        help="Sample source when --image is omitted (default: auto).",
    )

    demo_depth = demo_sub.add_parser(
        "depth", help="Monocular depth inference demo (Depth Anything / MiDaS / DPT; relative depth)."
    )
    demo_depth.add_argument("--image", default=None, help="Input image path (default: a bundled smoke image if present).")
    demo_depth.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/depth/<utc>).")
    demo_depth.add_argument("--device", default="auto", help="Torch device (cpu|cuda|mps|auto) (default: auto).")
    demo_depth.add_argument(
        "--model",
        default="depth_anything",
        choices=("depth_anything", "midas_small", "dpt_hybrid", "dpt_large"),
        help="Depth model preset (default: depth_anything).",
    )
    demo_depth.add_argument(
        "--compare",
        action="store_true",
        help="Run 3-model comparison (midas_small, dpt_hybrid, depth_anything) and write suffixed outputs.",
    )
    demo_depth.add_argument(
        "--invert",
        action="store_true",
        help="Invert visualization (closer=brighter) (default: enabled).",
    )
    demo_depth.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="Disable inversion (farther=brighter).",
    )
    demo_depth.set_defaults(invert=True)

    demo_ttt = demo_sub.add_parser(
        "ttt",
        help="TTT improvement micro-demo (few-shot train + deterministic domain shift + mAP proxy).",
    )
    demo_ttt.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/ttt/<utc>).")
    demo_ttt.add_argument("--dataset-root", default=str(Path("data") / "smoke"), help="Source YOLO dataset root (default: data/smoke).")
    demo_ttt.add_argument("--split", default="val", help="Split name under images/ and labels/ (default: val).")
    demo_ttt.add_argument("--max-images", type=int, default=10, help="Max images used for training/eval (default: 10).")
    demo_ttt.add_argument(
        "--corruption",
        choices=("gaussian_blur", "gaussian_noise", "brightness", "contrast", "jpeg"),
        default="gaussian_noise",
        help="Domain-shift corruption preset (default: gaussian_noise).",
    )
    demo_ttt.add_argument("--severity", type=int, default=3, help="Corruption severity 1..5 (default: 3).")
    demo_ttt.add_argument("--seed", type=int, default=2026, help="Deterministic seed (default: 2026).")
    demo_ttt.add_argument("--train-seed", type=int, default=0, help="Training RNG seed (default: 0).")
    demo_ttt.add_argument("--train-epochs", type=int, default=30, help="Few-shot training epochs (default: 30).")
    demo_ttt.add_argument("--train-batch-size", type=int, default=2, help="Few-shot training batch size (default: 2).")
    demo_ttt.add_argument("--train-lr", type=float, default=1e-3, help="Few-shot training learning rate (default: 1e-3).")
    demo_ttt.add_argument("--image-size", type=int, default=320, help="Train/infer image size (square, default: 320).")
    demo_ttt.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_ttt.add_argument(
        "--adapter-config",
        default="configs/yolo26_rtdetr_pose/yolo26n.json",
        help="RTDETRPose config JSON path used for train/infer (default: configs/yolo26_rtdetr_pose/yolo26n.json).",
    )
    demo_ttt.add_argument("--score-threshold", type=float, default=0.01, help="Detection score threshold (default: 0.01).")
    demo_ttt.add_argument("--max-detections", type=int, default=100, help="Max detections per image (default: 100).")
    demo_ttt.add_argument("--ttt-preset", choices=("safe",), default="safe", help="TTT preset (default: safe).")
    demo_ttt.add_argument("--force", action="store_true", help="Overwrite existing run_dir artifacts (train + predictions).")

    demo_tr = demo_sub.add_parser("train", help="Training demo (MNIST fine-tune; requires torch+torchvision).")
    demo_tr.add_argument(
        "--output",
        default=None,
        help="Output JSON path or run directory (default: demo_output/train/<utc>/train_demo_report.json).",
    )
    demo_tr.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_tr.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_tr.add_argument(
        "--data-dir",
        default=str(Path("data") / "torchvision"),
        help="Torchvision dataset root dir (default: data/torchvision).",
    )
    demo_tr.add_argument("--epochs", type=int, default=1, help="Epochs (default: 1).")
    demo_tr.add_argument("--max-steps", type=int, default=80, help="Max train steps (default: 80).")
    demo_tr.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    demo_tr.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4).")

    reg = sub.add_parser("registry", help="AI-first tool registry: list/show/validate/run tools from the canonical manifest.")
    reg_sub = reg.add_subparsers(dest="registry_command", required=True)
    reg_validate = reg_sub.add_parser("validate", help="Validate the canonical tool manifest (repo checkout required).")
    reg_validate.set_defaults(_fn=_cmd_registry_validate)
    reg_list = reg_sub.add_parser("list", help="List tools in the canonical manifest.")
    reg_list.add_argument("-j", "--json", action="store_true", help="Emit machine-readable JSON.")
    reg_list.add_argument("--tag", action="append", default=None, help="Filter by tag (repeatable, AND).")
    reg_list.add_argument("--contract", action="append", default=None, help="Filter by contract id (repeatable, AND).")
    reg_list.set_defaults(_fn=_cmd_registry_list)
    reg_show = reg_sub.add_parser("show", help="Show one tool spec from the canonical manifest.")
    reg_show.add_argument("id", help="Tool id from the manifest.")
    reg_show.add_argument("-j", "--json", action="store_true", help="Emit machine-readable JSON.")
    reg_show.set_defaults(_fn=_cmd_registry_show)
    reg_run = reg_sub.add_parser("run", help="Safely run a tool by id with allowlisted side effects (repo checkout required).")
    reg_run.add_argument("id", help="Tool id from the manifest.")
    reg_run.add_argument("-n", "--dry-run", action="store_true", help="Print the resolved command without executing.")
    reg_run.add_argument("--allow-network", action="store_true", help="Allow tools that require network access.")
    reg_run.add_argument("--allow-gpu", action="store_true", help="Allow tools that require GPU.")
    reg_run.add_argument("--allow-write-root", action="append", default=None, help="Allow writing under this repo-relative root (repeatable). Default: reports.")
    reg_run.add_argument("--allow-unsafe-paths", action="store_true", help="Allow absolute paths or '..' segments.")
    reg_run.add_argument("--allow-unknown-flags", action="store_true", help="Allow forwarding flags not declared in tool.inputs (not recommended for agents).")
    reg_run.add_argument("--allow-undeclared-effects", action="store_true", help="Allow running tools without tool.effects declarations (not recommended for agents).")
    reg_run.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the tool entrypoint.")
    reg_run.set_defaults(_fn=_cmd_registry_run)

    completion = sub.add_parser("completion", aliases=["comp"], help="Print shell completion script (bash/zsh).")
    completion.add_argument("--shell", choices=("bash", "zsh"), default="bash", help="Target shell (default: bash).")
    completion.add_argument(
        "--command",
        dest="completion_command",
        default="yolozu",
        help="Command name to bind completion to (default: yolozu).",
    )
    completion.add_argument("--output", default="-", help="Output path (default: stdout).")

    args, extra_argv = parser.parse_known_args(argv)
    if args.command not in ("train", "test") and extra_argv:
        parser.error(f"unrecognized arguments: {' '.join(extra_argv)}")
    if args.command == "guide":
        payload = _guide_payload(str(args.goal))
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_render_guide_text(str(args.goal)), end="")
        return 0
    if args.command == "train":
        if getattr(args, "import_from", None) and getattr(args, "external_backend", None):
            raise SystemExit("train cannot combine --import preview with --external-backend")
        if getattr(args, "import_from", None):
            _cmd_train_import_preview(args)
            if not getattr(args, "config", None):
                return 0
        if getattr(args, "external_backend", None):
            return _cmd_train_external(args, extra_args=list(extra_argv or []))
        if not getattr(args, "config", None):
            raise SystemExit("train config is required unless using --import preview-only mode")
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_train(config_path, extra_args=list(extra_argv or []))
    if args.command == "test":
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_test(config_path, extra_args=list(extra_argv or []))
    if args.command == "train-orchestrate":
        return _cmd_train_orchestrate(args)
    if args.command == "doctor":
        if getattr(args, "doctor_command", None) == "import":
            return _cmd_doctor_import(args)
        if getattr(args, "doctor_command", None) == "train-dataset":
            return _cmd_doctor_train_dataset(args)
        return _cmd_doctor(
            str(args.output),
            explain=bool(getattr(args, "explain", False)),
            proof=bool(getattr(args, "proof", False)),
            proof_dir=str(getattr(args, "proof_dir", "reports/doctor_proof")),
        )
    if args.command == "registry":
        fn = getattr(args, "_fn", None)
        if fn is None:
            raise SystemExit("missing registry handler")
        return int(fn(args))
    if args.command == "list":
        if args.list_command == "models":
            return _cmd_list_models(args)
        raise SystemExit("unknown list command")
    if args.command == "fetch":
        return _cmd_fetch_model(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "export-dataset":
        return _cmd_export_dataset(args)
    if args.command == "predict-images":
        return _cmd_predict_images(args)
    if args.command == "eval-coco":
        return _cmd_eval_coco(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "eval-long-tail":
        return _cmd_eval_long_tail(args)
    if args.command == "long-tail-recipe":
        return _cmd_long_tail_recipe(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "parity":
        return _cmd_parity(args)
    if args.command == "predictions":
        return _cmd_predictions(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "eval-instance-seg":
        return _cmd_eval_instance_seg(args)
    if args.command == "onnxrt":
        if args.onnxrt_command == "export":
            return _cmd_onnxrt_export(args)
        if args.onnxrt_command == "quantize":
            return _cmd_onnxrt_quantize(args)
        raise SystemExit("unknown onnxrt command")
    if args.command == "resources":
        return _cmd_resources(args)
    if args.command == "migrate":
        return _cmd_migrate(args)
    if args.command == "import":
        return _cmd_import(args)
    if args.command == "demo":
        return handle_demo_command(args)
    if args.command == "completion":
        rendered = write_completion(shell=str(args.shell), command=str(args.completion_command), output=str(args.output))
        if str(args.output).strip() == "-" or not str(args.output).strip():
            print(rendered, end="" if rendered.endswith("\n") else "\n")
        else:
            print(str(rendered))
        return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
