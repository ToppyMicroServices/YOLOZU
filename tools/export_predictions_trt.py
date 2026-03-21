import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.letterbox import compute_letterbox, input_xyxy_to_orig_xyxy, orig_xyxy_to_cxcywh_norm
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--engine", default=None, help="Path to TensorRT engine plan (required unless --dry-run).")
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size (square). If omitted, tries to infer from the TensorRT engine; defaults to 640.",
    )
    p.add_argument("--input-name", default="images", help="TensorRT input binding name (default: images).")
    p.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    p.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    p.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    p.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    p.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    p.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    p.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    p.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] (norm) wrt input_size (default: norm).",
    )
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    p.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    p.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    p.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS when decoding raw output.",
    )
    p.add_argument("--output", default="reports/predictions_trt.json", help="Where to write predictions JSON.")
    p.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    p.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")
    return p.parse_args(argv)


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_boxes_and_scores(outputs: dict[str, object], *, boxes_key: str, scores_key: str, class_key: str | None):
    boxes = outputs.get(boxes_key)
    scores = outputs.get(scores_key)
    if boxes is None:
        raise ValueError(f"missing boxes output: {boxes_key}")
    if scores is None:
        raise ValueError(f"missing scores output: {scores_key}")
    class_ids = outputs.get(class_key) if class_key else None
    return boxes, scores, class_ids


class _CudaBackend:
    def __init__(self, name: str, module):
        self.name = name
        self.module = module


def _to_int_handle(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    for attr in ("value", "handle"):
        try:
            attr_value = getattr(value, attr)
            if attr_value is not None:
                return int(attr_value)
        except (AttributeError, TypeError, ValueError):
            continue
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError(f"failed to convert CUDA handle/pointer to int: {value!r}") from exc


def _load_cuda_backend() -> _CudaBackend:
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401

        return _CudaBackend("pycuda", cuda)
    except Exception:
        pass
    try:
        from cuda.bindings import runtime as cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        pass
    try:
        from cuda import cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        raise RuntimeError("CUDA bindings not found (install pycuda or cuda-python)")


class _TrtRunner:
    def __init__(self, *, engine_path: Path, input_name: str):
        self.backend = _load_cuda_backend()
        self.stream = self._create_stream()

        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("tensorrt is required (pip install nvidia-tensorrt)") from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with engine_path.open("rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        self.input_name = input_name
        self.mode = "v2"
        self.io_names: list[str] = []
        self.output_names: list[str] = []

        self.bindings: list[int] = []
        self.binding_indices: dict[str, int] = {}
        self.device_buffers: dict[str, object] = {}
        self.host_buffers: dict[str, np.ndarray] = {}
        self.last_shapes: dict[str, tuple[int, ...]] = {}

        # TensorRT 10+ uses tensor-address v3 API; older uses bindings/v2.
        if hasattr(self.context, "execute_async_v3") and hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name"):
            self.mode = "v3"
            self.io_names = [str(self.engine.get_tensor_name(i)) for i in range(int(self.engine.num_io_tensors))]
            inputs = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if self.input_name not in inputs:
                raise ValueError(f"input tensor not found: {self.input_name} (available: {inputs})")
        else:
            self.mode = "v2"
            self.bindings = [0] * int(self.engine.num_bindings)
            self.io_names = [str(self.engine.get_binding_name(i)) for i in range(int(self.engine.num_bindings))]
            self.output_names = [
                str(self.engine.get_binding_name(i)) for i in range(int(self.engine.num_bindings)) if not self.engine.binding_is_input(i)
            ]
            self.binding_indices = {str(self.engine.get_binding_name(i)): i for i in range(int(self.engine.num_bindings))}
            if self.input_name not in self.binding_indices:
                raise ValueError(f"input binding not found: {self.input_name}")

    @staticmethod
    def _cuda_error_code(result: object) -> int:
        if isinstance(result, tuple):
            if not result:
                return 1
            result = result[0]
        try:
            return int(result)
        except Exception:
            return 1

    def _cuda_check(self, result: object, *, op: str) -> None:
        code = self._cuda_error_code(result)
        if code != 0:
            raise RuntimeError(f"{op} failed (error {code})")

    def _create_stream(self):
        if self.backend.name == "pycuda":
            return self.backend.module.Stream()
        err, stream = self.backend.module.cudaStreamCreate()
        self._cuda_check(err, op="cudaStreamCreate")
        return _to_int_handle(stream)

    def _alloc(self, name: str, shape: tuple[int, ...], dtype):
        size = int(np.prod(shape))
        host = np.empty(size, dtype=dtype)
        nbytes = host.nbytes
        if self.backend.name == "pycuda":
            device = self.backend.module.mem_alloc(nbytes)
        else:
            err, device = self.backend.module.cudaMalloc(nbytes)
            self._cuda_check(err, op=f"cudaMalloc({name})")
            device = _to_int_handle(device)
        self.host_buffers[name] = host
        self.device_buffers[name] = device
        self.last_shapes[name] = shape

    def _ensure_buffers(self, input_shape: tuple[int, ...]):
        if self.mode == "v3":
            self.context.set_input_shape(self.input_name, input_shape)
            for name in self.io_names:
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT and name == self.input_name:
                    shape = tuple(input_shape)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                if any(dim <= 0 for dim in shape):
                    raise RuntimeError(f"tensor shape unresolved for {name}: {shape}")
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                if name not in self.last_shapes or self.last_shapes[name] != shape:
                    self._alloc(name, shape, dtype)
                self.context.set_tensor_address(name, int(self.device_buffers[name]))
            return

        input_idx = self.binding_indices[self.input_name]
        self.context.set_binding_shape(input_idx, input_shape)
        for i in range(int(self.engine.num_bindings)):
            name = str(self.engine.get_binding_name(i))
            shape = tuple(self.context.get_binding_shape(i))
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(f"binding shape unresolved for {name}: {shape}")
            dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
            if name not in self.last_shapes or self.last_shapes[name] != shape:
                self._alloc(name, shape, dtype)
            self.bindings[i] = int(self.device_buffers[name])

    def infer(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        input_shape = tuple(input_tensor.shape)
        self._ensure_buffers(input_shape)

        input_name = self.input_name
        if self.mode == "v3":
            input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(input_name))
        else:
            input_idx = self.binding_indices[input_name]
            input_dtype = self.trt.nptype(self.engine.get_binding_dtype(input_idx))
        if input_tensor.dtype != input_dtype:
            input_tensor = input_tensor.astype(input_dtype)
        host_input = self.host_buffers[input_name]
        np.copyto(host_input.reshape(input_shape), input_tensor)

        if self.backend.name == "pycuda":
            self.backend.module.memcpy_htod_async(
                self.device_buffers[input_name], host_input, self.stream
            )
            if self.mode == "v3":
                ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execution failed")
            for name in self.output_names:
                host = self.host_buffers[name]
                self.backend.module.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
            self.stream.synchronize()
            outputs: dict[str, np.ndarray] = {}
            for name in self.output_names:
                host = self.host_buffers[name]
                outputs[name] = host.reshape(self.last_shapes[name]).copy()
            return outputs

        err = self.backend.module.cudaMemcpyAsync(
            _to_int_handle(self.device_buffers[input_name]),
            int(host_input.ctypes.data),
            host_input.nbytes,
            self.backend.module.cudaMemcpyKind.cudaMemcpyHostToDevice,
            _to_int_handle(self.stream),
        )
        self._cuda_check(err, op="cudaMemcpyAsync H2D")

        if self.mode == "v3":
            ok = self.context.execute_async_v3(stream_handle=_to_int_handle(self.stream))
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=_to_int_handle(self.stream))
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            host = self.host_buffers[name]
            err = self.backend.module.cudaMemcpyAsync(
                int(host.ctypes.data),
                _to_int_handle(self.device_buffers[name]),
                host.nbytes,
                self.backend.module.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                _to_int_handle(self.stream),
            )
            self._cuda_check(err, op=f"cudaMemcpyAsync D2H({name})")

        err = self.backend.module.cudaStreamSynchronize(_to_int_handle(self.stream))
        self._cuda_check(err, op="cudaStreamSynchronize")
        for name in self.output_names:
            host = self.host_buffers[name]
            outputs[name] = host.reshape(self.last_shapes[name]).copy()
        return outputs


def _shape_to_ints(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(v) for v in value)
    except Exception:
        return None


def _infer_engine_input_hw(runner: _TrtRunner) -> tuple[int | None, int | None, bool]:
    """Return (H, W, dynamic) best-effort from TensorRT engine input shape."""

    shape = None
    if runner.mode == "v3":
        try:
            shape = runner.engine.get_tensor_shape(runner.input_name)
        except Exception:
            shape = None
    else:
        try:
            idx = runner.binding_indices[runner.input_name]
            shape = runner.engine.get_binding_shape(int(idx))
        except Exception:
            shape = None

    dims = _shape_to_ints(shape)
    if not dims:
        return None, None, True

    dynamic = any(int(d) <= 0 for d in dims)
    if len(dims) >= 2:
        h, w = int(dims[-2]), int(dims[-1])
    else:
        return None, None, True

    if h <= 0 or w <= 0:
        return None, None, True
    return h, w, dynamic


def _resolve_trt_input_size(
    *,
    requested_imgsz: int | None,
    eng_h: int | None,
    eng_w: int | None,
    eng_dynamic: bool,
) -> tuple[int, str | None]:
    input_size = 640 if requested_imgsz is None else int(requested_imgsz)

    if eng_h is not None and eng_w is not None and eng_h != eng_w and not eng_dynamic:
        raise SystemExit(
            f"engine input is non-square ({eng_h}x{eng_w}); export_predictions_trt expects square. Rebuild engine for square inputs."
        )
    if requested_imgsz is None and eng_h is not None and eng_w is not None and eng_h == eng_w:
        input_size = int(eng_h)
    if eng_h is not None and eng_w is not None and eng_h == eng_w and not eng_dynamic and int(input_size) != int(eng_h):
        raise SystemExit(
            f"engine expects input {eng_h}x{eng_w} but imgsz={input_size}. Rebuild engine for the desired size or pass a matching --imgsz."
        )
    if requested_imgsz is None and eng_dynamic:
        return input_size, "[warn] TensorRT engine input is dynamic; using default imgsz=640. Pass --imgsz to pin preprocessing."
    return input_size, None


def _validate_decode_contract(*, exporter_name: str, boxes_format: str, boxes_scale: str, raw_output: str | None) -> None:
    if boxes_format != "xyxy":
        raise ValueError(f"only --boxes-format xyxy is supported in this {exporter_name}")
    if raw_output and boxes_scale != "abs":
        raise ValueError("--raw-output expects --boxes-scale abs (input-space pixels)")


def _split_combined_output(values, *, fmt: str):
    if fmt != "xyxy_score_class":
        raise ValueError(f"unsupported combined format: {fmt}")
    arr = values
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"unsupported combined output shape: {arr.shape}")
    boxes = arr[:, :4]
    scores = arr[:, 4]
    class_ids = arr[:, 5]
    return boxes, scores, class_ids


def _normalize_raw_output(values, *, fmt: str):
    if fmt != "yolo_84":
        raise ValueError(f"unsupported raw format: {fmt}")
    arr = values
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    if arr.shape[0] in (84, 85):
        arr = arr.T
    elif arr.shape[1] not in (84, 85):
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    return arr


def _xywh_to_xyxy(boxes):
    x, y, w, h = boxes.T
    x1 = x - (w / 2.0)
    y1 = y - (h / 2.0)
    x2 = x + (w / 2.0)
    y2 = y + (h / 2.0)
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _nms(boxes, scores, *, iou_thresh: float, max_det: int):
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    try:
        import torch  # type: ignore

        try:
            from ultralytics.utils.nms import TorchNMS  # type: ignore

            keep = TorchNMS.nms(
                torch.as_tensor(boxes, dtype=torch.float32),
                torch.as_tensor(scores, dtype=torch.float32),
                float(iou_thresh),
            )
            keep = keep[: int(max_det)].cpu().numpy().astype(np.int64)
            return keep
        except Exception:
            try:
                import torchvision  # type: ignore

                keep = torchvision.ops.nms(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    torch.as_tensor(scores, dtype=torch.float32),
                    float(iou_thresh),
                )
                keep = keep[: int(max_det)].cpu().numpy().astype(np.int64)
                return keep
            except Exception:
                return None
    except Exception:
        return None

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= float(iou_thresh)]
    return np.array(keep, dtype=np.int64)


def _decode_raw_output(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
):
    data = _normalize_raw_output(raw, fmt="yolo_84")
    if data.shape[1] <= 4:
        raise ValueError(f"raw output has no class scores: {data.shape}")
    boxes_xywh = data[:, :4]
    scores_all = data[:, 4:]
    class_ids = np.argmax(scores_all, axis=1)
    scores = np.max(scores_all, axis=1)

    keep = scores >= float(min_score)
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    if boxes_xyxy.size == 0:
        return boxes_xyxy, scores, class_ids

    max_nms = 30000
    if boxes_xyxy.shape[0] > max_nms:
        order = scores.argsort()[::-1][:max_nms]
        boxes_xyxy = boxes_xyxy[order]
        scores = scores[order]
        class_ids = class_ids[order]

    if not agnostic:
        max_wh = 7680.0
        offsets = class_ids.astype(np.float32) * max_wh
        boxes_nms = boxes_xyxy.copy()
        boxes_nms[:, 0] += offsets
        boxes_nms[:, 1] += offsets
        boxes_nms[:, 2] += offsets
        boxes_nms[:, 3] += offsets
    else:
        boxes_nms = boxes_xyxy

    keep_idx = _nms(boxes_nms, scores, iou_thresh=iou_thresh, max_det=max_det)
    return boxes_xyxy[keep_idx], scores[keep_idx], class_ids[keep_idx]


def _decode_raw_ultralytics(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
):
    try:
        import torch  # type: ignore
        from ultralytics.utils import nms as u_nms  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics + torch are required for raw-postprocess=ultralytics") from exc

    arr = raw
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    if arr.shape[1] in (84, 85) and arr.shape[0] not in (84, 85):
        arr = arr.T
    if arr.shape[0] not in (84, 85):
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    pred = torch.as_tensor(arr[None, ...], dtype=torch.float32)

    outputs = u_nms.non_max_suppression(
        pred,
        conf_thres=float(min_score),
        iou_thres=float(iou_thresh),
        classes=None,
        agnostic=bool(agnostic),
        max_det=int(max_det),
        nc=0,
        end2end=False,
        rotated=False,
        return_idxs=False,
    )
    if not outputs or outputs[0] is None or len(outputs[0]) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    out = outputs[0].detach().cpu().numpy()
    boxes = out[:, :4]
    scores = out[:, 4]
    class_ids = out[:, 5].astype(int)
    return boxes, scores, class_ids


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_size = 640 if args.imgsz is None else int(args.imgsz)
    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    predictions: list[dict[str, Any]] = []
    engine_path: Path | None = None

    if args.dry_run:
        predictions = [{"image": record["image"], "detections": []} for record in records]
        validate_predictions_entries(predictions, strict=args.strict)
    else:
        if not args.engine:
            raise SystemExit("--engine is required unless --dry-run is set")
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required for TensorRT exporter")
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for image loading (pip install opencv-python)") from exc

        engine_path = Path(args.engine)
        if not engine_path.is_absolute():
            engine_path = repo_root / engine_path
        if not engine_path.exists():
            raise SystemExit(f"engine not found: {engine_path}")

        runner = _TrtRunner(engine_path=engine_path, input_name=args.input_name)
        eng_h, eng_w, eng_dyn = _infer_engine_input_hw(runner)
        input_size, shape_warning = _resolve_trt_input_size(
            requested_imgsz=args.imgsz,
            eng_h=eng_h,
            eng_w=eng_w,
            eng_dynamic=eng_dyn,
        )
        if shape_warning:
            print(shape_warning, file=sys.stderr)

        _validate_decode_contract(
            exporter_name="exporter",
            boxes_format=str(args.boxes_format),
            boxes_scale=str(args.boxes_scale),
            raw_output=args.raw_output,
        )

        def preprocess(image_path: str):
            w, h = get_image_size(image_path)
            letterbox = compute_letterbox(orig_w=w, orig_h=h, input_size=input_size)

            img = cv2.imread(image_path)
            if img is None:
                raise RuntimeError(f"failed to load image: {image_path}")

            pad_w = float(input_size) - float(letterbox.new_w)
            pad_h = float(input_size) - float(letterbox.new_h)
            pad_x = pad_w / 2.0
            pad_y = pad_h / 2.0
            left = int(letterbox.pad_x)
            top = int(letterbox.pad_y)
            right = int(round(pad_x + 0.1))
            bottom = int(round(pad_y + 0.1))

            if (img.shape[1], img.shape[0]) != (letterbox.new_w, letterbox.new_h):
                img = cv2.resize(img, (letterbox.new_w, letterbox.new_h), interpolation=cv2.INTER_LINEAR)

            img = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            img = img[..., ::-1]  # BGR to RGB

            x = img.astype(np.float32) / 255.0  # (H,W,C)
            x = np.transpose(x, (2, 0, 1))  # (C,H,W)
            x = np.expand_dims(x, axis=0)  # (1,C,H,W)
            return x, (w, h), letterbox

        for record in records:
            image_path = record["image"]
            x, (orig_w, orig_h), letterbox = preprocess(image_path)
            outputs = runner.infer(x)

            if args.combined_output:
                combined = outputs.get(args.combined_output)
                if combined is None:
                    raise ValueError(f"missing combined output: {args.combined_output}")
                boxes_t, scores_t, class_t = _split_combined_output(
                    np.asarray(combined), fmt=str(args.combined_format)
                )
            elif args.raw_output:
                raw = outputs.get(args.raw_output)
                if raw is None:
                    raise ValueError(f"missing raw output: {args.raw_output}")
                if args.raw_postprocess == "ultralytics":
                    boxes_t, scores_t, class_t = _decode_raw_ultralytics(
                        np.asarray(raw),
                        min_score=float(args.min_score),
                        iou_thresh=float(args.nms_iou),
                        max_det=int(args.topk),
                        agnostic=bool(args.agnostic_nms),
                    )
                else:
                    boxes_t, scores_t, class_t = _decode_raw_output(
                        np.asarray(raw),
                        min_score=float(args.min_score),
                        iou_thresh=float(args.nms_iou),
                        max_det=int(args.topk),
                        agnostic=bool(args.agnostic_nms),
                    )
            else:
                boxes_t, scores_t, class_t = _resolve_boxes_and_scores(
                    outputs,
                    boxes_key=args.boxes_output,
                    scores_key=args.scores_output,
                    class_key=args.class_output,
                )

            boxes = np.asarray(boxes_t)
            scores = np.asarray(scores_t)
            class_ids = None if class_t is None else np.asarray(class_t)

            if scores.ndim == 2:
                class_ids = np.argmax(scores, axis=1)
                scores = np.max(scores, axis=1)
            elif scores.ndim != 1:
                raise ValueError(f"unsupported scores shape: {scores.shape}")

            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"unsupported boxes shape: {boxes.shape}")
            if class_ids is None:
                raise ValueError("class ids missing: provide --class-output or use (N,C) scores")

            scores = scores.astype(float)
            class_ids = class_ids.astype(int)

            if args.raw_output:
                idx = list(range(len(scores)))
            else:
                idx = [i for i, s in enumerate(scores.tolist()) if float(s) >= float(args.min_score)]
                idx.sort(key=lambda i: float(scores[i]), reverse=True)
                idx = idx[: max(0, int(args.topk))]

            detections = []
            use_ultra_scale = bool(args.raw_output and args.raw_postprocess == "ultralytics")
            if use_ultra_scale:
                try:
                    import torch  # type: ignore
                    from ultralytics.utils import ops as u_ops  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError("ultralytics + torch are required for raw-postprocess=ultralytics") from exc
            for i in idx:
                b = boxes[i].tolist()
                if args.boxes_scale == "norm":
                    x1, y1, x2, y2 = (
                        float(b[0]) * input_size,
                        float(b[1]) * input_size,
                        float(b[2]) * input_size,
                        float(b[3]) * input_size,
                    )
                else:
                    x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

                if use_ultra_scale:
                    scaled = u_ops.scale_boxes(
                        (input_size, input_size),
                        torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
                        (orig_h, orig_w),
                    )
                    sx1, sy1, sx2, sy2 = scaled[0].tolist()
                    orig_xyxy = (float(sx1), float(sy1), float(sx2), float(sy2))
                else:
                    orig_xyxy = input_xyxy_to_orig_xyxy(
                        (x1, y1, x2, y2),
                        letterbox=letterbox,
                        orig_w=orig_w,
                        orig_h=orig_h,
                    )
                bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)

                detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})

            predictions.append({"image": image_path, "detections": detections})

        validate_predictions_entries(predictions, strict=args.strict)

    meta = {
        "timestamp": _now_utc(),
        "exporter": "tensorrt",
        "protocol_id": "yolo26",
        "dry_run": bool(args.dry_run),
        "imgsz": input_size,
        "dataset": args.dataset,
        "split": manifest["split"],
        "max_images": args.max_images,
        "engine": None if engine_path is None else str(engine_path),
        "engine_sha256": None if engine_path is None or not engine_path.exists() else _sha256(engine_path),
        "input_name": args.input_name,
        "boxes_output": args.boxes_output,
        "scores_output": args.scores_output,
        "class_output": args.class_output,
        "combined_output": args.combined_output,
        "combined_format": args.combined_format,
        "raw_output": args.raw_output,
        "raw_format": args.raw_format,
        "raw_postprocess": args.raw_postprocess,
        "boxes_format": args.boxes_format,
        "boxes_scale": args.boxes_scale,
        "min_score": args.min_score,
        "topk": args.topk,
        "nms_iou": args.nms_iou,
        "agnostic_nms": args.agnostic_nms,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "note": "dry-run mode: no inference performed" if args.dry_run else None,
    }

    payload = {"predictions": predictions, "meta": meta} if args.wrap else predictions
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()
