"""Model adapter interface and built-in adapter implementations.

Adapters provide a uniform ``predict(records) -> list[dict]`` interface so
that the evaluation, TTA, TTT, and distillation pipelines stay backend-
agnostic.

Public adapters
---------------
ModelAdapter         -- abstract base.
DummyAdapter         -- returns empty detections (smoke-testing).
PrecomputedAdapter   -- loads predictions from a JSON file.
RTDETRPoseAdapter    -- runs the RT-DETR pose reference adapter (optional ``torch``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

__all__ = [
    "ModelAdapter",
    "DummyAdapter",
    "PrecomputedAdapter",
    "RTDETRPoseAdapter",
]

ENTRY_SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)


class ModelAdapter:
    """Abstract base adapter — subclass to integrate a new backend."""

    def predict(self, records: list[dict]) -> list[dict]:
        """Run inference on a batch of record dicts and return detections."""
        raise NotImplementedError

    def supports_ttt(self) -> bool:
        """Return ``True`` if this adapter supports test-time training."""
        return False

    def get_model(self):
        """Return the underlying model object, or ``None``."""
        return None

    def build_loader(self, records, *, batch_size: int = 1):
        """Yield preprocessed batches for TTT (torch tensors)."""
        raise RuntimeError("this adapter does not support TTT")


class DummyAdapter(ModelAdapter):
    def predict(self, records):
        return [
            {"schema_version": ENTRY_SCHEMA_VERSION, "image": record["image"], "detections": []}
            for record in records
        ]


class PrecomputedAdapter(ModelAdapter):
    """Adapter that returns detections loaded from a JSON file.

    This is useful when you run real inference elsewhere (torch/TRT/etc.)
    and want to evaluate the pipeline in this repo without heavyweight deps.

    Supported JSON formats:

    1) List of per-image entries:
       [{"image": "/abs/or/rel/path.jpg", "detections": [...]}, ...]

    2) Dict with top-level key:
       {"predictions": [ ...same as above... ]}

    3) Dict mapping image->detections:
       {"/path.jpg": [...], "000000000009.jpg": [...]}  # values are detections
    """

    def __init__(self, predictions_path):
        from pathlib import Path

        self.predictions_path = str(predictions_path)
        self._path = Path(predictions_path)
        self._index = None

    def _load(self):
        from yolozu.predictions.predictions import load_predictions_index

        self._index = load_predictions_index(self._path)

    def predict(self, records):
        from yolozu.core.image_keys import lookup_image_alias, require_image_key

        if self._index is None:
            self._load()

        outputs = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"records[{idx}] must be an object")
            image_key = require_image_key(record.get("image"), where=f"records[{idx}].image")
            dets = lookup_image_alias(self._index, image_key)
            outputs.append(
                {
                    "schema_version": ENTRY_SCHEMA_VERSION,
                    "image": image_key,
                    "detections": dets if dets is not None else [],
                }
            )
        return outputs


class RTDETRPoseAdapter(ModelAdapter):
    """Adapter that runs the RT-DETR pose reference adapter (optional dependency).

    This adapter is intentionally dependency-light at import time.
    If torch isn't installed, it raises a clear RuntimeError on first use.

    Output schema per image:
      {
        "image": <path>,
        "detections": [
          {
            "class_id": int,
            "score": float,
            "bbox": {"cx": float, "cy": float, "w": float, "h": float},
            "log_z": float,
            "rot6d": [float, ...],
            "log_sigma_z": float,        # optional (uncertainty head)
            "log_sigma_rot": float,      # optional (uncertainty head)
            "sigma_z": float,            # optional exp(log_sigma_z) convenience
            "sigma_rot": float,          # optional exp(log_sigma_rot) convenience
            "offsets": [float, float],
            "k_delta": [float, float, float, float],
          },
          ...
        ]
      }
    """

    def __init__(
        self,
        config_path="builtin:base",
        checkpoint_path=None,
        device="cpu",
        image_size=(320, 320),
        score_threshold=0.3,
        max_detections=50,
        infer_batch_size: int = 1,
        *,
        allow_partial_checkpoint: bool = False,
        lora_r: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        lora_target: str = "head",
        lora_freeze_base: bool = False,
        lora_train_bias: str = "none",
        compile_model: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        allow_compile_fallback: bool = False,
        amp: str = "off",
        channels_last: bool = False,
        use_inference_mode: bool = True,
        init_seed: int | None = None,
        repro_policy: str = "relaxed",
    ):
        self.config_path = str(config_path)
        self.checkpoint_path = checkpoint_path
        self.allow_partial_checkpoint = bool(allow_partial_checkpoint)
        if self.allow_partial_checkpoint and not self.checkpoint_path:
            raise ValueError(
                "allow_partial_checkpoint requires checkpoint_path"
            )
        self._checkpoint_report: dict | None = None
        self.device = device
        self.image_size = tuple(image_size)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.infer_batch_size = int(infer_batch_size) if infer_batch_size is not None else 1
        self._backend = None
        self._lora_report: dict | None = None
        self.compile_model = bool(compile_model)
        self.compile_backend = str(compile_backend)
        self.compile_mode = str(compile_mode)
        self.compile_fullgraph = bool(compile_fullgraph)
        self.compile_dynamic = compile_dynamic
        self.allow_compile_fallback = bool(allow_compile_fallback)
        self._compile_evidence: dict = {
            "requested": {
                "enabled": bool(self.compile_model),
                "backend": self.compile_backend if self.compile_model else None,
                "mode": self.compile_mode if self.compile_model else None,
                "fullgraph": self.compile_fullgraph if self.compile_model else None,
                "dynamic": self.compile_dynamic if self.compile_model else None,
                "allow_fallback": (
                    self.allow_compile_fallback if self.compile_model else False
                ),
            },
            "actual": {
                "status": (
                    "pending_first_execution"
                    if self.compile_model
                    else "not_requested"
                ),
                "backend": None if self.compile_model else "eager",
                "mode": None,
                "fullgraph": None if self.compile_model else False,
                "dynamic": None,
            },
            "evidence": {
                "compile_api_available": None,
                "setup_completed": False,
                "first_execution_completed": False,
                "fallback_execution_completed": False,
                "counter_source": None,
                "counter_delta": None,
                "graph_count": None,
                "graph_break_count": None,
                "captured_call_count": None,
            },
            "failure": None,
        }
        self.amp = str(amp).strip().lower()
        self.channels_last = bool(channels_last)
        self.use_inference_mode = bool(use_inference_mode)
        self.init_seed = (int(init_seed) if init_seed is not None else None)
        self.repro_policy = str(repro_policy).strip().lower()
        if self.repro_policy not in ("strict", "relaxed", "off"):
            raise ValueError("repro_policy must be one of: strict, relaxed, off")

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha) if lora_alpha is not None else None
        self.lora_dropout = float(lora_dropout)
        self.lora_target = str(lora_target)
        self.lora_freeze_base = bool(lora_freeze_base)
        self.lora_train_bias = str(lora_train_bias)

    def _ensure_backend(self):
        if self._backend is not None:
            return

        import sys
        from pathlib import Path

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "RTDETRPoseAdapter requires 'torch'. Install PyTorch (and optionally torchvision) to enable real inference."
            ) from exc

        from PIL import Image, ImageOps
        import numpy as np

        try:
            from rtdetr_pose.config import load_config
            from rtdetr_pose.factory import build_model
        except Exception:
            # Source-checkout fallback (when rtdetr_pose isn't installed as a package).
            import importlib

            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / "rtdetr_pose"
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                importlib.invalidate_caches()
                # If `repo_root` is already on sys.path, Python may have resolved `rtdetr_pose`
                # as a namespace package (repo_root/rtdetr_pose/*). Force a reload so
                # `rtdetr_pose.config` resolves to the real package under candidate.
                sys.modules.pop("rtdetr_pose", None)
                for key in list(sys.modules.keys()):
                    if key.startswith("rtdetr_pose."):
                        sys.modules.pop(key, None)
            from rtdetr_pose.config import load_config
            from rtdetr_pose.factory import build_model

        cfg = load_config(self.config_path)
        num_classes_fg = int(getattr(cfg.model, "num_classes", 80))

        if self.repro_policy == "strict":
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                # Some runtime combos may not expose strict deterministic toggles.
                logger.debug("Deterministic algorithms toggle unavailable: %s", exc)
            if hasattr(torch.backends, "cudnn"):
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception as exc:
                    logger.debug("cuDNN deterministic flags unavailable: %s", exc)

        # Keep reference-adapter baselines reproducible without forcing global RNG.
        if self.repro_policy == "off" or self.init_seed is None:
            model = build_model(cfg.model).eval()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.init_seed))
                model = build_model(cfg.model).eval()

        # Optional LoRA injection (useful for PEFT checkpoints and TTT adapter-only updates).
        lora_report: dict | None = None
        if int(self.lora_r) > 0:
            from rtdetr_pose.lora import apply_lora, count_trainable_params, mark_only_lora_as_trainable

            replaced = apply_lora(
                model,
                r=int(self.lora_r),
                alpha=(float(self.lora_alpha) if self.lora_alpha is not None else None),
                dropout=float(self.lora_dropout),
                target=str(self.lora_target),
            )

            trainable_info = None
            if bool(self.lora_freeze_base):
                trainable_info = mark_only_lora_as_trainable(model, train_bias=str(self.lora_train_bias))

            lora_report = {
                "enabled": True,
                "replaced": int(replaced),
                "r": int(self.lora_r),
                "alpha": (float(self.lora_alpha) if self.lora_alpha is not None else None),
                "dropout": float(self.lora_dropout),
                "target": str(self.lora_target),
                "freeze_base": bool(self.lora_freeze_base),
                "train_bias": str(self.lora_train_bias),
                "trainable_params": int(count_trainable_params(model)),
                "trainable_info": trainable_info,
            }
        else:
            lora_report = {"enabled": False}

        if self.checkpoint_path:
            from yolozu.inference.checkpoint_compatibility import (
                CheckpointCompatibilityError,
                load_checkpoint_compatible,
            )

            try:
                self._checkpoint_report = load_checkpoint_compatible(
                    model,
                    self.checkpoint_path,
                    config_identity=self.config_path,
                    allow_partial=self.allow_partial_checkpoint,
                )
            except CheckpointCompatibilityError as exc:
                self._checkpoint_report = exc.report
                raise

        model.to(self.device)

        # Optional torch.compile for inference speedup (PyTorch 2.x).
        if self.compile_model:
            from yolozu.inference.torch_export import (
                compile_for_inference,
                get_compile_evidence,
            )

            model = compile_for_inference(
                model,
                backend=self.compile_backend,
                mode=self.compile_mode,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
                allow_fallback=self.allow_compile_fallback,
            )
            compile_evidence = get_compile_evidence(model)
            if isinstance(compile_evidence, dict):
                self._compile_evidence = compile_evidence

        from yolozu.geometry.intrinsics import parse_intrinsics as _parse_intrinsics

        def preprocess(record_or_path):
            if isinstance(record_or_path, dict):
                from yolozu.core.image_keys import require_image_key

                path = require_image_key(record_or_path.get("image"), where="record.image")
            else:
                path = record_or_path
            with Image.open(path) as src_img:
                img = ImageOps.exif_transpose(src_img).convert("RGB")
            orig_w, orig_h = img.size
            dst_w, dst_h = int(self.image_size[0]), int(self.image_size[1])
            img = img.resize((dst_w, dst_h), resample=Image.BILINEAR)

            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise RuntimeError("invalid RGB image array")
            if isinstance(record_or_path, dict) and bool(record_or_path.get("_tta_hflip", False)):
                arr = arr[:, ::-1, :].copy()
            arr = arr / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)

            meta = {
                "orig_size": {"width": int(orig_w), "height": int(orig_h)},
                "input_size": {"width": int(dst_w), "height": int(dst_h)},
                "model_input_size": {"width": int(dst_w), "height": int(dst_h)},
                "scale_xy": {"x": float(dst_w) / float(orig_w) if orig_w else None, "y": float(dst_h) / float(orig_h) if orig_h else None},
                "method": "resize",
                "resize": {"algorithm": "bilinear", "mode": "stretch"},
                "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                "letterbox": False,
                "normalize": "0_1",
                "dtype": "float32",
                "color_order": "RGB",
                "exif_orientation": "normalized",
                "decoder": "Pillow",
                "tta_aug": {
                    "hflip": bool(isinstance(record_or_path, dict) and record_or_path.get("_tta_hflip", False)),
                },
            }

            intr = None
            if isinstance(record_or_path, dict):
                for key in ("intrinsics", "K_gt", "K"):
                    intr = _parse_intrinsics(record_or_path.get(key))
                    if intr is not None:
                        break
            if intr is not None and orig_w and orig_h:
                sx = float(dst_w) / float(orig_w)
                sy = float(dst_h) / float(orig_h)
                intr = {"fx": float(intr["fx"]) * sx, "fy": float(intr["fy"]) * sy, "cx": float(intr["cx"]) * sx, "cy": float(intr["cy"]) * sy}

            return x.unsqueeze(0) if x.ndim == 3 else x, meta, intr

        self._backend = {"torch": torch, "model": model, "preprocess": preprocess, "num_classes_fg": num_classes_fg}
        self._lora_report = lora_report

    def get_compile_evidence(self) -> dict:
        import copy

        if self._backend is not None:
            from yolozu.inference.torch_export import get_compile_evidence

            current = get_compile_evidence(self._backend["model"])
            if isinstance(current, dict):
                self._compile_evidence = current
        return copy.deepcopy(self._compile_evidence)

    def get_checkpoint_report(self) -> dict | None:
        import copy

        return copy.deepcopy(self._checkpoint_report)

    def require_compile_established(self) -> dict:
        report = self.get_compile_evidence()
        if not self.compile_model:
            return report
        status = ((report.get("actual") or {}).get("status"))
        if status == "compiled":
            return report
        if status == "fallback" and self.allow_compile_fallback:
            completed = bool(
                ((report.get("evidence") or {}).get("fallback_execution_completed"))
            )
            if completed:
                return report
            raise RuntimeError(
                "torch.compile fallback was selected but no eager model input completed"
            )
        if status == "pending_first_execution":
            raise RuntimeError(
                "torch.compile was requested but no model input completed; "
                "compiled execution cannot be established"
            )
        raise RuntimeError(
            f"torch.compile was requested but actual status is {status!r}"
        )

    def get_lora_report(self) -> dict | None:
        if self._backend is None and int(self.lora_r) > 0:
            self._ensure_backend()
        return self._lora_report

    def supports_ttt(self) -> bool:
        return True

    def get_model(self):
        self._ensure_backend()
        return self._backend["model"]

    def build_loader(self, records, *, batch_size: int = 1):
        self._ensure_backend()
        torch = self._backend["torch"]
        preprocess = self._backend["preprocess"]

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        batch = []
        for record in records:
            x, _, _ = preprocess(record)
            x = x.to(self.device)
            if self.channels_last and int(getattr(x, "ndim", 0)) == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            batch.append(x)
            if len(batch) >= int(batch_size):
                yield torch.cat(batch, dim=0)
                batch = []
        if batch:
            yield torch.cat(batch, dim=0)

    def predict(self, records):
        self._ensure_backend()
        torch = self._backend["torch"]
        model = self._backend["model"]
        preprocess = self._backend["preprocess"]
        from yolozu.core.image_keys import require_image_key

        if not isinstance(records, list):
            raise ValueError("records must be a list of dicts with key 'image'")

        batch_size = int(getattr(self, "infer_batch_size", 1) or 1)
        if batch_size <= 0:
            raise ValueError("infer_batch_size must be > 0")

        def _autocast_context(x_tensor):
            mode = str(getattr(self, "amp", "off") or "off").strip().lower()
            if mode in ("", "off", "none", "false", "0"):
                return nullcontext()
            if mode in ("fp16", "float16", "half"):
                dtype = torch.float16
            elif mode in ("bf16", "bfloat16"):
                dtype = torch.bfloat16
            else:
                return nullcontext()

            device_type = str(getattr(x_tensor, "device", "cpu")).split(":")[0]
            if device_type not in ("cpu", "cuda", "mps"):
                return nullcontext()
            try:
                return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
            except Exception:
                return nullcontext()

        def _run_model(x_tensor):
            grad_ctx = torch.inference_mode if bool(self.use_inference_mode) else torch.no_grad
            with grad_ctx():
                with _autocast_context(x_tensor):
                    result = model(x_tensor)
            evidence = getattr(model, "_yolozu_compile_evidence", None)
            if (
                isinstance(evidence, dict)
                and (evidence.get("actual") or {}).get("status") == "fallback"
            ):
                runtime = evidence.get("evidence")
                if isinstance(runtime, dict):
                    runtime["fallback_execution_completed"] = True
            return result

        def _decode_single(
            *,
            idx: int,
            out: dict,
            image_path: str,
            pp_meta: dict,
            intrinsics: dict | None,
        ) -> dict:
            logits = out["logits"][idx]
            bbox = out["bbox"][idx]
            log_z = out["log_z"][idx]
            rot6d = out["rot6d"][idx]

            log_sigma_z = out.get("log_sigma_z")
            log_sigma_rot = out.get("log_sigma_rot")
            if log_sigma_z is not None:
                log_sigma_z = log_sigma_z[idx].squeeze(-1)
            if log_sigma_rot is not None:
                log_sigma_rot = log_sigma_rot[idx].squeeze(-1)

            offsets = out["offsets"][idx]
            k_delta = out["k_delta"][idx]

            keypoints = out.get("keypoints")
            if keypoints is not None:
                keypoints = keypoints[idx]

            probs = torch.softmax(logits, dim=-1)

            # Background-aware scoring (RT-DETR style: last logit is "no-object").
            #
            # The model is built with `num_classes = fg + 1` (factory adds +1),
            # so we must not ignore the background logit when selecting detections.
            num_classes_fg = int(self._backend.get("num_classes_fg") or max(0, int(probs.shape[-1]) - 1))
            if int(probs.shape[-1]) > 1:
                num_classes_fg = max(0, min(int(num_classes_fg), int(probs.shape[-1]) - 1))
            else:
                num_classes_fg = int(probs.shape[-1])

            if num_classes_fg <= 0:
                # Degenerate config: no foreground classes.
                scores = torch.zeros((int(probs.shape[0]),), device=probs.device, dtype=probs.dtype)
                class_ids = torch.zeros((int(probs.shape[0]),), device=probs.device, dtype=torch.long)
                keep = torch.zeros_like(class_ids, dtype=torch.bool)
            else:
                probs_fg = probs[..., :num_classes_fg]
                scores, class_ids = torch.max(probs_fg, dim=-1)
                bg_scores = probs[..., -1] if int(probs.shape[-1]) > int(num_classes_fg) else None
                keep = (scores >= bg_scores) if bg_scores is not None else torch.ones_like(scores, dtype=torch.bool)

            # Mask out background-dominated queries so topk doesn't waste slots on them.
            scores_for_topk = scores.clone()
            scores_for_topk[~keep] = -1.0
            k = min(self.max_detections, int(scores_for_topk.shape[0]))
            top_scores, top_idx = torch.topk(scores_for_topk, k=k)

            detections: list[dict] = []
            for score, q_idx in zip(top_scores.tolist(), top_idx.tolist()):
                if score < self.score_threshold:
                    continue
                if not bool(keep[q_idx]):
                    continue
                cls_id = int(class_ids[q_idx].item())
                box = torch.sigmoid(bbox[q_idx]).tolist()

                # offsets/k_delta can be either per-query (Q,*) or global (*,)
                off_q = offsets[q_idx] if hasattr(offsets, "ndim") and int(offsets.ndim) > 1 else offsets
                kd_q = k_delta[q_idx] if hasattr(k_delta, "ndim") and int(k_delta.ndim) > 1 else k_delta

                det = {
                    "class_id": cls_id,
                    "score": float(score),
                    "bbox": {"cx": float(box[0]), "cy": float(box[1]), "w": float(box[2]), "h": float(box[3])},
                    "log_z": float(log_z[q_idx].item()),
                    "rot6d": [float(v) for v in rot6d[q_idx].tolist()],
                    "offsets": [float(v) for v in off_q.tolist()],
                    "k_delta": [float(v) for v in kd_q.tolist()],
                }

                if keypoints is not None:
                    try:
                        kp_xy = keypoints[q_idx]
                        det["keypoints"] = [{"x": float(x), "y": float(y), "v": 2} for x, y in kp_xy.tolist()]
                    except (AttributeError, IndexError, TypeError, ValueError):
                        det.pop("keypoints", None)

                if log_sigma_z is not None:
                    ls_z = float(log_sigma_z[q_idx].item())
                    det["log_sigma_z"] = ls_z
                    det["sigma_z"] = float(torch.exp(log_sigma_z[q_idx]).item())

                if log_sigma_rot is not None:
                    ls_r = float(log_sigma_rot[q_idx].item())
                    det["log_sigma_rot"] = ls_r
                    det["sigma_rot"] = float(torch.exp(log_sigma_rot[q_idx]).item())

                detections.append(det)

            detections.sort(
                key=lambda d: (
                    -float(d.get("score", 0.0)),
                    int(d.get("class_id", -1)),
                    float((d.get("bbox") or {}).get("cx", 0.0)),
                    float((d.get("bbox") or {}).get("cy", 0.0)),
                    float((d.get("bbox") or {}).get("w", 0.0)),
                    float((d.get("bbox") or {}).get("h", 0.0)),
                )
            )

            entry = {
                "schema_version": ENTRY_SCHEMA_VERSION,
                "image": image_path,
                "detections": detections,
                "image_size": pp_meta.get("input_size"),
                "image_w": int((pp_meta.get("input_size") or {}).get("width", 0)),
                "image_h": int((pp_meta.get("input_size") or {}).get("height", 0)),
                "orig_w": int((pp_meta.get("orig_size") or {}).get("width", 0)),
                "orig_h": int((pp_meta.get("orig_size") or {}).get("height", 0)),
                "model_input_w": int((pp_meta.get("model_input_size") or {}).get("width", 0)),
                "model_input_h": int((pp_meta.get("model_input_size") or {}).get("height", 0)),
                "preprocess": pp_meta,
                "preproc": pp_meta,
            }
            if intrinsics is not None:
                entry["intrinsics"] = intrinsics
            return entry

        outputs: list[dict] = []
        batch_x: list = []
        batch_meta: list[tuple[str, dict, dict | None]] = []

        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"records[{idx}] must be an object")
            image_path = require_image_key(record.get("image"), where=f"records[{idx}].image")
            x, pp_meta, intrinsics = preprocess(record)
            x = x.to(self.device)
            if self.channels_last and int(getattr(x, "ndim", 0)) == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            batch_x.append(x)
            batch_meta.append((image_path, pp_meta, intrinsics))

            if len(batch_x) >= batch_size:
                x_cat = torch.cat(batch_x, dim=0)
                if self.channels_last and int(getattr(x_cat, "ndim", 0)) == 4:
                    x_cat = x_cat.contiguous(memory_format=torch.channels_last)
                out = _run_model(x_cat)
                for i, (p, meta, intr) in enumerate(batch_meta):
                    outputs.append(_decode_single(idx=i, out=out, image_path=p, pp_meta=meta, intrinsics=intr))
                batch_x = []
                batch_meta = []

        if batch_x:
            x_cat = torch.cat(batch_x, dim=0)
            if self.channels_last and int(getattr(x_cat, "ndim", 0)) == 4:
                x_cat = x_cat.contiguous(memory_format=torch.channels_last)
            out = _run_model(x_cat)
            for i, (p, meta, intr) in enumerate(batch_meta):
                outputs.append(_decode_single(idx=i, out=out, image_path=p, pp_meta=meta, intrinsics=intr))

        return outputs
