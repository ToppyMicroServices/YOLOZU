"""Model adapter interface and built-in adapter implementations.

Adapters provide a uniform ``predict(records) -> list[dict]`` interface so
that the evaluation, TTA, TTT, and distillation pipelines stay backend-
agnostic.

Public adapters
---------------
ModelAdapter         -- abstract base.
DummyAdapter         -- returns empty detections (smoke-testing).
PrecomputedAdapter   -- loads predictions from a JSON file.
RTDETRPoseAdapter    -- runs the RT-DETR pose scaffold (optional ``torch``).
"""

from __future__ import annotations

from contextlib import nullcontext

__all__ = [
    "ModelAdapter",
    "DummyAdapter",
    "PrecomputedAdapter",
    "RTDETRPoseAdapter",
]


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
            {"image": record["image"], "detections": []} for record in records
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
            outputs.append({"image": image_key, "detections": dets if dets is not None else []})
        return outputs


class RTDETRPoseAdapter(ModelAdapter):
    """Adapter that runs the RT-DETR pose scaffold (optional dependency).

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
        lora_r: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        lora_target: str = "head",
        lora_freeze_base: bool = False,
        lora_train_bias: str = "none",
        compile_model: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        amp: str = "off",
        channels_last: bool = False,
        use_inference_mode: bool = True,
        init_seed: int | None = None,
        repro_policy: str = "relaxed",
    ):
        self.config_path = str(config_path)
        self.checkpoint_path = checkpoint_path
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

        from PIL import Image
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
            except Exception:
                # Some runtime combos may not expose strict deterministic toggles.
                pass
            if hasattr(torch.backends, "cudnn"):
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass

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
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                model_state = model.state_dict()
                filtered = {
                    k: v
                    for k, v in state.items()
                    if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
                }
                model.load_state_dict(filtered, strict=False)
            else:
                model.load_state_dict(state, strict=False)

        model.to(self.device)

        # Optional torch.compile for inference speedup (PyTorch 2.x).
        if self.compile_model:
            from yolozu.inference.torch_export import compile_for_inference

            model = compile_for_inference(
                model,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )

        from yolozu.geometry.intrinsics import parse_intrinsics as _parse_intrinsics

        def preprocess(record_or_path):
            if isinstance(record_or_path, dict):
                from yolozu.core.image_keys import require_image_key

                path = require_image_key(record_or_path.get("image"), where="record.image")
            else:
                path = record_or_path
            img = Image.open(path).convert("RGB")
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
                "scale_xy": {"x": float(dst_w) / float(orig_w) if orig_w else None, "y": float(dst_h) / float(orig_h) if orig_h else None},
                "method": "resize",
                "normalize": "0_1",
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
                    return model(x_tensor)

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

            # Prefer foreground classes; treat the last class as background.
            num_classes_fg = int(self._backend.get("num_classes_fg") or (probs.shape[-1] - 1))
            probs_fg = probs[..., :num_classes_fg]
            scores, class_ids = torch.max(probs_fg, dim=-1)
            k = min(self.max_detections, int(scores.shape[0]))
            top_scores, top_idx = torch.topk(scores, k=k)

            detections: list[dict] = []
            for score, q_idx in zip(top_scores.tolist(), top_idx.tolist()):
                if score < self.score_threshold:
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
                    except Exception:
                        pass

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
                "image": image_path,
                "detections": detections,
                "image_size": pp_meta.get("input_size"),
                "preprocess": pp_meta,
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
