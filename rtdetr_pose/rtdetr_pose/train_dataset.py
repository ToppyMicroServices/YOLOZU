"""Dataset and data collation for train_minimal."""

import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from rtdetr_pose.dataset import extract_full_gt_targets, depth_at_bbox_center
from yolozu.jitter import sample_intrinsics_jitter, sample_extrinsics_jitter

from rtdetr_pose.train_utils import (
    workspace_root,
    _rotation_matrix_from_rpy,
    apply_hsv_jitter,
    apply_grayscale,
    apply_gaussian_blur,
    generate_block_mask,
)


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        *,
        num_queries=300,
        num_classes=80,
        num_keypoints=0,
        keypoint_flip_pairs=(),
        image_size=640,
        seed=0,
        use_matcher=False,
        synthetic_pose=False,
        z_from_dobj=False,
        load_aux=False,
        depth_mode="none",
        depth_unit="unspecified",
        depth_scale=1.0,
        real_images=False,
        multiscale=False,
        scale_min=1.0,
        scale_max=1.0,
        hflip_prob=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        hsv_prob=1.0,
        gray_prob=0.0,
        gaussian_noise_std=0.0,
        gaussian_noise_prob=1.0,
        blur_prob=0.0,
        blur_sigma=0.0,
        blur_kernel=3,
        intrinsics_jitter=False,
        jitter_dfx=0.0,
        jitter_dfy=0.0,
        jitter_dcx=0.0,
        jitter_dcy=0.0,
        sim_jitter=False,
        sim_jitter_profile=None,
        sim_jitter_extrinsics=False,
        extrinsics_jitter=False,
        jitter_dx=0.0,
        jitter_dy=0.0,
        jitter_dz=0.0,
        jitter_droll=0.0,
        jitter_dpitch=0.0,
        jitter_dyaw=0.0,
        mim_mask_prob=0.0,
        mim_mask_size=16,
        mim_mask_value=0.0,
        derpp_enabled=False,
        derpp_teacher_key="derpp_teacher",
        derpp_keys=(),
    ):
        self.records = records
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.num_keypoints = int(num_keypoints)
        normalized_pairs = []
        seen_pairs = set()
        for pair in keypoint_flip_pairs or ():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                continue
            if a == b:
                continue
            if not (0 <= a < self.num_keypoints and 0 <= b < self.num_keypoints):
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            normalized_pairs.append((int(key[0]), int(key[1])))
        self.keypoint_flip_pairs = tuple(normalized_pairs)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.use_matcher = bool(use_matcher)
        self.synthetic_pose = bool(synthetic_pose)
        self.z_from_dobj = bool(z_from_dobj)
        self.load_aux = bool(load_aux)
        self.depth_mode = str(depth_mode or "none").strip().lower()
        self.depth_unit = str(depth_unit or "unspecified").strip().lower()
        self.depth_scale = float(depth_scale)
        self.mim_mask_prob = float(mim_mask_prob)
        self.mim_mask_size = int(mim_mask_size)
        self.mim_mask_value = float(mim_mask_value)
        self.real_images = bool(real_images)
        self.multiscale = bool(multiscale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hflip_prob = float(hflip_prob)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.hsv_prob = float(hsv_prob)
        self.gray_prob = float(gray_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.blur_prob = float(blur_prob)
        self.blur_sigma = float(blur_sigma)
        self.blur_kernel = int(blur_kernel)
        self.intrinsics_jitter = bool(intrinsics_jitter)
        self.jitter_dfx = float(jitter_dfx)
        self.jitter_dfy = float(jitter_dfy)
        self.jitter_dcx = float(jitter_dcx)
        self.jitter_dcy = float(jitter_dcy)
        self.sim_jitter = bool(sim_jitter)
        self.sim_jitter_profile = sim_jitter_profile
        self.sim_jitter_extrinsics = bool(sim_jitter_extrinsics)
        self.extrinsics_jitter = bool(extrinsics_jitter)
        self.jitter_dx = float(jitter_dx)
        self.jitter_dy = float(jitter_dy)
        self.jitter_dz = float(jitter_dz)
        self.jitter_droll = float(jitter_droll)
        self.jitter_dpitch = float(jitter_dpitch)
        self.jitter_dyaw = float(jitter_dyaw)
        self.derpp_enabled = bool(derpp_enabled)
        self.derpp_teacher_key = str(derpp_teacher_key) if derpp_teacher_key is not None else ""
        self.derpp_keys = tuple(str(k) for k in (derpp_keys or ()))
        self._derpp_teacher_warned: set[str] = set()

    def _load_derpp_teacher(self, value: Any) -> dict[str, "torch.Tensor"] | None:
        if not self.derpp_enabled or not self.derpp_teacher_key:
            return None
        if value is None:
            return None

        keys = tuple(k for k in self.derpp_keys if k)
        if not keys:
            return None

        def _maybe_squeeze(t: "torch.Tensor") -> "torch.Tensor":
            if t.ndim >= 1 and int(t.shape[0]) == 1:
                return t.squeeze(0)
            return t

        def _warn_once(msg: str) -> None:
            warned = getattr(self, "_derpp_teacher_warned", None)
            if not isinstance(warned, set):
                warned = set()
                setattr(self, "_derpp_teacher_warned", warned)
            if msg in warned:
                return
            warned.add(msg)
            print(msg, file=sys.stderr)

        if isinstance(value, dict):
            out: dict[str, torch.Tensor] = {}
            for k in keys:
                v = value.get(k)
                if isinstance(v, torch.Tensor):
                    out[k] = _maybe_squeeze(v.detach().to(dtype=torch.float32, device="cpu").clone())
                elif isinstance(v, (list, tuple)):
                    try:
                        out[k] = _maybe_squeeze(torch.tensor(v, dtype=torch.float32))
                    except Exception:
                        continue
            return out or None

        if isinstance(value, (str, Path)) and str(value):
            path = Path(value)
            if not path.is_absolute():
                path = (workspace_root / path).resolve()
                if not path.exists():
                    path = (workspace_root.parent / Path(value)).resolve()
            if not path.exists():
                return None
            if path.suffix.lower() == ".json":
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                return None
            if path.suffix.lower() in (".pt", ".pth"):
                try:
                    loaded = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                if isinstance(loaded, torch.Tensor):
                    if len(keys) != 1:
                        return None
                    return {keys[0]: _maybe_squeeze(loaded.detach().to(dtype=torch.float32, device="cpu").clone())}
                if isinstance(loaded, (list, tuple)):
                    if len(keys) != 1:
                        return None
                    try:
                        return {keys[0]: _maybe_squeeze(torch.tensor(loaded, dtype=torch.float32))}
                    except Exception:
                        return None
                return None
            if path.suffix.lower() == ".safetensors":
                try:
                    from safetensors.torch import load_file  # type: ignore
                except Exception:
                    _warn_once(
                        "DERPP teacher: safetensors is not installed; cannot load "
                        f"{path}. Install it with `python -m pip install safetensors`."
                    )
                    return None
                try:
                    loaded = load_file(str(path), device="cpu")
                except Exception:
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except Exception:
                    return None
                try:
                    loaded = np.load(path, allow_pickle=False)
                except Exception:
                    return None
                out: dict[str, torch.Tensor] = {}
                if hasattr(loaded, "files"):
                    for k in keys:
                        if k in loaded.files:
                            try:
                                out[k] = _maybe_squeeze(torch.from_numpy(loaded[k]).to(dtype=torch.float32))
                            except Exception:
                                continue
                else:
                    if len(keys) == 1:
                        try:
                            out[keys[0]] = _maybe_squeeze(torch.from_numpy(loaded).to(dtype=torch.float32))
                        except Exception:
                            return None
                return out or None
        return None

    def _load_rgb_image_tensor(self, image_path: Path, target_size: int) -> "torch.Tensor | None":
        if not image_path.exists():
            return None
        try:
            from PIL import Image
        except Exception as exc:
            raise SystemExit(
                "Pillow is required for --real-images. Install it (e.g. pip install Pillow) or omit --real-images."
            ) from exc
        try:
            import numpy as np
        except Exception as exc:
            raise SystemExit(
                "NumPy is required for --real-images. Install it (e.g. pip install numpy) or omit --real-images."
            ) from exc

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        if target_size > 0:
            img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        arr = arr / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_2d(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value
        if not self.load_aux:
            return None
        if isinstance(value, str):
            path = Path(value)
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                try:
                    import numpy as np
                    from PIL import Image
                except ImportError:
                    return None
                try:
                    loaded = np.array(Image.open(path), copy=True)
                except (OSError, ValueError):
                    return None
                if loaded.ndim == 2:
                    return loaded
                if loaded.ndim == 3 and loaded.shape[2] >= 1:
                    first = loaded[..., 0]
                    if all(np.array_equal(first, loaded[..., channel]) for channel in range(1, loaded.shape[2])):
                        return first
                return None
            if path.suffix.lower() == ".json":
                try:
                    return json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except ImportError:
                    return None
                try:
                    loaded = np.load(path)
                except OSError:
                    return None
                if hasattr(loaded, "files"):
                    if not loaded.files:
                        return None
                    return loaded[loaded.files[0]]
                return loaded
        return None

    @staticmethod
    def _raise_missing_synthetic_pose_intrinsics():
        raise RuntimeError("synthetic_pose requires K_gt")

    def _load_depth_sidecar(self, value, target_size: int, flip: bool):
        if self.depth_mode == "none" or value is None:
            return None, False

        depth = None
        if isinstance(value, torch.Tensor):
            depth = value.detach().to(dtype=torch.float32, device="cpu")
        elif isinstance(value, (list, tuple)):
            try:
                depth = torch.tensor(value, dtype=torch.float32)
            except Exception:
                depth = None
        elif isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                cand = (workspace_root / path).resolve()
                if cand.exists():
                    path = cand
                else:
                    cand2 = (workspace_root.parent / path).resolve()
                    if cand2.exists():
                        path = cand2
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".json":
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                        depth = torch.tensor(loaded, dtype=torch.float32)
                    except Exception:
                        depth = None
                elif suffix in (".npy", ".npz"):
                    try:
                        import numpy as np

                        loaded = np.load(path, allow_pickle=False)
                        if hasattr(loaded, "files"):
                            if loaded.files:
                                depth = torch.from_numpy(loaded[loaded.files[0]]).to(dtype=torch.float32)
                        else:
                            depth = torch.from_numpy(loaded).to(dtype=torch.float32)
                    except Exception:
                        depth = None
                else:
                    try:
                        from PIL import Image
                        import numpy as np

                        arr = np.asarray(Image.open(path), dtype=np.float32)
                        depth = torch.from_numpy(arr).to(dtype=torch.float32)
                    except Exception:
                        depth = None

        if depth is None:
            return None, False
        if depth.ndim == 3:
            if int(depth.shape[0]) in (1, 3, 4) and int(depth.shape[1]) > 4 and int(depth.shape[2]) > 4:
                depth = depth[0]
            else:
                depth = depth[..., 0]
        if depth.ndim != 2:
            return None, False

        depth = depth.contiguous()
        if target_size > 0 and (int(depth.shape[0]) != int(target_size) or int(depth.shape[1]) != int(target_size)):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(int(target_size), int(target_size)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        if flip:
            depth = torch.flip(depth, dims=(1,))

        depth = depth * float(self.depth_scale)
        finite_any = bool(torch.isfinite(depth).any().item())
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        positive_any = bool((depth > 0).any().item())
        valid = bool(finite_any and positive_any)
        return depth.unsqueeze(0), valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))

        base_size = max(1, int(self.image_size))
        scale = 1.0
        if self.multiscale:
            lo = min(self.scale_min, self.scale_max)
            hi = max(self.scale_min, self.scale_max)
            if hi > 0 and lo > 0:
                scale = float(torch.rand((), generator=gen) * (hi - lo) + lo)
        target_size = max(1, int(round(base_size * scale)))
        scale_factor = float(target_size) / float(base_size)

        flip = False
        if self.hflip_prob > 0:
            flip = bool(torch.rand((), generator=gen) < float(self.hflip_prob))

        # Keep the reference trainer runnable with minimal deps by falling back to
        # synthetic images unless real JPEG loading is explicitly enabled.
        image = None
        if self.real_images:
            image_path_raw = record.get("image_path")
            if image_path_raw:
                image = self._load_rgb_image_tensor(Path(str(image_path_raw)), target_size)
        if image is None:
            image = torch.rand(3, target_size, target_size, generator=gen)

        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if flip:
            image = torch.flip(image, dims=(2,))

        if self.hsv_prob > 0.0 and (self.hsv_h > 0.0 or self.hsv_s > 0.0 or self.hsv_v > 0.0):
            if bool(torch.rand((), generator=gen) < float(self.hsv_prob)):
                image = apply_hsv_jitter(
                    image,
                    generator=gen,
                    hgain=float(self.hsv_h),
                    sgain=float(self.hsv_s),
                    vgain=float(self.hsv_v),
                )

        if self.gray_prob > 0.0:
            if bool(torch.rand((), generator=gen) < float(self.gray_prob)):
                image = apply_grayscale(image)

        if self.blur_prob > 0.0 and self.blur_sigma > 0.0:
            if bool(torch.rand((), generator=gen) < float(self.blur_prob)):
                sigma = float(torch.rand((), generator=gen) * float(self.blur_sigma))
                sigma = max(sigma, 1e-3)
                image = apply_gaussian_blur(image, sigma=sigma, kernel_size=int(self.blur_kernel))

        if self.gaussian_noise_std > 0.0:
            prob = float(self.gaussian_noise_prob)
            if prob >= 1.0 or bool(torch.rand((), generator=gen) < prob):
                noise = torch.randn(image.shape, generator=gen, device=image.device, dtype=image.dtype)
                image = image + noise * float(self.gaussian_noise_std)
                image = torch.clamp(image, 0.0, 1.0)

        depth_tensor = None
        depth_valid = False
        if self.depth_mode != "none":
            depth_source = record.get("depth_path")
            if depth_source is None:
                depth_source = record.get("depth")
            depth_tensor, depth_valid = self._load_depth_sidecar(depth_source, target_size=target_size, flip=flip)

        image_raw = None
        mim_mask_ratio = None
        if self.mim_mask_prob and float(self.mim_mask_prob) > 0 and int(self.mim_mask_size) > 0:
            image_raw = image.clone()
            mask = generate_block_mask(
                target_size,
                target_size,
                patch_size=int(self.mim_mask_size),
                mask_prob=float(self.mim_mask_prob),
                generator=gen,
            )
            mim_mask_ratio = mask.float().mean()
            if bool(mask.any()):
                image = image.masked_fill(mask.unsqueeze(0).expand_as(image), float(self.mim_mask_value))

        instances = record.get("labels") or []
        if not instances:
            mask_value = record.get("mask")
            if mask_value is None:
                mask_value = record.get("M")
            if mask_value is None:
                mask_value = record.get("mask_path")
            mask_format = str(record.get("mask_format") or "")
            if not mask_format and bool(record.get("mask_instances", False)):
                mask_format = "instance"
            mask_class_id = record.get("mask_class_id")

            # Minimal mask-to-labels support for unit tests and inline-record use.
            if mask_value is not None and isinstance(mask_value, (list, tuple)) and mask_value:
                try:
                    h = len(mask_value)
                    w = len(mask_value[0]) if isinstance(mask_value[0], (list, tuple)) else 0
                except Exception:
                    h = 0
                    w = 0

                if h > 0 and w > 0:
                    unique_vals = set()
                    for row in mask_value:
                        if not isinstance(row, (list, tuple)):
                            continue
                        for v in row:
                            try:
                                unique_vals.add(int(v))
                            except Exception:
                                continue
                    unique_vals.discard(0)

                    derived = []
                    if mask_format.lower() in ("instance", "instances"):
                        class_id = int(mask_class_id) if mask_class_id is not None else 0
                        for inst_id in sorted(unique_vals):
                            x_min = y_min = None
                            x_max = y_max = None
                            for y, row in enumerate(mask_value):
                                if not isinstance(row, (list, tuple)):
                                    continue
                                for x, v in enumerate(row):
                                    try:
                                        if int(v) != int(inst_id):
                                            continue
                                    except Exception:
                                        continue
                                    x_min = x if x_min is None else min(x_min, x)
                                    x_max = x if x_max is None else max(x_max, x)
                                    y_min = y if y_min is None else min(y_min, y)
                                    y_max = y if y_max is None else max(y_max, y)
                            if x_min is None or y_min is None or x_max is None or y_max is None:
                                continue
                            cx = (x_min + x_max + 1) / 2.0 / float(w)
                            cy = (y_min + y_max + 1) / 2.0 / float(h)
                            bw = (x_max - x_min + 1) / float(w)
                            bh = (y_max - y_min + 1) / float(h)
                            derived.append({"class_id": class_id, "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
                    else:
                        # Treat mask values as class ids (semantic mask -> one bbox per class).
                        for class_val in sorted(unique_vals):
                            x_min = y_min = None
                            x_max = y_max = None
                            for y, row in enumerate(mask_value):
                                if not isinstance(row, (list, tuple)):
                                    continue
                                for x, v in enumerate(row):
                                    try:
                                        if int(v) != int(class_val):
                                            continue
                                    except Exception:
                                        continue
                                    x_min = x if x_min is None else min(x_min, x)
                                    x_max = x if x_max is None else max(x_max, x)
                                    y_min = y if y_min is None else min(y_min, y)
                                    y_max = y if y_max is None else max(y_max, y)
                            if x_min is None or y_min is None or x_max is None or y_max is None:
                                continue
                            cx = (x_min + x_max + 1) / 2.0 / float(w)
                            cy = (y_min + y_max + 1) / 2.0 / float(h)
                            bw = (x_max - x_min + 1) / float(w)
                            bh = (y_max - y_min + 1) / float(h)
                            derived.append(
                                {"class_id": int(class_val), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}}
                            )

                    if derived:
                        instances = derived
        if self.use_matcher:
            full = extract_full_gt_targets(record, num_instances=len(instances))
            gt_labels = []
            gt_bbox = []
            gt_z = []
            gt_z_mask = []
            gt_R = []
            gt_R_mask = []
            gt_t = []
            gt_t_mask = []
            gt_offsets = []
            gt_offsets_mask = []
            gt_keypoints = []
            gt_keypoints_mask = []
            gt_M_mask = []
            gt_D_obj_mask = []
            gt_M = []
            gt_D_obj = []

            # We don't decode images, so default HW is the generated tensor size.
            image_hw = torch.tensor([float(target_size), float(target_size)], dtype=torch.float32)

            # Prefer real intrinsics when present; else synthesize if requested.
            K_gt = None
            if full.get("K_gt") is not None:
                K_gt = torch.tensor(full["K_gt"], dtype=torch.float32)
            elif self.synthetic_pose:
                w = float(self.image_size)
                h = float(self.image_size)
                fx = w
                fy = w
                cx = w * 0.5
                cy = h * 0.5
                K_gt = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )

            if K_gt is not None and scale_factor != 1.0:
                K_gt = K_gt.clone()
                K_gt[0, 0] *= scale_factor
                K_gt[1, 1] *= scale_factor
                K_gt[0, 2] *= scale_factor
                K_gt[1, 2] *= scale_factor
            if K_gt is not None and flip:
                K_gt = K_gt.clone()
                K_gt[0, 2] = float(target_size - 1.0) - K_gt[0, 2]
            if K_gt is not None and self.sim_jitter and self.sim_jitter_profile:
                jitter = sample_intrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                K_gt = K_gt.clone()
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + float(jitter.get("dfx", 0.0)))
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + float(jitter.get("dfy", 0.0)))
                K_gt[0, 2] = K_gt[0, 2] + float(jitter.get("dcx", 0.0))
                K_gt[1, 2] = K_gt[1, 2] + float(jitter.get("dcy", 0.0))
            elif K_gt is not None and self.intrinsics_jitter:
                K_gt = K_gt.clone()
                dfx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfx)
                dfy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfy)
                dcx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcx)
                dcy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcy)
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + dfx)
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + dfy)
                K_gt[0, 2] = K_gt[0, 2] + dcx
                K_gt[1, 2] = K_gt[1, 2] + dcy

            for inst_i, inst in enumerate(instances):
                class_id = int(inst.get("class_id", -1))
                if not (0 <= class_id < self.num_classes):
                    continue
                bb = inst.get("bbox") or {}
                cx = float(bb.get("cx", 0.0))
                cy = float(bb.get("cy", 0.0))
                w = float(bb.get("w", 0.0))
                h = float(bb.get("h", 0.0))
                if flip:
                    cx = 1.0 - cx
                gt_labels.append(class_id)
                gt_bbox.append([cx, cy, w, h])

                if int(self.num_keypoints) > 0:
                    k_count = int(self.num_keypoints)
                    kps = inst.get("keypoints")
                    kp_xy = [[0.0, 0.0] for _ in range(k_count)]
                    kp_mask = [False for _ in range(k_count)]
                    if isinstance(kps, list) and kps:
                        for ki, kp in enumerate(kps[:k_count]):
                            if not isinstance(kp, dict):
                                continue
                            try:
                                x = float(kp.get("x", 0.0))
                                y = float(kp.get("y", 0.0))
                            except Exception:
                                continue
                            v = kp.get("v", 0.0)
                            try:
                                v_i = int(float(v))
                            except Exception:
                                v_i = 0
                            if flip:
                                x = 1.0 - x
                            kp_xy[ki] = [x, y]
                            kp_mask[ki] = v_i > 0
                    if flip and self.keypoint_flip_pairs:
                        for a, b in self.keypoint_flip_pairs:
                            kp_xy[a], kp_xy[b] = kp_xy[b], kp_xy[a]
                            kp_mask[a], kp_mask[b] = kp_mask[b], kp_mask[a]
                    gt_keypoints.append(kp_xy)
                    gt_keypoints_mask.append(kp_mask)

                # Real (t/R/offsets) if present, otherwise optional synthetic fallback.
                t_i = full.get("t_gt", [None])[inst_i] if full.get("t_gt") is not None else None
                r_i = full.get("R_gt", [None])[inst_i] if full.get("R_gt") is not None else None
                off_i = full.get("offsets_gt", [None])[inst_i] if full.get("offsets_gt") is not None else None

                m_i = full.get("M", [None])[inst_i] if full.get("M") is not None else None
                d_i = full.get("D_obj", [None])[inst_i] if full.get("D_obj") is not None else None
                m_loaded = self._load_2d(m_i)
                d_loaded = self._load_2d(d_i)
                gt_M_mask.append(bool(full.get("M_mask", [False])[inst_i]) if full.get("M_mask") else False)
                gt_D_obj_mask.append(
                    bool(full.get("D_obj_mask", [False])[inst_i]) if full.get("D_obj_mask") else False
                )
                gt_M.append(m_loaded)
                gt_D_obj.append(d_loaded)

                # z/t
                if t_i is not None:
                    t_val = [float(v) for v in t_i]
                    z_val = float(t_val[2])
                    gt_t.append(t_val)
                    gt_t_mask.append(True)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                elif self.synthetic_pose:
                    z_val = float(torch.rand((), generator=gen) * 0.9 + 0.1)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                    if K_gt is None:
                        self._raise_missing_synthetic_pose_intrinsics()
                    cx_n = float(bb.get("cx", 0.0))
                    cy_n = float(bb.get("cy", 0.0))
                    u = cx_n * float(image_hw[1])
                    v = cy_n * float(image_hw[0])
                    x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * z_val
                    y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * z_val
                    gt_t.append([x, y, z_val])
                    gt_t_mask.append(True)
                else:
                    # Optional: derive z (and optionally t) from D_obj at bbox center.
                    z_val = None
                    if self.z_from_dobj and d_loaded is not None:
                        z_val = depth_at_bbox_center(d_loaded, bb, mask=m_loaded)
                    if z_val is not None:
                        gt_z.append(float(z_val))
                        gt_z_mask.append(True)
                        if K_gt is not None:
                            cx_n = float(bb.get("cx", 0.0))
                            cy_n = float(bb.get("cy", 0.0))
                            u = cx_n * float(image_hw[1])
                            v = cy_n * float(image_hw[0])
                            x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * float(z_val)
                            y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * float(z_val)
                            gt_t.append([x, y, float(z_val)])
                            gt_t_mask.append(True)
                        else:
                            gt_t.append([0.0, 0.0, float(z_val)])
                            gt_t_mask.append(False)
                    else:
                        gt_z.append(0.0)
                        gt_z_mask.append(False)
                        gt_t.append([0.0, 0.0, 0.0])
                        gt_t_mask.append(False)

                # R
                if r_i is not None:
                    gt_R.append(torch.tensor(r_i, dtype=torch.float32))
                    gt_R_mask.append(True)
                elif self.synthetic_pose:
                    a = torch.randn(3, 3, generator=gen)
                    q, _ = torch.linalg.qr(a)
                    if torch.det(q) < 0:
                        q[:, 0] = -q[:, 0]
                    gt_R.append(q)
                    gt_R_mask.append(True)
                else:
                    gt_R.append(torch.eye(3, dtype=torch.float32))
                    gt_R_mask.append(False)

                # offsets
                if off_i is not None:
                    du = float(off_i[0]) * scale_factor
                    dv = float(off_i[1]) * scale_factor
                    if flip:
                        du = -du
                    gt_offsets.append([du, dv])
                    gt_offsets_mask.append(True)
                elif self.synthetic_pose:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(True)
                else:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(False)

            if (self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics) or self.extrinsics_jitter:
                if self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics:
                    jitter = sample_extrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                    dx = float(jitter.get("dx", 0.0))
                    dy = float(jitter.get("dy", 0.0))
                    dz = float(jitter.get("dz", 0.0))
                    droll = float(jitter.get("droll", 0.0))
                    dpitch = float(jitter.get("dpitch", 0.0))
                    dyaw = float(jitter.get("dyaw", 0.0))
                else:
                    dx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dx)
                    dy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dy)
                    dz = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dz)
                    droll = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_droll)
                    dpitch = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dpitch)
                    dyaw = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dyaw)

                for j in range(len(gt_t)):
                    if gt_t_mask[j]:
                        gt_t[j] = [gt_t[j][0] + dx, gt_t[j][1] + dy, gt_t[j][2] + dz]
                if any(gt_R_mask):
                    r_delta = _rotation_matrix_from_rpy(
                        math.radians(droll),
                        math.radians(dpitch),
                        math.radians(dyaw),
                    )
                    gt_R = [r_delta @ r if mask else r for r, mask in zip(gt_R, gt_R_mask)]

            num_inst = len(gt_labels)
            m_tensor = None
            d_tensor = None
            if self.load_aux and num_inst > 0:
                for item in gt_M:
                    if item is not None:
                        m_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if m_tensor is not None:
                    m_hw = tuple(m_tensor.shape)
                    stacked = []
                    for item in gt_M:
                        if item is None:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != m_hw:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    m_tensor = torch.stack(stacked, dim=0)
                    if tuple(m_tensor.shape[-2:]) != (target_size, target_size):
                        m_tensor = torch.nn.functional.interpolate(
                            m_tensor.unsqueeze(1),
                            size=(target_size, target_size),
                            mode="nearest",
                        ).squeeze(1)
                    if flip:
                        m_tensor = torch.flip(m_tensor, dims=(-1,))

                for item in gt_D_obj:
                    if item is not None:
                        d_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if d_tensor is not None:
                    d_hw = tuple(d_tensor.shape)
                    stacked = []
                    for item in gt_D_obj:
                        if item is None:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != d_hw:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    d_tensor = torch.stack(stacked, dim=0)
                    if tuple(d_tensor.shape[-2:]) != (target_size, target_size):
                        d_tensor = torch.nn.functional.interpolate(
                            d_tensor.unsqueeze(1),
                            size=(target_size, target_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    if flip:
                        d_tensor = torch.flip(d_tensor, dims=(-1,))
            if num_inst == 0:
                gt_labels_t = torch.empty((0,), dtype=torch.long)
                gt_bbox_t = torch.empty((0, 4), dtype=torch.float32)
                gt_z_t = torch.empty((0, 1), dtype=torch.float32)
                gt_z_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_R_t = torch.empty((0, 3, 3), dtype=torch.float32)
                gt_R_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_t_t = torch.empty((0, 3), dtype=torch.float32)
                gt_t_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_offsets_t = torch.empty((0, 2), dtype=torch.float32)
                gt_offsets_mask_t = torch.empty((0,), dtype=torch.bool)
                if int(self.num_keypoints) > 0:
                    k_count = int(self.num_keypoints)
                    gt_keypoints_t = torch.empty((0, k_count, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((0, k_count), dtype=torch.bool)
                else:
                    gt_keypoints_t = torch.empty((0, 0, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((0, 0), dtype=torch.bool)
                gt_M_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_D_obj_mask_t = torch.empty((0,), dtype=torch.bool)
            else:
                gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)
                gt_bbox_t = torch.tensor(gt_bbox, dtype=torch.float32)
                gt_z_t = torch.tensor(gt_z, dtype=torch.float32).unsqueeze(-1)
                gt_z_mask_t = torch.tensor(gt_z_mask, dtype=torch.bool)
                gt_R_t = torch.stack(gt_R, dim=0)
                gt_R_mask_t = torch.tensor(gt_R_mask, dtype=torch.bool)
                gt_t_t = torch.tensor(gt_t, dtype=torch.float32)
                gt_t_mask_t = torch.tensor(gt_t_mask, dtype=torch.bool)
                gt_offsets_t = torch.tensor(gt_offsets, dtype=torch.float32)
                gt_offsets_mask_t = torch.tensor(gt_offsets_mask, dtype=torch.bool)
                if int(self.num_keypoints) > 0:
                    gt_keypoints_t = torch.tensor(gt_keypoints, dtype=torch.float32)
                    gt_keypoints_mask_t = torch.tensor(gt_keypoints_mask, dtype=torch.bool)
                else:
                    gt_keypoints_t = torch.empty((num_inst, 0, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((num_inst, 0), dtype=torch.bool)
                gt_M_mask_t = torch.tensor(gt_M_mask, dtype=torch.bool)
                gt_D_obj_mask_t = torch.tensor(gt_D_obj_mask, dtype=torch.bool)

            targets = {
                "image_path": str(record.get("image_path", "") or ""),
                "gt_labels": gt_labels_t,
                "gt_bbox": gt_bbox_t,
                "gt_z": gt_z_t,
                "gt_z_mask": gt_z_mask_t,
                "gt_R": gt_R_t,
                "gt_R_mask": gt_R_mask_t,
                "gt_t": gt_t_t,
                "gt_t_mask": gt_t_mask_t,
                "gt_offsets": gt_offsets_t,
                "gt_offsets_mask": gt_offsets_mask_t,
                "gt_keypoints": gt_keypoints_t,
                "gt_keypoints_mask": gt_keypoints_mask_t,
                "gt_M_mask": gt_M_mask_t,
                "gt_D_obj_mask": gt_D_obj_mask_t,
                **({"gt_M": m_tensor} if m_tensor is not None else {}),
                **({"gt_D_obj": d_tensor} if d_tensor is not None else {}),
                **({"K_gt": K_gt} if K_gt is not None else {}),
                "image_hw": image_hw,
                "depth_valid": torch.tensor(bool(depth_valid), dtype=torch.bool),
            }
            derpp_teacher = self._load_derpp_teacher(record.get(self.derpp_teacher_key))
            if derpp_teacher is not None:
                targets["derpp_teacher"] = derpp_teacher
            out = {"image": image, "targets": targets}
            if depth_tensor is not None:
                out["depth"] = depth_tensor
            out["depth_valid"] = bool(depth_valid)
            if image_raw is not None and mim_mask_ratio is not None:
                out["image_raw"] = image_raw
                out["mim_mask_ratio"] = float(mim_mask_ratio)
            return out

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)
        bbox = torch.zeros((self.num_queries, 4), dtype=torch.float32)
        for qi, inst in enumerate(instances[: self.num_queries]):
            class_id = int(inst.get("class_id", -1))
            if 0 <= class_id < self.num_classes:
                labels[qi] = class_id
            bb = inst.get("bbox") or {}
            cx = float(bb.get("cx", 0.0))
            if flip:
                cx = 1.0 - cx
            bbox[qi, 0] = cx
            bbox[qi, 1] = float(bb.get("cy", 0.0))
            bbox[qi, 2] = float(bb.get("w", 0.0))
            bbox[qi, 3] = float(bb.get("h", 0.0))

        targets = {"labels": labels, "bbox": bbox}
        targets["image_path"] = str(record.get("image_path", "") or "")
        derpp_teacher = self._load_derpp_teacher(record.get(self.derpp_teacher_key))
        if derpp_teacher is not None:
            targets["derpp_teacher"] = derpp_teacher
        out = {"image": image, "targets": targets}
        if depth_tensor is not None:
            out["depth"] = depth_tensor
        if self.depth_mode != "none":
            out["depth_valid"] = bool(depth_valid)
        if image_raw is not None and mim_mask_ratio is not None:
            out["image_raw"] = image_raw
            out["mim_mask_ratio"] = float(mim_mask_ratio)
        return out


def _pad_field(targets, key, max_len, *, pad_value=0.0, dtype=None):
    if max_len == 0:
        return None
    tail = None
    for tgt in targets:
        value = tgt.get(key)
        if isinstance(value, torch.Tensor):
            if dtype is None:
                dtype = value.dtype
            tail = value.shape[1:]
            break
    if dtype is None:
        return None
    if tail is None:
        tail = ()

    rows = []
    for tgt in targets:
        value = tgt.get(key)
        if not isinstance(value, torch.Tensor):
            value = torch.empty((0, *tail), dtype=dtype)
        pad_len = max_len - value.shape[0]
        if pad_len < 0:
            _raise_padded_field_overflow(key, max_len, value.shape[0])
        if pad_len == 0:
            padded = value
        else:
            pad = torch.full((pad_len, *tail), pad_value, dtype=dtype)
            padded = torch.cat([value, pad], dim=0)
        rows.append(padded)
    return torch.stack(rows, dim=0)


def _raise_padded_field_overflow(key, max_len, value_len):
    raise ValueError(f"{key} has more instances than max_len ({value_len} > {max_len})")


def collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]

    extra: dict[str, torch.Tensor] = {}
    if any("image_raw" in item for item in batch):
        raws = []
        for item in batch:
            raw = item.get("image_raw")
            if isinstance(raw, torch.Tensor):
                raws.append(raw)
            else:
                raws.append(item["image"])
        extra["image_raw"] = torch.stack(raws, dim=0)
    if any("mim_mask_ratio" in item for item in batch):
        ratios = []
        for item in batch:
            try:
                ratios.append(float(item.get("mim_mask_ratio", 0.0)))
            except Exception:
                ratios.append(0.0)
        extra["mim_mask_ratio"] = torch.tensor(ratios, dtype=torch.float32)
    if any(("depth" in item) or ("depth_valid" in item) for item in batch):
        depth_rows = []
        depth_valid_rows = []
        for item in batch:
            depth_item = item.get("depth")
            if isinstance(depth_item, torch.Tensor) and depth_item.ndim == 3 and int(depth_item.shape[0]) == 1:
                depth_rows.append(depth_item.to(dtype=torch.float32))
                try:
                    depth_valid_rows.append(bool(item.get("depth_valid", True)))
                except Exception:
                    depth_valid_rows.append(True)
            else:
                depth_rows.append(torch.zeros((1, int(images.shape[-2]), int(images.shape[-1])), dtype=torch.float32))
                depth_valid_rows.append(False)
        extra["depth"] = torch.stack(depth_rows, dim=0)
        extra["depth_valid"] = torch.tensor(depth_valid_rows, dtype=torch.bool)

    if not targets or "gt_labels" not in targets[0]:
        return images, targets

    counts = torch.tensor(
        [int(tgt.get("gt_labels").shape[0]) if isinstance(tgt.get("gt_labels"), torch.Tensor) else 0 for tgt in targets],
        dtype=torch.long,
    )
    max_len = int(counts.max().item()) if counts.numel() else 0
    if max_len == 0:
        padded = {
            "gt_count": counts,
            "gt_mask": torch.zeros((len(targets), 0), dtype=torch.bool),
        }
        out = {"per_sample": targets, "padded": padded, **extra}
        return images, out

    padded = {
        "gt_count": counts,
        "gt_mask": (torch.arange(max_len).unsqueeze(0) < counts.unsqueeze(1)),
        "gt_labels": _pad_field(targets, "gt_labels", max_len, pad_value=-1, dtype=torch.long),
        "gt_bbox": _pad_field(targets, "gt_bbox", max_len, pad_value=0.0, dtype=torch.float32),
    }

    optional_fields = [
        ("gt_z", 0.0, torch.float32),
        ("gt_z_mask", False, torch.bool),
        ("gt_R", 0.0, torch.float32),
        ("gt_R_mask", False, torch.bool),
        ("gt_t", 0.0, torch.float32),
        ("gt_t_mask", False, torch.bool),
        ("gt_offsets", 0.0, torch.float32),
        ("gt_offsets_mask", False, torch.bool),
        ("gt_keypoints", 0.0, torch.float32),
        ("gt_keypoints_mask", False, torch.bool),
        ("gt_M_mask", False, torch.bool),
        ("gt_D_obj_mask", False, torch.bool),
        ("gt_M", 0.0, torch.float32),
        ("gt_D_obj", 0.0, torch.float32),
    ]
    for key, pad_value, dtype in optional_fields:
        if any(key in tgt for tgt in targets):
            padded[key] = _pad_field(targets, key, max_len, pad_value=pad_value, dtype=dtype)

    if any("depth_valid" in tgt for tgt in targets):
        depth_valid_list = []
        for tgt in targets:
            value = tgt.get("depth_valid")
            if isinstance(value, torch.Tensor):
                depth_valid_list.append(bool(value.to(dtype=torch.bool).item()))
            else:
                depth_valid_list.append(False)
        padded["depth_valid"] = torch.tensor(depth_valid_list, dtype=torch.bool)

    out = {"per_sample": targets, "padded": padded, **extra}
    return images, out
