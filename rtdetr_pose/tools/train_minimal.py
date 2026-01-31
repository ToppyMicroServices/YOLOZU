import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required; install requirements-test.txt") from exc

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.dataset import extract_full_gt_targets, depth_at_bbox_center
from rtdetr_pose.losses import Losses
from rtdetr_pose.training import build_query_aligned_targets
from rtdetr_pose.model import RTDETRPose


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        num_queries,
        num_classes,
        image_size,
        seed,
        use_matcher,
        synthetic_pose,
        z_from_dobj,
        load_aux,
    ):
        self.records = records
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.use_matcher = bool(use_matcher)
        self.synthetic_pose = bool(synthetic_pose)
        self.z_from_dobj = bool(z_from_dobj)
        self.load_aux = bool(load_aux)

    def _load_2d(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value
        if not self.load_aux:
            return None
        if isinstance(value, str):
            path = Path(value)
            if path.suffix.lower() == ".json":
                import json

                try:
                    return json.loads(path.read_text())
                except Exception:
                    return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except Exception:
                    return None
                try:
                    loaded = np.load(path)
                except Exception:
                    return None
                if hasattr(loaded, "files"):
                    if not loaded.files:
                        return None
                    return loaded[loaded.files[0]]
                return loaded
        return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # We intentionally do NOT decode JPEGs here (keeps deps minimal).
        # This is a training-loop scaffold: it exercises model/loss/optimizer plumbing.
        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))
        image = torch.rand(3, self.image_size, self.image_size, generator=gen)

        instances = record.get("labels") or []
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
            gt_M_mask = []
            gt_D_obj_mask = []

            # We don't decode images, so default HW is the generated tensor size.
            image_hw = torch.tensor([float(self.image_size), float(self.image_size)], dtype=torch.float32)

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

            for inst_i, inst in enumerate(instances):
                class_id = int(inst.get("class_id", -1))
                if not (0 <= class_id < self.num_classes):
                    continue
                bb = inst.get("bbox") or {}
                gt_labels.append(class_id)
                gt_bbox.append(
                    [
                        float(bb.get("cx", 0.0)),
                        float(bb.get("cy", 0.0)),
                        float(bb.get("w", 0.0)),
                        float(bb.get("h", 0.0)),
                    ]
                )

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
                        raise RuntimeError("synthetic_pose requires K_gt")
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
                    gt_offsets.append([float(off_i[0]), float(off_i[1])])
                    gt_offsets_mask.append(True)
                elif self.synthetic_pose:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(True)
                else:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(False)
            return {
                "image": image,
                "targets": {
                    "gt_labels": torch.tensor(gt_labels, dtype=torch.long),
                    "gt_bbox": torch.tensor(gt_bbox, dtype=torch.float32),
                    "gt_z": torch.tensor(gt_z, dtype=torch.float32).unsqueeze(-1),
                    "gt_z_mask": torch.tensor(gt_z_mask, dtype=torch.bool),
                    "gt_R": torch.stack(gt_R, dim=0),
                    "gt_R_mask": torch.tensor(gt_R_mask, dtype=torch.bool),
                    "gt_t": torch.tensor(gt_t, dtype=torch.float32),
                    "gt_t_mask": torch.tensor(gt_t_mask, dtype=torch.bool),
                    "gt_offsets": torch.tensor(gt_offsets, dtype=torch.float32),
                    "gt_offsets_mask": torch.tensor(gt_offsets_mask, dtype=torch.bool),
                    "gt_M_mask": torch.tensor(gt_M_mask, dtype=torch.bool),
                    "gt_D_obj_mask": torch.tensor(gt_D_obj_mask, dtype=torch.bool),
                    **({"K_gt": K_gt} if K_gt is not None else {}),
                    "image_hw": image_hw,
                },
            }

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)
        bbox = torch.zeros((self.num_queries, 4), dtype=torch.float32)
        for qi, inst in enumerate(instances[: self.num_queries]):
            class_id = int(inst.get("class_id", -1))
            if 0 <= class_id < self.num_classes:
                labels[qi] = class_id
            bb = inst.get("bbox") or {}
            bbox[qi, 0] = float(bb.get("cx", 0.0))
            bbox[qi, 1] = float(bb.get("cy", 0.0))
            bbox[qi, 2] = float(bb.get("w", 0.0))
            bbox[qi, 3] = float(bb.get("h", 0.0))

        return {"image": image, "targets": {"labels": labels, "bbox": bbox}}


def collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    return images, targets




def main():
    parser = argparse.ArgumentParser(description="Minimal RTDETRPose training scaffold (CPU).")
    parser.add_argument("--dataset-root", type=str, default="", help="Path to data/coco128")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=30, help="Cap steps per epoch")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N steps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Shuffle dataset each epoch (default: true).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic data order via DataLoader generator (seeded).",
    )
    parser.add_argument("--use-matcher", action="store_true", help="Use Hungarian matching")
    parser.add_argument("--cost-cls", type=float, default=1.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-z", type=float, default=0.0, help="Optional matching cost for depth")
    parser.add_argument(
        "--cost-rot",
        type=float,
        default=0.0,
        help="Optional matching cost for rotation (geodesic angle)",
    )
    parser.add_argument(
        "--synthetic-pose",
        action="store_true",
        help="Generate synthetic z/R GT per instance (scaffold only)",
    )
    parser.add_argument(
        "--z-from-dobj",
        action="store_true",
        help="When GT t is missing, derive z (and optionally t if K is available) from D_obj at bbox center",
    )
    parser.add_argument(
        "--load-aux",
        action="store_true",
        help="Allow loading mask/depth arrays from paths (.json/.npy) for z-from-dobj; default keeps lazy paths",
    )
    parser.add_argument(
        "--cost-t",
        type=float,
        default=0.0,
        help="Optional matching cost for translation recovered from (bbox, offsets, z, K')",
    )
    parser.add_argument(
        "--debug-losses",
        action="store_true",
        help="Print loss dict breakdown on step 1",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = repo_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = repo_root.parent / "data" / "coco128"

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if not records:
        raise SystemExit(
            f"No records found under {dataset_root}. "
            "Fetch coco128 first: bash tools/fetch_coco128.sh"
        )

    stats = {
        "mask": 0,
        "depth": 0,
        "pose": 0,
        "intrinsics": 0,
        "cad_points": 0,
    }
    for rec in records:
        if rec.get("mask_path") is not None:
            stats["mask"] += 1
        if rec.get("depth_path") is not None:
            stats["depth"] += 1
        if rec.get("R_gt") is not None or rec.get("t_gt") is not None or rec.get("pose") is not None:
            stats["pose"] += 1
        if rec.get("K_gt") is not None or rec.get("intrinsics") is not None:
            stats["intrinsics"] += 1
        if rec.get("cad_points") is not None:
            stats["cad_points"] += 1
    print(
        "dataset_stats "
        + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    )

    ds = ManifestDataset(
        records,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        image_size=args.image_size,
        seed=args.seed,
        use_matcher=args.use_matcher,
        synthetic_pose=args.synthetic_pose,
        z_from_dobj=args.z_from_dobj,
        load_aux=args.load_aux,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        generator=(torch.Generator().manual_seed(int(args.seed)) if args.deterministic else None),
    )

    model = RTDETRPose(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=2,
        nhead=4,
    )
    losses_fn = Losses()

    device = torch.device("cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(int(args.epochs)):
        running = 0.0
        steps = 0
        for images, targets in loader:
            images = images.to(device)

            out = model(images)

            if args.use_matcher:
                aligned = build_query_aligned_targets(
                    out["logits"],
                    out["bbox"],
                    targets,
                    num_queries=args.num_queries,
                    cost_cls=args.cost_cls,
                    cost_bbox=args.cost_bbox,
                    log_z_pred=out.get("log_z"),
                    rot6d_pred=out.get("rot6d"),
                    cost_z=args.cost_z,
                    cost_rot=args.cost_rot,
                    offsets_pred=out.get("offsets"),
                    k_delta=out.get("k_delta"),
                    cost_t=args.cost_t,
                )
                out = dict(out)
                # For box regression we train in normalized space.
                out["bbox"] = aligned["bbox_norm"]
                targets = {
                    "labels": aligned["labels"],
                    "bbox": aligned["bbox"],
                    "mask": aligned["mask"],
                    "z_gt": aligned["z_gt"],
                    "z_mask": aligned["z_mask"],
                    "R_gt": aligned["R_gt"],
                    "rot_mask": aligned["rot_mask"],
                    "offsets": aligned["offsets"],
                    "off_mask": aligned["off_mask"],
                    "t_gt": aligned["t_gt"],
                    "K_gt": aligned["K_gt"],
                    "image_hw": aligned["image_hw"],
                    "K_mask": aligned["K_mask"],
                    "t_mask": aligned["t_mask"],
                    "M_mask": aligned.get("M_mask"),
                    "D_obj_mask": aligned.get("D_obj_mask"),
                }
            else:
                # legacy padded targets
                targets = {
                    "labels": torch.stack([t["labels"] for t in targets], dim=0).to(device),
                    "bbox": torch.stack([t["bbox"] for t in targets], dim=0).to(device),
                }

            loss_dict = losses_fn(out, targets)
            loss = loss_dict["loss"]

            if steps == 0 and args.debug_losses:
                printable = {
                    k: float(v.detach().cpu())
                    for k, v in loss_dict.items()
                    if hasattr(v, "detach")
                }
                print("loss_breakdown", " ".join(f"{k}={v:.6g}" for k, v in sorted(printable.items())))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.detach().cpu())
            steps += 1

            if steps == 1 or (args.log_every and steps % int(args.log_every) == 0):
                avg = running / steps
                print(f"epoch={epoch} step={steps} loss={avg:.4f}")

            if args.max_steps and steps >= int(args.max_steps):
                break

        avg = running / max(1, steps)
        print(f"epoch={epoch} done steps={steps} loss={avg:.4f}")


if __name__ == "__main__":
    main()
