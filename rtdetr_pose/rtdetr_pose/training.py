"""Training helpers for the RT-DETR pose scaffold."""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .matcher import linear_sum_assignment


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _rot6d_to_matrix(x):
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def build_query_aligned_targets(
    logits,
    bbox_pred,
    targets: List[Dict],
    num_queries: int,
    cost_cls: float = 1.0,
    cost_bbox: float = 5.0,
    log_z_pred=None,
    rot6d_pred=None,
    cost_z: float = 0.0,
    cost_rot: float = 0.0,
    offsets_pred=None,
    k_delta=None,
    cost_t: float = 0.0,
    keypoints_pred=None,
):
    """Match queries to GT per-sample and build query-aligned targets.

    targets: list of dicts with keys gt_labels (M,) and gt_bbox (M,4)
    returns: dict with labels (B,Q) and bbox (B,Q,4) plus bbox_norm (B,Q,4)
    """

    if torch is None:
        raise RuntimeError("torch is required for build_query_aligned_targets")

    batch, queries, _ = logits.shape
    if queries != num_queries:
        raise ValueError("num_queries mismatch")

    aligned_labels = []
    aligned_bbox = []
    aligned_z = []
    aligned_r = []
    aligned_offsets = []
    aligned_z_mask = []
    aligned_rot_mask = []
    aligned_off_mask = []
    aligned_m_mask = []
    aligned_dobj_mask = []
    aligned_t = []
    aligned_k = []
    aligned_hw = []
    aligned_k_mask = []
    aligned_t_mask = []
    aligned_mask = []
    aligned_keypoints = []
    aligned_keypoints_mask = []

    neg_log_prob = -torch.log_softmax(logits, dim=-1)
    bbox_norm = bbox_pred.sigmoid()

    # Reserve the last class index as the "no-object" / background class.
    background_class_id = int(logits.shape[-1]) - 1

    if rot6d_pred is not None:
        r_pred_all = _rot6d_to_matrix(rot6d_pred)
    else:
        r_pred_all = None

    for b in range(batch):
        gt_labels = targets[b].get("gt_labels")
        gt_bbox = targets[b].get("gt_bbox")
        gt_z = _first_present(targets[b], ("gt_z", "z_gt", "z", "depth"))
        gt_z_mask = targets[b].get("gt_z_mask")
        gt_r = _first_present(targets[b], ("gt_R", "R_gt", "R", "gt_r"))
        gt_r_mask = targets[b].get("gt_R_mask")
        gt_offsets = _first_present(targets[b], ("gt_offsets", "offsets", "offsets_gt"))
        gt_offsets_mask = targets[b].get("gt_offsets_mask")
        gt_t = _first_present(targets[b], ("gt_t", "t_gt", "t"))
        gt_t_mask = targets[b].get("gt_t_mask")
        gt_keypoints = _first_present(targets[b], ("gt_keypoints", "keypoints_gt", "keypoints"))
        gt_keypoints_mask = _first_present(targets[b], ("gt_keypoints_mask", "keypoints_mask"))
        gt_m_mask = targets[b].get("gt_M_mask")
        gt_dobj_mask = targets[b].get("gt_D_obj_mask")
        k_gt = _first_present(targets[b], ("K_gt", "K", "intrinsics"))
        image_hw = _first_present(targets[b], ("image_hw", "hw"))

        # Treat empty tensors as missing.
        if isinstance(gt_z, torch.Tensor) and gt_z.numel() == 0:
            gt_z = None
        if isinstance(gt_z_mask, torch.Tensor) and gt_z_mask.numel() == 0:
            gt_z_mask = None
        if isinstance(gt_r, torch.Tensor) and gt_r.numel() == 0:
            gt_r = None
        if isinstance(gt_r_mask, torch.Tensor) and gt_r_mask.numel() == 0:
            gt_r_mask = None
        if isinstance(gt_offsets, torch.Tensor) and gt_offsets.numel() == 0:
            gt_offsets = None
        if isinstance(gt_offsets_mask, torch.Tensor) and gt_offsets_mask.numel() == 0:
            gt_offsets_mask = None
        if isinstance(gt_t, torch.Tensor) and gt_t.numel() == 0:
            gt_t = None
        if isinstance(gt_t_mask, torch.Tensor) and gt_t_mask.numel() == 0:
            gt_t_mask = None
        if isinstance(gt_m_mask, torch.Tensor) and gt_m_mask.numel() == 0:
            gt_m_mask = None
        if isinstance(gt_dobj_mask, torch.Tensor) and gt_dobj_mask.numel() == 0:
            gt_dobj_mask = None
        if isinstance(gt_keypoints, torch.Tensor) and gt_keypoints.numel() == 0:
            gt_keypoints = None
        if isinstance(gt_keypoints_mask, torch.Tensor) and gt_keypoints_mask.numel() == 0:
            gt_keypoints_mask = None

        has_k = k_gt is not None
        has_t = gt_t is not None and (gt_t_mask is None or bool(gt_t_mask.any()))
        if gt_labels is None or gt_bbox is None or gt_labels.numel() == 0:
            labels_q = torch.full((num_queries,), background_class_id, dtype=torch.long, device=logits.device)
            bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
            mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            z_q = torch.zeros((num_queries, 1), dtype=torch.float32, device=logits.device)
            r_q = torch.eye(3, device=logits.device, dtype=torch.float32).unsqueeze(0).repeat(
                num_queries, 1, 1
            )
            off_q = torch.zeros((num_queries, 2), dtype=torch.float32, device=logits.device)
            z_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            rot_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            off_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            m_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            dobj_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            t_q = torch.zeros((num_queries, 3), dtype=torch.float32, device=logits.device)
            t_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
            aligned_labels.append(labels_q)
            aligned_bbox.append(bbox_q)
            aligned_mask.append(mask_q)
            aligned_z.append(z_q)
            aligned_r.append(r_q)
            aligned_offsets.append(off_q)
            aligned_z_mask.append(z_mask_q)
            aligned_rot_mask.append(rot_mask_q)
            aligned_off_mask.append(off_mask_q)
            aligned_m_mask.append(m_mask_q)
            aligned_dobj_mask.append(dobj_mask_q)
            aligned_t.append(t_q)
            aligned_t_mask.append(t_mask_q)

            if has_k:
                k_gt_t = torch.as_tensor(k_gt, dtype=torch.float32, device=logits.device)
                aligned_k_mask.append(torch.tensor(True, device=logits.device))
            else:
                k_gt_t = torch.zeros((3, 3), dtype=torch.float32, device=logits.device)
                aligned_k_mask.append(torch.tensor(False, device=logits.device))
            aligned_k.append(k_gt_t)
            if image_hw is not None:
                hw_t = torch.as_tensor(image_hw, dtype=torch.float32, device=logits.device)
            else:
                hw_t = torch.zeros((2,), dtype=torch.float32, device=logits.device)
            aligned_hw.append(hw_t)
            k_count = 0
            if keypoints_pred is not None and isinstance(keypoints_pred, torch.Tensor) and keypoints_pred.ndim == 4:
                k_count = int(keypoints_pred.shape[2])
            elif gt_keypoints is not None and isinstance(gt_keypoints, torch.Tensor) and gt_keypoints.ndim == 3:
                k_count = int(gt_keypoints.shape[1])
            aligned_keypoints.append(torch.zeros((num_queries, k_count, 2), dtype=torch.float32, device=logits.device))
            aligned_keypoints_mask.append(torch.zeros((num_queries, k_count), dtype=torch.bool, device=logits.device))
            continue

        gt_labels = gt_labels.to(device=logits.device)
        gt_bbox = gt_bbox.to(device=logits.device)
        if gt_z is not None:
            gt_z = gt_z.to(device=logits.device, dtype=torch.float32)
            if gt_z.ndim == 1:
                gt_z = gt_z.unsqueeze(-1)
        if gt_z_mask is not None:
            gt_z_mask = gt_z_mask.to(device=logits.device, dtype=torch.bool)
        if gt_r is not None:
            gt_r = gt_r.to(device=logits.device, dtype=torch.float32)
        if gt_r_mask is not None:
            gt_r_mask = gt_r_mask.to(device=logits.device, dtype=torch.bool)
        if gt_offsets is not None:
            gt_offsets = gt_offsets.to(device=logits.device, dtype=torch.float32)
        if gt_offsets_mask is not None:
            gt_offsets_mask = gt_offsets_mask.to(device=logits.device, dtype=torch.bool)
        if gt_t is not None:
            gt_t = gt_t.to(device=logits.device, dtype=torch.float32)
        if gt_t_mask is not None:
            gt_t_mask = gt_t_mask.to(device=logits.device, dtype=torch.bool)
        if gt_keypoints is not None:
            gt_keypoints = gt_keypoints.to(device=logits.device, dtype=torch.float32)
        if gt_keypoints_mask is not None:
            gt_keypoints_mask = gt_keypoints_mask.to(device=logits.device, dtype=torch.bool)
        if gt_m_mask is not None:
            gt_m_mask = gt_m_mask.to(device=logits.device, dtype=torch.bool)
        if gt_dobj_mask is not None:
            gt_dobj_mask = gt_dobj_mask.to(device=logits.device, dtype=torch.bool)
        if k_gt is not None:
            k_gt = torch.as_tensor(k_gt, dtype=torch.float32, device=logits.device)
        if image_hw is not None:
            image_hw = torch.as_tensor(image_hw, dtype=torch.float32, device=logits.device)

        # (Q, M) -> (M, Q)
        cls_cost = neg_log_prob[b][:, gt_labels].transpose(0, 1)
        # (M, Q)
        bbox_cost = torch.abs(bbox_norm[b].unsqueeze(0) - gt_bbox.unsqueeze(1)).sum(dim=-1)
        cost = cost_cls * cls_cost + cost_bbox * bbox_cost

        if cost_z and log_z_pred is not None and gt_z is not None:
            log_z_gt = torch.log(torch.clamp(gt_z, min=1e-6))
            z_cost = torch.abs(log_z_pred[b].unsqueeze(0) - log_z_gt.unsqueeze(1)).sum(dim=-1)
            if gt_z_mask is not None:
                z_cost = z_cost * gt_z_mask.to(dtype=z_cost.dtype).unsqueeze(-1)
            cost = cost + float(cost_z) * z_cost

        if cost_rot and r_pred_all is not None and gt_r is not None:
            # Geodesic angle via trace(R_pred^T R_gt) == sum(R_pred * R_gt)
            trace = (r_pred_all[b].unsqueeze(0) * gt_r.unsqueeze(1)).sum(dim=(-1, -2))
            cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
            rot_cost = torch.acos(cos_theta)
            if gt_r_mask is not None:
                rot_cost = rot_cost * gt_r_mask.to(dtype=rot_cost.dtype).unsqueeze(-1)
            cost = cost + float(cost_rot) * rot_cost

        if cost_t and offsets_pred is not None and log_z_pred is not None and has_t and has_k:
            if image_hw is None:
                raise ValueError("cost_t requires targets[b]['image_hw']")

            # Intrinsics (fx,fy,cx,cy) from 3x3 K.
            fx = k_gt[0, 0]
            fy = k_gt[1, 1]
            cx = k_gt[0, 2]
            cy = k_gt[1, 2]
            fx = fx.clamp(min=1e-6)
            fy = fy.clamp(min=1e-6)
            if k_delta is not None:
                # Apply per-image delta (B,4): (dfx, dfy, dcx, dcy)
                dfx, dfy, dcx, dcy = k_delta[b]
                fx = fx * (1.0 + dfx)
                fy = fy * (1.0 + dfy)
                cx = cx + dcx
                cy = cy + dcy

            h, w = image_hw
            uv = bbox_norm[b][:, 0:2] * torch.stack((w, h), dim=0)
            u = uv[:, 0]
            v = uv[:, 1]
            du = offsets_pred[b][:, 0]
            dv = offsets_pred[b][:, 1]
            z = torch.exp(log_z_pred[b].squeeze(-1)).clamp(min=1e-6)
            u_p = u + du
            v_p = v + dv
            x = (u_p - cx) / fx * z
            y = (v_p - cy) / fy * z
            t_pred = torch.stack((x, y, z), dim=-1)  # (Q,3)
            # (M,Q)
            t_cost = torch.abs(t_pred.unsqueeze(0) - gt_t.unsqueeze(1)).sum(dim=-1)
            if gt_t_mask is not None:
                t_cost = t_cost * gt_t_mask.to(dtype=t_cost.dtype).unsqueeze(-1)
            cost = cost + float(cost_t) * t_cost

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().tolist())
        labels_q = torch.full((num_queries,), background_class_id, dtype=torch.long, device=logits.device)
        bbox_q = torch.zeros((num_queries, 4), dtype=torch.float32, device=logits.device)
        mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        z_q = torch.zeros((num_queries, 1), dtype=torch.float32, device=logits.device)
        r_q = torch.eye(3, device=logits.device, dtype=torch.float32).unsqueeze(0).repeat(
            num_queries, 1, 1
        )
        off_q = torch.zeros((num_queries, 2), dtype=torch.float32, device=logits.device)
        z_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        rot_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        off_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        m_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        dobj_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        t_q = torch.zeros((num_queries, 3), dtype=torch.float32, device=logits.device)
        t_mask_q = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)
        k_count = 0
        if keypoints_pred is not None and isinstance(keypoints_pred, torch.Tensor) and keypoints_pred.ndim == 4:
            k_count = int(keypoints_pred.shape[2])
        elif gt_keypoints is not None and isinstance(gt_keypoints, torch.Tensor) and gt_keypoints.ndim == 3:
            k_count = int(gt_keypoints.shape[1])
        kp_q = torch.zeros((num_queries, k_count, 2), dtype=torch.float32, device=logits.device)
        kp_mask_q = torch.zeros((num_queries, k_count), dtype=torch.bool, device=logits.device)
        for r, c in zip(row_ind, col_ind):
            if 0 <= c < num_queries:
                labels_q[c] = gt_labels[r]
                bbox_q[c] = gt_bbox[r]
                mask_q[c] = True
                if gt_z is not None:
                    z_q[c] = gt_z[r]
                    if gt_z_mask is None:
                        z_mask_q[c] = True
                    else:
                        z_mask_q[c] = bool(gt_z_mask[r])
                if gt_r is not None:
                    r_q[c] = gt_r[r]
                    if gt_r_mask is None:
                        rot_mask_q[c] = True
                    else:
                        rot_mask_q[c] = bool(gt_r_mask[r])
                if gt_offsets is not None:
                    off_q[c] = gt_offsets[r]
                    if gt_offsets_mask is None:
                        off_mask_q[c] = True
                    else:
                        off_mask_q[c] = bool(gt_offsets_mask[r])
                if gt_m_mask is not None:
                    m_mask_q[c] = bool(gt_m_mask[r])
                if gt_dobj_mask is not None:
                    dobj_mask_q[c] = bool(gt_dobj_mask[r])
                if gt_t is not None:
                    t_q[c] = gt_t[r]
                    if gt_t_mask is None:
                        t_mask_q[c] = True
                    else:
                        t_mask_q[c] = bool(gt_t_mask[r])
                if gt_keypoints is not None and int(kp_q.shape[1]) > 0:
                    kp_q[c] = gt_keypoints[r]
                    if gt_keypoints_mask is None:
                        kp_mask_q[c] = True
                    else:
                        kp_mask_q[c] = gt_keypoints_mask[r].to(dtype=torch.bool)
        aligned_labels.append(labels_q)
        aligned_bbox.append(bbox_q)
        aligned_mask.append(mask_q)
        aligned_z.append(z_q)
        aligned_r.append(r_q)
        aligned_offsets.append(off_q)
        aligned_z_mask.append(z_mask_q)
        aligned_rot_mask.append(rot_mask_q)
        aligned_off_mask.append(off_mask_q)
        aligned_m_mask.append(m_mask_q)
        aligned_dobj_mask.append(dobj_mask_q)
        aligned_t.append(t_q)
        aligned_t_mask.append(t_mask_q)
        aligned_keypoints.append(kp_q)
        aligned_keypoints_mask.append(kp_mask_q)

        if has_k:
            aligned_k.append(k_gt)
            aligned_k_mask.append(torch.tensor(True, device=logits.device))
        else:
            aligned_k.append(torch.zeros((3, 3), dtype=torch.float32, device=logits.device))
            aligned_k_mask.append(torch.tensor(False, device=logits.device))
        if image_hw is not None:
            aligned_hw.append(image_hw)
        else:
            aligned_hw.append(torch.zeros((2,), dtype=torch.float32, device=logits.device))

    return {
        "labels": torch.stack(aligned_labels, dim=0),
        "bbox": torch.stack(aligned_bbox, dim=0),
        "bbox_norm": bbox_norm,
        "mask": torch.stack(aligned_mask, dim=0),
        "z_gt": torch.stack(aligned_z, dim=0),
        "z_mask": torch.stack(aligned_z_mask, dim=0),
        "R_gt": torch.stack(aligned_r, dim=0),
        "rot_mask": torch.stack(aligned_rot_mask, dim=0),
        "offsets": torch.stack(aligned_offsets, dim=0),
        "off_mask": torch.stack(aligned_off_mask, dim=0),
        "M_mask": torch.stack(aligned_m_mask, dim=0),
        "D_obj_mask": torch.stack(aligned_dobj_mask, dim=0),
        "t_gt": torch.stack(aligned_t, dim=0),
        "K_gt": torch.stack(aligned_k, dim=0),
        "image_hw": torch.stack(aligned_hw, dim=0),
        "K_mask": torch.stack(aligned_k_mask, dim=0),
        "t_mask": torch.stack(aligned_t_mask, dim=0),
        "keypoints_gt": torch.stack(aligned_keypoints, dim=0),
        "keypoints_mask": torch.stack(aligned_keypoints_mask, dim=0),
    }
