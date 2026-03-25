try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)
    F = SimpleNamespace()

def _rot6d_to_matrix(rot6d_pred):
    from .model import rot6d_to_matrix

    return rot6d_to_matrix(rot6d_pred)


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _valid_mask_from_targets(targets):
    mask = targets.get("mask")
    if mask is not None:
        return mask
    labels = _first_present(targets, ("labels", "class_gt"))
    if labels is None:
        return None
    return labels != -1


def _masked_mean(value, mask):
    if mask is None:
        return value.mean()
    if mask.numel() == 0 or not bool(mask.any()):
        return value.sum() * 0.0
    return value[mask].mean()


def geodesic_distance(r1, r2):
    if torch is None:
        raise RuntimeError("torch is required for geodesic_distance")
    rel = torch.matmul(r1.transpose(-1, -2), r2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_theta)


def symmetry_rotation_loss(rot6d_pred, r_gt, sym_rots):
    if torch is None:
        raise RuntimeError("torch is required for symmetry_rotation_loss")
    r_pred = _rot6d_to_matrix(rot6d_pred)
    losses = []
    for s in sym_rots:
        r_eq = torch.matmul(r_gt, s)
        losses.append(geodesic_distance(r_pred, r_eq))
    return torch.stack(losses, dim=0).min(dim=0).values.mean()


def rotation_loss(rot6d_pred, r_gt):
    if torch is None:
        raise RuntimeError("torch is required for rotation_loss")
    r_pred = _rot6d_to_matrix(rot6d_pred)
    return geodesic_distance(r_pred, r_gt).mean()


def log_depth_loss(log_z_pred, z_gt):
    if torch is None:
        raise RuntimeError("torch is required for log_depth_loss")
    log_z_gt = torch.log(torch.clamp(z_gt, min=1e-6))
    return torch.abs(log_z_pred - log_z_gt).mean()


def plane_loss(t_gt, plane):
    if torch is None:
        raise RuntimeError("torch is required for plane_loss")
    n = torch.tensor(plane["n"], device=t_gt.device, dtype=t_gt.dtype)
    d = torch.tensor(plane["d"], device=t_gt.device, dtype=t_gt.dtype)
    return torch.abs(torch.einsum("...i,i->...", t_gt, n) + d).mean()


def upright_loss(rot6d_pred, roll_range, pitch_range):
    if torch is None:
        raise RuntimeError("torch is required for upright_loss")
    r_pred = _rot6d_to_matrix(rot6d_pred)
    roll = torch.atan2(r_pred[..., 2, 1], r_pred[..., 2, 2])
    pitch = torch.asin(torch.clamp(-r_pred[..., 2, 0], -1.0, 1.0))
    roll_min, roll_max = roll_range
    pitch_min, pitch_max = pitch_range
    roll_penalty = torch.clamp(roll - roll_max, min=0) + torch.clamp(roll_min - roll, min=0)
    pitch_penalty = torch.clamp(pitch - pitch_max, min=0) + torch.clamp(pitch_min - pitch, min=0)
    return (roll_penalty + pitch_penalty).mean()


def mim_reconstruction_loss(recon_feat, teacher_feat, mask=None):
    """MIM reconstruction loss with optional masking.
    
    Args:
        recon_feat: (B, C, H, W) reconstructed features
        teacher_feat: (B, C, H, W) teacher features (geometry-derived)
        mask: (H, W) or (B, H, W) optional mask (True = compute loss on these locations)
        
    Returns:
        loss: scalar L1 loss between reconstructed and teacher features
    """
    if torch is None:
        raise RuntimeError("torch is required for mim_reconstruction_loss")
    
    if teacher_feat is None:
        # No teacher available, return zero loss
        return recon_feat.sum() * 0.0
    
    # L1 loss
    diff = torch.abs(recon_feat - teacher_feat)
    
    if mask is not None:
        # Apply mask: compute loss only on masked locations
        mask_expanded = mask.unsqueeze(0).unsqueeze(0) if mask.ndim == 2 else mask.unsqueeze(1)
        mask_expanded = mask_expanded.expand_as(diff).to(dtype=torch.bool)
        
        if not mask_expanded.any():
            return diff.sum() * 0.0
        
        return diff[mask_expanded].mean()
    
    return diff.mean()


def entropy_loss(logits):
    """Entropy loss for geometric consistency (lower entropy = more confident).
    
    Args:
        logits: (B, N, C) classification logits
        
    Returns:
        loss: scalar entropy loss
    """
    if torch is None or F is None:
        raise RuntimeError("torch is required for entropy_loss")
    
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(torch.clamp(probs, min=1e-12))
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy


class Losses(nn.Module):
    def __init__(self, weights=None, *, task_aligner: str = "none"):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for Losses")
        self.task_aligner = str(task_aligner)
        self.weights = {
            "cls": 1.0,
            "box": 1.0,
            # DETR-style background ("no-object") reweighting. Without this, the
            # background class can dominate training because Q >> objects/image.
            # Typical DETR defaults use ~0.1 for eos/no-object.
            "eos_coef": 0.1,
            "kp": 1.0,
            "z": 1.0,
            "rot": 1.0,
            "off": 1.0,
            "k": 1.0,
            "z_prior": 0.1,
            "plane": 0.1,
            "upright": 0.1,
            "t": 0.1,
            "mim": 0.1,
            "entropy": 0.01,
        }
        if weights:
            self.weights.update(weights)
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets):
        if torch is None:
            raise RuntimeError("torch is required for Losses")
        losses = {}
        total = 0.0

        valid = _valid_mask_from_targets(targets)
        z_mask = targets.get("z_mask")
        d_obj_mask = targets.get("D_obj_mask")
        m_mask = targets.get("M_mask")
        rot_mask = targets.get("rot_mask")
        off_mask = targets.get("off_mask")

        logits = outputs.get("logits")
        labels = _first_present(targets, ("labels", "class_gt"))
        if logits is not None and labels is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = labels.reshape(-1)

            # Classification supervision should apply to *all* queries that have
            # a concrete label, including the "no-object" background class.
            # Do not reuse the bbox/pose valid mask here.
            valid_cls = labels_flat != -1

            if not bool(valid_cls.any()):
                loss_cls = logits_flat.sum() * 0.0
            else:
                weight = None
                # Downweight the background/no-object class (last index) to avoid collapse.
                try:
                    eos_coef = float(self.weights.get("eos_coef", 0.1))
                except Exception:
                    eos_coef = 0.1
                if logits_flat.shape[-1] > 1 and eos_coef > 0.0 and eos_coef < 1.0:
                    weight = torch.ones((int(logits_flat.shape[-1]),), device=logits_flat.device, dtype=torch.float32)
                    weight[-1] = float(eos_coef)
                loss_cls = F.cross_entropy(logits_flat[valid_cls], labels_flat[valid_cls], weight=weight)
            losses["loss_cls"] = loss_cls
            total = total + self.weights["cls"] * loss_cls

        bbox_pred = outputs.get("bbox")
        bbox_gt = targets.get("bbox")
        if bbox_pred is not None and bbox_gt is not None:
            if valid is None:
                loss_box = self.l1(bbox_pred, bbox_gt)
            else:
                diff = torch.abs(bbox_pred - bbox_gt).sum(dim=-1)
                loss_box = _masked_mean(diff, valid)
            losses["loss_box"] = loss_box
            total = total + self.weights["box"] * loss_box

        keypoints_pred = outputs.get("keypoints")
        keypoints_gt = _first_present(targets, ("keypoints_gt", "gt_keypoints", "keypoints"))
        keypoints_mask = _first_present(targets, ("keypoints_mask", "gt_keypoints_mask"))
        if keypoints_pred is not None and keypoints_gt is not None:
            kp_mask_for_loss = None
            if keypoints_mask is not None:
                keypoints_mask = keypoints_mask.to(device=keypoints_pred.device, dtype=torch.bool)
                kp_mask_for_loss = keypoints_mask
            if valid is not None:
                v = valid.to(device=keypoints_pred.device, dtype=torch.bool).unsqueeze(-1)
                kp_mask_for_loss = v if kp_mask_for_loss is None else (kp_mask_for_loss & v)

            diff = torch.abs(keypoints_pred - keypoints_gt).sum(dim=-1)  # (B,Q,K)
            if kp_mask_for_loss is None:
                loss_kp = diff.mean()
            else:
                if kp_mask_for_loss.numel() == 0 or not bool(kp_mask_for_loss.any()):
                    loss_kp = diff.sum() * 0.0
                else:
                    loss_kp = diff[kp_mask_for_loss].mean()
            losses["loss_kp"] = loss_kp
            total = total + self.weights["kp"] * loss_kp

        log_z_pred = outputs.get("log_z")
        log_sigma_z = outputs.get("log_sigma_z")
        z_gt = _first_present(targets, ("z_gt", "depth", "z"))
        if log_z_pred is not None and z_gt is not None:
            if isinstance(z_gt, torch.Tensor) and z_gt.ndim == log_z_pred.ndim + 1 and z_gt.shape[-1] == 1:
                z_gt = z_gt.squeeze(-1)

            mask_for_z = valid
            if z_mask is not None:
                z_mask = z_mask.to(device=log_z_pred.device, dtype=torch.bool)
                mask_for_z = z_mask if mask_for_z is None else (mask_for_z & z_mask)
            if d_obj_mask is not None:
                d_obj_mask = d_obj_mask.to(device=log_z_pred.device, dtype=torch.bool)
                mask_for_z = d_obj_mask if mask_for_z is None else (mask_for_z & d_obj_mask)
            if m_mask is not None:
                m_mask = m_mask.to(device=log_z_pred.device, dtype=torch.bool)
                mask_for_z = m_mask if mask_for_z is None else (mask_for_z & m_mask)

            if valid is None:
                if mask_for_z is None:
                    loss_z = log_depth_loss(log_z_pred, z_gt)
                else:
                    if not bool(mask_for_z.any()):
                        loss_z = log_z_pred.sum() * 0.0
                    else:
                        loss_z = log_depth_loss(log_z_pred[mask_for_z], z_gt[mask_for_z])
            else:
                if mask_for_z is not None and not bool(mask_for_z.any()):
                    loss_z = log_z_pred.sum() * 0.0
                else:
                    mask_sel = valid if mask_for_z is None else mask_for_z
                    loss_z = log_depth_loss(log_z_pred[mask_sel], z_gt[mask_sel])
            losses["loss_z"] = loss_z
            if self.task_aligner == "uncertainty" and log_sigma_z is not None:
                sigma = log_sigma_z
                if isinstance(sigma, torch.Tensor) and sigma.shape[-1] == 1:
                    sigma = sigma.squeeze(-1)
                if mask_for_z is not None:
                    sigma = sigma[mask_for_z]
                if sigma.numel() == 0:
                    sigma_mean = loss_z.detach() * 0.0
                else:
                    sigma_mean = sigma.mean()
                loss_z_aligned = torch.exp(-sigma_mean) * loss_z + sigma_mean
                losses["loss_z_aligned"] = loss_z_aligned
                total = total + self.weights["z"] * loss_z_aligned
            else:
                total = total + self.weights["z"] * loss_z

        rot_pred = outputs.get("rot6d")
        log_sigma_rot = outputs.get("log_sigma_rot")
        r_gt = _first_present(targets, ("R_gt", "r_gt"))
        sym_rots = targets.get("sym_rots")
        if rot_pred is not None and r_gt is not None:
            mask_for_rot = valid
            if rot_mask is not None:
                rot_mask = rot_mask.to(device=rot_pred.device, dtype=torch.bool)
                mask_for_rot = rot_mask if mask_for_rot is None else (mask_for_rot & rot_mask)

            rot_pred_in = rot_pred
            r_gt_in = r_gt
            loss_rot = rot_pred.sum() * 0.0
            if mask_for_rot is not None:
                if not bool(mask_for_rot.any()):
                    loss_rot = rot_pred.sum() * 0.0
                    rot_pred_in = None
                    r_gt_in = None
                else:
                    rot_pred_in = rot_pred[mask_for_rot]
                    r_gt_in = r_gt[mask_for_rot]

            if rot_pred_in is not None and r_gt_in is not None:
                if sym_rots is not None:
                    loss_rot = symmetry_rotation_loss(rot_pred_in, r_gt_in, sym_rots)
                else:
                    loss_rot = rotation_loss(rot_pred_in, r_gt_in)
            losses["loss_rot"] = loss_rot
            if self.task_aligner == "uncertainty" and log_sigma_rot is not None:
                sigma = log_sigma_rot
                if isinstance(sigma, torch.Tensor) and sigma.shape[-1] == 1:
                    sigma = sigma.squeeze(-1)
                if mask_for_rot is not None:
                    sigma = sigma[mask_for_rot]
                if sigma.numel() == 0:
                    sigma_mean = loss_rot.detach() * 0.0
                else:
                    sigma_mean = sigma.mean()
                loss_rot_aligned = torch.exp(-sigma_mean) * loss_rot + sigma_mean
                losses["loss_rot_aligned"] = loss_rot_aligned
                total = total + self.weights["rot"] * loss_rot_aligned
            else:
                total = total + self.weights["rot"] * loss_rot

        offsets_pred = outputs.get("offsets")
        offsets_gt = _first_present(targets, ("offsets", "offsets_gt"))
        if offsets_pred is not None and offsets_gt is not None:
            mask_for_off = valid
            if off_mask is not None:
                off_mask = off_mask.to(device=offsets_pred.device, dtype=torch.bool)
                mask_for_off = off_mask if mask_for_off is None else (mask_for_off & off_mask)

            if mask_for_off is None:
                loss_off = self.l1(offsets_pred, offsets_gt)
            else:
                diff = torch.abs(offsets_pred - offsets_gt).sum(dim=-1)
                loss_off = _masked_mean(diff, mask_for_off)
            losses["loss_off"] = loss_off
            total = total + self.weights["off"] * loss_off

        # Intrinsics-aware translation consistency loss.
        # Uses predicted (bbox center, offsets, z, k_delta) to recover t and compares to t_gt.
        k_gt = _first_present(targets, ("K_gt", "K", "intrinsics"))
        t_gt = _first_present(targets, ("t_gt", "t"))
        image_hw = _first_present(targets, ("image_hw", "hw"))
        k_mask = targets.get("K_mask")
        t_mask = targets.get("t_mask")
        bbox_gt_for_t = targets.get("bbox")
        log_z_pred_for_t = outputs.get("log_z")
        k_delta_pred = outputs.get("k_delta")
        if (
            bbox_gt_for_t is not None
            and offsets_pred is not None
            and log_z_pred_for_t is not None
            and k_gt is not None
            and t_gt is not None
            and image_hw is not None
        ):
            if (t_mask is not None and not bool(t_mask.any())) or (
                k_mask is not None and not bool(k_mask.any())
            ):
                # No translation supervision available in this batch.
                pass
            else:
                # Parse K: (B,3,3) or (3,3)
                if k_gt.ndim == 2:
                    k_gt_b = k_gt.unsqueeze(0).expand(bbox_gt_for_t.shape[0], -1, -1)
                else:
                    k_gt_b = k_gt

                fx = k_gt_b[:, 0, 0].clamp(min=1e-6)
                fy = k_gt_b[:, 1, 1].clamp(min=1e-6)
                cx = k_gt_b[:, 0, 2]
                cy = k_gt_b[:, 1, 2]

                if image_hw.ndim == 1:
                    hw_b = image_hw.unsqueeze(0).expand(bbox_gt_for_t.shape[0], -1)
                else:
                    hw_b = image_hw
                h = hw_b[:, 0].unsqueeze(1)
                w = hw_b[:, 1].unsqueeze(1)

                u = bbox_gt_for_t[..., 0] * w
                v = bbox_gt_for_t[..., 1] * h
                z = torch.exp(log_z_pred_for_t).clamp(min=1e-6).squeeze(-1)

                du = offsets_pred[..., 0]
                dv = offsets_pred[..., 1]

                # Apply k_delta if available (B,4).
                if k_delta_pred is not None:
                    dfx = k_delta_pred[:, 0].unsqueeze(1)
                    dfy = k_delta_pred[:, 1].unsqueeze(1)
                    dcx = k_delta_pred[:, 2].unsqueeze(1)
                    dcy = k_delta_pred[:, 3].unsqueeze(1)
                    fx = fx.unsqueeze(1) * (1.0 + dfx)
                    fy = fy.unsqueeze(1) * (1.0 + dfy)
                    cx = cx.unsqueeze(1) + dcx
                    cy = cy.unsqueeze(1) + dcy
                else:
                    fx = fx.unsqueeze(1)
                    fy = fy.unsqueeze(1)
                    cx = cx.unsqueeze(1)
                    cy = cy.unsqueeze(1)

                u_p = u + du
                v_p = v + dv
                x = (u_p - cx) / fx * z
                y = (v_p - cy) / fy * z
                t_pred = torch.stack((x, y, z), dim=-1)

                mask_for_t = valid
                if t_mask is not None:
                    mask_for_t = t_mask if mask_for_t is None else (mask_for_t & t_mask)

                if mask_for_t is not None:
                    if not bool(mask_for_t.any()):
                        loss_t = t_pred.sum() * 0.0
                    else:
                        diff = torch.abs(t_pred - t_gt).sum(dim=-1)
                        loss_t = _masked_mean(diff, mask_for_t)
                else:
                    loss_t = torch.abs(t_pred - t_gt).mean()

                losses["loss_t"] = loss_t
                total = total + self.weights["t"] * loss_t

        k_pred = outputs.get("k_delta")
        k_gt = _first_present(targets, ("k_delta", "k_delta_gt"))
        if k_pred is not None and k_gt is not None:
            loss_k = self.l1(k_pred, k_gt)
            losses["loss_k"] = loss_k
            total = total + self.weights["k"] * loss_k

        z_prior = targets.get("z_prior")
        if log_z_pred is not None and z_prior is not None:
            loss_z_prior = self.l1(log_z_pred, torch.log(torch.clamp(z_prior, min=1e-6)))
            losses["loss_z_prior"] = loss_z_prior
            total = total + self.weights["z_prior"] * loss_z_prior

        t_gt = targets.get("t_gt")
        plane = targets.get("plane")
        if t_gt is not None and plane is not None:
            loss_plane = plane_loss(t_gt, plane)
            losses["loss_plane"] = loss_plane
            total = total + self.weights["plane"] * loss_plane

        upright = targets.get("upright_range")
        if rot_pred is not None and upright is not None:
            loss_upright = upright_loss(rot_pred, upright["roll"], upright["pitch"])
            losses["loss_upright"] = loss_upright
            total = total + self.weights["upright"] * loss_upright

        # Masked reconstruction loss (MIM)
        mim_outputs = outputs.get("mim")
        if mim_outputs is not None:
            recon_feat = mim_outputs.get("recon_feat")
            teacher_feat = mim_outputs.get("teacher_feat")
            feature_mask = mim_outputs.get("mask")
            
            if recon_feat is not None and teacher_feat is not None:
                loss_mim = mim_reconstruction_loss(recon_feat, teacher_feat, mask=feature_mask)
                losses["loss_mim"] = loss_mim
                total = total + self.weights["mim"] * loss_mim
            
            # Entropy loss for geometric consistency
            entropy = mim_outputs.get("entropy")
            if entropy is not None:
                losses["loss_entropy"] = entropy
                total = total + self.weights["entropy"] * entropy

        losses["loss"] = total
        return losses
