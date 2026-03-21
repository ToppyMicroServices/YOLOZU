"""Hessian-based solver for refining regression head predictions.

This module provides per-image iterative refinement of regression outputs
(depth, rotation, offsets) using second-order optimization (Gauss-Newton).
Unlike the L-BFGS calibration which optimizes dataset-wide scale and intrinsics,
this solver refines individual predictions using available geometric constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True)
class HessianSolverConfig:
    """Configuration for Hessian-based regression refinement.
    
    Attributes:
        max_iterations: Maximum number of Gauss-Newton iterations.
        convergence_threshold: Stop when parameter update norm is below this.
        damping: Levenberg-Marquardt damping factor (0 = pure Gauss-Newton).
        refine_depth: Whether to refine log_z predictions.
        refine_rotation: Whether to refine rot6d predictions.
        refine_offsets: Whether to refine center offsets.
        use_geometric_constraints: Use plane/upright constraints if available.
        w_depth: Weight for depth residuals.
        w_rotation: Weight for rotation residuals.
        w_offsets: Weight for offset residuals.
        w_plane: Weight for plane constraint (if enabled).
        w_upright: Weight for upright constraint (if enabled).
    """
    
    max_iterations: int = 5
    convergence_threshold: float = 1e-4
    damping: float = 1e-3
    refine_depth: bool = True
    refine_rotation: bool = True
    refine_offsets: bool = True
    use_geometric_constraints: bool = False
    w_depth: float = 1.0
    w_rotation: float = 1.0
    w_offsets: float = 1.0
    w_plane: float = 0.1
    w_upright: float = 0.1


@dataclass(frozen=True)
class RefinementResult:
    """Result of Hessian refinement for a single detection.
    
    Attributes:
        log_z: Refined log depth (or original if not refined).
        rot6d: Refined rotation 6D representation (or original if not refined).
        offsets: Refined center offsets (or original if not refined).
        iterations: Number of iterations performed.
        converged: Whether the solver converged.
        final_residual: Final residual norm.
    """
    
    log_z: float | None
    rot6d: list[float] | None
    offsets: list[float] | None
    iterations: int
    converged: bool
    final_residual: float


def _rot6d_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rot6d: [..., 6] tensor of 6D rotation representation.
        
    Returns:
        [..., 3, 3] rotation matrix.
    """
    import torch
    from torch.nn import functional as F
    
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _geodesic_distance_torch(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distance between rotation matrices.
    
    Args:
        r1: [..., 3, 3] rotation matrix.
        r2: [..., 3, 3] rotation matrix.
        
    Returns:
        [...] geodesic distance in radians.
    """
    import torch
    
    rel = torch.matmul(r1.transpose(-1, -2), r2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cos_theta)


def _basename_key(path_like: Any) -> str:
    text = str(path_like)
    if not text:
        return ""
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def refine_detection_hessian(
    detection: dict[str, Any],
    *,
    config: HessianSolverConfig,
    gt_depth: float | None = None,
    gt_rotation: list[list[float]] | None = None,
    plane_constraint: dict[str, Any] | None = None,
    upright_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Refine a single detection using Gauss-Newton with Hessian approximation.
    
    This function performs iterative refinement of regression outputs using
    available supervision signals and geometric constraints. It uses the
    Gauss-Newton method with optional Levenberg-Marquardt damping.
    
    Args:
        detection: Detection dictionary with 'log_z', 'rot6d', 'offsets' fields.
        config: Solver configuration.
        gt_depth: Optional ground truth depth for supervision.
        gt_rotation: Optional ground truth rotation matrix (3x3) for supervision.
        plane_constraint: Optional plane constraint {'n': [nx, ny, nz], 'd': d}.
        upright_bounds: Optional upright bounds ((roll_min, roll_max), (pitch_min, pitch_max)).
        
    Returns:
        Updated detection dictionary with refined predictions and metadata.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        raise RuntimeError("torch is required for Hessian refinement")
    
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Extract current predictions.
    log_z_val = detection.get("log_z")
    rot6d_val = detection.get("rot6d")
    offsets_val = detection.get("offsets")
    
    # Initialize parameters to refine.
    params = []
    param_types = []
    
    if config.refine_depth and log_z_val is not None:
        log_z_param = torch.tensor([float(log_z_val)], dtype=dtype, device=device, requires_grad=True)
        params.append(log_z_param)
        param_types.append("log_z")
    else:
        log_z_param = None
        
    if config.refine_rotation and rot6d_val is not None:
        try:
            rot6d_list = [float(x) for x in rot6d_val]
            if len(rot6d_list) == 6:
                rot6d_param = torch.tensor(rot6d_list, dtype=dtype, device=device, requires_grad=True)
                params.append(rot6d_param)
                param_types.append("rot6d")
            else:
                rot6d_param = None
        except (TypeError, ValueError):
            rot6d_param = None
    else:
        rot6d_param = None
        
    if config.refine_offsets and offsets_val is not None:
        try:
            offsets_list = [float(x) for x in offsets_val]
            if len(offsets_list) == 2:
                offsets_param = torch.tensor(offsets_list, dtype=dtype, device=device, requires_grad=True)
                params.append(offsets_param)
                param_types.append("offsets")
            else:
                offsets_param = None
        except (TypeError, ValueError):
            offsets_param = None
    else:
        offsets_param = None
    
    # If nothing to refine, return original.
    if not params:
        return detection
    
    # Build supervision tensors.
    z_gt_tensor = None
    if gt_depth is not None:
        z_gt_tensor = torch.tensor(float(gt_depth), dtype=dtype, device=device)
        
    r_gt_tensor = None
    if gt_rotation is not None:
        try:
            r_flat = [float(x) for row in gt_rotation for x in row]
            if len(r_flat) == 9:
                r_gt_tensor = torch.tensor(r_flat, dtype=dtype, device=device).reshape(3, 3)
        except (TypeError, ValueError):
            r_gt_tensor = None
    
    # Gauss-Newton iterations.
    converged = False
    final_residual = float("inf")
    iterations_run = 0
    
    for iteration in range(config.max_iterations):
        iterations_run = int(iteration + 1)
        # Compute residuals.
        residuals = []
        weights = []
        
        # Depth residual.
        if log_z_param is not None and z_gt_tensor is not None:
            z_pred = torch.exp(log_z_param[0]).clamp(min=1e-6)
            r_depth = z_pred - z_gt_tensor
            residuals.append(r_depth)
            weights.append(config.w_depth)
        
        # Rotation residual.
        if rot6d_param is not None and r_gt_tensor is not None:
            r_pred = _rot6d_to_matrix_torch(rot6d_param.unsqueeze(0)).squeeze(0)
            geo_dist = _geodesic_distance_torch(r_pred.unsqueeze(0), r_gt_tensor.unsqueeze(0)).squeeze(0)
            residuals.append(geo_dist)
            weights.append(config.w_rotation)
        
        # Offset residual (regularization to keep offsets small).
        if offsets_param is not None:
            r_offsets = (offsets_param * offsets_param).sum().sqrt()
            residuals.append(r_offsets)
            weights.append(config.w_offsets * 0.01)  # Weak regularization.
        
        # Plane constraint (if using geometric constraints).
        if config.use_geometric_constraints and plane_constraint is not None:
            # Plane constraint would need translation which requires intrinsics.
            # Skip for now as this is a simplified implementation.
            pass
        
        # Upright constraint.
        if config.use_geometric_constraints and upright_bounds is not None and rot6d_param is not None:
            r_pred = _rot6d_to_matrix_torch(rot6d_param.unsqueeze(0)).squeeze(0)
            roll = torch.atan2(r_pred[2, 1], r_pred[2, 2])
            pitch = torch.asin(torch.clamp(-r_pred[2, 0], -1.0, 1.0))
            
            (roll_min, roll_max), (pitch_min, pitch_max) = upright_bounds
            roll_penalty = torch.clamp(roll - roll_max, min=0) + torch.clamp(roll_min - roll, min=0)
            pitch_penalty = torch.clamp(pitch - pitch_max, min=0) + torch.clamp(pitch_min - pitch, min=0)
            r_upright = roll_penalty + pitch_penalty
            residuals.append(r_upright)
            weights.append(config.w_upright)
        
        if not residuals:
            final_residual = 0.0
            break
        
        n_params = int(sum(int(p.numel()) for p in params))
        
        # Ensure all residuals are scalar and weighted.
        scalar_residuals = []
        for r, w in zip(residuals, weights):
            # Ensure residual is scalar.
            if r.numel() > 1:
                r_scalar = r.mean()  # Reduce to scalar if needed.
            else:
                r_scalar = r.squeeze()
            scalar_residuals.append(w * r_scalar)
        
        # Total loss for convergence check.
        total_residual = sum(r * r for r in scalar_residuals)
        final_residual = float(total_residual.sqrt().item())
        
        # Compute gradients for each residual.
        J_rows = []
        for r in scalar_residuals:
            # Compute gradient of this residual w.r.t. all parameters.
            grads = torch.autograd.grad(r, params, retain_graph=True, create_graph=False, allow_unused=True)
            grad_row = torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten() 
                                 for g, p in zip(grads, params)])
            J_rows.append(grad_row)
        
        if not J_rows:
            break
        
        J = torch.stack(J_rows, dim=0)  # [n_residuals, n_params]
        r_vec = torch.stack([r for r in scalar_residuals], dim=0)  # [n_residuals]
        
        # Gauss-Newton with Levenberg-Marquardt damping:
        # (J^T J + λI) Δp = -J^T r
        JtJ = torch.matmul(J.t(), J)
        Jtr = torch.matmul(J.t(), r_vec)
        
        # Add damping.
        damping_matrix = config.damping * torch.eye(n_params, dtype=dtype, device=device)
        A = JtJ + damping_matrix
        
        # Solve for parameter update.
        try:
            delta_p = torch.linalg.solve(A, -Jtr)
        except RuntimeError:
            # Singular matrix, stop.
            break
        
        # Check convergence.
        update_norm = delta_p.norm().item()
        
        # Check for NaN.
        if math.isnan(update_norm) or math.isnan(final_residual):
            # Numerical instability, stop refinement.
            break
        
        if update_norm < config.convergence_threshold:
            converged = True
            break
        
        # Update parameters (detach and re-create to avoid autograd graph explosion).
        offset = 0
        new_params = []
        for i, param in enumerate(params):
            size = param.numel()
            delta = delta_p[offset:offset + size].reshape(param.shape)
            updated = param.data + delta
            new_param = updated.clone().detach().requires_grad_(True)
            new_params.append(new_param)
            offset += size
        
        params = new_params
        # Re-assign to individual param variables.
        param_idx = 0
        if "log_z" in param_types:
            log_z_param = params[param_idx]
            param_idx += 1
        if "rot6d" in param_types:
            rot6d_param = params[param_idx]
            param_idx += 1
        if "offsets" in param_types:
            offsets_param = params[param_idx]
            param_idx += 1
    
    # Extract refined values.
    refined_log_z = None
    if log_z_param is not None:
        refined_log_z = float(log_z_param[0].detach().cpu().item())
    
    refined_rot6d = None
    if rot6d_param is not None:
        refined_rot6d = [float(x) for x in rot6d_param.detach().cpu().tolist()]
    
    refined_offsets = None
    if offsets_param is not None:
        refined_offsets = [float(x) for x in offsets_param.detach().cpu().tolist()]
    
    # Build result.
    result = RefinementResult(
        log_z=refined_log_z,
        rot6d=refined_rot6d,
        offsets=refined_offsets,
        iterations=int(iterations_run),
        converged=converged,
        final_residual=final_residual,
    )
    
    # Update detection with refined values.
    refined_detection = dict(detection)
    if refined_log_z is not None:
        refined_detection["log_z"] = refined_log_z
    if refined_rot6d is not None:
        refined_detection["rot6d"] = refined_rot6d
    if refined_offsets is not None:
        refined_detection["offsets"] = refined_offsets
    
    # Add refinement metadata.
    refined_detection["hessian_refinement"] = {
        "iterations": result.iterations,
        "converged": result.converged,
        "final_residual": result.final_residual,
    }
    
    return refined_detection


def refine_predictions_hessian(
    predictions: list[dict[str, Any]],
    records: list[dict[str, Any]] | None = None,
    *,
    config: HessianSolverConfig,
) -> list[dict[str, Any]]:
    """Refine predictions using Hessian-based optimization.
    
    This function applies per-detection refinement to all predictions.
    If records are provided, uses available ground truth for supervision.
    
    Args:
        predictions: List of prediction dictionaries with 'image' and 'detections'.
        records: Optional list of dataset records with ground truth.
        config: Solver configuration.
        
    Returns:
        List of refined predictions.
    """
    # Build record index if provided.
    record_index: dict[str, dict[str, Any]] = {}
    if records is not None:
        for record in records:
            if not isinstance(record, dict):
                continue
            image = record.get("image")
            if image:
                record_index[str(image)] = record
                # Also index by basename.
                base = _basename_key(image)
                if base and base not in record_index:
                    record_index[base] = record
    
    refined_predictions = []
    
    for pred_entry in predictions:
        if not isinstance(pred_entry, dict):
            refined_predictions.append(pred_entry)
            continue
        
        image = pred_entry.get("image")
        dets = pred_entry.get("detections")
        if not isinstance(dets, list):
            refined_predictions.append(pred_entry)
            continue
        
        # Find corresponding record if available.
        record = None
        if image and record_index:
            record = record_index.get(str(image))
            if record is None:
                base = _basename_key(image)
                record = record_index.get(base)
        
        # Refine each detection.
        refined_dets = []
        for det in dets:
            if not isinstance(det, dict):
                refined_dets.append(det)
                continue
            
            # Extract supervision from record if available.
            # NOTE: This is a simplified matching strategy that uses the first GT instance.
            # For production use with multi-object scenes, implement proper detection-to-GT
            # matching (e.g., IoU-based Hungarian matching or class_id + spatial proximity).
            gt_depth = None
            gt_rotation = None
            
            if record is not None:
                labels = record.get("labels")
                if isinstance(labels, list) and labels:
                    # Try to match by class_id first, otherwise use first label.
                    matched_label = None
                    det_class = det.get("class_id")
                    
                    if det_class is not None:
                        # Find first label with matching class_id.
                        for label in labels:
                            if isinstance(label, dict) and label.get("class_id") == det_class:
                                matched_label = label
                                break
                    
                    # Fallback to first label if no class match.
                    if matched_label is None and labels:
                        matched_label = labels[0]
                    
                    if matched_label is not None and isinstance(matched_label, dict):
                        # Try to extract depth from t_gt.
                        t_gt = matched_label.get("t_gt")
                        if isinstance(t_gt, (list, tuple)) and len(t_gt) >= 3:
                            try:
                                gt_depth = float(t_gt[2])  # Z component.
                            except (TypeError, ValueError):
                                gt_depth = None
                        
                        # Extract rotation.
                        r_gt = matched_label.get("R_gt")
                        if isinstance(r_gt, (list, tuple)) and len(r_gt) == 3:
                            try:
                                gt_rotation = [[float(x) for x in row] for row in r_gt]
                            except (TypeError, ValueError):
                                gt_rotation = None
            
            # Refine detection.
            refined_det = refine_detection_hessian(
                det,
                config=config,
                gt_depth=gt_depth,
                gt_rotation=gt_rotation,
            )
            refined_dets.append(refined_det)
        
        # Build refined entry.
        refined_entry = dict(pred_entry)
        refined_entry["detections"] = refined_dets
        refined_predictions.append(refined_entry)
    
    return refined_predictions
