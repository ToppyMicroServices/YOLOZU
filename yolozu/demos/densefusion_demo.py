from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _require_torch_cuda() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("densefusion demo requires torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("densefusion demo requires a CUDA-enabled torch build")
    return torch


def _ensure_repo(*, repo_dir: Path, auto_download: bool) -> Path:
    if repo_dir.exists():
        return repo_dir
    if not auto_download:
        raise RuntimeError(f"densefusion repo not found: {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "https://github.com/hz-ants/DenseFusion.git", str(repo_dir)]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone DenseFusion failed: {proc.stderr.strip()}")
    return repo_dir


def _ensure_assets(*, repo_dir: Path, auto_download: bool) -> tuple[Path, Path, Path]:
    dataset_root = repo_dir / "datasets" / "linemod" / "Linemod_preprocessed"
    ckpt_dir = repo_dir / "trained_checkpoints" / "linemod"
    if dataset_root.exists() and ckpt_dir.exists():
        return dataset_root, ckpt_dir, repo_dir
    if not auto_download:
        raise RuntimeError("densefusion assets missing; run DenseFusion download.sh first")
    script = repo_dir / "download.sh"
    if not script.exists():
        raise RuntimeError(f"densefusion download.sh not found: {script}")
    proc = subprocess.run(["bash", str(script)], cwd=str(repo_dir), check=False)
    if proc.returncode != 0:
        raise RuntimeError("densefusion download.sh failed")
    if not dataset_root.exists() or not ckpt_dir.exists():
        raise RuntimeError("densefusion assets not found after download.sh")
    return dataset_root, ckpt_dir, repo_dir


def _object_id_from_name(name: str) -> int:
    mapping = {
        "ape": 1,
        "benchvise": 2,
        "bowl": 3,
        "camera": 4,
        "can": 5,
        "cat": 6,
        "cup": 7,
        "driller": 8,
        "duck": 9,
        "eggbox": 10,
        "glue": 11,
        "holepuncher": 12,
        "iron": 13,
        "lamp": 14,
        "phone": 15,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise RuntimeError(f"unknown linemod object: {name}")
    return mapping[key]


def _project_points(*, pts: Any, fx: float, fy: float, cx: float, cy: float) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for x, y, z in pts:
        if z <= 1e-9:
            continue
        u = int(fx * (x / z) + cx)
        v = int(fy * (y / z) + cy)
        out.append((u, v))
    return out


def _bbox_corners(pts: Any) -> list[list[float]]:
    import numpy as np

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    return [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]


def _draw_bbox(*, cv2: Any, img: Any, corners_2d: list[tuple[int, int]]) -> None:
    if len(corners_2d) < 8:
        return
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        cv2.line(img, corners_2d[i], corners_2d[j], (0, 255, 0), 2)


def run_densefusion_demo(
    *,
    run_dir: str,
    image: str | None,
    object_name: str,
    auto_download: bool,
    densefusion_root: str | None,
    model_path: str | None,
    refine_model_path: str | None,
) -> str:
    torch = _require_torch_cuda()
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("densefusion demo requires opencv-python and numpy") from exc

    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(densefusion_root) if densefusion_root else Path("demo_output") / "pose" / "_densefusion"
    repo_dir = _ensure_repo(repo_dir=repo_dir, auto_download=auto_download)
    dataset_root, ckpt_dir, repo_dir = _ensure_assets(repo_dir=repo_dir, auto_download=auto_download)

    if model_path is None:
        model_path = str(ckpt_dir / "pose_model_9_0.01310166542980859.pth")
    if refine_model_path is None:
        refine_model_path = str(ckpt_dir / "pose_refine_model_493_0.006761023565178073.pth")

    sys.path.insert(0, str(repo_dir))

    from datasets.linemod.dataset import PoseDataset as PoseDatasetLinemod  # type: ignore
    from lib.network import PoseNet, PoseRefineNet  # type: ignore
    from lib.transformations import quaternion_matrix, quaternion_from_matrix  # type: ignore

    dataset = PoseDatasetLinemod("eval", 500, False, str(dataset_root), 0.0, True)
    obj_id = _object_id_from_name(object_name)

    index = None
    for i, obj in enumerate(dataset.list_obj):
        if int(obj) == int(obj_id):
            index = i
            break
    if index is None:
        raise RuntimeError(f"object {object_name} not found in dataset")

    points, choose, img, target, model_points, idx = dataset[index]
    if len(points.size()) == 2:
        raise RuntimeError("densefusion demo: detection missing for sample")

    estimator = PoseNet(num_points=500, num_obj=13).cuda().eval()
    refiner = PoseRefineNet(num_points=500, num_obj=13).cuda().eval()
    estimator.load_state_dict(torch.load(model_path))
    refiner.load_state_dict(torch.load(refine_model_path))

    points = points.cuda()
    choose = choose.cuda()
    img = img.cuda()
    idx = idx.cuda()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, 500, 1)
    pred_c = pred_c.view(1, 500)
    _, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(500, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).detach().cpu().numpy()
    my_t = (points.view(500, 1, 3) + pred_t)[which_max[0]].view(-1).detach().cpu().numpy()

    for _ in range(2):
        T = torch.from_numpy(my_t.astype(np.float32)).cuda().view(1, 3).repeat(500, 1).view(1, 500, 3)
        my_mat = quaternion_matrix(my_r)
        R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).detach().cpu().numpy()
        my_t_2 = pred_t.view(-1).detach().cpu().numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2
        my_mat = my_mat @ my_mat_2
        my_r = quaternion_from_matrix(my_mat, True)
        my_t = np.array([my_mat[0][3], my_mat[1][3], my_mat[2][3]])

    R = quaternion_matrix(my_r)[:3, :3]
    model_pts = model_points.detach().cpu().numpy()
    pred_pts = (model_pts @ R.T) + my_t

    img_path = Path(dataset.list_rgb[index])
    rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError("densefusion demo: failed to read rgb image")

    fx = float(dataset.cam_fx)
    fy = float(dataset.cam_fy)
    cx = float(dataset.cam_cx)
    cy = float(dataset.cam_cy)

    bbox3d = _bbox_corners(pred_pts)
    bbox2d = _project_points(pts=bbox3d, fx=fx, fy=fy, cx=cx, cy=cy)

    overlay = rgb.copy()
    _draw_bbox(cv2=cv2, img=overlay, corners_2d=bbox2d)
    axis_len = float(np.linalg.norm(pred_pts.max(axis=0) - pred_pts.min(axis=0))) * 0.2
    if axis_len <= 0:
        axis_len = 0.05
    axes = np.array(
        [
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ]
    )
    axes = (axes @ R.T) + my_t
    axes2d = _project_points(pts=axes, fx=fx, fy=fy, cx=cx, cy=cy)
    if len(axes2d) >= 4:
        cv2.line(overlay, axes2d[0], axes2d[1], (0, 0, 255), 2)
        cv2.line(overlay, axes2d[0], axes2d[2], (0, 255, 0), 2)
        cv2.line(overlay, axes2d[0], axes2d[3], (255, 0, 0), 2)

    image_out = run_dir_p / "pose_input.png"
    overlay_out = run_dir_p / "pose_overlay.png"
    report_out = run_dir_p / "pose_demo_report.json"

    cv2.imwrite(str(image_out), rgb)
    cv2.imwrite(str(overlay_out), overlay)

    payload = {
        "kind": "pose6d_demo",
        "settings": {
            "backend": "densefusion",
            "run_dir": str(run_dir_p),
            "image": str(img_path),
            "object": object_name,
            "densefusion_root": str(repo_dir),
            "dataset_root": str(dataset_root),
            "model": str(model_path),
            "refine_model": str(refine_model_path),
            "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        },
        "result": {
            "r_quat": [float(v) for v in my_r],
            "t_xyz": [float(v) for v in my_t.tolist()],
            "artifacts": {"image": str(image_out), "overlay": str(overlay_out)},
        },
    }

    report_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return str(report_out)
