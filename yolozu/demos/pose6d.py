from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg"

ARUCO_DEFAULT_DICT = "DICT_4X4_50"


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _require_deps() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo pose requires opencv-python and numpy") from exc
    return cv2, np


def _make_chessboard_image(*, np: Any, cv2: Any, pattern_cols: int, pattern_rows: int, square_px: int) -> Any:
    board_cols = pattern_cols + 1
    board_rows = pattern_rows + 1
    width = board_cols * square_px
    height = board_rows * square_px
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(board_rows):
        for x in range(board_cols):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * square_px, y * square_px),
                    ((x + 1) * square_px, (y + 1) * square_px),
                    255,
                    -1,
                )
    return img


def _warp_to_canvas(*, np: Any, cv2: Any, img: Any, canvas_w: int, canvas_h: int) -> Any:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [int(canvas_w * 0.20), int(canvas_h * 0.15)],
            [int(canvas_w * 0.78), int(canvas_h * 0.12)],
            [int(canvas_w * 0.85), int(canvas_h * 0.82)],
            [int(canvas_w * 0.12), int(canvas_h * 0.86)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (canvas_w, canvas_h), borderValue=255)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def _download_sample(*, target: Path) -> bool:
    import urllib.request

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, str(target))
    except Exception:
        return False
    return target.exists()


def _ensure_aruco_support(cv2: Any) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("demo pose (aruco) requires opencv-contrib-python")
    return cv2.aruco


def _get_aruco_dict(*, aruco: Any, name: str) -> Any:
    key = str(name).strip().upper()
    if not key.startswith("DICT_"):
        key = f"DICT_{key}"
    dict_id = getattr(aruco, key, None)
    if dict_id is None:
        raise RuntimeError(f"unknown aruco dictionary: {name}")
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def _make_aruco_image(*, cv2: Any, aruco: Any, dict_name: str, marker_id: int, marker_px: int) -> Any:
    aruco_dict = _get_aruco_dict(aruco=aruco, name=dict_name)
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(aruco_dict, int(marker_id), int(marker_px))
    else:
        marker = aruco.drawMarker(aruco_dict, int(marker_id), int(marker_px))
    canvas = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return canvas


def _make_textured_bg(*, np: Any, cv2: Any, width: int, height: int) -> Any:
    rng = np.random.default_rng(1234)
    noise = rng.normal(loc=210.0, scale=18.0, size=(height, width, 1)).astype("float32")
    noise = np.clip(noise, 0.0, 255.0)
    grad = np.linspace(0.9, 1.1, height, dtype="float32").reshape(height, 1, 1)
    base = noise * grad
    base = np.clip(base, 0.0, 255.0)
    base = np.repeat(base, 3, axis=2)
    tint = np.array([0.96, 0.98, 1.02], dtype="float32").reshape(1, 1, 3)
    base = np.clip(base * tint, 0.0, 255.0).astype("uint8")
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    return base


def _warp_marker_to_bg(*, np: Any, cv2: Any, marker: Any, canvas_w: int, canvas_h: int) -> tuple[Any, Any]:
    h, w = marker.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [int(canvas_w * 0.22), int(canvas_h * 0.20)],
            [int(canvas_w * 0.72), int(canvas_h * 0.15)],
            [int(canvas_w * 0.78), int(canvas_h * 0.78)],
            [int(canvas_w * 0.18), int(canvas_h * 0.80)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(marker, M, (canvas_w, canvas_h), borderValue=255)
    mask = cv2.warpPerspective(
        np.ones((h, w), dtype="uint8") * 255,
        M,
        (canvas_w, canvas_h),
        borderValue=0,
    )
    return warped, mask


def _ensure_sample_image(
    *,
    cv2: Any,
    np: Any,
    sample_dir: Path,
    pattern_cols: int,
    pattern_rows: int,
    sample_source: str,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"chessboard_{pattern_cols}x{pattern_rows}.png"
    if sample_source in ("auto", "download"):
        downloaded_path = sample_dir / "chessboard_download.jpg"
        if downloaded_path.exists():
            return downloaded_path
        if _download_sample(target=downloaded_path):
            return downloaded_path
        if sample_source == "download":
            return downloaded_path
    if sample_path.exists():
        return sample_path
    base = _make_chessboard_image(np=np, cv2=cv2, pattern_cols=pattern_cols, pattern_rows=pattern_rows, square_px=40)
    canvas = _warp_to_canvas(np=np, cv2=cv2, img=base, canvas_w=640, canvas_h=480)
    cv2.imwrite(str(sample_path), canvas)
    return sample_path


def _ensure_aruco_sample(
    *,
    cv2: Any,
    np: Any,
    sample_dir: Path,
    dict_name: str,
    marker_id: int,
    sample_source: str,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"aruco_{dict_name.lower()}_{marker_id}.png"
    if sample_path.exists():
        return sample_path
    aruco = _ensure_aruco_support(cv2)
    marker = _make_aruco_image(cv2=cv2, aruco=aruco, dict_name=dict_name, marker_id=marker_id, marker_px=260)
    canvas = _make_textured_bg(np=np, cv2=cv2, width=640, height=480)
    warped, mask = _warp_marker_to_bg(np=np, cv2=cv2, marker=marker, canvas_w=640, canvas_h=480)

    shadow = cv2.dilate(mask, np.ones((9, 9), dtype="uint8"), iterations=1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 9.0)
    shadow = (shadow.astype("float32") / 255.0) * 60.0
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c].astype("float32") - shadow, 0.0, 255.0).astype("uint8")

    inv = cv2.bitwise_not(mask)
    canvas = cv2.bitwise_and(canvas, canvas, mask=inv)
    canvas = cv2.add(canvas, warped)
    cv2.imwrite(str(sample_path), canvas)
    return sample_path


def _default_camera_matrix(*, np: Any, width: int, height: int) -> Any:
    fx = float(max(width, height))
    fy = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotmat_to_rot6d(R: Any) -> list[float]:
    r = R.astype(float)
    a1 = r[:, 0].tolist()
    a2 = r[:, 1].tolist()
    return [float(v) for v in (a1 + a2)]


def _find_corners(*, cv2: Any, img_gray: Any, pattern_size: tuple[int, int] | None) -> tuple[bool, Any, tuple[int, int] | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    candidates = [(9, 6), (7, 6), (6, 5), (5, 4)]
    if pattern_size is not None:
        candidates = [pattern_size]
    for cols, rows in candidates:
        found, corners = cv2.findChessboardCorners(img_gray, (cols, rows), flags)
        if found:
            return True, corners, (cols, rows)
    return False, None, None


def run_pose6d_demo(
    *,
    image: str | None = None,
    run_dir: str | None = None,
    backend: str = "chessboard",
    densefusion_root: str | None = None,
    densefusion_object: str = "ape",
    densefusion_auto_download: bool = True,
    densefusion_model: str | None = None,
    densefusion_refine_model: str | None = None,
    pattern_cols: int | None = None,
    pattern_rows: int | None = None,
    square_size: float = 0.04,
    camera_fx: float | None = None,
    camera_fy: float | None = None,
    camera_cx: float | None = None,
    camera_cy: float | None = None,
    sample_source: str = "auto",
    aruco_dict: str = ARUCO_DEFAULT_DICT,
    aruco_id: int = 23,
    marker_length: float = 0.05,
) -> str:
    cv2, np = _require_deps()
    backend_name = str(backend).strip().lower()
    aruco_dict_name = str(aruco_dict)

    if run_dir:
        run_dir_p = Path(run_dir)
    else:
        run_dir_p = Path("demo_output") / "pose" / _utc_run_id()
    run_dir_p.mkdir(parents=True, exist_ok=True)

    if backend_name == "densefusion":
        from yolozu.demos.densefusion_demo import run_densefusion_demo

        return run_densefusion_demo(
            run_dir=str(run_dir_p),
            image=image,
            object_name=str(densefusion_object),
            auto_download=bool(densefusion_auto_download),
            densefusion_root=densefusion_root,
            model_path=densefusion_model,
            refine_model_path=densefusion_refine_model,
        )

    img_path = Path(image) if image else None
    if img_path is None:
        if backend_name == "aruco":
            img_path = _ensure_aruco_sample(
                cv2=cv2,
                np=np,
                sample_dir=Path("demo_output") / "pose" / "_samples",
                dict_name=str(aruco_dict),
                marker_id=int(aruco_id),
                sample_source=str(sample_source),
            )
        else:
            img_path = _ensure_sample_image(
                cv2=cv2,
                np=np,
                sample_dir=Path("demo_output") / "pose" / "_samples",
                pattern_cols=int(pattern_cols) if pattern_cols else 6,
                pattern_rows=int(pattern_rows) if pattern_rows else 5,
                sample_source=str(sample_source),
            )

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"pose demo: failed to read image: {img_path}")
    height, width = img.shape[:2]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if any(v is not None for v in (camera_fx, camera_fy, camera_cx, camera_cy)):
        fx = float(camera_fx) if camera_fx is not None else float(max(width, height))
        fy = float(camera_fy) if camera_fy is not None else float(max(width, height))
        cx = float(camera_cx) if camera_cx is not None else float(width) / 2.0
        cy = float(camera_cy) if camera_cy is not None else float(height) / 2.0
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        camera_matrix = _default_camera_matrix(np=np, width=width, height=height)
    dist = np.zeros((5, 1), dtype=np.float32)
    overlay = img.copy()

    if backend_name == "aruco":
        aruco = _ensure_aruco_support(cv2)
        aruco_dict_obj = _get_aruco_dict(aruco=aruco, name=aruco_dict_name)
        if hasattr(aruco, "DetectorParameters"):
            detector_params = aruco.DetectorParameters()
        else:
            detector_params = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict_obj, parameters=detector_params)
        if ids is None or len(corners) == 0:
            raise RuntimeError("pose demo: aruco markers not found")
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, float(marker_length), camera_matrix, dist)
        rvec = rvecs[0].reshape(3, 1)
        tvec = tvecs[0].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        rot6d = _rotmat_to_rot6d(R)
        aruco.drawDetectedMarkers(overlay, corners, ids)
        axis_len = float(marker_length) * 0.6
        cv2.drawFrameAxes(overlay, camera_matrix, dist, rvec, tvec, axis_len)
        extra = {
            "aruco": {
                "dict": aruco_dict_name,
                "marker_id": int(ids[0][0]) if ids is not None else None,
                "marker_length": float(marker_length),
            }
        }
        num_points = int(len(corners[0]))
    else:
        psize = None
        if pattern_cols is not None and pattern_rows is not None:
            psize = (int(pattern_cols), int(pattern_rows))

        found, corners, resolved_psize = _find_corners(cv2=cv2, img_gray=img_gray, pattern_size=psize)
        if not found or corners is None or resolved_psize is None:
            raise RuntimeError("pose demo: chessboard corners not found; check --pattern-cols/--pattern-rows")

        cols, rows = resolved_psize
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

        objp = np.zeros((rows * cols, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(square_size)

        ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("pose demo: solvePnP failed")

        R, _ = cv2.Rodrigues(rvec)
        rot6d = _rotmat_to_rot6d(R)
        cv2.drawChessboardCorners(overlay, (cols, rows), corners, True)
        axis_len = float(square_size) * max(2.0, min(cols, rows))
        cv2.drawFrameAxes(overlay, camera_matrix, dist, rvec, tvec, axis_len)
        extra = {
            "chessboard": {
                "pattern_cols": int(cols),
                "pattern_rows": int(rows),
                "square_size": float(square_size),
            }
        }
        num_points = int(corners.shape[0])

    image_out = run_dir_p / "pose_input.png"
    overlay_out = run_dir_p / "pose_overlay.png"
    report_out = run_dir_p / "pose_demo_report.json"

    cv2.imwrite(str(image_out), img)
    cv2.imwrite(str(overlay_out), overlay)

    payload = {
        "kind": "pose6d_demo",
        "settings": {
            "image": str(img_path),
            "run_dir": str(run_dir_p),
            "backend": backend_name,
            "pattern_cols": int(pattern_cols) if pattern_cols is not None else None,
            "pattern_rows": int(pattern_rows) if pattern_rows is not None else None,
            "square_size": float(square_size),
            "aruco_dict": aruco_dict_name,
            "aruco_id": int(aruco_id),
            "marker_length": float(marker_length),
            "sample_source": str(sample_source),
            "camera": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
        },
        "result": {
            "found": True,
            "num_points": int(num_points),
            "rvec": [float(v) for v in rvec.reshape(-1).tolist()],
            "tvec": [float(v) for v in tvec.reshape(-1).tolist()],
            "R": [[float(v) for v in row] for row in R.tolist()],
            "rot6d": rot6d,
            "t_xyz": [float(v) for v in tvec.reshape(-1).tolist()],
            "backend_meta": extra,
            "artifacts": {
                "image": str(image_out),
                "overlay": str(overlay_out),
            },
        },
    }

    report_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return str(report_out)
