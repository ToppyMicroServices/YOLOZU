"""Dynamic int8 quantisation of ONNX models.

Uses ONNX Runtime's dynamic quantisation (no calibration data
required) to produce a smaller, faster int8 model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = ["quantize_onnx_dynamic"]


def quantize_onnx_dynamic(
    *,
    onnx_in: str | Path,
    onnx_out: str | Path,
    weight_type: str = "qint8",
    per_channel: bool = False,
    reduce_range: bool = False,
    op_types_to_quantize: Iterable[str] | None = None,
    use_external_data_format: bool = False,
) -> Path:
    """Quantize an ONNX model using ONNXRuntime dynamic quantization.

    This is a lightweight, dependency-friendly option to generate int8-ish ONNX
    artifacts for CPU inference. It does not require torchao and does not run
    calibration data.
    """

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime.quantization is required (install `yolozu[onnxrt]`)") from exc

    onnx_in_path = Path(onnx_in).expanduser()
    if not onnx_in_path.is_absolute():
        onnx_in_path = Path.cwd() / onnx_in_path
    if not onnx_in_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_in_path}")

    onnx_out_path = Path(onnx_out).expanduser()
    if not onnx_out_path.is_absolute():
        onnx_out_path = Path.cwd() / onnx_out_path
    onnx_out_path.parent.mkdir(parents=True, exist_ok=True)

    wt = str(weight_type).strip().lower()
    if wt == "qint8":
        wt_enum = QuantType.QInt8
    elif wt == "quint8":
        wt_enum = QuantType.QUInt8
    else:
        raise ValueError("--weight-type must be one of: qint8, quint8")

    ops = None
    if op_types_to_quantize is not None:
        ops = [str(v) for v in op_types_to_quantize if str(v).strip()]

    quantize_dynamic(
        model_input=str(onnx_in_path),
        model_output=str(onnx_out_path),
        op_types_to_quantize=ops,
        per_channel=bool(per_channel),
        reduce_range=bool(reduce_range),
        weight_type=wt_enum,
        use_external_data_format=bool(use_external_data_format),
    )

    return onnx_out_path

