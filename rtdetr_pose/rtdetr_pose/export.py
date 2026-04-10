from __future__ import annotations


def export_onnx(
    model,
    dummy_input,
    output_path,
    *,
    opset_version: int = 18,
    input_name: str = "images",
    dynamic_hw: bool = False,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for export") from exc
    try:
        import onnx  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnx is required for torch.onnx.export (pip install onnx)") from exc

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model, output_keys: list[str]):
            super().__init__()
            self.model = model
            self.output_keys = list(output_keys)

        def forward(self, x):
            out = self.model(x)
            return tuple(out[k] for k in self.output_keys)

    with torch.no_grad():
        sample_out = model(dummy_input)
    if not isinstance(sample_out, dict):
        raise RuntimeError("export_onnx expects model(dummy_input) to return a dict of tensors")

    canonical_keys = [
        "logits",
        "bbox",
        "log_z",
        "rot6d",
        "offsets",
        "k_delta",
        "keypoints",
        "log_sigma_z",
        "log_sigma_rot",
    ]
    output_keys = [
        key
        for key in canonical_keys
        if key in sample_out and isinstance(sample_out[key], torch.Tensor)
    ]
    if not output_keys:
        raise RuntimeError("export_onnx found no tensor outputs to export")

    wrapper = ExportWrapper(model, output_keys).eval()
    input_name = str(input_name) if input_name else "images"
    input_dynamic_axes: dict[int, str] = {0: "batch"}
    if bool(dynamic_hw):
        input_dynamic_axes[2] = "height"
        input_dynamic_axes[3] = "width"
    dyn_axes = {
        input_name: input_dynamic_axes,
    }
    for key in output_keys:
        tensor = sample_out[key]
        if int(tensor.ndim) >= 1:
            dyn_axes[key] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=[input_name],
        output_names=output_keys,
        opset_version=int(opset_version),
        dynamic_axes=dyn_axes,
    )
