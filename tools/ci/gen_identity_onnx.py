import argparse
from pathlib import Path


def _parse_shape(value: str) -> list[int]:
    raw = str(value).replace(",", "x").lower()
    parts = [p.strip() for p in raw.split("x") if p.strip()]
    if not parts:
        raise ValueError(f"invalid shape: {value!r}")
    dims = [int(p) for p in parts]
    if any(d <= 0 for d in dims):
        raise ValueError(f"invalid shape: {value!r}")
    return dims


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a tiny identity ONNX model for TRT smoke tests.")
    p.add_argument("--out", required=True, help="Where to write the ONNX model.")
    p.add_argument("--input-name", default="images", help="Input tensor name.")
    p.add_argument("--output-name", default="output0", help="Output tensor name.")
    p.add_argument("--shape", default="1x3x64x64", help="Tensor shape (e.g. 1x3x64x64).")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset to target.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import onnx
    from onnx import TensorProto, helper

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shape = _parse_shape(str(args.shape))
    input_name = str(args.input_name)
    output_name = str(args.output_name)

    x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", inputs=[input_name], outputs=[output_name])
    graph = helper.make_graph([node], "identity_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", int(args.opset))])
    onnx.save(model, str(out_path))

    print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
