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
    p = argparse.ArgumentParser(description="Generate a tiny ONNX model that outputs a dummy (1,N,6) det tensor.")
    p.add_argument("--out", required=True, help="Where to write the ONNX model.")
    p.add_argument("--input-name", default="images", help="Input tensor name.")
    p.add_argument("--output-name", default="output0", help="Output tensor name.")
    p.add_argument("--shape", default="1x3x64x64", help="Input tensor shape (e.g. 1x3x64x64).")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset to target (default: 17).")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Small scale for input-dependent delta added to constant dets (keeps input from being pruned).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import onnx
    from onnx import TensorProto, helper

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_name = str(args.input_name)
    output_name = str(args.output_name)
    input_shape = _parse_shape(str(args.shape))

    # Constant detections: (1,2,6) in xyxy_score_class (normalized xyxy).
    # Values are chosen away from 0/1 so adding a small delta keeps them valid.
    const_vals = [
        0.2,
        0.2,
        0.8,
        0.8,
        0.9,
        0.0,
        0.1,
        0.1,
        0.3,
        0.3,
        0.5,
        1.0,
    ]

    x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 2, 6])

    init_const = helper.make_tensor("const_dets", TensorProto.FLOAT, [1, 2, 6], const_vals)
    init_alpha = helper.make_tensor("alpha", TensorProto.FLOAT, [], [float(args.alpha)])

    mean = "mean"
    delta = "delta"

    reduce = helper.make_node(
        "ReduceMean",
        inputs=[input_name],
        outputs=[mean],
        axes=list(range(len(input_shape))),
        keepdims=0,
    )
    mul = helper.make_node("Mul", inputs=[mean, "alpha"], outputs=[delta])
    add = helper.make_node("Add", inputs=["const_dets", delta], outputs=[output_name])

    graph = helper.make_graph([reduce, mul, add], "dummy_dets_graph", [x], [y], initializer=[init_const, init_alpha])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", int(args.opset))])
    onnx.save(model, str(out_path))

    print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

