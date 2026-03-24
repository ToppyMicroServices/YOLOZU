import argparse
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a tiny YOLO-format dataset for CI smoke.")
    p.add_argument("--out", required=True, help="Dataset root to create (will write images/ + labels/).")
    p.add_argument("--split", default="val", help="Split name under images/ and labels/ (default: val).")
    p.add_argument("--image-stem", default="000001", help="Image/label stem (default: 000001).")
    p.add_argument("--hw", default="48x64", help="Image size HxW (default: 48x64).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out)

    try:
        h_str, w_str = str(args.hw).replace(",", "x").lower().split("x", 1)
        h, w = int(h_str), int(w_str)
    except ValueError as exc:
        raise SystemExit(f"invalid --hw: {args.hw!r} (expected HxW, e.g. 48x64)") from exc

    from PIL import Image
    import numpy as np

    images = out_root / "images" / str(args.split)
    labels = out_root / "labels" / str(args.split)
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 255
    Image.fromarray(img).save(images / f"{args.image_stem}.jpg", quality=90)
    (labels / f"{args.image_stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    print("wrote", str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

