from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _require_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo train requires torch+torchvision") from exc
    return torch, torchvision, transforms


def run_train_demo(
    *,
    output: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    data_dir: str | Path = str(Path("data") / "torchvision"),
    epochs: int = 1,
    max_steps: int = 80,
    batch_size: int = 64,
    lr: float = 3e-4,
    num_workers: int = 0,
    output_name: str = "train_demo_report.json",
) -> Path:
    """Practical training demo (MNIST + ResNet18 fine-tune).

    - CPU-friendly: bounded by max_steps.
    - Produces: checkpoint.pt + JSON report.
    - Downloads MNIST on first run.
    """

    torch, torchvision, transforms = _require_deps()

    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))

    if output is None:
        run_dir = Path("demo_output") / "train" / _utc_run_id()
    else:
        p = Path(output)
        run_dir = p if p.suffix == "" else p.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    # MNIST -> 1ch 28x28; resize to 224 and repeat channels for ResNet.
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_ds = torchvision.datasets.MNIST(root=str(data_root), train=True, download=True, transform=tfm)
    val_ds = torchvision.datasets.MNIST(root=str(data_root), train=False, download=True, transform=tfm)

    # Keep this bounded and deterministic-ish on CPU.
    train_subset = torch.utils.data.Subset(train_ds, list(range(min(len(train_ds), 4096))))
    val_subset = torch.utils.data.Subset(val_ds, list(range(min(len(val_ds), 1024))))

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )

    torch_device = torch.device(str(device))

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.to(torch_device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.CrossEntropyLoss()

    def _eval_accuracy() -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(torch_device)
                yb = yb.to(torch_device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return float(correct) / float(max(1, total))

    train_losses: list[float] = []
    steps = 0
    model.train()
    start = time.time()
    for _epoch in range(max(1, int(epochs))):
        for xb, yb in train_loader:
            xb = xb.to(torch_device)
            yb = yb.to(torch_device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu().item()))
            steps += 1
            if steps >= int(max_steps):
                break
        if steps >= int(max_steps):
            break

    train_sec = float(time.time() - start)

    val_acc = _eval_accuracy()

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "seed": int(seed),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "steps": int(steps),
        },
        ckpt_path,
    )

    payload = {
        "kind": "train_demo",
        "schema_version": 1,
        "settings": {
            "seed": int(seed),
            "device": str(torch_device),
            "data_dir": str(data_root),
            "epochs": int(epochs),
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "run_dir": str(run_dir),
        },
        "meta": {
            "backend": "torchvision.resnet18_mnist_finetune",
            "torch": getattr(torch, "__version__", None),
            "torchvision": getattr(torchvision, "__version__", None),
        },
        "result": {
            "train": {
                "steps": int(steps),
                "loss_mean": float(sum(train_losses) / max(1, len(train_losses))),
                "loss_last": float(train_losses[-1]) if train_losses else None,
                "wall_sec": train_sec,
            },
            "val": {
                "acc": float(val_acc),
                "samples": int(len(val_subset)),
            },
            "artifacts": {
                "checkpoint": str(ckpt_path),
            },
        },
    }

    out_path = Path(output) if (output is not None and Path(output).suffix) else (run_dir / str(output_name))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
