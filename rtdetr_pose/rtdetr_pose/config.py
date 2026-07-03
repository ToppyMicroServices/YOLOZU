import json
from dataclasses import dataclass, field
import importlib.resources
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    root: str
    split: str = "train2017"
    format: str = "yolo"


@dataclass
class ModelConfig:
    num_classes: int = 80
    num_keypoints: int = 0
    enable_mim: bool = False
    mim_geom_channels: int = 2
    depth_mode: str = "none"
    depth_dropout: float = 0.0
    hidden_dim: int = 256
    num_queries: int = 300
    use_uncertainty: bool = False
    backbone_name: str = "cspresnet"
    stem_channels: int = 32
    backbone_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    stage_blocks: list[int] = field(default_factory=lambda: [1, 2, 2])
    use_sppf: bool = True
    use_level_embed: bool = True
    num_encoder_layers: int = 1
    num_decoder_layers: int = 3
    nhead: int = 8
    encoder_dim_feedforward: int | None = None
    decoder_dim_feedforward: int | None = None
    activation_preset: str = "default"
    backbone_activation: str = "silu"
    head_activation: str = "silu"
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)
    backbone_norm: str = "bn"
    backbone: dict[str, Any] = field(default_factory=dict)
    projector: dict[str, Any] = field(default_factory=dict)
    graph_refine: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "default"
    task_aligner: str = "none"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainConfig:
    batch_size: int = 2
    lr: float = 1e-4
    epochs: int = 1


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(path):
    text = None
    if isinstance(path, str):
        value = str(path)
        if value.startswith("builtin:") or value.startswith("pkg:"):
            name = value.split(":", 1)[1].strip()
            if not name:
                raise ValueError("builtin config name is empty")
            if not name.endswith(".json"):
                name = f"{name}.json"
            rel = name if "/" in name else f"configs/{name}"
            try:
                text = (
                    importlib.resources.files("rtdetr_pose")
                    .joinpath(rel)
                    .read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise FileNotFoundError(f"builtin config not found: {rel}") from exc

    if text is None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")

    data = json.loads(text)
    dataset = DatasetConfig(**data["dataset"])
    model = ModelConfig(**data.get("model", {}))
    train = TrainConfig(**data.get("train", {}))
    loss = LossConfig(**data.get("loss", {}))
    return Config(dataset=dataset, model=model, train=train, loss=loss)
