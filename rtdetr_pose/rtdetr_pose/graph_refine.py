from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - torch is optional at package import time
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)
    F = SimpleNamespace()


SUPPORTED_GRAPH_REFINERS = ("gcnv2", "gcnv3")


def normalize_graph_refine_config(value: dict | None) -> dict:
    cfg = dict(value) if isinstance(value, dict) else {}
    default_mode = cfg.get("version", "none") if bool(cfg.get("enabled", False)) else "none"
    mode = str(cfg.get("mode", cfg.get("name", default_mode)) or "none").strip().lower().replace("-", "_")
    version = str(cfg.get("version", mode) or mode).strip().lower().replace("-", "_")

    if mode in ("", "none", "off", "false", "disabled"):
        return {"enabled": False}
    if mode in ("gcn", "graph"):
        version = "gcnv2"
    if version in ("v2", "gcn_v2"):
        version = "gcnv2"
    if version in ("v3", "gcn_v3"):
        version = "gcnv3"
    if version not in SUPPORTED_GRAPH_REFINERS:
        raise ValueError(
            f"unsupported model.graph_refine version: {version} "
            f"(supported: {', '.join(SUPPORTED_GRAPH_REFINERS)})"
        )

    layers = int(cfg.get("layers", 1) or 1)
    if layers < 1:
        raise ValueError("model.graph_refine.layers must be >= 1")
    topk = int(cfg.get("topk", 0) or 0)
    if topk < 0:
        raise ValueError("model.graph_refine.topk must be >= 0")
    dropout = float(cfg.get("dropout", 0.0) or 0.0)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("model.graph_refine.dropout must be >= 0 and < 1")

    return {
        "enabled": True,
        "version": version,
        "layers": layers,
        "topk": topk,
        "dropout": dropout,
    }


def _masked_attention(sim, topk: int):
    if topk <= 0 or topk >= int(sim.shape[-1]):
        return F.softmax(sim, dim=-1)

    values, indices = torch.topk(sim, k=int(topk), dim=-1)
    masked = torch.full_like(sim, float("-inf"))
    masked = masked.scatter(-1, indices, values)
    return F.softmax(masked, dim=-1)


class _GraphConvV2Block(nn.Module):
    def __init__(self, hidden_dim: int, *, topk: int = 0, dropout: float = 0.0):
        super().__init__()
        self.topk = int(topk)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.message = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.update = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )

    def forward(self, x):
        h = self.norm(x)
        sim = torch.matmul(h, h.transpose(-1, -2)) / math.sqrt(float(h.shape[-1]))
        weights = _masked_attention(sim, self.topk)
        agg = torch.matmul(weights, self.message(h))
        return x + self.update(torch.cat([h, agg], dim=-1))


class _GraphConvV3Block(nn.Module):
    def __init__(self, hidden_dim: int, *, topk: int = 0, dropout: float = 0.0):
        super().__init__()
        self.topk = int(topk)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.message = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.gate = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.update = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )

    def forward(self, x):
        h = self.norm(x)
        sim = torch.matmul(h, h.transpose(-1, -2)) / math.sqrt(float(h.shape[-1]))
        weights = _masked_attention(sim, self.topk)
        agg = torch.matmul(weights, self.message(h))
        gate = torch.sigmoid(self.gate(torch.cat([h, agg], dim=-1)))
        mixed = h + gate * agg
        return x + self.update(mixed)


class QueryGraphRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        version: str,
        layers: int = 1,
        topk: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for QueryGraphRefiner")
        version = str(version).lower()
        if version == "gcnv2":
            block_cls = _GraphConvV2Block
        elif version == "gcnv3":
            block_cls = _GraphConvV3Block
        else:
            raise ValueError(f"unsupported graph refiner: {version}")
        self.version = version
        self.blocks = nn.ModuleList(
            [block_cls(int(hidden_dim), topk=int(topk), dropout=float(dropout)) for _ in range(int(layers))]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
