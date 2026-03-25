try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(functional=SimpleNamespace(normalize=None))


def rot6d_to_matrix(x):
    if torch is None or getattr(nn, "functional", None) is None:
        raise RuntimeError("torch is required for rot6d_to_matrix")
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
