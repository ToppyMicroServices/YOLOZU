"""Continual-learning regularisers (EWC and Synaptic Intelligence).

Provides diagonal EWC (Elastic Weight Consolidation) and SI (Synaptic
Intelligence) accumulators, penalty functions, and state
serialisation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "EwcState",
    "EwcAccumulator",
    "ewc_penalty",
    "save_ewc_state",
    "load_ewc_state",
    "SiState",
    "SiAccumulator",
    "si_penalty",
    "save_si_state",
    "load_si_state",
]

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for yolozu.continual_regularizers")


def _named_trainable_params(model: Any) -> list[tuple[str, "torch.Tensor"]]:
    _require_torch()
    out: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(param, torch.Tensor):
            continue
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if not bool(param.is_floating_point()):
            continue
        out.append((name, param))
    return out


@dataclass(frozen=True)
class EwcState:
    """Diagonal EWC state: fisher importance + reference parameters (theta*)."""

    schema_version: int
    fisher: dict[str, "torch.Tensor"]
    theta_star: dict[str, "torch.Tensor"]
    steps: int

    def to(self, device: "torch.device") -> "EwcState":
        _require_torch()
        return EwcState(
            schema_version=int(self.schema_version),
            fisher={k: v.to(device) for k, v in self.fisher.items()},
            theta_star={k: v.to(device) for k, v in self.theta_star.items()},
            steps=int(self.steps),
        )


class EwcAccumulator:
    def __init__(self) -> None:
        _require_torch()
        self._sum_sq_grads: dict[str, torch.Tensor] = {}
        self._steps: int = 0

    @property
    def steps(self) -> int:
        return int(self._steps)

    def accumulate_from_grads(self, model: Any) -> None:
        _require_torch()
        for name, param in _named_trainable_params(model):
            grad = getattr(param, "grad", None)
            if not isinstance(grad, torch.Tensor):
                continue
            g = grad.detach()
            if not g.is_floating_point():
                continue
            g2 = g.float() * g.float()
            acc = self._sum_sq_grads.get(name)
            if acc is None:
                self._sum_sq_grads[name] = g2.to(device="cpu")
            else:
                self._sum_sq_grads[name] = acc + g2.to(device="cpu")
        self._steps += 1

    def finalize(self, model: Any) -> EwcState:
        _require_torch()
        denom = max(1, int(self._steps))
        fisher = {k: (v / float(denom)).clone() for k, v in self._sum_sq_grads.items()}
        theta_star = {name: param.detach().to(device="cpu").clone() for name, param in _named_trainable_params(model)}
        return EwcState(schema_version=1, fisher=fisher, theta_star=theta_star, steps=int(self._steps))


def ewc_penalty(model: Any, state: EwcState) -> "torch.Tensor":
    _require_torch()
    total = None
    for name, param in _named_trainable_params(model):
        fisher = state.fisher.get(name)
        theta = state.theta_star.get(name)
        if fisher is None or theta is None:
            continue
        fisher_t = fisher.to(device=param.device, dtype=torch.float32)
        theta_t = theta.to(device=param.device, dtype=param.dtype)
        diff = (param - theta_t).float()
        term = (fisher_t * diff * diff).sum()
        total = term if total is None else (total + term)
    if total is None:
        reference = next((p for _, p in _named_trainable_params(model)), None)
        if reference is None:  # pragma: no cover
            return torch.tensor(0.0)
        return torch.zeros((), device=reference.device, dtype=torch.float32)
    return total


def save_ewc_state(path: str | Path, state: EwcState) -> None:
    _require_torch()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": int(state.schema_version),
            "kind": "ewc_state",
            "steps": int(state.steps),
            "fisher": state.fisher,
            "theta_star": state.theta_star,
        },
        str(p),
    )


def load_ewc_state(path: str | Path) -> EwcState:
    _require_torch()
    raw = torch.load(str(path), map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("ewc state must be a dict")
    if str(raw.get("kind") or "ewc_state") != "ewc_state":
        raise ValueError("not an ewc_state file")
    schema_version = int(raw.get("schema_version") or 1)
    if schema_version != 1:
        raise ValueError(f"unsupported ewc schema_version: {schema_version}")
    fisher = raw.get("fisher") if isinstance(raw.get("fisher"), dict) else {}
    theta_star = raw.get("theta_star") if isinstance(raw.get("theta_star"), dict) else {}
    steps = int(raw.get("steps") or 0)
    fisher_t: dict[str, torch.Tensor] = {str(k): v for k, v in fisher.items() if isinstance(v, torch.Tensor)}
    theta_t: dict[str, torch.Tensor] = {str(k): v for k, v in theta_star.items() if isinstance(v, torch.Tensor)}
    return EwcState(schema_version=1, fisher=fisher_t, theta_star=theta_t, steps=steps)


@dataclass(frozen=True)
class SiState:
    """Synaptic Intelligence state: omega importance + reference parameters (theta*)."""

    schema_version: int
    omega: dict[str, "torch.Tensor"]
    theta_star: dict[str, "torch.Tensor"]
    epsilon: float
    steps: int

    def to(self, device: "torch.device") -> "SiState":
        _require_torch()
        return SiState(
            schema_version=int(self.schema_version),
            omega={k: v.to(device) for k, v in self.omega.items()},
            theta_star={k: v.to(device) for k, v in self.theta_star.items()},
            epsilon=float(self.epsilon),
            steps=int(self.steps),
        )


class SiAccumulator:
    def __init__(self, *, epsilon: float = 1e-3) -> None:
        _require_torch()
        if not (float(epsilon) > 0.0):
            raise ValueError("epsilon must be > 0")
        self._epsilon = float(epsilon)
        self._theta_start: dict[str, torch.Tensor] | None = None
        self._omega_total: dict[str, torch.Tensor] = {}
        self._w: dict[str, torch.Tensor] = {}
        self._steps: int = 0
        self._params_before: dict[str, torch.Tensor] = {}
        self._grads_before: dict[str, torch.Tensor] = {}

    @property
    def steps(self) -> int:
        return int(self._steps)

    def load_state(self, state: SiState) -> None:
        _require_torch()
        self._omega_total = {k: v.clone() for k, v in state.omega.items()}
        self._theta_start = {k: v.clone() for k, v in state.theta_star.items()}

    def begin_task(self, model: Any) -> None:
        _require_torch()
        self._theta_start = {name: param.detach().to(device="cpu").clone() for name, param in _named_trainable_params(model)}
        self._w = {}
        self._params_before = {}
        self._grads_before = {}
        self._steps = 0

    def capture_before_step(self, model: Any) -> None:
        _require_torch()
        self._params_before = {}
        self._grads_before = {}
        for name, param in _named_trainable_params(model):
            grad = getattr(param, "grad", None)
            if not isinstance(grad, torch.Tensor):
                continue
            self._params_before[name] = param.detach().to(device="cpu").clone()
            self._grads_before[name] = grad.detach().to(device="cpu").float().clone()

    def update_after_step(self, model: Any) -> None:
        _require_torch()
        for name, param in _named_trainable_params(model):
            p0 = self._params_before.get(name)
            g = self._grads_before.get(name)
            if p0 is None or g is None:
                continue
            p1 = param.detach().to(device="cpu").clone()
            delta = p1 - p0
            w_prev = self._w.get(name)
            contrib = (-g) * delta.float()
            self._w[name] = contrib if w_prev is None else (w_prev + contrib)
        self._steps += 1

    def finalize(self, model: Any) -> SiState:
        _require_torch()
        if self._theta_start is None:
            self.begin_task(model)
        assert self._theta_start is not None
        theta_end = {name: param.detach().to(device="cpu").clone() for name, param in _named_trainable_params(model)}
        eps = float(self._epsilon)
        omega_new: dict[str, torch.Tensor] = {}
        for name, theta0 in self._theta_start.items():
            theta1 = theta_end.get(name)
            if theta1 is None:
                continue
            delta_total = (theta1 - theta0).float()
            denom = delta_total * delta_total + eps
            w = self._w.get(name)
            if w is None:
                continue
            omega_new[name] = (w / denom).clone()

        omega_total = dict(self._omega_total)
        for name, om in omega_new.items():
            prev = omega_total.get(name)
            omega_total[name] = om if prev is None else (prev + om)
        self._omega_total = omega_total
        self._theta_start = theta_end
        return SiState(schema_version=1, omega=omega_total, theta_star=theta_end, epsilon=eps, steps=int(self._steps))


def si_penalty(model: Any, state: SiState) -> "torch.Tensor":
    _require_torch()
    total = None
    for name, param in _named_trainable_params(model):
        omega = state.omega.get(name)
        theta = state.theta_star.get(name)
        if omega is None or theta is None:
            continue
        omega_t = omega.to(device=param.device, dtype=torch.float32)
        theta_t = theta.to(device=param.device, dtype=param.dtype)
        diff = (param - theta_t).float()
        term = (omega_t * diff * diff).sum()
        total = term if total is None else (total + term)
    if total is None:
        reference = next((p for _, p in _named_trainable_params(model)), None)
        if reference is None:  # pragma: no cover
            return torch.tensor(0.0)
        return torch.zeros((), device=reference.device, dtype=torch.float32)
    return total


def save_si_state(path: str | Path, state: SiState) -> None:
    _require_torch()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": int(state.schema_version),
            "kind": "si_state",
            "steps": int(state.steps),
            "epsilon": float(state.epsilon),
            "omega": state.omega,
            "theta_star": state.theta_star,
        },
        str(p),
    )


def load_si_state(path: str | Path) -> SiState:
    _require_torch()
    raw = torch.load(str(path), map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("si state must be a dict")
    if str(raw.get("kind") or "si_state") != "si_state":
        raise ValueError("not a si_state file")
    schema_version = int(raw.get("schema_version") or 1)
    if schema_version != 1:
        raise ValueError(f"unsupported si schema_version: {schema_version}")
    omega = raw.get("omega") if isinstance(raw.get("omega"), dict) else {}
    theta_star = raw.get("theta_star") if isinstance(raw.get("theta_star"), dict) else {}
    steps = int(raw.get("steps") or 0)
    epsilon = float(raw.get("epsilon") or 1e-3)
    omega_t: dict[str, torch.Tensor] = {str(k): v for k, v in omega.items() if isinstance(v, torch.Tensor)}
    theta_t: dict[str, torch.Tensor] = {str(k): v for k, v in theta_star.items() if isinstance(v, torch.Tensor)}
    return SiState(schema_version=1, omega=omega_t, theta_star=theta_t, epsilon=epsilon, steps=steps)

