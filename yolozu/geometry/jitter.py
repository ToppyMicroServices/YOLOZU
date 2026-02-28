"""Random perturbation (jitter) sampling for camera parameters.

Used for data-augmentation and robustness testing of intrinsics /
extrinsics pipelines.
"""

import random

__all__ = [
    "default_jitter_profile",
    "sample_intrinsics_jitter",
    "sample_extrinsics_jitter",
    "jitter_off",
]


def default_jitter_profile():
    return {
        "intrinsics": {"dfx": 0.02, "dfy": 0.02, "dcx": 4.0, "dcy": 4.0},
        "extrinsics": {"dx": 0.01, "dy": 0.01, "dz": 0.02, "droll": 1.0, "dpitch": 1.0, "dyaw": 2.0},
        "rolling_shutter": {"enabled": False, "line_delay": 0.0},
    }


def sample_intrinsics_jitter(profile, seed=None):
    rng = random.Random(seed)
    intr = profile.get("intrinsics", {})
    return {
        "dfx": rng.uniform(-intr.get("dfx", 0.0), intr.get("dfx", 0.0)),
        "dfy": rng.uniform(-intr.get("dfy", 0.0), intr.get("dfy", 0.0)),
        "dcx": rng.uniform(-intr.get("dcx", 0.0), intr.get("dcx", 0.0)),
        "dcy": rng.uniform(-intr.get("dcy", 0.0), intr.get("dcy", 0.0)),
    }


def sample_extrinsics_jitter(profile, seed=None):
    rng = random.Random(seed)
    ext = profile.get("extrinsics", {})
    return {
        "dx": rng.uniform(-ext.get("dx", 0.0), ext.get("dx", 0.0)),
        "dy": rng.uniform(-ext.get("dy", 0.0), ext.get("dy", 0.0)),
        "dz": rng.uniform(-ext.get("dz", 0.0), ext.get("dz", 0.0)),
        "droll": rng.uniform(-ext.get("droll", 0.0), ext.get("droll", 0.0)),
        "dpitch": rng.uniform(-ext.get("dpitch", 0.0), ext.get("dpitch", 0.0)),
        "dyaw": rng.uniform(-ext.get("dyaw", 0.0), ext.get("dyaw", 0.0)),
    }


def jitter_off():
    return {"dfx": 0.0, "dfy": 0.0, "dcx": 0.0, "dcy": 0.0}
