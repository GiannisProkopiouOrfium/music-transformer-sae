"""
Smooth Steering for DiffMean: gradual lambda ramp-up / decay.

Wraps the existing SteeringHook with a schedule so that lambda transitions
smoothly from 0 → lambda_target over ``n_ramp`` generation steps, optionally
decays to ``lambda_maintain`` over ``n_decay`` steps, avoiding the audible
discontinuity that occurs with an abrupt, constant-lambda intervention.

Schedule options
----------------
- **linear**: ``min(1, t / n_ramp)``
- **cosine**: ``0.5 * (1 - cos(pi * min(1, t / n_ramp)))``
- **sigmoid**: ``sigmoid(k * (t / n_ramp - 0.5))`` (normalised to [0, 1])

Usage
-----
Drop-in replacement for ``SteeringHook`` — same ``register`` / ``remove`` API.

    from smooth_steering_dm import SmoothSteeringHook

    hook = SmoothSteeringHook(
        steering_vector=vec,
        alpha=2.0,
        schedule="cosine",
        n_ramp=64,
        n_decay=32,
        lambda_maintain=0.3,           # maintenance fraction of alpha
    )
    hook.register(target_module)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ─── Schedule functions ──────────────────────────────────────────────────────


def _linear_schedule(progress: float) -> float:
    """Linear ramp: progress ∈ [0, 1] → [0, 1]."""
    return min(1.0, max(0.0, progress))


def _cosine_schedule(progress: float) -> float:
    """Cosine ramp: smooth ease-in/ease-out."""
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _sigmoid_schedule(progress: float, k: float = 10.0) -> float:
    """Sigmoid ramp: S-curve centred at progress=0.5."""
    progress = min(1.0, max(0.0, progress))
    raw = 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))
    # Normalise so that schedule(0)=0 and schedule(1)=1
    low = 1.0 / (1.0 + math.exp(-k * (0.0 - 0.5)))
    high = 1.0 / (1.0 + math.exp(-k * (1.0 - 0.5)))
    return (raw - low) / (high - low)


SCHEDULE_FNS = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
    "sigmoid": _sigmoid_schedule,
}


# ─── Smooth Steering Hook ───────────────────────────────────────────────────


class SmoothSteeringHook:
    """Drop-in replacement for ``SteeringHook`` with a time-varying envelope.

    Envelope phases
    ~~~~~~~~~~~~~~~
    1. **Ramp-up** (steps 0 … n_ramp−1):
       ``effective_alpha = alpha * schedule(t / n_ramp)``
    2. **Hold** (steps n_ramp … until decay starts):
       ``effective_alpha = alpha``
    3. **Decay** (n_decay steps, optional):
       ``effective_alpha = alpha → alpha * lambda_maintain``
       where ``lambda_maintain`` is a fraction (default 1.0 = no decay).

    Parameters
    ----------
    steering_vector : torch.Tensor
        Steering direction (shape ``(dim,)``).
    alpha : float
        Target (peak) scaling factor.
    schedule : str
        One of ``"linear"``, ``"cosine"``, ``"sigmoid"``.
    n_ramp : int
        Number of generation steps for the ramp-up phase.
    n_decay : int
        Number of steps for the decay phase (0 = no decay).
    lambda_maintain : float
        Fraction of ``alpha`` to maintain after decay (0–1).
    intervention_position : str
        ``"last"`` (default, only last token) or ``"all"``.
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float,
        schedule: str = "cosine",
        n_ramp: int = 64,
        n_decay: int = 0,
        lambda_maintain: float = 1.0,
        intervention_position: str = "last",
    ):
        if schedule not in SCHEDULE_FNS:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose from {list(SCHEDULE_FNS)}"
            )
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.schedule_name = schedule
        self.schedule_fn = SCHEDULE_FNS[schedule]
        self.n_ramp = max(1, n_ramp)
        self.n_decay = max(0, n_decay)
        self.lambda_maintain = min(1.0, max(0.0, lambda_maintain))
        self.intervention_position = intervention_position
        self.hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

        # Step counter — incremented each time the hook fires
        self._step = 0

    def reset(self):
        """Reset step counter (call before each generation)."""
        self._step = 0

    @property
    def effective_alpha(self) -> float:
        """Compute the current effective alpha given the step counter."""
        t = self._step

        # Phase 1: ramp-up
        if t < self.n_ramp:
            return self.alpha * self.schedule_fn(t / self.n_ramp)

        # Phase 2: hold (if no decay, stays here forever)
        hold_end = self.n_ramp + self.n_decay
        if self.n_decay == 0 or t < self.n_ramp:
            return self.alpha

        # Phase 3: decay
        if t < hold_end:
            decay_progress = (t - self.n_ramp) / self.n_decay
            # Interpolate from 1.0 → lambda_maintain using same schedule
            return self.alpha * (
                1.0 - (1.0 - self.lambda_maintain) * self.schedule_fn(decay_progress)
            )

        # Phase 4: maintenance
        return self.alpha * self.lambda_maintain

    def make_hook_fn(self):
        """Create the forward-hook function (same API as ``SteeringHook``)."""

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                actual_output = output[0]
                is_tuple = True
            else:
                actual_output = output
                is_tuple = False

            eff = self.effective_alpha
            self._step += 1

            if eff == 0.0:
                return output  # no-op before first real step

            sv = self.steering_vector.to(actual_output.device)

            if self.intervention_position == "last":
                actual_output[:, -1, :] = actual_output[:, -1, :] + eff * sv
            elif self.intervention_position == "all":
                actual_output = actual_output + eff * sv

            if is_tuple:
                return (actual_output,) + output[1:]
            return actual_output

        return hook_fn

    # ── register / remove (same interface as SteeringHook) ───────────────

    def register(self, module: nn.Module):
        """Register this hook on a module."""
        self.reset()
        self.hook_handle = module.register_forward_hook(self.make_hook_fn())

    def remove(self):
        """Remove this hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
