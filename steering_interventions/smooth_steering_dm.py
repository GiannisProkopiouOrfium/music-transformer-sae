"""
Smooth Steering for DiffMean: time-varying lambda envelopes.

Wraps the existing SteeringHook with a schedule so that lambda follows a
chosen envelope, avoiding the audible discontinuity that occurs with an
abrupt, constant-lambda intervention.

Modes
-----
- **ramp_up**: gradual 0 → λ over ``n_ramp`` steps (original smooth steering)
- **delayed_onset**: 0 for ``n_delay`` steps, then instant full λ
- **pulse**: full λ for ``n_pulse`` steps, then 0 (fire-and-forget)
- **ramp_down**: full λ immediately, decays to ``lambda_maintain × λ`` over
  ``n_decay`` steps
- **gradual**: linear ramp 0 → λ over ``n_ramp`` steps (ignores schedule).
  Set ``n_ramp`` = ``continuation_len`` for a full-piece ramp.
- **warmup_hold**: schedule-shaped ramp 0 → λ over ``n_ramp``, then hold.
  Uses the configured schedule (cosine by default) for a smooth S-curve.

Schedule options (for ramp_up / ramp_down curves)
--------------------------------------------------
- **linear**: ``min(1, t / n)``
- **cosine**: ``0.5 * (1 - cos(π · min(1, t / n)))``
- **sigmoid**: ``σ(k · (t/n − 0.5))``  (normalised to [0, 1])

Usage
-----
    from smooth_steering_dm import SmoothSteeringHook

    # Ramp-up (original)
    hook = SmoothSteeringHook(vec, alpha=2.0, mode="ramp_up", n_ramp=64)

    # Delayed onset — 16 natural tokens then full steering
    hook = SmoothSteeringHook(vec, alpha=2.0, mode="delayed_onset", n_delay=16)

    # Pulse — steer for 64 tokens then stop
    hook = SmoothSteeringHook(vec, alpha=2.0, mode="pulse", n_pulse=64)

    # Ramp-down — full strength then decay
    hook = SmoothSteeringHook(vec, alpha=2.0, mode="ramp_down",
                               n_decay=32, lambda_maintain=0.3)

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

VALID_MODES = {
    "ramp_up",
    "delayed_onset",
    "pulse",
    "ramp_down",
    "gradual",
    "warmup_hold",
}


# ─── Smooth Steering Hook ───────────────────────────────────────────────────


class SmoothSteeringHook:
    """Drop-in replacement for ``SteeringHook`` with a time-varying envelope.

    Parameters
    ----------
    steering_vector : torch.Tensor
        Steering direction (shape ``(dim,)``).
    alpha : float
        Target (peak) scaling factor.
    mode : str
        ``"ramp_up"`` | ``"delayed_onset"`` | ``"pulse"`` | ``"ramp_down"``.
    schedule : str
        One of ``"linear"``, ``"cosine"``, ``"sigmoid"``
        (used by ramp_up and ramp_down modes).
    n_ramp : int
        Steps for ramp-up phase (mode=ramp_up).
    n_delay : int
        Steps of silence before full onset (mode=delayed_onset).
    n_pulse : int
        Steps of full steering before stopping (mode=pulse).
    n_decay : int
        Steps for decay phase (mode=ramp_up with decay, or mode=ramp_down).
    lambda_maintain : float
        Fraction of alpha to maintain after decay (0–1).
    intervention_position : str
        ``"last"`` (default, only last token) or ``"all"``.
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float,
        mode: str = "ramp_up",
        schedule: str = "cosine",
        n_ramp: int = 64,
        n_delay: int = 16,
        n_pulse: int = 64,
        n_decay: int = 0,
        lambda_maintain: float = 1.0,
        intervention_position: str = "last",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {VALID_MODES}")
        if schedule not in SCHEDULE_FNS:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose from {list(SCHEDULE_FNS)}"
            )
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.mode = mode
        self.schedule_name = schedule
        self.schedule_fn = SCHEDULE_FNS[schedule]
        self.n_ramp = max(1, n_ramp)
        self.n_delay = max(0, n_delay)
        self.n_pulse = max(1, n_pulse)
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

        if self.mode == "ramp_up":
            return self._ramp_up_alpha(t)
        elif self.mode == "delayed_onset":
            return self._delayed_onset_alpha(t)
        elif self.mode == "pulse":
            return self._pulse_alpha(t)
        elif self.mode == "ramp_down":
            return self._ramp_down_alpha(t)
        elif self.mode == "gradual":
            return self._gradual_alpha(t)
        elif self.mode == "warmup_hold":
            return self._warmup_hold_alpha(t)
        return self.alpha

    def _ramp_up_alpha(self, t: int) -> float:
        """0 → α over n_ramp, then hold, optionally decay."""
        if t < self.n_ramp:
            return self.alpha * self.schedule_fn(t / self.n_ramp)
        if self.n_decay == 0:
            return self.alpha
        hold_end = self.n_ramp + self.n_decay
        if t < hold_end:
            decay_progress = (t - self.n_ramp) / self.n_decay
            return self.alpha * (
                1.0 - (1.0 - self.lambda_maintain) * self.schedule_fn(decay_progress)
            )
        return self.alpha * self.lambda_maintain

    def _delayed_onset_alpha(self, t: int) -> float:
        """0 for n_delay steps, then full α."""
        if t < self.n_delay:
            return 0.0
        return self.alpha

    def _pulse_alpha(self, t: int) -> float:
        """Full α for n_pulse steps, then 0."""
        if t < self.n_pulse:
            return self.alpha
        return 0.0

    def _ramp_down_alpha(self, t: int) -> float:
        """Full α immediately, then decay to λ_maintain × α over n_decay."""
        if self.n_decay == 0:
            return self.alpha
        if t < self.n_decay:
            decay_progress = t / self.n_decay
            return self.alpha * (
                1.0 - (1.0 - self.lambda_maintain) * self.schedule_fn(decay_progress)
            )
        return self.alpha * self.lambda_maintain

    def _gradual_alpha(self, t: int) -> float:
        """Linear ramp 0 → α over n_ramp steps, then hold.

        Always uses a linear curve regardless of schedule setting.
        Designed for n_ramp = continuation_len (full-piece ramp).
        """
        if t < self.n_ramp:
            return self.alpha * (t / self.n_ramp)
        return self.alpha

    def _warmup_hold_alpha(self, t: int) -> float:
        """Schedule-shaped ramp 0 → α over n_ramp, then hold at α.

        Uses the configured schedule (cosine by default) for an S-curve
        ramp.  Designed for n_ramp ≈ 75% of continuation_len.
        """
        if t < self.n_ramp:
            return self.alpha * self.schedule_fn(t / self.n_ramp)
        return self.alpha

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
                return output  # no-op

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
