"""
Smooth Steering for SAS: gradual lambda ramp-up / decay.

Wraps ``SASSteeringHook`` with a time-varying envelope so that the
steering strength transitions smoothly from 0 → λ_target over
``n_ramp`` generation steps, then optionally decays to
``λ_target * lambda_maintain``.

This mirrors ``smooth_steering_dm.py``'s schedule logic but operates in
the sparse SAE feature space (Algorithm 2).

Usage
-----
Drop-in replacement for ``register_steering_hooks``:

    from smooth_steering_sas import register_smooth_steering_hooks

    handles = register_smooth_steering_hooks(
        model, sae_models, sas_vectors,
        concept="average_pitch",
        steering_strength=1.0,
        layers_to_steer=[10],
        schedule="cosine",
        n_ramp=64,
        n_decay=0,
        lambda_maintain=1.0,
    )
"""

import math
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ─── Schedule functions (same as smooth_steering_dm) ─────────────────────────


def _linear_schedule(progress: float) -> float:
    return min(1.0, max(0.0, progress))


def _cosine_schedule(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _sigmoid_schedule(progress: float, k: float = 10.0) -> float:
    progress = min(1.0, max(0.0, progress))
    raw = 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))
    low = 1.0 / (1.0 + math.exp(-k * (0.0 - 0.5)))
    high = 1.0 / (1.0 + math.exp(-k * (1.0 - 0.5)))
    return (raw - low) / (high - low)


SCHEDULE_FNS = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
    "sigmoid": _sigmoid_schedule,
}


# ─── Smooth SAS Steering Hook ───────────────────────────────────────────────


class SmoothSASSteeringHook:
    """SAS steering hook with a time-varying lambda envelope.

    The envelope follows the same 4-phase pattern as the DiffMean variant:
    ramp-up → hold → decay → maintenance.

    Parameters
    ----------
    sae_models : dict
        Layer index → trained SAE model.
    sas_vectors : dict
        Layer index → SAS vector (numpy array, shape ``(4096,)``).
    concept : str
        Concept name (for logging).
    steering_strength : float
        Target (peak) λ.
    layers_to_steer : list[int] or None
        Layers to apply steering.  ``None`` = all available.
    schedule : str
        ``"linear"`` | ``"cosine"`` | ``"sigmoid"``.
    n_ramp : int
        Steps for the ramp-up phase.
    n_decay : int
        Steps for the decay phase (0 = no decay).
    lambda_maintain : float
        Fraction of ``steering_strength`` to maintain after decay.
    """

    def __init__(
        self,
        sae_models: Dict,
        sas_vectors: Dict,
        concept: str,
        steering_strength: float,
        layers_to_steer: Optional[List[int]] = None,
        schedule: str = "cosine",
        n_ramp: int = 64,
        n_decay: int = 0,
        lambda_maintain: float = 1.0,
    ):
        if schedule not in SCHEDULE_FNS:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose from {list(SCHEDULE_FNS)}"
            )

        self.sae_models = sae_models
        self.sas_vectors = sas_vectors
        self.concept = concept
        self.steering_strength = steering_strength
        self.layers_to_steer = layers_to_steer
        self.schedule_fn = SCHEDULE_FNS[schedule]
        self.n_ramp = max(1, n_ramp)
        self.n_decay = max(0, n_decay)
        self.lambda_maintain = min(1.0, max(0.0, lambda_maintain))

        # Convert SAS vectors to torch tensors
        self.sas_vectors_torch: Dict[int, torch.Tensor] = {
            layer_idx: torch.from_numpy(vec).float()
            for layer_idx, vec in sas_vectors.items()
        }

        self.device = None
        self._initialized = False
        self._step = 0

    def reset(self):
        """Reset step counter (call before each generation)."""
        self._step = 0

    def _initialize_device(self, activations: torch.Tensor):
        if not self._initialized:
            self.device = activations.device
            for layer_idx in self.sas_vectors_torch:
                self.sas_vectors_torch[layer_idx] = self.sas_vectors_torch[
                    layer_idx
                ].to(self.device)
            self._initialized = True

    @property
    def effective_lambda(self) -> float:
        """Current effective steering strength given the step counter."""
        t = self._step
        lam = self.steering_strength

        # Phase 1: ramp-up
        if t < self.n_ramp:
            return lam * self.schedule_fn(t / self.n_ramp)

        # Phase 2: hold (if no decay)
        if self.n_decay == 0:
            return lam

        # Phase 3: decay
        hold_end = self.n_ramp + self.n_decay
        if t < hold_end:
            decay_progress = (t - self.n_ramp) / self.n_decay
            return lam * (
                1.0 - (1.0 - self.lambda_maintain) * self.schedule_fn(decay_progress)
            )

        # Phase 4: maintenance
        return lam * self.lambda_maintain

    def __call__(self, module, input, output, layer_idx: int):
        """Forward hook implementing Algorithm 2 with smooth lambda."""
        is_tuple_output = isinstance(output, tuple)
        if is_tuple_output:
            actual_output = output[0]
            other_outputs = output[1:]
        else:
            actual_output = output
            other_outputs = None

        # Check if we should steer this layer
        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.sae_models or layer_idx not in self.sas_vectors_torch:
            return output

        self._initialize_device(actual_output)

        # Compute effective lambda and advance step counter
        # NOTE: step is shared across layers — incremented once per layer-0 call
        eff_lam = self.effective_lambda
        if layer_idx == min(self.sae_models.keys()):
            self._step += 1

        if eff_lam == 0.0:
            return output

        sae = self.sae_models[layer_idx]
        v_sas = self.sas_vectors_torch[layer_idx]

        a_l = actual_output
        batch_size, seq_len, d_model = a_l.shape
        a_flat = a_l.reshape(-1, d_model)

        # Algorithm 2 with smooth lambda
        f_a = sae.encode(a_flat)
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        s_l = f_a + eff_lam * v_sas.unsqueeze(0)

        s_l_relu = torch.relu(s_l)
        s_l_activated = sae.topk(s_l_relu)
        a_prime = sae.decode(s_l_activated)

        a_steered = (a_prime + delta).reshape(batch_size, seq_len, d_model)

        if is_tuple_output:
            return (a_steered,) + other_outputs
        return a_steered


# ─── Registration helper (drop-in for register_steering_hooks) ───────────────


def register_smooth_steering_hooks(
    model: nn.Module,
    sae_models: Dict,
    sas_vectors: Dict,
    concept: str,
    steering_strength: float,
    layers_to_steer: Optional[List[int]] = None,
    schedule: str = "cosine",
    n_ramp: int = 64,
    n_decay: int = 0,
    lambda_maintain: float = 1.0,
) -> List:
    """Register smooth SAS steering hooks — same return type as
    ``steered_generator_sas.register_steering_hooks``.

    Returns
    -------
    handles : list
        Hook handles for cleanup with ``remove_hooks(handles)``.
    """
    hook = SmoothSASSteeringHook(
        sae_models=sae_models,
        sas_vectors=sas_vectors,
        concept=concept,
        steering_strength=steering_strength,
        layers_to_steer=layers_to_steer,
        schedule=schedule,
        n_ramp=n_ramp,
        n_decay=n_decay,
        lambda_maintain=lambda_maintain,
    )

    handles = []
    layers = model.decoder.net.attn_layers.layers

    for layer_idx, layer in enumerate(layers):
        if isinstance(layer, nn.ModuleList) and len(layer) > 1:
            target_module = layer[1]
        else:
            target_module = layer

        handle = target_module.register_forward_hook(
            lambda module, inp, out, idx=layer_idx: hook(module, inp, out, idx)
        )
        handles.append(handle)

    return handles
