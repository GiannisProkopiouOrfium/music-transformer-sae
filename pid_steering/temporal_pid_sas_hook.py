"""Temporal PID Controller for Sparse Activation Steering (SAS).

Solves the fundamental failure of static smooth steering in SAS:
when a fractional lambda is applied during a cosine ramp-up envelope,
the steering signal falls below the SAE's Top-K ReLU threshold and is
zeroed out entirely. The temporal PID controller monitors this failure
and dynamically adjusts lambda across autoregressive generation steps.

The controller operates at a single layer (typically Layer 10) and
maintains state across the token generation sequence:

    At each generation step t:
    1. Encode activations → apply lambda_t * v_SAS → ReLU → Top-K → Decode + Δ
    2. Measure error: did target features survive Top-K?
       e(t) = target_magnitude - actual_feature_activation
       If features zeroed by Top-K → e(t) is large positive
    3. PID computes lambda_{t+1}:
       - If features keep failing: integral accumulates → pushes lambda higher
       - Once features breach threshold: derivative damps to prevent explosion
    4. Optional beat-wise clock for musical-time consistency.

Reference: "Activation Steering with a Feedback Controller"
           Nguyen et al., ICLR 2026 — adapted to temporal domain
"""

import logging
import math
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_controller import TemporalPIDController
import config_pid

logger = logging.getLogger(__name__)


class TemporalPIDSASHook:
    """Forward hook implementing temporal PID-controlled SAS steering.

    Extends Algorithm 2 (SAS Vectors in Inference) with a PID controller
    that dynamically adjusts the steering multiplier lambda_t at each
    autoregressive generation step.
    """

    def __init__(
        self,
        sae_model,
        sas_vector: np.ndarray,
        target_feature_indices: np.ndarray,
        target_magnitude: float,
        Kp: float = 1.0,
        Ki: float = 0.2,
        Kd: float = 0.1,
        max_I: float = 10.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,
        beat_mode: bool = False,
        n_ramp_beats: int = 32,
        skip_conditioning: bool = False,
    ):
        """Initialize the temporal PID SAS hook.

        Args:
            sae_model: Trained SparseAutoencoder for the target layer.
            sas_vector: SAS steering vector (sparse_dim,) numpy array.
            target_feature_indices: Indices of features we want to activate
                (from frequency filtering). Used to measure PID error.
            target_magnitude: Desired mean activation magnitude of target
                features (used as the PID setpoint).
            Kp, Ki, Kd, max_I: PID controller gains.
            lambda_min, lambda_max: Output clamp for the PID multiplier.
            beat_mode: If True, advance PID clock per musical beat (not token).
            n_ramp_beats: Number of beats for the desired ramp envelope.
            skip_conditioning: If True, skip the first multi-token forward
                pass (conditioning prefix). Activated by conditioned generation.
        """
        self.sae_model = sae_model
        self.sas_vector = torch.from_numpy(sas_vector).float()
        self.target_feature_indices = target_feature_indices
        self.target_magnitude = target_magnitude

        # PID controller
        self.pid = TemporalPIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

        # Conditioning skip
        self._skip_conditioning = skip_conditioning
        self._conditioning_skipped = False

        # Beat-mode tracking
        self.beat_mode = beat_mode
        self.n_ramp_beats = n_ramp_beats
        self._last_beat_value = None
        self._beat_count = 0

        # Diagnostics
        self.lambda_trajectory = []
        self.error_trajectory = []
        self.feature_activation_trajectory = []

        # Device initialization
        self.device = None
        self._initialized = False

    def _initialize_device(self, activations):
        if not self._initialized:
            self.device = activations.device
            self.sas_vector = self.sas_vector.to(self.device)
            self._initialized = True

    def reset(self):
        """Reset controller state for a new generation."""
        self.pid.reset()
        self._last_beat_value = None
        self._beat_count = 0
        self.lambda_trajectory = []
        self.error_trajectory = []
        self.feature_activation_trajectory = []
        self._conditioning_skipped = False

    def _compute_target_envelope(self) -> float:
        """Compute the desired target magnitude based on ramp progress.

        Uses a cosine schedule to ramp the target from 0 to target_magnitude
        over n_ramp_beats. The PID controller then ensures this target is met
        despite Top-K filtering.
        """
        if self.n_ramp_beats <= 0:
            return self.target_magnitude

        t = self._beat_count if self.beat_mode else len(self.lambda_trajectory)

        if t >= self.n_ramp_beats:
            return self.target_magnitude

        # Cosine ramp: 0 → target_magnitude
        progress = t / self.n_ramp_beats
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.target_magnitude * scale

    def __call__(self, module, input, output, layer_idx):
        """Forward hook implementing PID-controlled Algorithm 2."""
        is_tuple_output = isinstance(output, tuple)
        if is_tuple_output:
            actual_output = output[0]
            other_outputs = output[1:]
        else:
            actual_output = output
            other_outputs = None

        self._initialize_device(actual_output)

        # Skip the conditioning prefix pass (multi-token)
        batch_size, seq_len, d_model = actual_output.shape
        if self._skip_conditioning and not self._conditioning_skipped and seq_len > 1:
            self._conditioning_skipped = True
            return output  # pass through unmodified

        sae = self.sae_model
        v_sas = self.sas_vector  # (sparse_dim,)

        a_l = actual_output  # (batch, seq, 512)
        a_flat = a_l.reshape(-1, d_model)  # (batch*seq, 512)

        # Beat tracking (check if beat changed in the last generated token)
        if self.beat_mode and seq_len > 0:
            # The beat dimension is index 1 in the 6-tuple token
            # We can only track this if we can see the input tokens
            # For now, increment beat_count per hook call in beat_mode
            # (will be refined with a pre-hook observer if needed)
            pass

        # Step 1: Encode to sparse space
        f_a = sae.encode(a_flat)  # (batch*seq, sparse_dim)

        # Step 2: Compute correction term Δ
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed  # (batch*seq, 512)

        # Step 3: Measure current feature activation (before steering)
        # Look at target feature indices in the sparse representation
        target_activations = f_a[
            :, self.target_feature_indices
        ]  # (batch*seq, n_targets)
        actual_magnitude = target_activations.mean().item()

        # Compute desired target (ramp envelope)
        desired_magnitude = self._compute_target_envelope()

        # Error signal: positive = under-activated (need more steering)
        error = desired_magnitude - actual_magnitude
        self.error_trajectory.append(error)
        self.feature_activation_trajectory.append(actual_magnitude)

        # PID computes dynamic lambda
        lambda_t = self.pid.compute(error)
        self.lambda_trajectory.append(lambda_t)

        # Step 4: Apply steering with PID-computed lambda
        s_l = f_a + lambda_t * v_sas.unsqueeze(0)  # (batch*seq, sparse_dim)

        # Step 5: ReLU + Top-K (must match SAE training)
        s_l_relu = torch.relu(s_l)
        s_l_activated = sae.topk(s_l_relu)
        a_prime = sae.decode(s_l_activated)  # (batch*seq, 512)

        # Step 6: Add correction term
        a_steered = a_prime + delta  # (batch*seq, 512)
        a_steered = a_steered.reshape(batch_size, seq_len, d_model)

        if is_tuple_output:
            return (a_steered,) + other_outputs
        return a_steered

    def get_diagnostics(self) -> Dict:
        """Return diagnostic trajectories for analysis and plotting."""
        return {
            "lambda_trajectory": self.lambda_trajectory,
            "error_trajectory": self.error_trajectory,
            "feature_activation_trajectory": self.feature_activation_trajectory,
            "pid_gains": self.pid.gains,
            "n_steps": len(self.lambda_trajectory),
        }
