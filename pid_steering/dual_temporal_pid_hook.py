"""Dual-Concept Temporal PID SAS Hook.

Two independent PID controllers — one for pitch, one for duration — each
dynamically adjusting its own λ_t.  At each generation step:

  1. Encode activations via SAE
  2. Measure pitch features → PID_pitch computes λ_pitch_t
  3. Measure duration features → PID_duration computes λ_duration_t
  4. Compose: s = f(a) + λ_pitch_t * v_pitch + λ_duration_t * v_duration
  5. ReLU → TopK (with expanded K=2×) → Decode + Δ

Uses Gram-Schmidt orthogonalization (duration w.r.t. pitch) for vector
composition, matching the best dual strategy from the ISMIR paper (gs_ek2).
The expanded K budget (2×) prevents Top-K resource competition between
the two concepts.

Reference: extends Nguyen et al. (ICLR 2026) temporal PID to dual-concept.
"""

import logging
import math
import pathlib
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pid_steering.pid_controller import TemporalPIDController

logger = logging.getLogger(__name__)


def gram_schmidt_orthogonalize(
    v_primary: np.ndarray, v_secondary: np.ndarray
) -> np.ndarray:
    """Orthogonalize v_secondary w.r.t. v_primary via Gram-Schmidt.

    Returns v_secondary⊥ = v_secondary - proj(v_secondary → v_primary).
    """
    dot = np.dot(v_secondary, v_primary)
    norm_sq = np.dot(v_primary, v_primary)
    if norm_sq < 1e-12:
        return v_secondary.copy()
    projection = (dot / norm_sq) * v_primary
    return v_secondary - projection


def get_target_feature_indices(sas_vector: np.ndarray, top_n: int = 32) -> np.ndarray:
    """Extract indices of top features by magnitude from SAS vector."""
    magnitudes = np.abs(sas_vector)
    nonzero = np.where(magnitudes > 0)[0]
    if len(nonzero) == 0:
        return np.arange(min(top_n, len(sas_vector)))
    top_n = min(top_n, len(nonzero))
    top_indices = np.argsort(magnitudes[nonzero])[-top_n:]
    return nonzero[top_indices]


class DualTemporalPIDSASHook:
    """Forward hook with two independent PID controllers for dual-concept steering.

    Each concept has its own:
    - SAS vector (optionally GS-orthogonalized)
    - Target feature indices for error measurement
    - PID controller computing its own λ_t
    - Diagnostic trajectories

    The composed steering is: s = f(a) + λ_pitch * v_pitch + λ_dur * v_dur
    Re-sparsified with expanded K (2×) to prevent Top-K competition.
    """

    def __init__(
        self,
        sae_model,
        pitch_vector: np.ndarray,
        duration_vector: np.ndarray,
        target_magnitude: float = 1.0,
        Kp: float = 1.0,
        Ki: float = 0.05,
        Kd: float = 0.01,
        max_I: float = 10.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,
        n_ramp_beats: int = 64,
        orthogonalize: bool = True,
        k_multiplier: float = 2.0,
        skip_conditioning: bool = False,
    ):
        """
        Args:
            sae_model: Trained SAE for the target layer.
            pitch_vector: SAS vector for pitch (4096,).
            duration_vector: SAS vector for duration (4096,).
            target_magnitude: Desired feature activation setpoint.
            Kp, Ki, Kd, max_I: PID gains (shared for both controllers).
            lambda_min, lambda_max: Clamp for each PID output.
            n_ramp_beats: Cosine ramp length.
            orthogonalize: Apply GS to duration w.r.t. pitch.
            k_multiplier: TopK expansion factor (default 2×).
            skip_conditioning: Skip first multi-token pass (for conditioned gen).
        """
        self.sae_model = sae_model
        self.k_multiplier = k_multiplier

        # Orthogonalize duration w.r.t. pitch (matching gs_ek2 strategy)
        if orthogonalize:
            duration_vector = gram_schmidt_orthogonalize(pitch_vector, duration_vector)

        self.pitch_vector = torch.from_numpy(pitch_vector).float()
        self.duration_vector = torch.from_numpy(duration_vector).float()

        # Target features for each concept
        self.pitch_feature_indices = get_target_feature_indices(pitch_vector)
        self.duration_feature_indices = get_target_feature_indices(duration_vector)

        self.target_magnitude = target_magnitude
        self.n_ramp_beats = n_ramp_beats

        # Two independent PID controllers
        self.pid_pitch = TemporalPIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
        self.pid_duration = TemporalPIDController(
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

        # Diagnostics
        self.pitch_lambda_trajectory = []
        self.duration_lambda_trajectory = []
        self.pitch_error_trajectory = []
        self.duration_error_trajectory = []
        self.pitch_activation_trajectory = []
        self.duration_activation_trajectory = []

        self.device = None
        self._initialized = False

    def _initialize_device(self, activations):
        if not self._initialized:
            self.device = activations.device
            self.pitch_vector = self.pitch_vector.to(self.device)
            self.duration_vector = self.duration_vector.to(self.device)
            self._initialized = True

    def reset(self):
        """Reset both controllers for a new generation."""
        self.pid_pitch.reset()
        self.pid_duration.reset()
        self.pitch_lambda_trajectory = []
        self.duration_lambda_trajectory = []
        self.pitch_error_trajectory = []
        self.duration_error_trajectory = []
        self.pitch_activation_trajectory = []
        self.duration_activation_trajectory = []
        self._conditioning_skipped = False

    def _compute_target_envelope(self) -> float:
        """Cosine ramp target: 0 → target_magnitude over n_ramp_beats."""
        if self.n_ramp_beats <= 0:
            return self.target_magnitude
        t = len(self.pitch_lambda_trajectory)
        if t >= self.n_ramp_beats:
            return self.target_magnitude
        progress = t / self.n_ramp_beats
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.target_magnitude * scale

    def __call__(self, module, input, output, layer_idx):
        """Forward hook: dual PID-controlled SAS with expanded K."""
        is_tuple = isinstance(output, tuple)
        actual_output = output[0] if is_tuple else output
        other_outputs = output[1:] if is_tuple else None

        self._initialize_device(actual_output)

        batch_size, seq_len, d_model = actual_output.shape

        # Skip conditioning prefix
        if self._skip_conditioning and not self._conditioning_skipped and seq_len > 1:
            self._conditioning_skipped = True
            return output

        sae = self.sae_model
        a_flat = actual_output.reshape(-1, d_model)

        # 1. Encode
        f_a = sae.encode(a_flat)

        # 2. Correction Δ
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        # 3. Measure each concept's feature activation
        pitch_act = f_a[:, self.pitch_feature_indices].mean().item()
        dur_act = f_a[:, self.duration_feature_indices].mean().item()

        desired = self._compute_target_envelope()

        # 4. PID for pitch
        pitch_error = desired - pitch_act
        lambda_pitch = self.pid_pitch.compute(pitch_error)
        self.pitch_error_trajectory.append(pitch_error)
        self.pitch_lambda_trajectory.append(lambda_pitch)
        self.pitch_activation_trajectory.append(pitch_act)

        # 5. PID for duration
        dur_error = desired - dur_act
        lambda_dur = self.pid_duration.compute(dur_error)
        self.duration_error_trajectory.append(dur_error)
        self.duration_lambda_trajectory.append(lambda_dur)
        self.duration_activation_trajectory.append(dur_act)

        # 6. Compose steering: f(a) + λ_p * v_p + λ_d * v_d
        s_l = (
            f_a
            + lambda_pitch * self.pitch_vector.unsqueeze(0)
            + lambda_dur * self.duration_vector.unsqueeze(0)
        )

        # 7. ReLU + expanded TopK
        s_l = torch.relu(s_l)
        original_k = sae.topk.k
        expanded_k = min(int(original_k * self.k_multiplier), s_l.shape[-1])
        sae.topk.k = expanded_k
        try:
            s_l = sae.topk(s_l)
        finally:
            sae.topk.k = original_k

        # 8. Decode + correct
        a_steered = sae.decode(s_l) + delta
        a_steered = a_steered.reshape(batch_size, seq_len, d_model)

        if is_tuple:
            return (a_steered,) + other_outputs
        return a_steered

    def get_diagnostics(self) -> Dict:
        """Return per-concept diagnostic trajectories."""
        return {
            "pitch_lambda_trajectory": self.pitch_lambda_trajectory,
            "duration_lambda_trajectory": self.duration_lambda_trajectory,
            "pitch_error_trajectory": self.pitch_error_trajectory,
            "duration_error_trajectory": self.duration_error_trajectory,
            "pitch_activation_trajectory": self.pitch_activation_trajectory,
            "duration_activation_trajectory": self.duration_activation_trajectory,
            "n_steps": len(self.pitch_lambda_trajectory),
        }
