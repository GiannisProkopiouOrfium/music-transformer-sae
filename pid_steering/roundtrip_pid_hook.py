"""Round-Trip PID Hook for Multi-Phase Temporal Steering.

Extends TemporalPIDSASHook with support for multiple steering phases
within a single generation. Each phase can have a different direction
(up/down) and its own ramp schedule. The PID controller resets at
each phase transition.

Typical use case: steer away from a conditioned prefix, hold,
then steer back toward the original characteristics.

Example schedule:
    phases = [
        {"tokens": 64, "direction": "up",   "ramp_steps": 64},
        {"tokens": 64, "direction": "hold",  "ramp_steps": 0},
        {"tokens": 64, "direction": "down",  "ramp_steps": 64},
    ]
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


class RoundTripPIDSASHook:
    """Forward hook implementing multi-phase PID-controlled SAS steering.

    Supports a sequence of phases, each with:
      - A token budget (how many generation steps)
      - A direction ("up", "down", or "hold")
      - A ramp duration (PID setpoint ramp within the phase)

    At each phase transition, the PID controller resets (I=0) and the
    SAS vector is flipped if the direction changes. During "hold" phases,
    the controller continues with the same vector and a flat setpoint.
    """

    def __init__(
        self,
        sae_model,
        sas_vector_up: np.ndarray,
        target_feature_indices_up: np.ndarray,
        target_feature_indices_down: np.ndarray,
        phases: List[Dict],
        target_magnitude: float = 2.0,
        Kp: float = 1.0,
        Ki: float = 0.05,
        Kd: float = 0.01,
        max_I: float = 10.0,
        lambda_max: float = 3.0,
        skip_conditioning: bool = True,
    ):
        """Initialize the round-trip PID hook.

        Args:
            sae_model: Trained SparseAutoencoder for the target layer.
            sas_vector_up: SAS vector for the "up" direction (sparse_dim,).
            target_feature_indices_up: Feature indices for up-direction error.
            target_feature_indices_down: Feature indices for down-direction error.
            phases: List of phase dicts, each with keys:
                "tokens": int — number of generation steps in this phase
                "direction": str — "up", "down", or "hold"
                "ramp_steps": int — cosine ramp duration (0 = instant setpoint)
            target_magnitude: PID setpoint magnitude.
            Kp, Ki, Kd, max_I: PID controller gains.
            lambda_max: Output clamp for lambda_t.
            skip_conditioning: Skip the first multi-token forward pass.
        """
        self.sae_model = sae_model
        self.sas_vector_up = torch.from_numpy(sas_vector_up).float()
        self.sas_vector_down = -self.sas_vector_up.clone()
        self.target_feature_indices_up = target_feature_indices_up
        self.target_feature_indices_down = target_feature_indices_down
        self.target_magnitude = target_magnitude

        # PID gains (stored for reset)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_I = max_I
        self.lambda_max = lambda_max

        # Phase schedule
        self.phases = phases
        self._validate_phases()

        # PID controller
        self.pid = TemporalPIDController(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            lambda_min=0.0,
            lambda_max=lambda_max,
        )

        # Conditioning skip
        self._skip_conditioning = skip_conditioning
        self._conditioning_skipped = False

        # State
        self._global_step = 0  # total continuation tokens generated
        self._phase_idx = 0  # current phase index
        self._phase_step = 0  # steps within current phase

        # Diagnostics
        self.lambda_trajectory = []
        self.error_trajectory = []
        self.feature_activation_trajectory = []
        self.phase_trajectory = []  # which phase each step belongs to

        # Device
        self.device = None
        self._initialized = False

    def _validate_phases(self):
        for i, p in enumerate(self.phases):
            assert "tokens" in p, f"Phase {i} missing 'tokens'"
            assert "direction" in p, f"Phase {i} missing 'direction'"
            assert p["direction"] in (
                "up",
                "down",
                "hold",
            ), f"Phase {i} direction must be 'up', 'down', or 'hold'"
            if "ramp_steps" not in p:
                p["ramp_steps"] = 0

    def _initialize_device(self, activations):
        if not self._initialized:
            self.device = activations.device
            self.sas_vector_up = self.sas_vector_up.to(self.device)
            self.sas_vector_down = self.sas_vector_down.to(self.device)
            self._initialized = True

    def reset(self):
        """Reset all state for a new generation."""
        self.pid.reset()
        self._global_step = 0
        self._phase_idx = 0
        self._phase_step = 0
        self.lambda_trajectory = []
        self.error_trajectory = []
        self.feature_activation_trajectory = []
        self.phase_trajectory = []
        self._conditioning_skipped = False

    def _get_current_phase(self) -> Optional[Dict]:
        """Get the current phase dict, or None if all phases are done."""
        if self._phase_idx >= len(self.phases):
            return None
        return self.phases[self._phase_idx]

    def _advance_phase_if_needed(self):
        """Check if current phase is exhausted and advance to next."""
        phase = self._get_current_phase()
        if phase is None:
            return

        if self._phase_step >= phase["tokens"]:
            # Transition to next phase
            self._phase_idx += 1
            self._phase_step = 0

            new_phase = self._get_current_phase()
            if new_phase is not None:
                old_dir = phase["direction"]
                new_dir = new_phase["direction"]
                # Reset PID on direction change
                if new_dir != old_dir and new_dir != "hold":
                    self.pid.reset()
                    logger.debug(
                        f"Phase transition {self._phase_idx - 1} → {self._phase_idx}: "
                        f"{old_dir} → {new_dir}, PID reset"
                    )

    def _get_active_vector_and_indices(self):
        """Return the SAS vector and target indices for the current phase."""
        phase = self._get_current_phase()
        if phase is None or phase["direction"] == "hold":
            # During hold, maintain the PREVIOUS direction's vector
            # Look backward to find the last non-hold direction
            for i in range(self._phase_idx - 1, -1, -1):
                if self.phases[i]["direction"] != "hold":
                    if self.phases[i]["direction"] == "up":
                        return self.sas_vector_up, self.target_feature_indices_up
                    else:
                        return self.sas_vector_down, self.target_feature_indices_down
            # Default to up if no prior non-hold phase
            return self.sas_vector_up, self.target_feature_indices_up

        if phase["direction"] == "up":
            return self.sas_vector_up, self.target_feature_indices_up
        else:  # "down"
            return self.sas_vector_down, self.target_feature_indices_down

    def _compute_target_envelope(self) -> float:
        """Compute the PID setpoint for the current phase and step."""
        phase = self._get_current_phase()
        if phase is None:
            return self.target_magnitude

        ramp_steps = phase.get("ramp_steps", 0)

        if phase["direction"] == "hold":
            # Flat setpoint at target magnitude
            return self.target_magnitude

        if ramp_steps <= 0:
            return self.target_magnitude

        t = self._phase_step
        if t >= ramp_steps:
            return self.target_magnitude

        # Cosine ramp: 0 → target_magnitude within this phase
        progress = t / ramp_steps
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.target_magnitude * scale

    def __call__(self, module, input, output, layer_idx):
        """Forward hook implementing multi-phase PID-controlled SAS."""
        is_tuple_output = isinstance(output, tuple)
        if is_tuple_output:
            actual_output = output[0]
            other_outputs = output[1:]
        else:
            actual_output = output
            other_outputs = None

        self._initialize_device(actual_output)

        # Skip conditioning prefix (multi-token pass)
        batch_size, seq_len, d_model = actual_output.shape
        if self._skip_conditioning and not self._conditioning_skipped and seq_len > 1:
            self._conditioning_skipped = True
            return output

        # Check if all phases are exhausted — pass through unsteered
        self._advance_phase_if_needed()
        phase = self._get_current_phase()

        if phase is None:
            # All phases done, pass through without steering
            self.lambda_trajectory.append(0.0)
            self.error_trajectory.append(0.0)
            self.feature_activation_trajectory.append(0.0)
            self.phase_trajectory.append(-1)
            return output

        sae = self.sae_model
        v_sas, target_indices = self._get_active_vector_and_indices()

        a_l = actual_output
        a_flat = a_l.reshape(-1, d_model)

        # Step 1: Encode to sparse space
        f_a = sae.encode(a_flat)

        # Step 2: Correction term
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        # Step 3: Measure target feature activation
        target_activations = f_a[:, target_indices]
        actual_magnitude = target_activations.mean().item()

        # Compute setpoint
        desired_magnitude = self._compute_target_envelope()
        error = desired_magnitude - actual_magnitude

        # PID update
        lambda_t = self.pid.compute(error)

        # Record diagnostics
        self.lambda_trajectory.append(lambda_t)
        self.error_trajectory.append(error)
        self.feature_activation_trajectory.append(actual_magnitude)
        self.phase_trajectory.append(self._phase_idx)

        # Step 4: Apply steering
        s_l = f_a + lambda_t * v_sas.unsqueeze(0)
        s_l_relu = torch.relu(s_l)
        s_l_activated = sae.topk(s_l_relu)
        a_prime = sae.decode(s_l_activated)
        a_steered = a_prime + delta
        a_steered = a_steered.reshape(batch_size, seq_len, d_model)

        # Advance counters
        self._global_step += 1
        self._phase_step += 1

        if is_tuple_output:
            return (a_steered,) + other_outputs
        return a_steered

    def get_diagnostics(self) -> Dict:
        """Return diagnostic trajectories."""
        return {
            "lambda_trajectory": self.lambda_trajectory,
            "error_trajectory": self.error_trajectory,
            "feature_activation_trajectory": self.feature_activation_trajectory,
            "phase_trajectory": self.phase_trajectory,
            "phases": self.phases,
            "pid_gains": {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd},
            "n_steps": len(self.lambda_trajectory),
        }

    @property
    def total_tokens(self) -> int:
        """Total number of continuation tokens across all phases."""
        return sum(p["tokens"] for p in self.phases)
