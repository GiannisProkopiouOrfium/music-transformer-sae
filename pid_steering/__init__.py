"""PID Activation Steering for Symbolic Music Generation.

Implements Proportional-Integral-Derivative (PID) controllers for
activation steering in the Multitrack Music Transformer (MMT).

Two steering paradigms:
- Spatial PID: Layer-by-layer control for Dense DiffMean steering
- Temporal PID: Token-by-token control for Sparse Activation Steering (SAS)
"""

from pid_steering.pid_controller import SpatialPIDController, TemporalPIDController

__all__ = [
    "SpatialPIDController",
    "TemporalPIDController",
]
