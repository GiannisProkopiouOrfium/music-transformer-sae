"""PID Controllers for Activation Steering.

Implements discrete-time PID controllers adapted from Nguyen et al. (ICLR 2026)
for the symbolic music domain:

- SpatialPIDController: Layer-by-layer control for Dense DiffMean steering.
  Operates across the 12 sublayers during a single token's forward pass.

- TemporalPIDController: Token-by-token control for Sparse Activation Steering.
  Operates across autoregressive generation steps at a single layer.

Both controllers implement anti-windup (integral clamping) to prevent
accumulator explosion under strong autoregressive resistance.

Reference: "Activation Steering with a Feedback Controller"
           Nguyen et al., ICLR 2026 (arXiv:2510.04309)
"""

import torch


class SpatialPIDController:
    """Layer-by-layer PID controller for Dense DiffMean steering.

    Computes a dynamic steering vector u(k) at each sublayer k:
        u(k) = Kp * e(k) + Ki * integral(k) + Kd * (e(k) - e(k-1))

    where e(k) is the error signal (difference-in-means vector) at sublayer k.

    The controller resets its state at the start of each token's forward pass
    and accumulates across the 12 sublayers within that pass.

    For non-sequential mode, the error e(k) is the pre-computed DiffMean vector
    r(k) at each layer. For sequential mode, e(k) is dynamically re-computed
    after each layer's intervention.
    """

    def __init__(self, Kp=1.0, Ki=0.3, Kd=0.1, max_I=5.0, dim=512):
        """Initialize the spatial PID controller.

        Args:
            Kp: Proportional gain. 1.0 = standard DiffMean strength.
            Ki: Integral gain. Accumulates past errors to remove steady-state bias.
            Kd: Derivative gain. Damps rapid changes to reduce overshoot.
            max_I: Anti-windup bound for the integral accumulator (element-wise).
            dim: Dimension of the activation space (512 for MMT).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_I = max_I
        self.dim = dim

        # Internal state (reset per token forward pass)
        self._integral = torch.zeros(dim)
        self._prev_error = torch.zeros(dim)
        self._step = 0

    def reset(self):
        """Reset controller state. Call at the start of each token's forward pass."""
        self._integral.zero_()
        self._prev_error.zero_()
        self._step = 0

    def compute(self, error):
        """Compute the PID steering vector for the current sublayer.

        Args:
            error: Error signal e(k) at the current sublayer.
                   For non-sequential PID, this is the pre-computed DiffMean
                   vector r(k). Shape: (dim,) tensor or numpy array.

        Returns:
            u(k): The PID-enhanced steering vector. Shape: (dim,) tensor.
        """
        if not isinstance(error, torch.Tensor):
            error = torch.as_tensor(error, dtype=torch.float32)

        # Move error to same device as internal state
        device = self._integral.device
        error = error.to(device)

        # Proportional term: Kp * e(k)
        p_term = self.Kp * error

        # Integral term: Ki * Σ_{j=0}^{k-1} e(j) (previous errors only, per Eq. 18)
        i_term = self.Ki * self._integral

        # Update integral accumulator with current error for next step
        self._integral = self._integral + error
        self._integral = self._integral.clamp(-self.max_I, self.max_I)

        # Derivative term: Kd * (e(k) - e(k-1))
        if self._step == 0:
            d_term = torch.zeros_like(error)
        else:
            d_term = self.Kd * (error - self._prev_error)

        # Update state
        self._prev_error = error.clone()
        self._step += 1

        # u(k) = P + I + D
        return p_term + i_term + d_term

    def to(self, device):
        """Move controller state to a device."""
        self._integral = self._integral.to(device)
        self._prev_error = self._prev_error.to(device)
        return self

    @property
    def gains(self):
        """Return current gain values."""
        return {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd, "max_I": self.max_I}


class TemporalPIDController:
    """Token-by-token PID controller for Sparse Activation Steering.

    Computes a dynamic steering multiplier λ_t at each autoregressive step t:
        λ_t = Kp * e(t) + Ki * integral(t) + Kd * (e(t) - e(t-1))

    where e(t) measures whether target sparse features survived the Top-K
    filter at time step t. If features are zeroed out by Top-K, e(t) remains
    high and the integral term accumulates, pushing λ_t higher until it
    breaches the threshold. Once features activate, the derivative term
    damps λ_t to prevent explosion.

    The controller maintains state across the full autoregressive generation
    and resets at the start of each new generation.
    """

    def __init__(
        self, Kp=1.0, Ki=0.10, Kd=0.05, max_I=10.0, lambda_min=0.0, lambda_max=5.0
    ):
        """Initialize the temporal PID controller.

        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
            max_I: Anti-windup bound for the integral sum.
            lambda_min: Minimum allowed steering multiplier.
            lambda_max: Maximum allowed steering multiplier.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_I = max_I
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Internal state (reset per generation)
        self._integral = 0.0
        self._prev_error = 0.0
        self._step = 0

    def reset(self):
        """Reset controller state. Call at the start of each generation."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._step = 0

    def compute(self, error):
        """Compute the dynamic steering multiplier for the current token.

        Args:
            error: Scalar error signal e(t). Positive = target features are
                   under-expressed (below Top-K threshold). Negative = features
                   are over-expressed.

        Returns:
            lambda_t: The PID-computed steering multiplier (scalar, clamped).
        """
        error = float(error)

        # Proportional term
        p_term = self.Kp * error

        # Integral term: uses sum of previous errors (per paper Eq. 18)
        i_term = self.Ki * self._integral

        # Update integral with current error for next step, with anti-windup
        self._integral += error
        self._integral = max(-self.max_I, min(self.max_I, self._integral))

        # Derivative term
        if self._step == 0:
            d_term = 0.0
        else:
            d_term = self.Kd * (error - self._prev_error)

        # Update state
        self._prev_error = error
        self._step += 1

        # Compute and clamp output
        lambda_t = p_term + i_term + d_term
        lambda_t = max(self.lambda_min, min(self.lambda_max, lambda_t))

        return lambda_t

    @property
    def gains(self):
        """Return current gain values."""
        return {
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "max_I": self.max_I,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
        }
