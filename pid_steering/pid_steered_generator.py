"""PID-Enhanced Steered Generator for Dense DiffMean Steering.

This module provides generation with PID-enhanced steering vectors.
It supports two modes:

1. Pre-computed PID: Vectors are computed offline by pid_vector_calculator.py
   and applied identically to standard DiffMean hooks (zero inference overhead).

2. Online Spatial PID: The PID controller runs during the forward pass,
   dynamically computing u(k) at each sublayer using the live error signal
   e(k) = mu_target(k) - h(k). This requires cached per-layer target means.

Both modes use the same hook registration pattern as the existing
SteeredGenerator in steering_interventions/steered_generator.py.
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_controller import SpatialPIDController
import config_pid

import music_x_transformers
import representation
import utils

logger = logging.getLogger(__name__)


class PIDSteeringHook:
    """Hook that applies a PID-computed steering vector during forward pass.

    For pre-computed mode, this is functionally identical to SteeringHook.
    For online mode, it dynamically computes the steering vector via PID.
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float,
        intervention_position: str = "last",
    ):
        """Initialize with a pre-computed PID vector.

        Args:
            steering_vector: PID-enhanced steering vector (dim,).
            alpha: Scaling factor for the steering vector.
            intervention_position: "last" (next-token prediction) or "all".
        """
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.intervention_position = intervention_position
        self.hook_handle = None

    def make_hook_fn(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                actual_output = output[0]
                is_tuple = True
            else:
                actual_output = output
                is_tuple = False

            if self.intervention_position == "last":
                actual_output[:, -1, :] = actual_output[
                    :, -1, :
                ] + self.alpha * self.steering_vector.to(actual_output.device)
            elif self.intervention_position == "all":
                actual_output = actual_output + self.alpha * self.steering_vector.to(
                    actual_output.device
                )

            if is_tuple:
                return (actual_output,) + output[1:]
            return actual_output

        return hook_fn

    def register(self, module: nn.Module):
        self.hook_handle = module.register_forward_hook(self.make_hook_fn())

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class OnlinePIDSteeringHook:
    """Hook that dynamically computes PID steering during the forward pass.

    Unlike PIDSteeringHook (pre-computed), this hook measures the actual
    hidden state at each sublayer, computes the error against a target mean,
    and applies the PID control law in real time.

    Requires per-layer target mean activations (mu_target) to be provided.
    """

    def __init__(
        self,
        target_mean: torch.Tensor,
        controller: SpatialPIDController,
        layer_idx: int,
        alpha: float = 1.0,
        intervention_position: str = "last",
    ):
        """Initialize the online PID hook.

        Args:
            target_mean: Target mean activation for this sublayer (dim,).
            controller: Shared SpatialPIDController instance.
            layer_idx: This hook's sublayer index (for controller ordering).
            alpha: Global scaling factor applied after PID computation.
            intervention_position: "last" or "all".
        """
        self.target_mean = target_mean
        self.controller = controller
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.intervention_position = intervention_position
        self.hook_handle = None

    def make_hook_fn(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                actual_output = output[0]
                is_tuple = True
            else:
                actual_output = output
                is_tuple = False

            device = actual_output.device
            target = self.target_mean.to(device)

            if self.intervention_position == "last":
                # Error: target - current hidden state at last token
                h_last = actual_output[:, -1, :]  # (batch, dim)
                # Use batch mean for error computation
                e_k = target - h_last.mean(dim=0)
                u_k = self.controller.compute(e_k)
                actual_output[:, -1, :] = actual_output[:, -1, :] + self.alpha * u_k
            elif self.intervention_position == "all":
                h_mean = actual_output.mean(dim=(0, 1))  # (dim,)
                e_k = target - h_mean
                u_k = self.controller.compute(e_k)
                actual_output = actual_output + self.alpha * u_k

            if is_tuple:
                return (actual_output,) + output[1:]
            return actual_output

        return hook_fn

    def register(self, module: nn.Module):
        self.hook_handle = module.register_forward_hook(self.make_hook_fn())

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class PIDSteeredGenerator:
    """Generator with PID-enhanced steering vector interventions.

    Supports both pre-computed PID vectors and online PID control.
    """

    def __init__(self, model: nn.Module, encoding: Dict):
        """Initialize the PID steered generator.

        Args:
            model: The MusicXTransformer model.
            encoding: Encoding dictionary.
        """
        self.model = model
        self.encoding = encoding
        self.hooks = []

        # Navigate model structure (same pattern as SteeredGenerator)
        if hasattr(self.model, "decoder"):
            decoder_wrapper = self.model.decoder
            if hasattr(decoder_wrapper, "net"):
                self.transformer = decoder_wrapper.net
            else:
                raise ValueError("Cannot find 'net' in model.decoder")
        elif hasattr(self.model, "net"):
            self.transformer = self.model.net
        else:
            raise ValueError("Cannot navigate model structure")

        if not hasattr(self.transformer, "attn_layers"):
            raise ValueError("Cannot find attn_layers in transformer")

        self.attn_layers = self.transformer.attn_layers
        if not hasattr(self.attn_layers, "layers"):
            raise ValueError("Cannot find layers in attn_layers")

        self.num_layers = len(self.attn_layers.layers)
        logger.info(f"PID generator initialized with {self.num_layers} sublayers")

    def _get_target_module(self, layer_idx):
        """Get the target module for hook registration at a given sublayer."""
        layer_module_list = self.attn_layers.layers[layer_idx]
        if isinstance(layer_module_list, nn.ModuleList) and len(layer_module_list) > 1:
            return layer_module_list[1]
        return layer_module_list

    def apply_precomputed_steering(
        self,
        pid_vectors: Dict[int, torch.Tensor],
        alpha: float,
        target_layers: Optional[List[int]] = None,
        intervention_position: str = "last",
    ):
        """Apply pre-computed PID steering vectors via hooks.

        Args:
            pid_vectors: Dict mapping sublayer_idx -> PID vector.
            alpha: Scaling factor.
            target_layers: Sublayers to intervene on (None = all available).
            intervention_position: "last" or "all".
        """
        self.remove_steering()

        if target_layers is None:
            target_layers = sorted(pid_vectors.keys())

        for layer_idx in target_layers:
            if layer_idx not in pid_vectors or layer_idx >= self.num_layers:
                continue

            hook = PIDSteeringHook(pid_vectors[layer_idx], alpha, intervention_position)
            hook.register(self._get_target_module(layer_idx))
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} PID hooks (alpha={alpha})")

    def apply_online_steering(
        self,
        target_means: Dict[int, torch.Tensor],
        Kp: float = 1.0,
        Ki: float = 0.3,
        Kd: float = 0.1,
        max_I: float = 5.0,
        alpha: float = 1.0,
        target_layers: Optional[List[int]] = None,
        intervention_position: str = "last",
    ):
        """Apply online PID steering with dynamic error computation.

        Args:
            target_means: Dict mapping sublayer_idx -> target mean activation.
            Kp, Ki, Kd, max_I: PID controller gains.
            alpha: Global scaling factor.
            target_layers: Sublayers to intervene on (None = all available).
            intervention_position: "last" or "all".
        """
        self.remove_steering()

        if target_layers is None:
            target_layers = sorted(target_means.keys())

        # Create a shared controller (state accumulates across sublayers)
        controller = SpatialPIDController(
            Kp=Kp, Ki=Ki, Kd=Kd, max_I=max_I, dim=config_pid.MODEL_DIM
        )

        for layer_idx in target_layers:
            if layer_idx not in target_means or layer_idx >= self.num_layers:
                continue

            hook = OnlinePIDSteeringHook(
                target_mean=target_means[layer_idx],
                controller=controller,
                layer_idx=layer_idx,
                alpha=alpha,
                intervention_position=intervention_position,
            )
            hook.register(self._get_target_module(layer_idx))
            self.hooks.append(hook)

        # Store controller reference so we can reset it per generation step
        self._online_controller = controller
        logger.info(
            f"Registered {len(self.hooks)} online PID hooks "
            f"(Kp={Kp}, Ki={Ki}, Kd={Kd})"
        )

    def remove_steering(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._online_controller = None

    def generate(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with PID steering active.

        Hooks must be applied before calling this method via
        apply_precomputed_steering() or apply_online_steering().

        Args:
            start_tokens: Starting tokens (batch, n, 6).
            seq_len: Number of tokens to generate.
            **kwargs: Additional arguments for model.generate().

        Returns:
            Generated token sequence.
        """
        try:
            generated = self.model.generate(start_tokens, seq_len, **kwargs)
        finally:
            self.remove_steering()

        return generated

    def generate_samples(
        self,
        pid_vectors: Dict[int, torch.Tensor],
        alpha: float,
        n_samples: int = 5,
        seq_len: int = 512,
        target_layers: Optional[List[int]] = None,
        device: torch.device = None,
        **gen_kwargs,
    ) -> List[np.ndarray]:
        """Generate multiple samples with pre-computed PID steering.

        Args:
            pid_vectors: PID-enhanced steering vectors.
            alpha: Steering strength.
            n_samples: Number of samples to generate.
            seq_len: Max sequence length.
            target_layers: Sublayers to intervene on.
            device: Torch device.
            **gen_kwargs: Additional generation kwargs.

        Returns:
            List of generated token arrays (each shape: seq_len, 6).
        """
        if device is None:
            device = next(self.model.parameters()).device

        sos = self.encoding["type_code_map"]["start-of-song"]
        eos = self.encoding["type_code_map"]["end-of-song"]

        gen_kwargs.setdefault("eos_token", eos)
        gen_kwargs.setdefault("temperature", config_pid.TEMPERATURE)
        gen_kwargs.setdefault("filter_logits_fn", "top_k")
        gen_kwargs.setdefault("filter_thres", config_pid.FILTER_THRESHOLD)
        gen_kwargs.setdefault("monotonicity_dim", ("type", "beat"))

        sequences = []
        for i in range(n_samples):
            # Apply hooks fresh for each sample
            self.apply_precomputed_steering(pid_vectors, alpha, target_layers)

            start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start[:, 0, 0] = sos

            with torch.no_grad():
                generated = self.model.generate(start, seq_len, **gen_kwargs)

            self.remove_steering()

            full_seq = torch.cat((start, generated), 1).cpu().numpy()[0]
            sequences.append(full_seq)

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_samples} samples")

        return sequences


def load_model(checkpoint_path, train_args_path, encoding_path, device):
    """Load the trained MMT model."""
    logger.info(f"Loading model from {checkpoint_path}")

    train_args = utils.load_json(train_args_path)
    encoding = representation.load_encoding(encoding_path)

    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info("Model loaded successfully")
    return model, encoding, train_args
