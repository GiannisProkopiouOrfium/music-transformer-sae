"""Temporal PID SAS Generator.

Wraps the MMT generation loop with a TemporalPIDSASHook at the target
layer (default: Layer 10). Handles SAE loading, vector loading, hook
registration, and multi-sample generation with diagnostics.
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.temporal_pid_sas_hook import TemporalPIDSASHook
import config_pid

import music_x_transformers
import representation
import utils

logger = logging.getLogger(__name__)


def load_sae_model(checkpoint_path, k, device):
    """Load a trained SAE model."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "sparse_steering"))
    from sae_model import SparseAutoencoder

    sae = SparseAutoencoder(
        input_dim=config_pid.MODEL_DIM,
        sparse_dim=config_pid.SPARSE_DIM,
        k=k,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        sae.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        sae.load_state_dict(checkpoint, strict=False)
    sae.to(device)
    sae.eval()
    return sae


def load_sas_vector(path, layer_idx=None):
    """Load a pre-computed SAS steering vector.

    Supports two formats:
    - .npy file: single numpy array
    - .pt file: dict {layer_idx: array/tensor} (from compute_sas_vectors.py)
    """
    path = str(path)
    if path.endswith(".pt"):
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if layer_idx is not None and layer_idx in data:
                vec = data[layer_idx]
            else:
                # Default to highest available layer
                layer_idx = max(data.keys())
                vec = data[layer_idx]
            return vec.numpy() if hasattr(vec, "numpy") else vec
        return data.numpy() if hasattr(data, "numpy") else data
    else:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            return data
        return data.item()


def get_target_feature_indices(sas_vector, top_n=32):
    """Extract indices of the most important features from the SAS vector.

    These are the features the PID controller monitors to ensure
    they survive the Top-K filter.

    Args:
        sas_vector: SAS steering vector (sparse_dim,).
        top_n: Number of top features to monitor.

    Returns:
        Array of feature indices.
    """
    magnitudes = np.abs(sas_vector)
    # Get indices of features with non-zero magnitude
    nonzero = np.nonzero(magnitudes)[0]
    if len(nonzero) <= top_n:
        return nonzero

    # Take the top_n by magnitude
    top_indices = np.argsort(magnitudes[nonzero])[-top_n:]
    return nonzero[top_indices]


def get_adaptive_k(layer_idx: int) -> int:
    """Get the adaptive K value for a given layer (matching SAE training)."""
    if layer_idx == 0:
        return 32
    elif layer_idx < 4:
        return 64
    elif layer_idx < 8:
        return 96
    else:
        return 128


class TemporalPIDSASGenerator:
    """Generator with temporal PID-controlled SAS steering."""

    def __init__(self, model, encoding, sae_model, sas_vector, target_layer=10):
        """Initialize the temporal PID SAS generator.

        Args:
            model: MusicXTransformer model.
            encoding: Encoding dictionary.
            sae_model: Trained SAE for the target layer.
            sas_vector: SAS steering vector (sparse_dim,) numpy array.
            target_layer: Sublayer index for SAS intervention (default: 10).
        """
        self.model = model
        self.encoding = encoding
        self.sae_model = sae_model
        self.sas_vector = sas_vector
        self.target_layer = target_layer

        # Navigate model structure
        if hasattr(self.model, "decoder"):
            self.transformer = self.model.decoder.net
        else:
            self.transformer = self.model.net

        self.attn_layers = self.transformer.attn_layers
        self.num_layers = len(self.attn_layers.layers)

        # Extract target features for PID error measurement
        self.target_feature_indices = get_target_feature_indices(sas_vector)
        logger.info(f"Monitoring {len(self.target_feature_indices)} target features")

    def _get_target_module(self, layer_idx):
        layer_module_list = self.attn_layers.layers[layer_idx]
        if isinstance(layer_module_list, nn.ModuleList) and len(layer_module_list) > 1:
            return layer_module_list[1]
        return layer_module_list

    def generate_with_temporal_pid(
        self,
        n_samples: int = 5,
        seq_len: int = 512,
        target_magnitude: float = 1.0,
        Kp: float = 1.0,
        Ki: float = 0.2,
        Kd: float = 0.1,
        max_I: float = 10.0,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,
        beat_mode: bool = False,
        n_ramp_beats: int = 32,
        device: torch.device = None,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate samples with temporal PID-controlled SAS steering.

        Args:
            n_samples: Number of samples to generate.
            seq_len: Maximum sequence length.
            target_magnitude: Desired feature activation (PID setpoint).
            Kp, Ki, Kd, max_I: PID controller gains.
            lambda_min, lambda_max: Output clamps for lambda_t.
            beat_mode: Advance PID per beat (not per token).
            n_ramp_beats: Number of beats for the desired ramp envelope.
            device: Torch device.

        Returns:
            (sequences, diagnostics_list) — generated token arrays and
            per-sample PID diagnostic dicts.
        """
        if device is None:
            device = next(self.model.parameters()).device

        sos = self.encoding["type_code_map"]["start-of-song"]
        eos = self.encoding["type_code_map"]["end-of-song"]

        sequences = []
        diagnostics_list = []

        for i in range(n_samples):
            # Create fresh PID hook for each sample
            pid_hook = TemporalPIDSASHook(
                sae_model=self.sae_model,
                sas_vector=self.sas_vector,
                target_feature_indices=self.target_feature_indices,
                target_magnitude=target_magnitude,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd,
                max_I=max_I,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                beat_mode=beat_mode,
                n_ramp_beats=n_ramp_beats,
            )
            pid_hook.reset()

            # Register hook at target layer
            target_module = self._get_target_module(self.target_layer)
            handle = target_module.register_forward_hook(
                lambda mod, inp, out, hook=pid_hook: hook(
                    mod, inp, out, self.target_layer
                )
            )

            try:
                start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
                start[:, 0, 0] = sos

                with torch.no_grad():
                    generated = self.model.generate(
                        start,
                        seq_len,
                        eos_token=eos,
                        temperature=config_pid.TEMPERATURE,
                        filter_logits_fn="top_k",
                        filter_thres=config_pid.FILTER_THRESHOLD,
                        monotonicity_dim=("type", "beat"),
                    )

                full_seq = torch.cat((start, generated), 1).cpu().numpy()[0]
                sequences.append(full_seq)
                diagnostics_list.append(pid_hook.get_diagnostics())

            finally:
                handle.remove()

            if (i + 1) % 5 == 0:
                diag = diagnostics_list[-1]
                avg_lambda = (
                    np.mean(diag["lambda_trajectory"])
                    if diag["lambda_trajectory"]
                    else 0
                )
                logger.info(
                    f"Generated {i + 1}/{n_samples} samples "
                    f"(avg λ_t={avg_lambda:.3f})"
                )

        return sequences, diagnostics_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate with temporal PID-controlled SAS steering"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--lambda_max", type=float, default=5.0)
    parser.add_argument("--beat_mode", action="store_true")
    parser.add_argument("--n_ramp_beats", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "temporal_sas",
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    from pid_steering.pid_steered_generator import load_model

    model, encoding, train_args = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    # Load SAE for target layer
    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    # Load SAS vector (format: {concept}_sas_vectors.pt with dict keyed by layer)
    vector_path = config_pid.SAS_VECTORS_DIR / f"{args.concept}_sas_vectors.pt"
    sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

    # Create generator
    generator = TemporalPIDSASGenerator(
        model, encoding, sae_model, sas_vector, args.target_layer
    )

    # Generate
    sequences, diagnostics = generator.generate_with_temporal_pid(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        target_magnitude=args.target_magnitude,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        beat_mode=args.beat_mode,
        n_ramp_beats=args.n_ramp_beats,
        device=device,
    )

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, seq in enumerate(sequences):
        np.save(args.output_dir / f"sample_{i}.npy", seq)

    # Save diagnostics
    import json

    for i, diag in enumerate(diagnostics):
        diag_path = args.output_dir / f"diagnostics_{i}.json"
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2, default=float)

    logger.info(f"Saved {len(sequences)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
