"""Step D: Steered Generation with Interventions.

This module:
1. Loads pretrained model and steering vectors
2. Uses PyTorch hooks to intervene during generation
3. Applies steering vectors at specified layers with scaling factor alpha
4. Generates music with controlled attributes
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils


class SteeringHook:
    """Hook class to apply steering vectors during forward pass."""

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float,
        intervention_position: str = "last",
    ):
        """Initialize the steering hook.

        Args:
            steering_vector: The steering vector to add (shape: dim)
            alpha: Scaling factor for the steering vector
            intervention_position: Where to apply intervention ("last" or "all")
        """
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.intervention_position = intervention_position
        self.hook_handle = None

    def make_hook_fn(self):
        """Create the hook function.

        Returns:
            Hook function to be registered
        """

        def hook_fn(module, input, output):
            # output shape: (batch_size, seq_len, dim)

            if self.intervention_position == "last":
                # Apply intervention only to the last token
                # This is the summary position used for next-token prediction
                output[:, -1, :] = output[
                    :, -1, :
                ] + self.alpha * self.steering_vector.to(output.device)

            elif self.intervention_position == "all":
                # Apply intervention to all tokens
                output = output + self.alpha * self.steering_vector.to(output.device)

            return output

        return hook_fn

    def register(self, module: nn.Module):
        """Register this hook on a module.

        Args:
            module: The module to register the hook on
        """
        self.hook_handle = module.register_forward_hook(self.make_hook_fn())

    def remove(self):
        """Remove this hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class SteeredGenerator:
    """Generator with steering vector interventions."""

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: Dict[int, torch.Tensor],
        encoding: Dict,
    ):
        """Initialize the steered generator.

        Args:
            model: The MusicXTransformer model
            steering_vectors: Dictionary mapping layer_idx -> steering_vector
            encoding: Encoding dictionary
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.encoding = encoding
        self.hooks = []

        # Get the attention layers for hook registration
        if hasattr(self.model, "net"):
            self.transformer = self.model.net
        else:
            self.transformer = self.model

        if not hasattr(self.transformer, "attn_layers"):
            raise ValueError("Cannot find attn_layers in model")

        self.attn_layers = self.transformer.attn_layers

        if not hasattr(self.attn_layers, "layers"):
            raise ValueError("Cannot find layers in attn_layers")

        self.num_layers = len(self.attn_layers.layers)
        logging.info(f"Initialized steered generator with {self.num_layers} layers")

    def apply_steering(
        self,
        alpha: float,
        target_layers: Optional[List[int]] = None,
        intervention_position: str = "last",
    ):
        """Apply steering to specified layers.

        Args:
            alpha: Scaling factor for steering vectors
            target_layers: Which layers to intervene on (None = all layers)
            intervention_position: Where to apply intervention ("last" or "all")
        """
        # Remove any existing hooks first
        self.remove_steering()

        if target_layers is None:
            target_layers = list(range(self.num_layers))

        logging.info(f"Applying steering with alpha={alpha} to layers {target_layers}")

        for layer_idx in target_layers:
            if layer_idx not in self.steering_vectors:
                logging.warning(f"No steering vector for layer {layer_idx}, skipping")
                continue

            if layer_idx >= self.num_layers:
                logging.warning(f"Layer {layer_idx} out of range, skipping")
                continue

            # Create hook
            hook = SteeringHook(
                self.steering_vectors[layer_idx], alpha, intervention_position
            )

            # Register on the layer
            layer_module = self.attn_layers.layers[layer_idx]
            hook.register(layer_module)

            self.hooks.append(hook)

        logging.info(f"Registered {len(self.hooks)} steering hooks")

    def remove_steering(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        alpha: float = 0.0,
        target_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        """Generate with steering intervention.

        Args:
            start_tokens: Starting tokens (shape: batch, n, d)
            seq_len: Length to generate
            alpha: Steering strength
            target_layers: Which layers to intervene on
            **kwargs: Additional arguments for model.generate()

        Returns:
            Generated sequence
        """
        # Apply steering
        self.apply_steering(alpha, target_layers)

        # Generate
        try:
            generated = self.model.generate(start_tokens, seq_len, **kwargs)
        finally:
            # Always remove hooks after generation
            self.remove_steering()

        return generated


def load_steering_vectors(filepath: pathlib.Path) -> Dict[int, torch.Tensor]:
    """Load steering vectors from file.

    Args:
        filepath: Path to .pt file

    Returns:
        Dictionary mapping layer_idx -> steering_vector tensor
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    steering_vectors = checkpoint["steering_vectors"]
    metadata = checkpoint.get("metadata", {})

    logging.info(f"Loaded steering vectors from: {filepath}")
    logging.info(f"Concept: {metadata.get('concept', 'unknown')}")
    logging.info(f"Number of layers: {len(steering_vectors)}")

    return steering_vectors, metadata


def generate_with_steering(
    model: nn.Module,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    start_tokens: torch.Tensor,
    seq_len: int,
    alpha: float,
    target_layers: Optional[List[int]] = None,
    eos_token: int = None,
    temperature: float = 1.0,
    filter_logits_fn: str = "top_k",
    filter_thres: float = 0.9,
    monotonicity_dim: Optional[tuple] = None,
) -> torch.Tensor:
    """Helper function to generate with steering.

    Args:
        model: The model
        steering_vectors: Steering vectors
        encoding: Encoding dictionary
        start_tokens: Starting tokens
        seq_len: Generation length
        alpha: Steering strength
        target_layers: Layers to intervene on
        eos_token: End-of-sequence token
        temperature: Sampling temperature
        filter_logits_fn: Sampling filter
        filter_thres: Filter threshold
        monotonicity_dim: Monotonicity dimensions

    Returns:
        Generated tokens
    """
    generator = SteeredGenerator(model, steering_vectors, encoding)

    generated = generator.generate(
        start_tokens,
        seq_len,
        alpha=alpha,
        target_layers=target_layers,
        eos_token=eos_token,
        temperature=temperature,
        filter_logits_fn=filter_logits_fn,
        filter_thres=filter_thres,
        monotonicity_dim=monotonicity_dim,
    )

    return generated


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="Generate with steering interventions")
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to use"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None, help="Model checkpoint path"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.0],
        help="Steering strength values to test",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "generated_samples",
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup device
    if args.gpu is not None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = (
            config.OUTPUT_DIR
            / "steering_vectors"
            / f"{args.concept}_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    steering_vectors, metadata = load_steering_vectors(steering_path)

    # Move steering vectors to device
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Load model
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

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

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Loaded model")

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    # Test generation with different alphas
    for alpha in args.alpha:
        logging.info(f"\nGenerating with alpha={alpha}")

        output_subdir = args.output_dir / f"alpha_{alpha}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        for i in range(args.n_samples):
            # Create start tokens
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos

            # Generate
            generated = generate_with_steering(
                model,
                steering_vectors,
                encoding,
                tgt_start,
                args.seq_len,
                alpha,
                target_layers=None,  # Use all layers
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Combine start and generated
            full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]

            # Save as NPY
            np.save(output_subdir / f"sample_{i}.npy", full_seq)

            # Convert to music and save as MIDI
            music = representation.decode(full_seq, encoding)
            music.write(output_subdir / f"sample_{i}.mid")

            logging.info(f"Saved sample {i} to {output_subdir}")

    logging.info("Generation complete!")


if __name__ == "__main__":
    main()
