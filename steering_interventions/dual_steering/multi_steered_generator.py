#!/usr/bin/env python3
"""Multi-Concept Steered Generator for Dual Steering.

Extends the single-concept SteeredGenerator to support simultaneous control
of pitch and modality using composed steering vectors.

Usage:
    from vector_composition import VectorComposer

    # Create composer
    composer = VectorComposer(pitch_vectors, modality_vectors)

    # Generate with dual steering
    generator = MultiSteeringGenerator(
        model=model,
        composer=composer,
        alpha_pitch=1.5,
        alpha_modality=2.0,
        strategy="direct",  # or "gram_schmidt"
        intervention_position="last"
    )

    outputs = generator.generate(
        primer=primer,
        target_seq_length=512,
        beam=0,
        beam_chance=1.0
    )
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from steered_generator import SteeringHook
from vector_composition import CompositionStrategy, VectorComposer

InterventionPosition = Literal["last", "all"]


class MultiSteeringGenerator:
    """Generator with simultaneous pitch and modality steering."""

    def __init__(
        self,
        model: torch.nn.Module,
        composer: VectorComposer,
        alpha_pitch: float,
        alpha_modality: float,
        strategy: CompositionStrategy = "direct",
        intervention_position: InterventionPosition = "last",
        layer_range: Optional[List[int]] = None,
    ):
        """Initialize multi-concept steering generator.

        Args:
            model: The transformer model to apply steering to
            composer: VectorComposer with pitch and modality vectors
            alpha_pitch: Scaling factor for pitch steering
            alpha_modality: Scaling factor for modality steering
            strategy: Vector composition strategy ("direct" or "gram_schmidt")
            intervention_position: Where to apply steering ("last" or "all" tokens)
            layer_range: Optional list of layer indices to apply steering to.
                        If None, applies to all layers.
        """
        self.model = model
        self.composer = composer
        self.alpha_pitch = alpha_pitch
        self.alpha_modality = alpha_modality
        self.strategy = strategy
        self.intervention_position = intervention_position

        # Compose vectors using specified strategy
        self.composed_vectors = composer.compose(alpha_pitch, alpha_modality, strategy)

        # Determine which layers to apply steering to
        if layer_range is None:
            self.target_layers = sorted(self.composed_vectors.keys())
        else:
            self.target_layers = [
                layer for layer in layer_range if layer in self.composed_vectors
            ]

        logging.info(
            f"MultiSteeringGenerator initialized: strategy={strategy}, "
            f"α_pitch={alpha_pitch}, α_modality={alpha_modality}, "
            f"position={intervention_position}, layers={len(self.target_layers)}"
        )

        # Get the attention layers for hook registration
        # Navigate through the model structure: MusicXTransformer -> .decoder -> .net -> .attn_layers
        if hasattr(self.model, "decoder"):
            decoder_wrapper = self.model.decoder
            if hasattr(decoder_wrapper, "net"):
                self.transformer = decoder_wrapper.net
            else:
                raise ValueError("Cannot find 'net' in model.decoder")
        elif hasattr(self.model, "net"):
            # Fallback: direct access to net
            self.transformer = self.model.net
        else:
            raise ValueError(
                "Cannot navigate model structure - no 'decoder' or 'net' attribute"
            )

        if not hasattr(self.transformer, "attn_layers"):
            raise ValueError("Cannot find attn_layers in transformer")

        self.attn_layers = self.transformer.attn_layers

        if not hasattr(self.attn_layers, "layers"):
            raise ValueError("Cannot find layers in attn_layers")

        # Store hooks
        self.hooks = []
        self.is_active = False

    def apply_steering(self):
        """Apply steering hooks to model layers."""
        if self.is_active:
            logging.warning("Steering already active, skipping")
            return

        # Get model layers (already resolved in __init__)
        layers = self.attn_layers.layers

        # Apply hooks to target layers
        for layer_idx in self.target_layers:
            if layer_idx >= len(layers):
                logging.warning(
                    f"Layer {layer_idx} out of range (model has {len(layers)} layers)"
                )
                continue

            # Create hook with composed steering vector
            # Note: alpha is already applied in compose(), so we pass alpha=1.0
            hook = SteeringHook(
                steering_vector=self.composed_vectors[layer_idx],
                alpha=1.0,  # Alpha already applied during composition
                intervention_position=self.intervention_position,
            )

            # Register on the layer
            # Each layer is a ModuleList containing [prenorm, attention/feedforward, residual]
            # We want to hook the attention/feedforward module (typically index 1)
            layer_module_list = layers[layer_idx]
            if (
                isinstance(layer_module_list, torch.nn.ModuleList)
                and len(layer_module_list) > 1
            ):
                # Hook the Attention/FeedForward module at index 1
                target_module = layer_module_list[1]
            else:
                # Fallback: hook the whole layer
                target_module = layer_module_list

            hook.register(target_module)
            self.hooks.append(hook)

        self.is_active = True
        logging.info(f"Applied {len(self.hooks)} steering hooks")

    def remove_steering(self):
        """Remove all steering hooks from model."""
        if not self.is_active:
            return

        for hook in self.hooks:
            hook.remove()

        self.hooks = []
        self.is_active = False
        logging.info("Removed all steering hooks")

    def generate(
        self, start_tokens: torch.Tensor, seq_len: int, **kwargs
    ) -> torch.Tensor:
        """Generate with steering applied.

        Args:
            start_tokens: Input primer sequence [batch, seq_len, features]
            seq_len: Target sequence length to generate
            **kwargs: Additional arguments for model.generate()

        Returns:
            Generated sequence [batch, seq_len, features]
        """
        try:
            self.apply_steering()
            with torch.no_grad():
                output = self.model.generate(start_tokens, seq_len, **kwargs)
            return output
        finally:
            self.remove_steering()

    def get_steering_info(self) -> Dict:
        """Get information about current steering configuration.

        Returns:
            Dictionary with steering configuration details
        """
        return {
            "strategy": self.strategy,
            "alpha_pitch": self.alpha_pitch,
            "alpha_modality": self.alpha_modality,
            "intervention_position": self.intervention_position,
            "target_layers": self.target_layers,
            "num_layers": len(self.target_layers),
            "is_active": self.is_active,
        }

    def __enter__(self):
        """Context manager entry: apply steering."""
        self.apply_steering()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: remove steering."""
        self.remove_steering()

    def __del__(self):
        """Cleanup: ensure hooks are removed."""
        if self.is_active:
            self.remove_steering()


def create_multi_steering_generator(
    model: torch.nn.Module,
    pitch_vectors_path: str,
    modality_vectors_path: str,
    alpha_pitch: float,
    alpha_modality: float,
    strategy: CompositionStrategy = "direct",
    intervention_position: InterventionPosition = "last",
    layer_range: Optional[List[int]] = None,
) -> MultiSteeringGenerator:
    """Create MultiSteeringGenerator from vector files.

    Args:
        model: Transformer model
        pitch_vectors_path: Path to pitch steering vectors (.pt)
        modality_vectors_path: Path to modality steering vectors (.pt)
        alpha_pitch: Pitch scaling factor
        alpha_modality: Modality scaling factor
        strategy: Composition strategy
        intervention_position: Where to apply steering
        layer_range: Optional layer indices to target

    Returns:
        MultiSteeringGenerator instance
    """
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(pitch_vectors_path, modality_vectors_path)

    return MultiSteeringGenerator(
        model=model,
        composer=composer,
        alpha_pitch=alpha_pitch,
        alpha_modality=alpha_modality,
        strategy=strategy,
        intervention_position=intervention_position,
        layer_range=layer_range,
    )


if __name__ == "__main__":
    """Test multi-steering generator creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Test multi-steering generator")
    parser.add_argument(
        "--pitch_vectors",
        default="steering_interventions/outputs/steering_vectors/average_pitch_steering_vectors.pt",
        help="Path to pitch vectors",
    )
    parser.add_argument(
        "--modality_vectors",
        default="steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt",
        help="Path to modality vectors",
    )
    parser.add_argument(
        "--alpha_pitch", type=float, default=1.5, help="Pitch scaling factor"
    )
    parser.add_argument(
        "--alpha_modality", type=float, default=2.0, help="Modality scaling factor"
    )
    parser.add_argument(
        "--strategy",
        choices=["direct", "gram_schmidt"],
        default="direct",
        help="Composition strategy",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("MULTI-STEERING GENERATOR TEST")
    print("=" * 60)

    # Load model (placeholder - would need actual model in production)
    print("\nNote: This is a configuration test only.")
    print("To actually generate, provide a loaded model instance.")

    # Create composer
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(args.pitch_vectors, args.modality_vectors)

    print(f"\nComposer loaded with {len(composer.layers)} layers")
    print(f"Strategy: {args.strategy}")
    print(f"Alpha values: pitch={args.alpha_pitch}, modality={args.alpha_modality}")

    # Analyze composition
    analysis = composer.analyze_composition(
        args.strategy, args.alpha_pitch, args.alpha_modality
    )

    print("\nComposed vector statistics:")
    print(f"  Mean norm:              {analysis['mean_norm']:.4f}")
    print(f"  Std norm:               {analysis['std_norm']:.4f}")
    print(f"  Mean pitch contrib:     {analysis['mean_pitch_contrib']:.4f}")
    print(f"  Mean modality contrib:  {analysis['mean_modality_contrib']:.4f}")

    print("\n" + "=" * 60)
    print("Configuration validated successfully!")
    print("=" * 60)
