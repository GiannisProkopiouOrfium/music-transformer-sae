#!/usr/bin/env python3
"""Vector Composition Strategies for Dual-Concept Steering.

This module implements different strategies for combining two steering vectors
(pitch and modality) based on Phase 0 orthogonality analysis results.

Strategies implemented:
1. Direct Addition: Simple weighted sum of vectors
2. Gram-Schmidt Orthogonalization: Remove pitch component from modality vector

Usage:
    composer = VectorComposer(pitch_vectors, modality_vectors)

    # Direct addition
    combined = composer.compose(
        alpha_pitch=1.5,
        alpha_modality=2.0,
        strategy="direct"
    )

    # Gram-Schmidt (orthogonalize modality w.r.t. pitch)
    combined = composer.compose(
        alpha_pitch=1.5,
        alpha_modality=2.0,
        strategy="gram_schmidt"
    )
"""

import logging
from typing import Dict, Literal

import torch

CompositionStrategy = Literal["direct", "gram_schmidt"]


class VectorComposer:
    """Compose two steering vectors using different strategies."""

    def __init__(
        self,
        pitch_vectors: Dict[int, torch.Tensor],
        modality_vectors: Dict[int, torch.Tensor],
    ):
        """Initialize composer with pitch and modality steering vectors.

        Args:
            pitch_vectors: Dictionary mapping layer indices to pitch steering vectors
            modality_vectors: Dictionary mapping layer indices to modality steering vectors

        Raises:
            ValueError: If layer indices don't match between the two vector sets
        """
        self.pitch_vectors = pitch_vectors
        self.modality_vectors = modality_vectors

        # Verify layers match
        pitch_layers = set(pitch_vectors.keys())
        modality_layers = set(modality_vectors.keys())

        if pitch_layers != modality_layers:
            raise ValueError(
                f"Layer mismatch: pitch has {len(pitch_layers)} layers, "
                f"modality has {len(modality_layers)} layers"
            )

        self.layers = sorted(pitch_layers)
        logging.info(f"VectorComposer initialized with {len(self.layers)} layers")

    def compose(
        self,
        alpha_pitch: float,
        alpha_modality: float,
        strategy: CompositionStrategy = "direct",
    ) -> Dict[int, torch.Tensor]:
        """Compose pitch and modality vectors using specified strategy.

        Args:
            alpha_pitch: Scaling factor for pitch steering vector
            alpha_modality: Scaling factor for modality steering vector
            strategy: Composition strategy ("direct" or "gram_schmidt")

        Returns:
            Dictionary mapping layer indices to composed steering vectors

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy == "direct":
            return self._direct_addition(alpha_pitch, alpha_modality)
        elif strategy == "gram_schmidt":
            return self._gram_schmidt(alpha_pitch, alpha_modality)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be 'direct' or 'gram_schmidt'"
            )

    def _direct_addition(
        self, alpha_pitch: float, alpha_modality: float
    ) -> Dict[int, torch.Tensor]:
        """Direct addition: combined = α_pitch * v_pitch + α_modality * v_modality.

        This is the baseline strategy. Simply scales each vector by its alpha
        and adds them together. Works well when vectors are already orthogonal.

        Args:
            alpha_pitch: Scaling factor for pitch vector
            alpha_modality: Scaling factor for modality vector

        Returns:
            Dictionary of combined vectors per layer
        """
        combined = {}

        for layer in self.layers:
            v_pitch = self.pitch_vectors[layer]
            v_modality = self.modality_vectors[layer]

            # Simple weighted sum
            combined[layer] = alpha_pitch * v_pitch + alpha_modality * v_modality

        return combined

    def _gram_schmidt(
        self, alpha_pitch: float, alpha_modality: float
    ) -> Dict[int, torch.Tensor]:
        """Gram-Schmidt orthogonalization: Remove pitch component from modality.

        This strategy orthogonalizes the modality vector with respect to the pitch
        vector before combining them. This ensures the two concepts don't interfere.

        Process:
        1. Keep pitch vector as-is: v_pitch_orth = v_pitch
        2. Remove pitch component from modality: v_modality_orth = v_modality - proj(v_modality onto v_pitch)
        3. Combine: combined = α_pitch * v_pitch_orth + α_modality * v_modality_orth

        Args:
            alpha_pitch: Scaling factor for pitch vector
            alpha_modality: Scaling factor for modality vector

        Returns:
            Dictionary of combined vectors per layer
        """
        combined = {}

        for layer in self.layers:
            v_pitch = self.pitch_vectors[layer].flatten()
            v_modality = self.modality_vectors[layer].flatten()

            # Orthogonalize modality w.r.t. pitch using Gram-Schmidt
            # projection = (v_modality · v_pitch / ||v_pitch||²) * v_pitch
            pitch_norm_sq = torch.dot(v_pitch, v_pitch)

            if pitch_norm_sq > 1e-8:
                projection = (torch.dot(v_modality, v_pitch) / pitch_norm_sq) * v_pitch
                v_modality_orth = v_modality - projection
            else:
                # Edge case: pitch vector is near-zero, use original modality
                v_modality_orth = v_modality
                logging.warning(
                    f"Layer {layer}: Pitch vector near-zero, skipping orthogonalization"
                )

            # Combine orthogonalized vectors
            combined[layer] = alpha_pitch * v_pitch + alpha_modality * v_modality_orth

        return combined

    def analyze_composition(
        self, strategy: CompositionStrategy, alpha_pitch: float, alpha_modality: float
    ) -> Dict[str, float]:
        """Analyze properties of composed vectors for given strategy.

        Args:
            strategy: Composition strategy to analyze
            alpha_pitch: Pitch scaling factor
            alpha_modality: Modality scaling factor

        Returns:
            Dictionary with analysis metrics:
            - mean_norm: Average L2 norm of combined vectors
            - std_norm: Standard deviation of norms
            - mean_pitch_contrib: Average contribution of pitch component
            - mean_modality_contrib: Average contribution of modality component
        """
        combined = self.compose(alpha_pitch, alpha_modality, strategy)

        norms = []
        pitch_contribs = []
        modality_contribs = []

        for layer in self.layers:
            combined_vec = combined[layer].flatten()
            v_pitch = (alpha_pitch * self.pitch_vectors[layer]).flatten()
            v_modality = (alpha_modality * self.modality_vectors[layer]).flatten()

            # Compute norm
            norms.append(float(combined_vec.norm()))

            # Compute contributions (as fraction of combined norm)
            combined_norm = combined_vec.norm()
            if combined_norm > 1e-8:
                pitch_contribs.append(float(v_pitch.norm() / combined_norm))
                modality_contribs.append(float(v_modality.norm() / combined_norm))
            else:
                pitch_contribs.append(0.0)
                modality_contribs.append(0.0)

        return {
            "mean_norm": float(torch.tensor(norms).mean()),
            "std_norm": float(torch.tensor(norms).std()),
            "mean_pitch_contrib": float(torch.tensor(pitch_contribs).mean()),
            "mean_modality_contrib": float(torch.tensor(modality_contribs).mean()),
        }


def load_and_create_composer(
    pitch_vectors_path: str, modality_vectors_path: str
) -> VectorComposer:
    """Load steering vectors and create composer.

    Args:
        pitch_vectors_path: Path to pitch steering vectors (.pt file)
        modality_vectors_path: Path to modality steering vectors (.pt file)

    Returns:
        VectorComposer instance
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from steered_generator import load_steering_vectors

    pitch_vectors, _ = load_steering_vectors(pitch_vectors_path)
    modality_vectors, _ = load_steering_vectors(modality_vectors_path)

    return VectorComposer(pitch_vectors, modality_vectors)


if __name__ == "__main__":
    """Quick test of composition strategies."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vector composition strategies")
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create composer
    composer = load_and_create_composer(args.pitch_vectors, args.modality_vectors)

    # Test both strategies
    for strategy in ["direct", "gram_schmidt"]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print("=" * 60)

        analysis = composer.analyze_composition(
            strategy, args.alpha_pitch, args.alpha_modality
        )

        print(f"Alpha values: pitch={args.alpha_pitch}, modality={args.alpha_modality}")
        print("\nCombined vector statistics:")
        print(f"  Mean norm:              {analysis['mean_norm']:.4f}")
        print(f"  Std norm:               {analysis['std_norm']:.4f}")
        print(f"  Mean pitch contrib:     {analysis['mean_pitch_contrib']:.4f}")
        print(f"  Mean modality contrib:  {analysis['mean_modality_contrib']:.4f}")
