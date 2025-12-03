#!/usr/bin/env python3
"""Vector Composition Strategies for Dual-Concept Steering.

This module implements different strategies for combining two steering vectors
(pitch and duration) based on Phase 0 orthogonality analysis results.

Strategies implemented:
1. Direct Addition: Simple weighted sum of vectors
2. Gram-Schmidt Duration: Remove pitch component from duration vector
3. Gram-Schmidt Pitch: Remove duration component from pitch vector

Usage:
    composer = VectorComposer(pitch_vectors, duration_vectors)

    # Direct addition
    combined = composer.compose(
        alpha_pitch=1.5,
        alpha_duration=2.0,
        strategy="direct"
    )

    # Gram-Schmidt (orthogonalize duration w.r.t. pitch)
    combined = composer.compose(
        alpha_pitch=1.5,
        alpha_duration=2.0,
        strategy="gram_schmidt_duration"
    )

    # Gram-Schmidt (orthogonalize pitch w.r.t. duration)
    combined = composer.compose(
        alpha_pitch=1.5,
        alpha_duration=2.0,
        strategy="gram_schmidt_pitch"
    )
"""

import logging
from typing import Dict, Literal

import torch

CompositionStrategy = Literal["direct", "gram_schmidt_duration", "gram_schmidt_pitch"]


class VectorComposer:
    """Compose two steering vectors using different strategies."""

    def __init__(
        self,
        pitch_vectors: Dict[int, torch.Tensor],
        duration_vectors: Dict[int, torch.Tensor],
    ):
        """Initialize composer with pitch and duration steering vectors.

        Args:
            pitch_vectors: Dictionary mapping layer indices to pitch steering vectors
            duration_vectors: Dictionary mapping layer indices to duration steering vectors

        Raises:
            ValueError: If layer indices don't match between the two vector sets
        """
        self.pitch_vectors = pitch_vectors
        self.duration_vectors = duration_vectors

        # Verify layers match
        pitch_layers = set(pitch_vectors.keys())
        duration_layers = set(duration_vectors.keys())

        if pitch_layers != duration_layers:
            raise ValueError(
                f"Layer mismatch: pitch has {len(pitch_layers)} layers, "
                f"duration has {len(duration_layers)} layers"
            )

        self.layers = sorted(pitch_layers)
        logging.info(f"VectorComposer initialized with {len(self.layers)} layers")

    def compose(
        self,
        alpha_pitch: float,
        alpha_duration: float,
        strategy: CompositionStrategy = "direct",
    ) -> Dict[int, torch.Tensor]:
        """Compose pitch and duration vectors using specified strategy.

        Args:
            alpha_pitch: Scaling factor for pitch steering vector
            alpha_duration: Scaling factor for duration steering vector
            strategy: Composition strategy ("direct", "gram_schmidt_duration", or "gram_schmidt_pitch")

        Returns:
            Dictionary mapping layer indices to composed steering vectors

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy == "direct":
            return self._direct_addition(alpha_pitch, alpha_duration)
        elif strategy == "gram_schmidt_duration":
            return self._gram_schmidt_duration(alpha_pitch, alpha_duration)
        elif strategy == "gram_schmidt_pitch":
            return self._gram_schmidt_pitch(alpha_pitch, alpha_duration)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be 'direct', 'gram_schmidt_duration', or 'gram_schmidt_pitch'"
            )

    def _direct_addition(
        self, alpha_pitch: float, alpha_duration: float
    ) -> Dict[int, torch.Tensor]:
        """Direct addition: combined = α_pitch * v_pitch + α_duration * v_duration.

        This is the baseline strategy. Simply scales each vector by its alpha
        and adds them together. Works well when vectors are already orthogonal.

        Args:
            alpha_pitch: Scaling factor for pitch vector
            alpha_duration: Scaling factor for duration vector

        Returns:
            Dictionary of combined vectors per layer
        """
        combined = {}

        for layer in self.layers:
            v_pitch = self.pitch_vectors[layer]
            v_duration = self.duration_vectors[layer]

            # Simple weighted sum
            combined[layer] = alpha_pitch * v_pitch + alpha_duration * v_duration

        return combined

    def _gram_schmidt_duration(
        self, alpha_pitch: float, alpha_duration: float
    ) -> Dict[int, torch.Tensor]:
        """Gram-Schmidt orthogonalization: Remove pitch component from duration.

        This strategy orthogonalizes the duration vector with respect to the pitch
        vector before combining them. This ensures the two concepts don't interfere.

        Process:
        1. Keep pitch vector as-is: v_pitch_orth = v_pitch
        2. Remove pitch component from duration: v_duration_orth = v_duration - proj(v_duration onto v_pitch)
        3. Combine: combined = α_pitch * v_pitch_orth + α_duration * v_duration_orth

        Args:
            alpha_pitch: Scaling factor for pitch vector
            alpha_duration: Scaling factor for duration vector

        Returns:
            Dictionary of combined vectors per layer
        """
        combined = {}

        for layer in self.layers:
            v_pitch = self.pitch_vectors[layer].flatten()
            v_duration = self.duration_vectors[layer].flatten()

            # Orthogonalize duration w.r.t. pitch using Gram-Schmidt
            # projection = (v_duration · v_pitch / ||v_pitch||²) * v_pitch
            pitch_norm_sq = torch.dot(v_pitch, v_pitch)

            if pitch_norm_sq > 1e-8:
                projection = (torch.dot(v_duration, v_pitch) / pitch_norm_sq) * v_pitch
                v_duration_orth = v_duration - projection
            else:
                # Edge case: pitch vector is near-zero, use original duration
                v_duration_orth = v_duration
                logging.warning(
                    f"Layer {layer}: Pitch vector near-zero, skipping orthogonalization"
                )

            # Combine orthogonalized vectors
            combined[layer] = alpha_pitch * v_pitch + alpha_duration * v_duration_orth

        return combined

    def _gram_schmidt_pitch(
        self, alpha_pitch: float, alpha_duration: float
    ) -> Dict[int, torch.Tensor]:
        """Gram-Schmidt orthogonalization: Remove duration component from pitch.

        This strategy orthogonalizes the pitch vector with respect to the duration
        vector before combining them. This is the opposite direction from
        _gram_schmidt_duration.

        Process:
        1. Keep duration vector as-is: v_duration_orth = v_duration
        2. Remove duration component from pitch: v_pitch_orth = v_pitch - proj(v_pitch onto v_duration)
        3. Combine: combined = α_pitch * v_pitch_orth + α_duration * v_duration_orth

        Args:
            alpha_pitch: Scaling factor for pitch vector
            alpha_duration: Scaling factor for duration vector

        Returns:
            Dictionary of combined vectors per layer
        """
        combined = {}

        for layer in self.layers:
            v_pitch = self.pitch_vectors[layer].flatten()
            v_duration = self.duration_vectors[layer].flatten()

            # Orthogonalize pitch w.r.t. duration using Gram-Schmidt
            # projection = (v_pitch · v_duration / ||v_duration||²) * v_duration
            duration_norm_sq = torch.dot(v_duration, v_duration)

            if duration_norm_sq > 1e-8:
                projection = (
                    torch.dot(v_pitch, v_duration) / duration_norm_sq
                ) * v_duration
                v_pitch_orth = v_pitch - projection
            else:
                # Edge case: duration vector is near-zero, use original pitch
                v_pitch_orth = v_pitch
                logging.warning(
                    f"Layer {layer}: Duration vector near-zero, skipping orthogonalization"
                )

            # Combine orthogonalized vectors
            combined[layer] = alpha_pitch * v_pitch_orth + alpha_duration * v_duration

        return combined

    def analyze_composition(
        self, strategy: CompositionStrategy, alpha_pitch: float, alpha_duration: float
    ) -> Dict[str, float]:
        """Analyze properties of composed vectors for given strategy.

        Args:
            strategy: Composition strategy to analyze
            alpha_pitch: Pitch scaling factor
            alpha_duration: Duration scaling factor

        Returns:
            Dictionary with analysis metrics:
            - mean_norm: Average L2 norm of combined vectors
            - std_norm: Standard deviation of norms
            - mean_pitch_contrib: Average contribution of pitch component
            - mean_duration_contrib: Average contribution of duration component
        """
        combined = self.compose(alpha_pitch, alpha_duration, strategy)

        norms = []
        pitch_contribs = []
        duration_contribs = []

        for layer in self.layers:
            combined_vec = combined[layer].flatten()
            v_pitch = (alpha_pitch * self.pitch_vectors[layer]).flatten()
            v_duration = (alpha_duration * self.duration_vectors[layer]).flatten()

            # Compute norm
            norms.append(float(combined_vec.norm()))

            # Compute contributions (as fraction of combined norm)
            combined_norm = combined_vec.norm()
            if combined_norm > 1e-8:
                pitch_contribs.append(float(v_pitch.norm() / combined_norm))
                duration_contribs.append(float(v_duration.norm() / combined_norm))
            else:
                pitch_contribs.append(0.0)
                duration_contribs.append(0.0)

        return {
            "mean_norm": float(torch.tensor(norms).mean()),
            "std_norm": float(torch.tensor(norms).std()),
            "mean_pitch_contrib": float(torch.tensor(pitch_contribs).mean()),
            "mean_duration_contrib": float(torch.tensor(duration_contribs).mean()),
        }


def load_and_create_composer(
    pitch_vectors_path: str, duration_vectors_path: str
) -> VectorComposer:
    """Load steering vectors and create composer.

    Args:
        pitch_vectors_path: Path to pitch steering vectors (.pt file)
        duration_vectors_path: Path to duration steering vectors (.pt file)

    Returns:
        VectorComposer instance
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from steered_generator import load_steering_vectors

    pitch_vectors, _ = load_steering_vectors(pitch_vectors_path)
    duration_vectors, _ = load_steering_vectors(duration_vectors_path)

    return VectorComposer(pitch_vectors, duration_vectors)


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
        "--duration_vectors",
        default="steering_interventions/outputs/steering_vectors/average_duration_steering_vectors.pt",
        help="Path to duration vectors",
    )
    parser.add_argument(
        "--alpha_pitch", type=float, default=1.5, help="Pitch scaling factor"
    )
    parser.add_argument(
        "--alpha_duration", type=float, default=2.0, help="Duration scaling factor"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create composer
    composer = load_and_create_composer(args.pitch_vectors, args.duration_vectors)

    # Test all three strategies
    for strategy in ["direct", "gram_schmidt_duration", "gram_schmidt_pitch"]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print("=" * 60)

        analysis = composer.analyze_composition(
            strategy, args.alpha_pitch, args.alpha_duration
        )

        print(f"Alpha values: pitch={args.alpha_pitch}, duration={args.alpha_duration}")
        print("\nCombined vector statistics:")
        print(f"  Mean norm:              {analysis['mean_norm']:.4f}")
        print(f"  Std norm:               {analysis['std_norm']:.4f}")
        print(f"  Mean pitch contrib:     {analysis['mean_pitch_contrib']:.4f}")
        print(f"  Mean duration contrib:  {analysis['mean_duration_contrib']:.4f}")
