#!/usr/bin/env python3
"""
Test LiMuF intervention with Feature 182 (steady pulse patterns).

This script demonstrates how to use LiMuFs to control musical generation,
specifically testing Feature 182 which detects steady pulse patterns.

Usage:
    python test_feature_182_intervention.py --limuf-dir limufs_layer3 \
                                          --model-path exp/sod/ape/checkpoints/best_model.pt \
                                          --output-dir intervention_results
"""

import argparse
import torch
from pathlib import Path
import sys
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from limuf_extractor import LiMuFExtractor
from limuf_interventions import LiMuFInterventionWrapper, _load_music_model


def create_start_tokens(model, device: str = "cuda") -> torch.Tensor:
    """
    Create start tokens for generation.

    This is a simplified version - adapt based on your actual token structure.
    """
    # Get model's encoding information
    decoder = model.decoder if hasattr(model, "decoder") else model

    # Create a simple start-of-song token sequence
    # Shape: [batch_size, seq_len, n_dimensions]
    # Assuming 6 dimensions: [type, beat, position, pitch, duration, instrument]

    start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)

    # Set start-of-song type (assuming this is token 1)
    start_tokens[0, 0, 0] = 1  # type dimension
    # Other dimensions can remain 0 for start-of-song

    return start_tokens


def save_generation_result(
    output_dir: Path,
    generated_sequence: torch.Tensor,
    strength: float,
    feature_id: int,
    description: str,
):
    """Save generated sequence and metadata."""

    # Create filename
    strength_str = (
        f"plus{abs(strength):.1f}" if strength >= 0 else f"minus{abs(strength):.1f}"
    )
    filename = f"feature_{feature_id}_strength_{strength_str}"

    # Save the tensor
    torch.save(
        {
            "generated": generated_sequence.cpu(),
            "strength": strength,
            "feature_id": feature_id,
            "description": description,
            "shape": list(generated_sequence.shape),
        },
        output_dir / f"{filename}.pt",
    )

    # Save readable metadata
    metadata = {
        "feature_id": feature_id,
        "description": description,
        "intervention_strength": strength,
        "sequence_shape": list(generated_sequence.shape),
        "sequence_length": (
            generated_sequence.shape[1] if len(generated_sequence.shape) > 1 else 1
        ),
    }

    with open(output_dir / f"{filename}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   💾 Saved: {filename}.pt")


def analyze_generation_differences(
    baseline: torch.Tensor, intervention: torch.Tensor, strength: float
) -> dict:
    """
    Analyze differences between baseline and intervention generations.

    This is a simple analysis - you can extend this with more sophisticated
    musical pattern analysis.
    """
    # Basic sequence analysis
    analysis = {
        "strength": strength,
        "length_diff": (
            intervention.shape[1] - baseline.shape[1]
            if len(intervention.shape) > 1
            else 0
        ),
        "token_differences": 0,
        "unique_tokens_baseline": 0,
        "unique_tokens_intervention": 0,
    }

    if len(baseline.shape) > 1 and len(intervention.shape) > 1:
        # Count token differences (simple comparison)
        min_length = min(baseline.shape[1], intervention.shape[1])
        if min_length > 0:
            baseline_trimmed = baseline[:, :min_length, :]
            intervention_trimmed = intervention[:, :min_length, :]

            # Count positions where tokens differ
            differences = (baseline_trimmed != intervention_trimmed).any(dim=-1)
            analysis["token_differences"] = differences.sum().item()
            analysis["difference_percentage"] = (
                differences.sum().item() / min_length
            ) * 100

            # Count unique tokens in each sequence
            analysis["unique_tokens_baseline"] = len(
                torch.unique(
                    baseline_trimmed.view(-1, baseline_trimmed.shape[-1]), dim=0
                )
            )
            analysis["unique_tokens_intervention"] = len(
                torch.unique(
                    intervention_trimmed.view(-1, intervention_trimmed.shape[-1]), dim=0
                )
            )

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Test Feature 182 (steady pulse) intervention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Feature 182 with default settings
  python test_feature_182_intervention.py --limuf-dir limufs_layer3 \\
                                         --model-path exp/sod/ape/checkpoints/best_model.pt

  # Test with custom sequence length and device
  python test_feature_182_intervention.py --limuf-dir limufs_layer3 \\
                                         --model-path exp/sod/ape/checkpoints/best_model.pt \\
                                         --seq-len 256 --device cpu
        """,
    )

    parser.add_argument(
        "--limuf-dir", required=True, help="Directory containing extracted LiMuFs"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to trained music model"
    )
    parser.add_argument(
        "--output-dir",
        default="intervention_results",
        help="Output directory for results",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Generation sequence length"
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        default=182,
        help="Feature ID to test (default: 182 for steady pulse)",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Intervention strengths to test",
    )

    args = parser.parse_args()

    print("🎼 TESTING FEATURE 182 INTERVENTION (Steady Pulse)")
    print("=" * 60)
    print(f"LiMuF Directory: {args.limuf_dir}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Feature ID: {args.feature_id}")
    print(f"Test Strengths: {args.strengths}")

    # Validate input paths
    limuf_dir = Path(args.limuf_dir)
    model_path = Path(args.model_path)

    if not limuf_dir.exists():
        print(f"❌ LiMuF directory not found: {limuf_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Load LiMuFs
    print(f"\n📁 Loading LiMuFs from {limuf_dir}")
    try:
        limufs, metadata = LiMuFExtractor.load_limufs(str(limuf_dir))
    except Exception as e:
        print(f"❌ Failed to load LiMuFs: {e}")
        sys.exit(1)

    # Check if target feature is available
    if args.feature_id not in limufs:
        available_features = sorted(limufs.keys())
        print(f"❌ Feature {args.feature_id} not found in LiMuFs")
        print(f"Available features: {available_features}")
        sys.exit(1)

    print(f"✅ Feature {args.feature_id} available")
    print(f"Total available features: {len(limufs)}")

    # Load music model
    print(f"\n🎵 Loading music model from {model_path}")
    try:
        model = _load_music_model(str(model_path), args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Create intervention wrapper
    print(f"\n🎛️  Creating intervention wrapper...")
    intervention_model = LiMuFInterventionWrapper(
        model=model,
        limufs=limufs,
        intervention_layer=3,  # Based on your layer 3 analysis
    )

    # Create start tokens
    print(f"🎼 Creating start tokens...")
    try:
        start_tokens = create_start_tokens(model, args.device)
        print(f"✅ Start tokens created: {start_tokens.shape}")
    except Exception as e:
        print(f"❌ Failed to create start tokens: {e}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Test different intervention strengths
    print(f"\n🧪 Testing intervention strengths: {args.strengths}")
    print(f"🎯 Target: Feature {args.feature_id} (steady pulse patterns)")

    results = {}
    baseline_generation = None

    for strength in args.strengths:
        print(f"\n🎛️  Generating with strength {strength:+.1f}")

        try:
            if abs(strength) < 1e-6:  # Essentially zero
                # Baseline generation (no intervention)
                print("   (Baseline - no intervention)")
                generated = model.generate(start_tokens, args.seq_len, temperature=1.0)
                baseline_generation = generated
                description = "baseline_no_intervention"
            else:
                # Generate with intervention
                generated = intervention_model.generate_with_intervention(
                    start_tokens=start_tokens,
                    seq_len=args.seq_len,
                    feature_id=args.feature_id,
                    strength=strength,
                    temperature=1.0,
                )
                if strength > 0:
                    description = f"promote_steady_pulse_strength_{strength}"
                else:
                    description = f"suppress_steady_pulse_strength_{abs(strength)}"

            # Save the result
            save_generation_result(
                output_dir, generated, strength, args.feature_id, description
            )

            results[strength] = generated

            # Basic analysis if we have baseline
            if baseline_generation is not None and abs(strength) > 1e-6:
                analysis = analyze_generation_differences(
                    baseline_generation, generated, strength
                )
                print(
                    f"   📊 Difference from baseline: {analysis['difference_percentage']:.1f}% of tokens"
                )

        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            continue

    # Save experiment summary
    summary = {
        "feature_id": args.feature_id,
        "feature_description": "steady_pulse_patterns",
        "tested_strengths": args.strengths,
        "successful_generations": len(results),
        "sequence_length": args.seq_len,
        "model_path": str(model_path),
        "limuf_dir": str(limuf_dir),
        "device": args.device,
    }

    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 60)
    print("✅ INTERVENTION TEST COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎼 Generated sequences: {len(results)}")
    print(f"🔍 Analysis files:")
    print(f"   - experiment_summary.json: Test configuration and results")
    print(f"   - feature_{args.feature_id}_strength_*.pt: Generated sequences")
    print(f"   - *_metadata.json: Sequence metadata")

    if len(results) > 1:
        print(f"\n📊 Quick Analysis:")
        print(
            f"   Compare the generated sequences to observe how Feature {args.feature_id}"
        )
        print(f"   (steady pulse patterns) affects the musical output.")
        print(f"   Positive strengths should promote steady, metronomic rhythms.")
        print(f"   Negative strengths should suppress regular pulse patterns.")

    print(f"\n🎛️  Next steps:")
    print(f"   1. Listen to/analyze the generated sequences")
    print(
        f"   2. Try other features: python test_feature_interventions.py --limuf-dir {args.limuf_dir}"
    )
    print(f"   3. Integrate into your own generation pipeline")


if __name__ == "__main__":
    main()
