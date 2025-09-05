#!/usr/bin/env python3
"""
Test LiMuF interventions for multiple musical features.

This script allows you to test interventions on any of your extracted LiMuFs,
not just Feature 182.

Usage:
    python test_feature_interventions.py --limuf-dir limufs_layer3 \
                                        --model-path exp/sod/ape/checkpoints/best_model.pt \
                                        --features 182 997 855
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
    """Create start tokens for generation."""
    # Simple start-of-song token - adapt based on your tokenization
    start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    start_tokens[0, 0, 0] = 1  # Start-of-song type
    return start_tokens


def load_feature_descriptions(interpretations_path: str) -> dict:
    """Load feature descriptions from interpretations file."""
    try:
        with open(interpretations_path, "r") as f:
            data = json.load(f)

        descriptions = {}
        for feature_id, interp in data.get("interpretations", {}).items():
            descriptions[int(feature_id)] = {
                "description": interp.get("interpretation", "No description"),
                "category": interp.get("musical_category", "unknown"),
                "confidence": interp.get("confidence_score", 0),
            }

        return descriptions
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Test LiMuF interventions for multiple musical features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific features
  python test_feature_interventions.py --limuf-dir limufs_layer3 \\
                                      --model-path exp/sod/ape/checkpoints/best_model.pt \\
                                      --features 182 997 855

  # Test all available features
  python test_feature_interventions.py --limuf-dir limufs_layer3 \\
                                      --model-path exp/sod/ape/checkpoints/best_model.pt \\
                                      --test-all

  # Test with custom strengths
  python test_feature_interventions.py --limuf-dir limufs_layer3 \\
                                      --model-path exp/sod/ape/checkpoints/best_model.pt \\
                                      --features 182 \\
                                      --strengths -1.5 0.0 1.5
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
        default="multi_feature_interventions",
        help="Output directory for results",
    )
    parser.add_argument(
        "--features", nargs="+", type=int, help="Specific feature IDs to test"
    )
    parser.add_argument(
        "--test-all", action="store_true", help="Test all available features"
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[-1.0, 0.0, 1.0],
        help="Intervention strengths to test",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Generation sequence length"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--interpretations",
        help="Path to interpretations JSON for feature descriptions",
    )

    args = parser.parse_args()

    print("🎼 TESTING MULTIPLE LiMuF INTERVENTIONS")
    print("=" * 50)

    # Validate inputs
    limuf_dir = Path(args.limuf_dir)
    model_path = Path(args.model_path)

    if not limuf_dir.exists():
        print(f"❌ LiMuF directory not found: {limuf_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Load LiMuFs
    print(f"📁 Loading LiMuFs from {limuf_dir}")
    try:
        limufs, metadata = LiMuFExtractor.load_limufs(str(limuf_dir))
        print(f"✅ Loaded {len(limufs)} LiMuFs")
    except Exception as e:
        print(f"❌ Failed to load LiMuFs: {e}")
        sys.exit(1)

    # Load feature descriptions if available
    feature_descriptions = {}
    if args.interpretations:
        feature_descriptions = load_feature_descriptions(args.interpretations)
        print(f"📖 Loaded descriptions for {len(feature_descriptions)} features")

    # Determine which features to test
    if args.test_all:
        test_features = sorted(limufs.keys())
        print(f"🎯 Testing all {len(test_features)} available features")
    elif args.features:
        test_features = []
        for feature_id in args.features:
            if feature_id in limufs:
                test_features.append(feature_id)
            else:
                print(f"⚠️  Feature {feature_id} not found in LiMuFs, skipping")

        if not test_features:
            print("❌ No valid features to test")
            sys.exit(1)

        print(f"🎯 Testing {len(test_features)} specified features: {test_features}")
    else:
        print("❌ Must specify either --features or --test-all")
        sys.exit(1)

    # Load model
    print(f"🎵 Loading music model...")
    try:
        model = _load_music_model(str(model_path), args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Create intervention wrapper
    intervention_model = LiMuFInterventionWrapper(
        model=model, limufs=limufs, intervention_layer=3
    )

    # Create start tokens
    start_tokens = create_start_tokens(model, args.device)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Test each feature
    print(f"\n🧪 Testing {len(test_features)} features with strengths {args.strengths}")

    all_results = {}

    for i, feature_id in enumerate(test_features):
        print(f"\n{'='*50}")
        print(f"🎵 Feature {feature_id} ({i+1}/{len(test_features)})")

        # Show feature description if available
        if feature_id in feature_descriptions:
            desc_info = feature_descriptions[feature_id]
            print(f"   Category: {desc_info['category']}")
            print(f"   Confidence: {desc_info['confidence']}")
            print(f"   Description: {desc_info['description'][:100]}...")

        feature_results = {}

        for strength in args.strengths:
            print(f"\n   🎛️  Strength {strength:+.1f}: ", end="")

            try:
                if abs(strength) < 1e-6:  # Baseline
                    generated = model.generate(
                        start_tokens, args.seq_len, temperature=1.0
                    )
                    print("Baseline ✅")
                else:
                    generated = intervention_model.generate_with_intervention(
                        start_tokens=start_tokens,
                        seq_len=args.seq_len,
                        feature_id=feature_id,
                        strength=strength,
                        temperature=1.0,
                    )
                    action = "promote" if strength > 0 else "suppress"
                    print(f"{action.capitalize()} ✅")

                # Save result
                filename = f"feature_{feature_id}_strength_{strength:+.1f}".replace(
                    "+", "plus"
                ).replace("-", "minus")
                torch.save(
                    {
                        "generated": generated.cpu(),
                        "feature_id": feature_id,
                        "strength": strength,
                        "shape": list(generated.shape),
                    },
                    output_dir / f"{filename}.pt",
                )

                feature_results[strength] = generated.shape

            except Exception as e:
                print(f"❌ ({e})")
                continue

        all_results[feature_id] = feature_results
        print(f"   📊 Generated {len(feature_results)}/{len(args.strengths)} sequences")

    # Save experiment summary
    summary = {
        "experiment_type": "multi_feature_intervention_test",
        "tested_features": test_features,
        "tested_strengths": args.strengths,
        "sequence_length": args.seq_len,
        "total_features_tested": len(test_features),
        "total_generations": sum(len(results) for results in all_results.values()),
        "feature_descriptions": feature_descriptions,
        "results_summary": {
            str(fid): {str(s): list(shape) for s, shape in results.items()}
            for fid, results in all_results.items()
        },
    }

    with open(output_dir / "multi_feature_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*50}")
    print("✅ MULTI-FEATURE INTERVENTION TEST COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎼 Features tested: {len(test_features)}")
    print(f"🎛️  Strengths tested: {len(args.strengths)}")

    total_generations = sum(len(results) for results in all_results.values())
    print(f"📊 Total generations: {total_generations}")

    if feature_descriptions:
        print(f"\n🎯 Tested Feature Categories:")
        categories = {}
        for fid in test_features:
            if fid in feature_descriptions:
                cat = feature_descriptions[fid]["category"]
                categories[cat] = categories.get(cat, 0) + 1

        for category, count in sorted(categories.items()):
            print(f"   {category}: {count} features")

    print(f"\n📂 Output Files:")
    print(f"   - feature_*_strength_*.pt: Generated sequences")
    print(f"   - multi_feature_experiment_summary.json: Complete experiment summary")

    print(f"\n🔍 Analysis Suggestions:")
    print(f"   1. Compare sequences with different strengths for the same feature")
    print(f"   2. Look for consistent patterns across similar feature categories")
    print(f"   3. Identify which features produce the most noticeable changes")


if __name__ == "__main__":
    main()
