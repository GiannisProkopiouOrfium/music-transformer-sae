#!/usr/bin/env python3
"""
Extract Linear Music Features (LiMuFs) from SAE interpretations.

This script extracts LiMuFs from your existing SAE features and interpretations,
creating linear directions that can be used for controlled music generation.

Usage:
    python extract_limufs.py --sae-model exp/sod/ape/sae_models/sae_layer_2048d.pt \
                           --activations exp/sod/ape/activations/activations_train_layers_3.h5 \
                           --interpretations interpretation/enhanced_diversity/enhanced_diversity_report.json \
                           --output-dir limufs_layer3
"""

import argparse
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mmt.limuf_extractor import LiMuFExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract Linear Music Features (LiMuFs) from SAE interpretations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract LiMuFs for rhythmic and harmonic features from layer 3
  python extract_limufs.py --sae-model exp/sod/ape/sae_models/sae_layer_2048d.pt \\
                          --activations exp/sod/ape/activations/activations_train_layers_3.h5 \\
                          --interpretations interpretation/enhanced_diversity/enhanced_diversity_report.json \\
                          --output-dir limufs_layer3 \\
                          --categories rhythmic_specific harmonic_specific

  # Extract all available categories with GPU
  python extract_limufs.py --sae-model exp/sod/ape/sae_models/sae_layer_2048d.pt \\
                          --activations exp/sod/ape/activations/activations_train_layers_3.h5 \\
                          --interpretations interpretation/enhanced_diversity/enhanced_diversity_report.json \\
                          --output-dir limufs_layer3 \\
                          --device cuda
        """,
    )

    parser.add_argument(
        "--sae-model", required=True, help="Path to trained SAE model (.pt file)"
    )
    parser.add_argument(
        "--activations", required=True, help="Path to activations h5 file"
    )
    parser.add_argument(
        "--interpretations",
        required=True,
        help="Path to enhanced_diversity_report JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="limufs",
        help="Output directory for LiMuFs (default: limufs)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["rhythmic_specific", "harmonic_specific", "textural_specific"],
        help="Musical categories to extract (default: rhythmic_specific harmonic_specific textural_specific)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to extract (default: all)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate extracted LiMuFs quality"
    )

    args = parser.parse_args()

    print("🎼 LINEAR MUSIC FEATURES (LiMuFs) EXTRACTION")
    print("=" * 60)
    print(f"SAE Model: {args.sae_model}")
    print(f"Activations: {args.activations}")
    print(f"Interpretations: {args.interpretations}")
    print(f"Output: {args.output_dir}")
    print(f"Categories: {args.categories}")
    print(f"Device: {args.device}")
    if args.max_features:
        print(f"Max features: {args.max_features}")

    # Validate input files
    sae_path = Path(args.sae_model)
    activations_path = Path(args.activations)
    interpretations_path = Path(args.interpretations)

    if not sae_path.exists():
        print(f"❌ SAE model not found: {sae_path}")
        sys.exit(1)

    if not activations_path.exists():
        print(f"❌ Activations file not found: {activations_path}")
        sys.exit(1)

    if not interpretations_path.exists():
        print(f"❌ Interpretations file not found: {interpretations_path}")
        sys.exit(1)

    # Load interpretations
    print(f"\n📖 Loading interpretations from {interpretations_path}")
    try:
        with open(interpretations_path, "r") as f:
            interpretation_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load interpretations: {e}")
        sys.exit(1)

    feature_interpretations = interpretation_data.get("interpretations", {})
    print(f"✅ Found {len(feature_interpretations)} interpreted features")

    # Filter by categories
    filtered_features = {}
    for feature_id, interpretation in feature_interpretations.items():
        category = interpretation.get("musical_category", "unknown")
        if category in args.categories:
            filtered_features[feature_id] = interpretation

    print(f"📊 Features by category:")
    for category in args.categories:
        count = sum(
            1
            for interp in filtered_features.values()
            if interp.get("musical_category") == category
        )
        print(f"   {category}: {count} features")

    if not filtered_features:
        print("❌ No features found for specified categories")
        sys.exit(1)

    # Initialize LiMuF extractor
    print(f"\n🔧 Initializing LiMuF extractor...")
    try:
        extractor = LiMuFExtractor(
            sae_model_path=str(sae_path),
            activations_path=str(activations_path),
            device=args.device,
        )
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)

    # Extract LiMuFs for specified categories
    print(f"\n🎵 Extracting LiMuFs...")
    try:
        limufs = extractor.extract_multiple_limufs(
            feature_interpretations=filtered_features,
            categories=args.categories,
            max_features=args.max_features,
        )
    except Exception as e:
        print(f"❌ Failed to extract LiMuFs: {e}")
        sys.exit(1)

    if not limufs:
        print("❌ No LiMuFs were successfully extracted")
        sys.exit(1)

    # Validate LiMuFs if requested
    if args.validate:
        print(f"\n🔍 Validating LiMuF quality...")
        validation_results = {}

        for feature_id in limufs.keys():
            try:
                metrics = extractor.validate_limuf_quality(feature_id)
                validation_results[feature_id] = metrics

                correlation = metrics["correlation"]
                separation = metrics["separation_score"]

                if correlation > 0.7 and separation > 1.0:
                    quality = "✅ Excellent"
                elif correlation > 0.5 and separation > 0.5:
                    quality = "✓ Good"
                elif correlation > 0.3:
                    quality = "⚠️ Moderate"
                else:
                    quality = "❌ Poor"

                print(
                    f"   Feature {feature_id}: {quality} (corr={correlation:.3f}, sep={separation:.3f})"
                )

            except Exception as e:
                print(f"   Feature {feature_id}: ❌ Validation failed: {e}")
                validation_results[feature_id] = None

    # Save results
    output_path = Path(args.output_dir)
    print(f"\n💾 Saving results to {output_path}")
    try:
        extractor.save_limufs(str(output_path))

        # Save validation results if available
        if args.validate and validation_results:
            validation_file = output_path / "validation_results.json"
            with open(validation_file, "w") as f:
                json.dump(validation_results, f, indent=2)
            print(f"   Validation results saved to: {validation_file}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        sys.exit(1)

    # Print summary
    print(f"\n" + "=" * 60)
    print("✅ LiMuF EXTRACTION COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print(f"🎼 Extracted LiMuFs: {len(limufs)}")
    print(
        f"📊 Success rate: {len(limufs)}/{len(filtered_features)} ({len(limufs)/len(filtered_features)*100:.1f}%)"
    )

    # Show extracted features
    if limufs:
        print(f"\n🎯 Successfully Extracted Features:")
        for feature_id in sorted(limufs.keys()):
            interp = filtered_features.get(str(feature_id), {})
            category = interp.get("musical_category", "unknown")
            description = interp.get("interpretation", "No description")[:80]
            confidence = interp.get("confidence_score", 0)

            print(
                f"  Feature {feature_id:4d} ({category:15s}, conf={confidence:2d}): {description}..."
            )

    print(f"\n🎛️  Next steps:")
    print(
        f"   1. Test interventions with: python test_feature_interventions.py --limuf-dir {output_path}"
    )
    print(f"   2. Or use in your own code: LiMuFInterventionWrapper(model, limufs)")


if __name__ == "__main__":
    main()
