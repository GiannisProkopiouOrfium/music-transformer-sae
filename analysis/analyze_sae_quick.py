#!/usr/bin/env python3
"""
Quick SAE Analysis Runner

This script provides easy access to SAE analysis functionality.
Run this after training to get comprehensive analysis and visualizations.

Usage:
    # Analyze specific model (auto-detect activations)
    python analyze_sae_quick.py --model exp/sod/ape/sae_models/sae_layer_2048d.pt
    
    # Analyze with specific activations file
    python analyze_sae_quick.py --model exp/sod/ape/sae_models/sae_layer_2048d.pt \
        --activations exp/sod/ape/activations/activations_layers_3.h5
    
    # Analyze with custom output directory
    python analyze_sae_quick.py --model exp/sod/ape/sae_models/sae_layer_2048d.pt \
        --activations exp/sod/ape/activations/activations_layers_3.h5 --output results/
    
    # Quick analysis (fewer samples)
    python analyze_sae_quick.py --model exp/sod/ape/sae_models/sae_layer_2048d.pt \
        --activations exp/sod/ape/activations/activations_layers_3.h5 --quick
"""

import argparse
import pathlib
import sys

# Add current directory to path
sys.path.append(str(pathlib.Path(__file__).parent))

from sae.analyze_trained_sae import SAEAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Quick SAE Analysis Tool")
    parser.add_argument("--model", required=True, help="Path to trained SAE model")
    parser.add_argument(
        "--activations", help="Path to activations file (e.g., activations_layers_3.h5)"
    )
    parser.add_argument("--output", help="Output directory (default: auto-detect)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick analysis with fewer samples"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Validate model path
    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1

    # Set output directory
    if args.output:
        output_dir = pathlib.Path(args.output)
    else:
        # Auto-detect: create analysis_results next to model
        output_dir = model_path.parent / "analysis_results"

    # Set analysis parameters
    subsample = 2000 if args.quick else 5000

    print("🎼 QUICK SAE ANALYSIS")
    print("=" * 40)
    print(f"📦 Model: {model_path}")
    print(f"📁 Output: {output_dir}")
    print(f"⚡ Quick mode: {args.quick}")
    print("=" * 40)

    try:
        # Use provided activations path or auto-detect
        activations_path = None
        if args.activations:
            activations_path = args.activations
            if not pathlib.Path(activations_path).exists():
                print(f"❌ Activations file not found: {activations_path}")
                return 1
            print(
                f"🎯 Using provided activations: {pathlib.Path(activations_path).name}"
            )
        else:
            # Auto-detect activations path
            activations_dir = model_path.parent.parent / "activations"
            if activations_dir.exists():
                h5_files = list(activations_dir.glob("*.h5"))
                if h5_files:
                    activations_path = str(h5_files[0])
                    print(
                        f"🎯 Auto-detected activations: {pathlib.Path(activations_path).name}"
                    )
                else:
                    print("⚠️ No .h5 files found in activations directory")
            else:
                print(
                    "⚠️ No activations directory found - analysis will be limited to model weights and training history"
                )

        # Create analyzer
        analyzer = SAEAnalyzer(str(model_path), activations_path)

        # Run analysis
        if args.no_plots:
            print("📊 Running analysis without visualizations...")
            # Just run the analyses without creating plots
            weight_analysis = analyzer.analyze_model_weights()
            activation_analysis = analyzer.analyze_activations(subsample=subsample)
            training_analysis = analyzer.analyze_training_history()
            print("✅ Analysis complete (no visualizations)")
        else:
            print("📊 Running comprehensive analysis...")
            result_path = analyzer.create_comprehensive_report(str(output_dir))
            print(f"✅ Analysis complete! Results: {result_path}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
