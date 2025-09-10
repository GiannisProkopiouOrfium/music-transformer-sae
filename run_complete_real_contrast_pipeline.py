#!/usr/bin/env python3
"""
🎼 COMPLETE REAL CONTRAST LIMUF PIPELINE

This script runs the complete pipeline:
1. Extract real contrast LiMuFs from SAE (182 vs 1743)
2. Test interventions with strengths [-2, -1, 0, +1, +2]
3. Convert to audio for listening
4. Analyze results

Uses your existing infrastructure:
- SAE: exp/sod/ape/sae_models/sae_layer_2048d.pt
- Activations: exp/sod/ape/activations/activations_layers_3.h5
- Model: exp/sod/ape/checkpoints/best_model.pt
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=check, text=True, capture_output=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print("stdout:", e.stdout[-500:])
        if e.stderr:
            print("stderr:", e.stderr[-500:])
        return False


def check_files():
    """Check if required files exist."""
    print("🔍 CHECKING REQUIRED FILES")
    print("=" * 40)

    required_files = [
        "exp/sod/ape/sae_models/sae_layer_2048d.pt",
        "exp/sod/ape/activations/activations_layers_3.h5",
        "exp/sod/ape/checkpoints/best_model.pt",
        "interpretation/enhanced_diversity/enhanced_diversity_report.json",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Run the complete real contrast LiMuF pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete real contrast LiMuF pipeline"
    )
    parser.add_argument(
        "--feature-a",
        type=int,
        default=182,
        help="First feature ID (positive direction)",
    )
    parser.add_argument(
        "--feature-b",
        type=int,
        default=1743,
        help="Second feature ID (negative direction)",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="SAE activation threshold"
    )
    parser.add_argument(
        "--strengths", default="-2.0,-1.0,0.0,1.0,2.0", help="Intervention strengths"
    )
    parser.add_argument(
        "--num-sequences", type=int, default=3, help="Sequences per strength"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip contrast extraction (use existing)",
    )
    parser.add_argument(
        "--skip-testing", action="store_true", help="Skip intervention testing"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    print("🎼 COMPLETE REAL CONTRAST LIMUF PIPELINE")
    print("=" * 50)
    print(f"Feature A (positive): {args.feature_a}")
    print(f"Feature B (negative): {args.feature_b}")
    print(f"Threshold: {args.threshold}")
    print(f"Strengths: {args.strengths}")
    print(f"Device: {args.device}")
    print()

    # Check prerequisites
    if not check_files():
        print("\n❌ Missing required files. Please check your setup.")
        return False

    print("\n✅ All required files found!")

    # Define paths
    contrast_dir = f"real_contrast_{args.feature_a}_vs_{args.feature_b}"
    test_dir = f"real_test_{args.feature_a}_vs_{args.feature_b}"
    contrast_name = f"feature_{args.feature_a}_vs_{args.feature_b}"

    success = True

    # Step 1: Extract real contrast LiMuF
    if not args.skip_extraction:
        cmd = [
            sys.executable,
            "extract_real_contrast_limufs_182_vs_1743.py",
            "--feature-a",
            str(args.feature_a),
            "--feature-b",
            str(args.feature_b),
            "--threshold",
            str(args.threshold),
            "--output-dir",
            contrast_dir,
            "--device",
            args.device,
        ]
        success = run_command(
            cmd, f"Extract real contrast LiMuF ({args.feature_a} vs {args.feature_b})"
        )
        if not success:
            print("❌ Contrast extraction failed")
            return False
    else:
        print(f"⏭️  Skipping extraction, using existing: {contrast_dir}/")

    # Step 2: Test interventions
    if not args.skip_testing and success:
        cmd = [
            sys.executable,
            "test_real_contrast_interventions.py",
            "--limuf-dir",
            contrast_dir,
            "--contrast-name",
            contrast_name,
            "--output-dir",
            test_dir,
            "--strengths",
            args.strengths,
            "--num-sequences",
            str(args.num_sequences),
            "--device",
            args.device,
        ]
        success = run_command(cmd, "Test real contrast interventions")
        if not success:
            print("❌ Intervention testing failed")
            return False
    elif args.skip_testing:
        print("⏭️  Skipping intervention testing")

    # Step 3: Convert to audio (if possible)
    if success and not args.skip_testing:
        try:
            # Check if conversion script exists
            if Path("mmt/convert_sequences_to_audio.py").exists():
                cmd = [
                    sys.executable,
                    "mmt/convert_sequences_to_audio.py",
                    "--results-dir",
                    test_dir,
                    "--contrast-name",
                    contrast_name,
                ]
                audio_success = run_command(
                    cmd, "Convert sequences to audio", check=False
                )
                if not audio_success:
                    print("⚠️  Audio conversion failed, but sequences are available")
            else:
                print("⚠️  Audio conversion script not found, skipping")
        except Exception as e:
            print(f"⚠️  Audio conversion error: {e}")

    # Step 4: Show results
    if success:
        print(f"\n🎯 PIPELINE COMPLETE!")
        print("=" * 50)

        if not args.skip_extraction:
            print(f"✅ Contrast LiMuF extracted: {contrast_dir}/limufs.pt")

        if not args.skip_testing:
            print(f"✅ Intervention results: {test_dir}/")

            # Show generated files
            test_path = Path(test_dir)
            if test_path.exists():
                generated_files = list(test_path.glob("*.pt"))
                print(f"✅ Generated {len(generated_files)} sequence files")

                # Show what each strength means
                print(f"\n🎵 SEMANTIC MEANINGS:")
                strengths = [float(s.strip()) for s in args.strengths.split(",")]
                for strength in strengths:
                    if strength > 0:
                        meaning = (
                            f"Promotes Feature {args.feature_a} (steady pulse patterns)"
                        )
                    elif strength < 0:
                        meaning = f"Promotes Feature {args.feature_b} (extended melodic sequences)"
                    else:
                        meaning = "Baseline (no intervention)"
                    print(f"   Strength {strength:+.1f}: {meaning}")

        print(f"\n🎧 NEXT STEPS:")
        print(f"   1. Listen to generated sequences (if audio conversion worked)")
        print(f"   2. Compare musical characteristics across strengths")
        print(f"   3. Verify semantic control matches expectations")
        print()
        print(f"📁 Key files:")
        print(f"   Contrast LiMuF: {contrast_dir}/limufs.pt")
        print(f"   Test summary: {test_dir}/test_summary.json")
        print(
            f"   Generated sequences: {test_dir}/feature_{args.feature_a}_vs_{args.feature_b}_strength_*.pt"
        )

        return True
    else:
        print("\n❌ Pipeline failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
