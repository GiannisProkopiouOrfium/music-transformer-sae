#!/usr/bin/env python3
"""
🎼 COMPLETE REAL CONTRAST LIMUF PIPELINE

This script runs the complete pipeline for Features 182 vs 1743:
1. Extract real contrast LiMuF using your existing SAE
2. Test interventions with strengths [-2, -1, 0, +1, +2]
3. Convert sequences to audio
4. Analyze results

Usage:
    python run_real_contrast_pipeline.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check_output=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        if check_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ SUCCESS!")
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
            print("✅ SUCCESS!")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_prerequisites():
    """Check that all required files exist."""
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 40)

    required_files = [
        "exp/sod/ape/sae_models/sae_layer_2048d.pt",
        "exp/sod/ape/activations/activations_layers_3.h5",
        "interpretation/enhanced_diversity/enhanced_diversity_report.json",
        "exp/sod/ape/checkpoints/best_model.pt",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")

    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files!")
        print("Please ensure you have:")
        print("1. Trained SAE model")
        print("2. Extracted activations")
        print("3. Feature interpretations")
        print("4. Trained music transformer")
        return False

    print("\n✅ All prerequisites found!")
    return True


def main():
    """Run the complete pipeline."""

    print("🎼 REAL CONTRAST LIMUF PIPELINE FOR FEATURES 182 vs 1743")
    print("=" * 70)
    print("This pipeline will:")
    print("1. Extract real contrast LiMuF using difference-in-means")
    print("2. Test interventions with strengths [-2, -1, 0, +1, +2]")
    print("3. Convert sequences to audio for analysis")
    print("4. Analyze musical differences")
    print()

    # Check prerequisites
    if not check_prerequisites():
        return False

    # Set environment
    print("\n🔧 Setting up environment...")
    import os

    os.environ["PYTHONPATH"] = "."

    success_count = 0
    total_steps = 4

    # Step 1: Extract real contrast LiMuF
    cmd = [
        sys.executable,
        "extract_real_contrast_limufs.py",
        "--sae-model-path",
        "exp/sod/ape/sae_models/sae_layer_2048d.pt",
        "--activations-path",
        "exp/sod/ape/activations/activations_layers_3.h5",
        "--interpretations-path",
        "interpretation/enhanced_diversity/enhanced_diversity_report.json",
        "--feature-a",
        "182",
        "--feature-b",
        "1743",
        "--activation-threshold",
        "2.0",
        "--output-dir",
        "real_contrast_limufs_182_vs_1743",
    ]

    if run_command(cmd, "STEP 1: Extract Real Contrast LiMuF (182 vs 1743)"):
        success_count += 1
    else:
        print("❌ Failed at Step 1. Aborting pipeline.")
        return False

    # Step 2: Test contrast interventions
    cmd = [
        sys.executable,
        "test_real_contrast_interventions.py",
        "--model-path",
        "exp/sod/ape/checkpoints/best_model.pt",
        "--contrast-limuf-path",
        "real_contrast_limufs_182_vs_1743/limufs.pt",
        "--output-dir",
        "real_contrast_interventions_182_vs_1743",
        "--strengths",
        "-2.0,-1.0,0.0,1.0,2.0",
        "--intervention-layer",
        "3",
        "--seq-len",
        "256",
        "--num-sequences",
        "3",
    ]

    if run_command(cmd, "STEP 2: Test Contrast Interventions"):
        success_count += 1
    else:
        print("⚠️  Step 2 failed, but continuing with available results...")

    # Step 3: Convert to audio (using existing script)
    try:
        cmd = [
            sys.executable,
            "mmt/convert_sequences_to_audio.py",
            "--results-dir",
            "real_contrast_interventions_182_vs_1743",
            "--contrast-name",
            "steady_pulse_vs_melodic_sequences",
        ]

        if run_command(cmd, "STEP 3: Convert Sequences to Audio"):
            success_count += 1
        else:
            print("⚠️  Audio conversion failed, but sequences were generated")
    except:
        print("⚠️  Audio conversion script not available")

    # Step 4: Analyze results (using existing script)
    try:
        cmd = [
            sys.executable,
            "mmt/audio_flamingo_analysis.py",
            "--results-dir",
            "real_contrast_interventions_182_vs_1743",
            "--contrast-name",
            "steady_pulse_vs_melodic_sequences",
        ]

        if run_command(cmd, "STEP 4: Analyze Musical Differences"):
            success_count += 1
        else:
            print("⚠️  Analysis script failed, but manual analysis is possible")
    except:
        print("⚠️  Analysis script not available")

    # Final summary
    print("\n" + "=" * 70)
    print("🎯 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"✅ Completed {success_count}/{total_steps} steps successfully")
    print()

    print("📁 GENERATED FILES:")
    print("   • real_contrast_limufs_182_vs_1743/limufs.pt - Contrast LiMuF")
    print("   • real_contrast_interventions_182_vs_1743/ - Generated sequences")
    print("   • *.wav files - Audio conversions (if available)")
    print()

    print("🎵 WHAT TO EXPECT:")
    print("   • strength -2.0: Strong melodic sequences (Feature 1743)")
    print("   • strength -1.0: Weak melodic tendency")
    print("   • strength  0.0: Baseline (normal model)")
    print("   • strength +1.0: Weak steady pulse (Feature 182)")
    print("   • strength +2.0: Strong steady pulse")
    print()

    print("🔍 NEXT STEPS:")
    print("   1. Listen to generated audio files")
    print("   2. Compare rhythmic vs melodic characteristics")
    print("   3. Adjust intervention strengths if needed")
    print("   4. Try other feature contrasts")

    if success_count >= 2:  # At least extraction and intervention worked
        print("\n🎉 MINIMUM VIABLE RESULTS ACHIEVED!")
        print("   You can now compare the musical differences between strengths.")
        return True
    else:
        print("\n❌ PIPELINE FAILED")
        print("   Check error messages above for troubleshooting.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
