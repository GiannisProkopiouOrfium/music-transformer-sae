#!/usr/bin/env python3
"""
Test LiMuF polarity - flip the intervention direction to see if it fixes the results.
"""

import torch
from pathlib import Path
import json
import argparse


def flip_limuf_direction(limuf_dir: str, feature_id: int, output_dir: str):
    """
    Create a copy of LiMuFs with Feature 182 direction flipped.
    
    Args:
        limuf_dir: Original LiMuF directory
        feature_id: Feature to flip
        output_dir: Output directory for flipped LiMuFs
    """
    print(f"🔄 Flipping direction for Feature {feature_id}")
    
    # Load original LiMuFs
    limuf_file = Path(limuf_dir) / "limufs.pt"
    if not limuf_file.exists():
        print(f"❌ LiMuF file not found: {limuf_file}")
        return
    
    data = torch.load(limuf_file, map_location="cpu")
    limufs = data["limufs"]
    metadata = data["metadata"]
    
    print(f"📁 Loaded {len(limufs)} LiMuFs")
    
    # Convert keys to integers for checking
    limuf_keys = {int(k) if isinstance(k, str) else k: v for k, v in limufs.items()}
    
    if feature_id not in limuf_keys:
        print(f"❌ Feature {feature_id} not found in LiMuFs")
        print(f"   Available features: {sorted(limuf_keys.keys())}")
        return
    
    # Flip the direction
    original_vector = limuf_keys[feature_id].clone()
    flipped_vector = -original_vector
    
    # Create new LiMuFs with flipped feature (preserve original key types)
    flipped_limufs = limufs.copy()
    
    # Find the original key format and update it
    for orig_key, vector in limufs.items():
        if (int(orig_key) if isinstance(orig_key, str) else orig_key) == feature_id:
            flipped_limufs[orig_key] = flipped_vector
            break
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save flipped LiMuFs
    flipped_data = {
        "limufs": flipped_limufs,
        "metadata": {
            **metadata,
            "flipped_features": [feature_id],
            "note": f"Feature {feature_id} direction flipped for testing"
        }
    }
    
    output_file = output_path / "limufs.pt"
    torch.save(flipped_data, output_file)
    
    print(f"✅ Saved flipped LiMuFs to: {output_file}")
    print(f"🔄 Feature {feature_id} vector flipped:")
    print(f"   Original norm: {torch.norm(original_vector):.4f}")
    print(f"   Flipped norm:  {torch.norm(flipped_vector):.4f}")
    print(f"   Dot product:   {torch.dot(original_vector.flatten(), flipped_vector.flatten()):.4f}")


def test_flipped_intervention(
    feature_id: int = 182,
    original_limuf_dir: str = "limufs_layer3",
    flipped_limuf_dir: str = "limufs_layer3_flipped",
    model_path: str = "exp/sod/ape/checkpoints/best_model.pt",
    output_dir: str = "feature_182_flipped_test"
):
    """
    Test intervention with flipped LiMuF direction.
    """
    print(f"\n🧪 Testing flipped intervention for Feature {feature_id}")
    print("=" * 50)
    
    # First, create flipped LiMuFs
    flip_limuf_direction(original_limuf_dir, feature_id, flipped_limuf_dir)
    
    # Test the flipped intervention
    import subprocess
    import sys
    
    print(f"\n🎼 Running intervention test with flipped Feature {feature_id}...")
    
    cmd = [
        sys.executable, "mmt/test_feature_182_intervention.py",
        "--limuf-dir", flipped_limuf_dir,
        "--model-path", model_path,
        "--output-dir", output_dir,
        "--feature-id", str(feature_id),
        "--seq-len", "256"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Flipped intervention test completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Flipped intervention test failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout[-500:]}")  # Last 500 chars
        if e.stderr:
            print(f"   stderr: {e.stderr[-500:]}")
        return False


def analyze_flipped_results(output_dir: str = "feature_182_flipped_test", feature_id: int = 182):
    """
    Analyze results from flipped intervention.
    """
    print(f"\n📊 Analyzing flipped results...")
    
    # Convert to audio and analyze
    import subprocess
    import sys
    
    # Convert sequences to audio
    cmd1 = [
        sys.executable, "mmt/convert_sequences_to_audio.py",
        "--results-dir", output_dir,
        "--feature-id", str(feature_id)
    ]
    
    # Analyze with audio flamingo
    cmd2 = [
        sys.executable, "mmt/audio_flamingo_analysis.py", 
        "--results-dir", output_dir,
        "--feature-id", str(feature_id)
    ]
    
    try:
        print("🎵 Converting flipped sequences to audio...")
        subprocess.run(cmd1, check=True)
        
        print("🤖 Analyzing flipped sequences...")
        result = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        # Show key results
        results_file = Path(output_dir) / f"audio_flamingo_analysis_feature_{feature_id}.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            verification = data.get("verification", {})
            scores = verification.get("scores", {})
            
            print(f"\n🎯 Flipped Results - Regularity Scores:")
            for name in ["suppress_strong", "suppress_weak", "baseline", "promote_weak", "promote_strong"]:
                if name in scores:
                    print(f"   {name:15s}: {scores[name]:.1f}/10")
            
            overall_success = verification.get("overall_success", False)
            print(f"\n🏆 Flipped Result: {'SUCCESS' if overall_success else 'STILL_NEEDS_INVESTIGATION'}")
            
            if overall_success:
                print("✅ Flipping the feature direction FIXED the issue!")
                print("✅ Feature 182 now works as expected for steady pulse control")
            else:
                print("⚠️  Flipping didn't fix the issue - may need further investigation")
            
            return overall_success
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test LiMuF direction flipping")
    parser.add_argument(
        "--feature-id",
        type=int,
        default=182,
        help="Feature ID to flip and test"
    )
    parser.add_argument(
        "--original-limuf-dir",
        default="limufs_layer3",
        help="Original LiMuF directory"
    )
    parser.add_argument(
        "--flipped-limuf-dir", 
        default="limufs_layer3_flipped",
        help="Output directory for flipped LiMuFs"
    )
    parser.add_argument(
        "--model-path",
        default="exp/sod/ape/checkpoints/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--output-dir",
        default="feature_182_flipped_test", 
        help="Output directory for test results"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing flipped results"
    )
    
    args = parser.parse_args()
    
    print("🔄 LIMUF DIRECTION TESTING")
    print("=" * 40)
    print(f"Feature ID: {args.feature_id}")
    print(f"Original LiMuFs: {args.original_limuf_dir}")
    print(f"Flipped LiMuFs: {args.flipped_limuf_dir}")
    print(f"Test Output: {args.output_dir}")
    
    if args.analyze_only:
        success = analyze_flipped_results(args.output_dir, args.feature_id)
    else:
        # Run full test
        success = test_flipped_intervention(
            args.feature_id,
            args.original_limuf_dir,
            args.flipped_limuf_dir,
            args.model_path,
            args.output_dir
        )
        
        if success:
            analyze_flipped_results(args.output_dir, args.feature_id)
    
    print(f"\n🎯 CONCLUSION:")
    if success:
        print("✅ Consider using the flipped LiMuFs for correct behavior")
        print(f"✅ Copy {args.flipped_limuf_dir}/limufs.pt over {args.original_limuf_dir}/limufs.pt")
    else:
        print("⚠️  Direction flipping didn't resolve the issue")
        print("   May need to investigate feature extraction methodology")


if __name__ == "__main__":
    main()
