#!/usr/bin/env python3
"""
Extract Residual Stream LiMuFs for musical concept contrasts.

This script demonstrates the LiReFs-style approach for musical features:
- Feature 182: Steady pulse patterns (metronomic consistency)
- Feature 1743: Extended melodic sequences (melodic development)

Usage:
    python extract_contrast_limufs.py
"""

import torch
from pathlib import Path
import json
import argparse

# Import the new residual stream extractor
from mmt.residual_stream_limuf_extractor import ResidualStreamLiMuFExtractor


def extract_steady_pulse_vs_melodic_sequences(
    sae_model_path: str = "sae/layer3_sae_model.pt",
    activations_path: str = "data/sae_activations_layer3.h5",
    interpretations_path: str = "interpretation/enhanced_diversity/enhanced_diversity_report.json",
    output_dir: str = "contrast_limufs_layer3",
    device: str = "cuda",
):
    """
    Extract contrast LiMuF: Steady Pulse vs. Extended Melodic Sequences

    This creates a direction vector where:
    - Positive intervention promotes steady pulse patterns (Feature 182)
    - Negative intervention promotes extended melodic sequences (Feature 1743)
    """
    print("🎼 EXTRACTING MUSICAL CONCEPT CONTRAST LIMUFS")
    print("=" * 60)
    print("Following LiReFs methodology for musical concept control")
    print()

    # Initialize extractor
    extractor = ResidualStreamLiMuFExtractor(
        sae_model_path=sae_model_path,
        activations_path=activations_path,
        interpretations_path=interpretations_path,
        device=device,
    )

    # Define the key musical contrast we want to control
    feature_contrasts = [
        # (feature_a_id, feature_b_id, contrast_name)
        (182, 1743, "steady_pulse_vs_melodic_sequences"),
        # Additional interesting contrasts from the interpretations
        (182, 889, "steady_pulse_vs_syncopation"),  # Steady vs syncopated rhythms
        (997, 231, "antiphonal_vs_octave_displacement"),  # Textural contrasts
        (855, 171, "dynamic_contrast_vs_chromatic_harmony"),  # Dynamic vs harmonic
    ]

    print(f"🎯 Extracting {len(feature_contrasts)} musical concept contrasts:")
    for feature_a, feature_b, name in feature_contrasts:
        # Get descriptions from interpretations
        interp_a = extractor.interpretations.get(str(feature_a), {}).get(
            "interpretation", f"Feature {feature_a}"
        )
        interp_b = extractor.interpretations.get(str(feature_b), {}).get(
            "interpretation", f"Feature {feature_b}"
        )

        print(f"   📊 {name}:")
        print(f"      Positive → {interp_a[:80]}...")
        print(f"      Negative → {interp_b[:80]}...")
        print()

    # Extract all contrasts
    extracted_limufs = extractor.extract_multiple_contrast_limufs(
        feature_contrasts=feature_contrasts,
        activation_threshold=2.0,  # Conservative threshold for reliable categorization
        min_samples=100,  # Minimum samples per category
        batch_size=5000,  # Memory-efficient processing
    )

    # Save results
    extractor.save_limufs(output_dir)

    # Show summary
    print(f"\n📊 EXTRACTION SUMMARY")
    print("=" * 40)
    print(f"Successfully extracted {len(extracted_limufs)} contrast LiMuFs:")

    for contrast_name, limuf_vector in extracted_limufs.items():
        stats = extractor.contrast_stats[contrast_name]
        print(f"\n🎯 {contrast_name}:")
        print(
            f"   Feature A: {stats['feature_a_id']} ({stats['feature_a_samples']} samples)"
        )
        print(
            f"   Feature B: {stats['feature_b_id']} ({stats['feature_b_samples']} samples)"
        )
        print(f"   Direction norm: {stats['direction_norm']:.4f}")
        print(f"   Vector shape: {limuf_vector.shape}")

    return extractor, extracted_limufs


def create_intervention_test_script(
    contrast_name: str = "steady_pulse_vs_melodic_sequences",
):
    """Create a test script for the extracted contrast LiMuF."""

    test_script = f'''#!/usr/bin/env python3
"""
Test {contrast_name} LiMuF interventions.
Generated automatically from extract_contrast_limufs.py
"""

import torch
from pathlib import Path
from mmt.residual_stream_limuf_extractor import ResidualStreamLiMuFExtractor

def test_{contrast_name.replace("-", "_")}_intervention():
    """Test intervention using the extracted contrast LiMuF."""
    
    # Load the extracted contrast LiMuFs
    limufs, metadata = ResidualStreamLiMuFExtractor.load_limufs("contrast_limufs_layer3")
    
    if "{contrast_name}" not in limufs:
        print("❌ Contrast LiMuF not found")
        return
        
    contrast_limuf = limufs["{contrast_name}"]
    
    print(f"🎼 Testing {{contrast_name}} intervention")
    print(f"   LiMuF shape: {{contrast_limuf.shape}}")
    print(f"   Direction norm: {{contrast_limuf.norm():.4f}}")
    
    # TODO: Integrate with your existing generation pipeline
    # Example intervention strengths to test:
    test_strengths = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for strength in test_strengths:
        print(f"\\n🎵 Testing strength: {{strength:+.1f}}")
        if strength > 0:
            print("   → Should promote STEADY PULSE patterns")
        elif strength < 0:
            print("   → Should promote EXTENDED MELODIC SEQUENCES")
        else:
            print("   → Baseline (no intervention)")
        
        # Your generation code would go here
        # intervened_activations = original_activations + strength * contrast_limuf
        
    print("\\n✅ Intervention test framework ready!")
    print("   Integrate with your generation pipeline to test musical concept control")

if __name__ == "__main__":
    test_{contrast_name.replace("-", "_")}_intervention()
'''

    # Save test script
    test_file = Path(f"test_{contrast_name}_intervention.py")
    with open(test_file, "w") as f:
        f.write(test_script)

    print(f"📝 Created test script: {test_file}")
    return test_file


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(
        description="Extract Musical Concept Contrast LiMuFs"
    )
    parser.add_argument(
        "--sae-model",
        default="sae/layer3_sae_model.pt",
        help="Path to trained SAE model",
    )
    parser.add_argument(
        "--activations",
        default="data/sae_activations_layer3.h5",
        help="Path to residual stream activations",
    )
    parser.add_argument(
        "--interpretations",
        default="interpretation/enhanced_diversity/enhanced_diversity_report.json",
        help="Path to feature interpretations",
    )
    parser.add_argument(
        "--output-dir",
        default="contrast_limufs_layer3",
        help="Output directory for contrast LiMuFs",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for extraction",
    )
    parser.add_argument(
        "--create-test-script",
        action="store_true",
        help="Create intervention test script",
    )

    args = parser.parse_args()

    print("🚀 MUSICAL CONCEPT CONTRAST LIMUF EXTRACTION")
    print("=" * 50)
    print("Based on LiReFs methodology:")
    print("📖 'Linear Reasoning Features' adapted for musical concepts")
    print()
    print(f"SAE Model: {args.sae_model}")
    print(f"Activations: {args.activations}")
    print(f"Interpretations: {args.interpretations}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Extract contrast LiMuFs
    extractor, limufs = extract_steady_pulse_vs_melodic_sequences(
        sae_model_path=args.sae_model,
        activations_path=args.activations,
        interpretations_path=args.interpretations,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Create test script if requested
    if args.create_test_script:
        create_intervention_test_script("steady_pulse_vs_melodic_sequences")

    print("\\n🎯 NEXT STEPS:")
    print("1. Review the extracted contrast LiMuFs")
    print("2. Integrate intervention functions with your generation pipeline")
    print("3. Test musical concept control during generation")
    print("4. Validate results with audio analysis")
    print()
    print("💡 Key Insight: These LiMuFs work in residual stream space")
    print("   where the model makes its actual decisions!")


if __name__ == "__main__":
    main()
