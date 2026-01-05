#!/usr/bin/env python3
"""
Quick test script to evaluate only Category 1 Duration samples.
Tests if improved prompt clarity helps with duration discrimination.
"""
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from evaluate_music_flamingo module in same directory
import evaluate_music_flamingo as efm

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    # Load config
    config = efm.load_config(Path("flamingo_eval/config.yaml"))

    # Setup model
    api_method = config["music_flamingo"]["api_method"]
    if api_method == "transformers":
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        model_id = "nvidia/music-flamingo-hf"
        device = config["music_flamingo"].get("device", "cuda:0")

        logging.info(f"Loading {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        transformer_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        )

        # Package as dict like evaluate_music_flamingo expects
        model = {"processor": processor, "model": transformer_model}
    else:
        raise ValueError("This test only supports transformers mode")

    # Output root for file paths
    output_root = Path(config["paths"]["output_root"])

    # Load manifest
    manifest_path = Path("flamingo_eval/manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter to Category 1 Duration samples only
    duration_samples = [
        s
        for s in manifest
        if s["category"] == "unconditional_single" and s["concept"] == "duration"
    ]

    print(f"\n{'='*70}")
    print(f"Testing Category 1 Duration Only ({len(duration_samples)} samples)")
    print(f"{'='*70}\n")

    results = []
    ratings = []

    for sample in duration_samples:
        print(f"\n[Sample: alpha={sample['alpha']}]")
        result = efm.evaluate_sample_category_1(sample, config, model, output_root)
        results.append(result)

        # Extract rating
        rating = int(result.get("final_rating", 0))
        ratings.append(rating)
        print(f"  → Rating: {rating}")

    # Check if all identical
    unique_ratings = set(ratings)
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Ratings: {ratings}")
    print(f"Unique ratings: {unique_ratings}")

    if len(unique_ratings) == 1:
        print(f"❌ FAILED: All samples rated identically as '{ratings[0]}'")
        print(f"   Model still not discriminating duration.")
    else:
        print(f"✅ SUCCESS: {len(unique_ratings)} different ratings detected!")
        print(f"   Model is now discriminating duration.")

    # Save results
    output_file = Path("flamingo_eval/duration_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
