#!/usr/bin/env python3
"""Expanded Test: Verify model discriminates between different samples.

This script:
1. Selects 3 diverse samples per category (e.g., positive/negative/baseline alphas)
2. Runs each sample ONCE (not 3x, since model is consistent)
3. Verifies model gives different responses for different samples

Usage:
    python flamingo_eval/expanded_test_evaluation.py --config flamingo_eval/config.yaml
"""

import argparse
import json
import logging
import pathlib
import sys

# Import evaluation functions
from evaluate_music_flamingo import (
    load_config,
    evaluate_sample_category_1,
    evaluate_sample_category_2,
    evaluate_sample_category_3,
    evaluate_sample_category_4,
)


def select_diverse_samples(manifest: list, category: str, config: dict) -> list:
    """Select 3 diverse samples from a category.

    Args:
        manifest: Full manifest list
        category: Category name
        config: Config dict

    Returns:
        List of 3 selected samples
    """
    category_samples = [s for s in manifest if s["category"] == category]

    if category == "unconditional_single":
        # Select: negative alpha, baseline (0.0), positive alpha
        selected = []

        # Get pitch samples with different alphas
        pitch_samples = [s for s in category_samples if s["concept"] == "pitch"]

        # Negative alpha (e.g., -1.5)
        neg = [s for s in pitch_samples if s["alpha"] < 0]
        if neg:
            selected.append(sorted(neg, key=lambda x: x["alpha"])[0])

        # Baseline (0.0)
        baseline = [s for s in pitch_samples if s["alpha"] == 0.0]
        if baseline:
            selected.append(baseline[0])

        # Positive alpha (e.g., +1.5)
        pos = [s for s in pitch_samples if s["alpha"] > 0]
        if pos:
            selected.append(sorted(pos, key=lambda x: x["alpha"], reverse=True)[0])

        return selected[:3]

    elif category == "unconditional_dual":
        # Select 3 different scenarios
        scenarios = ["low_long", "high_short", "baseline"]
        selected = []

        for scenario in scenarios:
            scenario_samples = [
                s for s in category_samples if s.get("scenario") == scenario
            ]
            if scenario_samples:
                selected.append(scenario_samples[0])

        return selected[:3]

    elif category == "conditional_single":
        # Select: pitch high_to_low, duration low_to_high, pitch baseline
        selected = []

        # Pitch high_to_low with strong alpha
        pitch_htl = [
            s
            for s in category_samples
            if s["concept"] == "pitch"
            and s.get("transition_type") == "high_to_low"
            and abs(s.get("alpha", 0)) > 1.0
        ]
        if pitch_htl:
            selected.append(pitch_htl[0])

        # Duration low_to_high with strong alpha
        dur_lth = [
            s
            for s in category_samples
            if s["concept"] == "duration"
            and s.get("transition_type") == "low_to_high"
            and abs(s.get("alpha", 0)) > 1.0
        ]
        if dur_lth:
            selected.append(dur_lth[0])

        # Pitch baseline
        pitch_baseline = [
            s
            for s in category_samples
            if s["concept"] == "pitch" and s.get("alpha") == 0.0
        ]
        if pitch_baseline:
            selected.append(pitch_baseline[0])

        return selected[:3]

    elif category == "conditional_dual":
        # Select 3 different transition scenarios
        scenarios = [
            "low_pitch_short_duration_to_high_long",
            "high_pitch_long_duration_to_low_short",
            "low_pitch_long_duration_to_high_short",
        ]

        selected = []
        for scenario in scenarios:
            scenario_samples = [
                s for s in category_samples if scenario in s.get("id", "")
            ]
            if scenario_samples:
                selected.append(scenario_samples[0])

        return selected[:3]

    # Fallback: just take first 3
    return category_samples[:3]


def main():
    parser = argparse.ArgumentParser(
        description="Expanded diversity test for Music Flamingo"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("flamingo_eval/config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("EXPANDED DIVERSITY TEST")
    print("=" * 70)
    print("Testing 3 diverse samples per category (single run each)")
    print("Goal: Verify model discriminates between different samples")
    print("=" * 70)

    # Load config
    config = load_config(args.config)

    # Temporarily set num_runs to 1
    original_num_runs = config["music_flamingo"]["num_runs"]
    config["music_flamingo"]["num_runs"] = 1

    # Load manifest
    output_root = pathlib.Path(config["paths"]["output_root"])
    manifest_path = output_root / "manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize model
    api_method = config["music_flamingo"]["api_method"]
    print(f"\nInitializing Music Flamingo ({api_method})...")

    if api_method == "transformers":
        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained("nvidia/music-flamingo-hf")
        transformer_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            "nvidia/music-flamingo-hf", device_map="auto", torch_dtype=torch.float16
        )
        model = {"processor": processor, "model": transformer_model}
        print("✅ Model loaded\n")
    else:
        print("❌ Only transformers method supported for this test")
        sys.exit(1)

    # Test categories
    categories = [
        ("unconditional_single", evaluate_sample_category_1),
        ("unconditional_dual", evaluate_sample_category_2),
        ("conditional_single", evaluate_sample_category_3),
        ("conditional_dual", evaluate_sample_category_4),
    ]

    all_results = []

    for category_name, eval_func in categories:
        print("=" * 70)
        print(f"CATEGORY: {category_name.upper().replace('_', ' ')}")
        print("=" * 70)

        # Select diverse samples
        samples = select_diverse_samples(manifest, category_name, config)

        if not samples:
            print(f"⚠️  No samples found for {category_name}\n")
            continue

        print(f"Testing {len(samples)} diverse samples:\n")

        category_responses = []

        for idx, sample in enumerate(samples, 1):
            sample_id = sample["id"]
            print(f"\n--- Sample {idx}: {sample_id} ---")

            # Print sample details
            if "alpha" in sample:
                print(f"Alpha: {sample['alpha']}")
            if "concept" in sample:
                print(f"Concept: {sample['concept']}")
            if "scenario" in sample:
                print(f"Scenario: {sample['scenario']}")
            if "transition_type" in sample:
                print(f"Transition: {sample['transition_type']}")

            # Evaluate
            result = eval_func(sample, config, model, output_root)
            all_results.append(result)

            # Display results
            response = result["responses"][0] if result.get("responses") else "N/A"
            print(f"\nResponse: {response[:120]}...")

            if category_name == "unconditional_single":
                rating = result.get("final_rating")
                print(f"Rating: {rating}")
                category_responses.append(str(rating))

            elif category_name == "unconditional_dual":
                pitch = result.get("final_pitch_rating")
                duration = result.get("final_duration_rating")
                print(f"Pitch rating: {pitch}, Duration rating: {duration}")
                category_responses.append(f"P{pitch}D{duration}")

            elif category_name == "conditional_single":
                direction = result.get("final_direction")
                print(f"Direction: {direction}")
                category_responses.append(direction)

            elif category_name == "conditional_dual":
                pitch_dir = result.get("final_pitch_direction")
                dur_dir = result.get("final_duration_direction")
                print(f"Pitch: {pitch_dir}, Duration: {dur_dir}")
                category_responses.append(f"{pitch_dir}+{dur_dir}")

        # Check diversity
        print(f"\n{'='*70}")
        print(f"DIVERSITY CHECK for {category_name}:")
        print(f"{'='*70}")
        print(f"Responses: {category_responses}")

        unique_responses = len(set(category_responses))
        total_responses = len(category_responses)

        if unique_responses == 1:
            print(f"❌ FAILED: All {total_responses} samples gave identical response!")
            print(f"   Model may not be discriminating between samples.")
        elif unique_responses == total_responses:
            print(f"✅ PASSED: All {total_responses} samples gave different responses")
            print(f"   Model is discriminating well!")
        else:
            print(f"⚠️  PARTIAL: {unique_responses}/{total_responses} unique responses")
            print(f"   Model shows some discrimination.")

        print()

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total samples tested: {len(all_results)}")
    print("\nConclusion:")
    print("If categories show diversity, model is working correctly.")
    print("If categories show identical responses, model may have issues.")
    print("=" * 70)

    # Restore original num_runs
    config["music_flamingo"]["num_runs"] = original_num_runs


if __name__ == "__main__":
    main()
