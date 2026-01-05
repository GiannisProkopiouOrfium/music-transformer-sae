#!/usr/bin/env python3
"""
Flexible test script to evaluate specific categories/concepts.
Allows testing prompt changes on a subset of samples before full evaluation.
"""
import json
import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from evaluate_music_flamingo module in same directory
import evaluate_music_flamingo as efm

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_category_display_name(category, concept=None):
    """Get human-readable category name."""
    if category == "unconditional_single":
        return f"Category 1: Unconditional Single ({concept})"
    elif category == "unconditional_dual":
        return "Category 2: Unconditional Dual"
    elif category == "conditional_single":
        return f"Category 3: Conditional Single ({concept})"
    elif category == "conditional_dual":
        return "Category 4: Conditional Dual"
    return category


def main():
    parser = argparse.ArgumentParser(
        description="Test specific category/concept prompts"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "unconditional_single",
            "unconditional_dual",
            "conditional_single",
            "conditional_dual",
        ],
        help="Category to test",
    )
    parser.add_argument(
        "--concept",
        choices=["pitch", "duration"],
        help="Concept to test (required for single categories)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of samples to test (default: all)"
    )

    args = parser.parse_args()

    # Validate concept requirement
    if args.category in ["unconditional_single", "conditional_single"]:
        if not args.concept:
            parser.error(f"{args.category} requires --concept (pitch or duration)")

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

    # Filter samples based on category and concept
    if args.category == "unconditional_single":
        test_samples = [
            s
            for s in manifest
            if s["category"] == "unconditional_single" and s["concept"] == args.concept
        ]
    elif args.category == "unconditional_dual":
        test_samples = [s for s in manifest if s["category"] == "unconditional_dual"]
    elif args.category == "conditional_single":
        test_samples = [
            s
            for s in manifest
            if s["category"] == "conditional_single" and s["concept"] == args.concept
        ]
    elif args.category == "conditional_dual":
        test_samples = [s for s in manifest if s["category"] == "conditional_dual"]

    # Apply limit if specified
    if args.limit:
        test_samples = test_samples[: args.limit]

    category_name = get_category_display_name(args.category, args.concept)
    print(f"\n{'='*70}")
    print(f"Testing {category_name}")
    print(f"Samples: {len(test_samples)}")
    print(f"{'='*70}\n")

    if not test_samples:
        print("❌ No samples found matching criteria!")
        return 1

    results = []
    all_responses = []

    # Determine evaluation function
    if args.category == "unconditional_single":
        eval_func = efm.evaluate_sample_category_1
    elif args.category == "unconditional_dual":
        eval_func = efm.evaluate_sample_category_2
    elif args.category == "conditional_single":
        eval_func = efm.evaluate_sample_category_3
    elif args.category == "conditional_dual":
        eval_func = efm.evaluate_sample_category_4

    for sample in test_samples:
        identifier = f"alpha={sample.get('alpha', sample.get('id'))}"
        print(f"\n[Sample: {identifier}]")
        result = eval_func(sample, config, model, output_root)
        results.append(result)

        # Extract and display result
        if "final_rating" in result:
            rating = result.get("final_rating")
            all_responses.append(str(rating))
            print(f"  → Rating: {rating}")
        elif "final_pitch_rating" in result:
            pitch = result.get("final_pitch_rating")
            duration = result.get("final_duration_rating")
            all_responses.append(f"P{pitch}D{duration}")
            print(f"  → Pitch: {pitch}, Duration: {duration}")
        elif "final_direction" in result:
            direction = result.get("final_direction")
            all_responses.append(direction)
            print(f"  → Direction: {direction}")
        elif "final_pitch_direction" in result:
            pitch_dir = result.get("final_pitch_direction")
            duration_dir = result.get("final_duration_direction")
            all_responses.append(f"{pitch_dir}+{duration_dir}")
            print(f"  → Pitch: {pitch_dir}, Duration: {duration_dir}")

    # Analysis
    unique_responses = set(all_responses)
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"All responses: {all_responses}")
    print(f"Unique responses: {unique_responses} ({len(unique_responses)} unique)")

    if len(unique_responses) == 1:
        print(f"❌ FAILED: All samples gave identical response: '{all_responses[0]}'")
        print(f"   Model not discriminating between samples.")
    elif len(unique_responses) == len(all_responses):
        print(f"✅ PERFECT: All {len(all_responses)} samples gave different responses!")
        print(f"   Model discriminating excellently.")
    else:
        print(
            f"⚠️  PARTIAL: {len(unique_responses)}/{len(all_responses)} unique responses"
        )
        print(f"   Model showing some discrimination.")

    # Save results
    output_filename = f"test_{args.category}"
    if args.concept:
        output_filename += f"_{args.concept}"
    output_filename += "_results.json"

    output_file = Path("flamingo_eval") / output_filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return 0 if len(unique_responses) > 1 else 1


if __name__ == "__main__":
    sys.exit(main())
