#!/usr/bin/env python3
"""Test Music Flamingo evaluation with one sample from each category."""

import json
import pathlib
import sys
import logging
import importlib.util

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))

# Import evaluation functions from evaluate_music_flamingo.py
# eval_script_path = pathlib.Path(__file__).parent / "evaluate_music_flamingo.py"
# spec = importlib.util.spec_from_file_location(
#     "evaluate_music_flamingo", eval_script_path
# )
# eval_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(eval_module)

# Import the functions we need
# evaluate_sample_category_1 = eval_module.evaluate_sample_category_1
# evaluate_sample_category_2 = eval_module.evaluate_sample_category_2
# evaluate_sample_category_3 = eval_module.evaluate_sample_category_3
# evaluate_sample_category_4 = eval_module.evaluate_sample_category_4

from evaluate_music_flamingo import (
    evaluate_sample_category_1,
    evaluate_sample_category_2,
    evaluate_sample_category_3,
    evaluate_sample_category_4,
)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("MUSIC FLAMINGO EVALUATION - TEST RUN")
    print("=" * 70)

    # Load manifest
    manifest_path = pathlib.Path("flamingo_eval/manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Select one sample from each category
    test_samples = {}
    for entry in manifest:
        cat = entry["category"]
        if cat not in test_samples:
            test_samples[cat] = entry

    print(f"\nSelected {len(test_samples)} test samples:")
    for cat, entry in test_samples.items():
        print(f"  {cat}: {entry['id']}")

    # Load config
    import yaml

    config_path = pathlib.Path("flamingo_eval/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize Music Flamingo client
    print(f"\nInitializing Music Flamingo client...")
    try:
        from gradio_client import Client

        model = Client("nvidia/music-flamingo")
        print(f"✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return 1

    # Test each category
    output_root = pathlib.Path(config["paths"]["output_root"])
    results = []

    for cat, sample in test_samples.items():
        print(f"\n{'='*70}")
        print(f"Testing Category: {cat.upper()}")
        print(f"Sample: {sample['id']}")
        print("=" * 70)

        try:
            if cat == "unconditional_single":
                result = evaluate_sample_category_1(sample, config, model, output_root)
            elif cat == "unconditional_dual":
                result = evaluate_sample_category_2(sample, config, model, output_root)
            elif cat == "conditional_single":
                result = evaluate_sample_category_3(sample, config, model, output_root)
            elif cat == "conditional_dual":
                result = evaluate_sample_category_4(sample, config, model, output_root)

            print(f"\n✅ SUCCESS for {cat}")
            print(
                f"Final rating: {result.get('final_rating', result.get('final_pitch_rating', 'N/A'))}"
            )
            results.append({"category": cat, "success": True, "result": result})

        except Exception as e:
            print(f"\n❌ FAILED for {cat}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"category": cat, "success": False, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results if r["success"])
    print(f"Passed: {success_count}/{len(results)}")

    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['category']}")
        if not result["success"]:
            print(f"      Error: {result.get('error', 'Unknown')}")

    if success_count == len(results):
        print(f"\n🎉 All tests passed! Ready for full evaluation.")
        print(f"\nTo run full evaluation:")
        print(
            f"  python flamingo_eval/evaluate_music_flamingo.py --config flamingo_eval/config.yaml --resume"
        )
        return 0
    else:
        print(
            f"\n⚠️  Some tests failed. Please fix errors before running full evaluation."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
