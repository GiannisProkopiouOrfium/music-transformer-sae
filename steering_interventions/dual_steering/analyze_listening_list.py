#!/usr/bin/env python3
"""Analyze listening list by scenario.

Shows top examples for each scenario separately by analyzing the full results file.
"""

import json
import pathlib
import sys
from collections import defaultdict


def main():
    if len(sys.argv) > 1:
        results_dir = pathlib.Path(sys.argv[1])
    else:
        results_dir = pathlib.Path("outputs/phase4_conditioned_quick")

    # Load full results to get per-scenario examples
    results_file = results_dir / "conditioned_results.json"

    if not results_file.exists():
        print(f"❌ File not found: {results_file}")
        return

    print(f"📖 Loading: {results_file}\n")

    with open(results_file, "r") as f:
        full_data = json.load(f)

    all_results = full_data["results"]

    # Score each result using the same criteria as extract_listening_list
    scored_results = []
    for r in all_results:
        if r.get("generated_n_notes", 0) < 10:
            continue

        # Exclude combinations where either alpha is zero (baseline)
        # We want both concepts actively steered
        if r["alpha_pitch"] == 0.0 or r["alpha_modality"] == 0.0:
            continue

        # Calculate audibility score
        pitch_magnitude = abs(r["pitch_change"]) / 10.0
        mode_change_score = 1.0 if r["mode_changed"] else 0.0
        confidence_score = r["generated_confidence"]

        # Calculate quality score
        degradation = r["degradation"]["total_degradation"]
        quality_score = max(0, 1.0 - degradation / 10.0)

        # Combined score
        audibility = (pitch_magnitude + mode_change_score + confidence_score) / 3.0
        overall_score = 0.6 * audibility + 0.4 * quality_score

        scored_results.append(
            {
                **r,
                "audibility_score": audibility,
                "quality_score": quality_score,
                "listening_score": overall_score,
            }
        )

    top_examples = scored_results

    # Group by scenario
    by_scenario = defaultdict(list)
    for example in top_examples:
        by_scenario[example["scenario"]].append(example)

    # Print header
    print("=" * 100)
    print("BEST LISTENING EXAMPLES BY SCENARIO")
    print("=" * 100)

    # Scenario names for nice display
    scenario_descriptions = {
        "low_pitch_minor_to_high_major": "Low Pitch Minor → High Major (fight both)",
        "high_pitch_major_to_low_minor": "High Pitch Major → Low Minor (fight both)",
        "low_pitch_major_to_high_minor": "Low Pitch Major → High Minor (fight both)",
        "high_pitch_minor_to_low_major": "High Pitch Minor → Low Major (fight both)",
    }

    # Print top 5 per scenario
    for scenario in sorted(by_scenario.keys()):
        examples = by_scenario[scenario]
        description = scenario_descriptions.get(scenario, scenario)

        print(f"\n{'='*100}")
        print(f"📊 SCENARIO: {description}")
        print(f"{'='*100}")
        print(f"Total examples in list: {len(examples)}\n")

        for i, ex in enumerate(examples[:5], 1):
            print(f"{i}. {ex['song_name']}")
            print(f"   Strategy: {ex['strategy']}")
            print(
                f"   α_pitch={ex['alpha_pitch']:+.1f}, α_modality={ex['alpha_modality']:+.1f}"
            )
            print(
                f"   Conditioning: pitch={ex['conditioning_pitch']:.1f}, mode={ex['conditioning_mode']}, conf={ex['conditioning_confidence']:.2f}"
            )
            print(
                f"   Generated: pitch={ex['generated_pitch_mean']:.1f}, mode={ex['generated_mode']}, conf={ex['generated_confidence']:.2f}"
            )
            print(
                f"   Change: Δpitch={ex['pitch_change']:+.1f} semitones, mode_changed={ex['mode_changed']}"
            )
            print(
                f"   Success: pitch={ex['pitch_steering_success']}, mode={ex['mode_steering_success']}, overall={ex['overall_success']}"
            )
            print(
                f"   Scores: listening={ex['listening_score']:.3f}, audibility={ex['audibility_score']:.2f}, quality={ex['quality_score']:.2f}"
            )
            print(f"   Degradation: {ex['degradation']['total_degradation']:.2f}")

            # Show file path
            scenario_name = ex["scenario"]
            strategy = ex["strategy"]
            alpha_p = ex["alpha_pitch"]
            alpha_m = ex["alpha_modality"]
            song_name = ex["song_name"]

            wav_path = f"{results_dir}/{scenario_name}/{strategy}/ap{alpha_p:+.1f}_am{alpha_m:+.1f}/{song_name}.wav"
            print(f"   🎵 Audio: {wav_path}")
            print()

    print("=" * 100)
    print("✅ Analysis complete!")
    print("=" * 100)

    # Summary statistics
    print("\n📈 SUMMARY BY SCENARIO:\n")
    for scenario in sorted(by_scenario.keys()):
        examples = by_scenario[scenario]
        description = scenario_descriptions.get(scenario, scenario)

        if examples:
            avg_score = sum(ex["listening_score"] for ex in examples) / len(examples)
            avg_pitch_change = sum(abs(ex["pitch_change"]) for ex in examples) / len(
                examples
            )
            mode_change_rate = sum(ex["mode_changed"] for ex in examples) / len(
                examples
            )
            success_rate = sum(ex["overall_success"] for ex in examples) / len(examples)

            print(f"{description}:")
            print(f"  Examples in top list: {len(examples)}")
            print(f"  Average listening score: {avg_score:.3f}")
            print(f"  Average pitch change: {avg_pitch_change:.1f} semitones")
            print(f"  Mode change rate: {mode_change_rate*100:.1f}%")
            print(f"  Success rate: {success_rate*100:.1f}%")
            print()


if __name__ == "__main__":
    main()
