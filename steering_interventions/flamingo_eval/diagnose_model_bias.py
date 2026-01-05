#!/usr/bin/env python3
"""Diagnose Music Flamingo response bias and audio sensitivity.

This script analyzes the evaluation results to detect:
1. Response diversity (are responses too similar?)
2. Position bias (does model always say "UP" regardless of audio?)
3. Baseline vs steering detection (can model tell them apart?)
4. Text template matching (is model just repeating patterns?)
"""

import json
import pathlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio (0-1)."""
    return SequenceMatcher(None, a, b).ratio()


def analyze_response_diversity(results: list) -> dict:
    """Check if responses are too similar (template matching)."""
    analysis = {
        "by_category": {},
        "overall": {
            "total_samples": len(results),
            "avg_similarity": 0.0,
            "high_similarity_count": 0,
        },
    }

    # Group by category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    all_similarities = []

    for cat, samples in by_cat.items():
        similarities = []

        for sample in samples:
            responses = sample.get("responses", [])
            if len(responses) >= 2:
                # Compare each pair of responses for same sample
                for i in range(len(responses) - 1):
                    sim = text_similarity(responses[i], responses[i + 1])
                    similarities.append(sim)
                    all_similarities.append(sim)

        if similarities:
            avg_sim = np.mean(similarities)
            high_sim = sum(1 for s in similarities if s > 0.9)

            analysis["by_category"][cat] = {
                "avg_similarity": avg_sim,
                "high_similarity_count": high_sim,
                "total_comparisons": len(similarities),
                "high_similarity_rate": (
                    high_sim / len(similarities) if similarities else 0
                ),
            }

    if all_similarities:
        analysis["overall"]["avg_similarity"] = np.mean(all_similarities)
        analysis["overall"]["high_similarity_count"] = sum(
            1 for s in all_similarities if s > 0.9
        )
        analysis["overall"]["high_similarity_rate"] = analysis["overall"][
            "high_similarity_count"
        ] / len(all_similarities)

    return analysis


def analyze_position_bias(results: list) -> dict:
    """Check if model has directional bias (always says UP/LONGER)."""
    analysis = {}

    # Category 1 & 2: Check rating distributions
    cat1_pitch = [
        r
        for r in results
        if r["category"] == "unconditional_single" and r["concept"] == "pitch"
    ]
    cat1_dur = [
        r
        for r in results
        if r["category"] == "unconditional_single" and r["concept"] == "duration"
    ]

    if cat1_pitch:
        pitch_ratings = [
            r.get("final_rating") for r in cat1_pitch if r.get("final_rating")
        ]
        analysis["cat1_pitch"] = {
            "distribution": dict(Counter(pitch_ratings)),
            "unique_values": len(set(pitch_ratings)),
            "is_constant": len(set(pitch_ratings)) == 1,
        }

    if cat1_dur:
        dur_ratings = [r.get("final_rating") for r in cat1_dur if r.get("final_rating")]
        analysis["cat1_duration"] = {
            "distribution": dict(Counter(dur_ratings)),
            "unique_values": len(set(dur_ratings)),
            "is_constant": len(set(dur_ratings)) == 1,
        }

    # Category 3: Check if model can detect decreases
    cat3 = [r for r in results if r["category"] == "conditional_single"]

    direction_counts = defaultdict(lambda: {"UP": 0, "DOWN": 0, "CONSTANT": 0})

    for r in cat3:
        concept = r.get("concept", "")
        direction = r.get("final_direction", "")
        transition = r.get("transition_type", "")

        if direction:
            direction_counts[f"{concept}_{transition}"][direction] += 1

    analysis["cat3_directional_bias"] = dict(direction_counts)

    # Category 3 specific: Check if model EVER says DOWN
    all_cat3_directions = [
        r.get("final_direction") for r in cat3 if r.get("final_direction")
    ]
    analysis["cat3_overall"] = {
        "distribution": dict(Counter(all_cat3_directions)),
        "has_down": "DOWN" in all_cat3_directions,
        "has_constant": "CONSTANT" in all_cat3_directions,
    }

    return analysis


def analyze_baseline_vs_steering(results: list) -> dict:
    """Check if model can distinguish baseline from steered samples."""
    analysis = {}

    # Category 3 & 4: Compare baseline detection
    cat3 = [r for r in results if r["category"] == "conditional_single"]

    baseline_correct = 0
    baseline_total = 0

    for r in cat3:
        alpha = r.get("alpha", 0)
        direction = r.get("final_direction", "")
        transition = r.get("transition_type", "")

        # For baseline (alpha=0), should say CONSTANT
        if alpha == 0:
            baseline_total += 1
            if direction == "CONSTANT":
                baseline_correct += 1

    if baseline_total > 0:
        analysis["baseline_detection"] = {
            "correct": baseline_correct,
            "total": baseline_total,
            "accuracy": baseline_correct / baseline_total,
        }

    return analysis


def analyze_response_templates(results: list) -> dict:
    """Find common response templates (indicates non-audio-driven responses)."""
    analysis = {"common_patterns": {}}

    # Extract first 50 chars of each response
    by_category = defaultdict(list)
    for r in results:
        responses = r.get("responses", [])
        cat = r["category"]
        for resp in responses:
            prefix = resp[:50] if resp else ""
            by_category[cat].append(prefix)

    for cat, prefixes in by_category.items():
        counter = Counter(prefixes)
        total = len(prefixes)
        most_common = counter.most_common(5)

        analysis["common_patterns"][cat] = {
            "total_responses": total,
            "unique_prefixes": len(counter),
            "top_5": [
                {"prefix": prefix, "count": count, "percentage": count / total * 100}
                for prefix, count in most_common
            ],
        }

    return analysis


def main():
    results_path = pathlib.Path("flamingo_eval/results.json")

    print("=" * 70)
    print("Music Flamingo Bias Diagnosis")
    print("=" * 70)

    with open(results_path) as f:
        results = json.load(f)

    print(f"\nAnalyzing {len(results)} samples...\n")

    # 1. Response diversity
    print("=" * 70)
    print("1. RESPONSE DIVERSITY (Are responses too similar?)")
    print("=" * 70)
    diversity = analyze_response_diversity(results)
    print(f"\nOverall avg similarity: {diversity['overall']['avg_similarity']:.3f}")
    print(
        f"High similarity rate (>0.9): {diversity['overall'].get('high_similarity_rate', 0):.1%}"
    )

    for cat, data in diversity["by_category"].items():
        print(f"\n{cat}:")
        print(f"  Avg similarity: {data['avg_similarity']:.3f}")
        print(
            f"  High similarity: {data['high_similarity_count']}/{data['total_comparisons']} ({data['high_similarity_rate']:.1%})"
        )

    # 2. Position bias
    print("\n" + "=" * 70)
    print("2. POSITIONAL BIAS (Does model always say UP/LONGER?)")
    print("=" * 70)
    bias = analyze_position_bias(results)

    if "cat1_pitch" in bias:
        print("\nCategory I Pitch ratings:")
        print(f"  Distribution: {bias['cat1_pitch']['distribution']}")
        print(f"  Unique values: {bias['cat1_pitch']['unique_values']}")
        print(f"  ⚠️ CONSTANT: {bias['cat1_pitch']['is_constant']}")

    if "cat1_duration" in bias:
        print("\nCategory I Duration ratings:")
        print(f"  Distribution: {bias['cat1_duration']['distribution']}")
        print(f"  Unique values: {bias['cat1_duration']['unique_values']}")
        print(f"  ⚠️ CONSTANT: {bias['cat1_duration']['is_constant']}")

    print("\nCategory III Directional Bias:")
    for trans, counts in bias.get("cat3_directional_bias", {}).items():
        print(f"  {trans}: {dict(counts)}")

    print(
        f"\nCategory III Overall: {bias.get('cat3_overall', {}).get('distribution', {})}"
    )
    print(f"  Can detect DOWN: {bias.get('cat3_overall', {}).get('has_down', False)}")

    # 3. Baseline detection
    print("\n" + "=" * 70)
    print("3. BASELINE DETECTION (Can model detect no change?)")
    print("=" * 70)
    baseline = analyze_baseline_vs_steering(results)

    if "baseline_detection" in baseline:
        acc = baseline["baseline_detection"]["accuracy"]
        print(f"\nBaseline (α=0) correctly identified as CONSTANT:")
        print(
            f"  {baseline['baseline_detection']['correct']}/{baseline['baseline_detection']['total']} ({acc:.1%})"
        )
    else:
        print("\nNo baseline samples found in data")

    # 4. Response templates
    print("\n" + "=" * 70)
    print("4. RESPONSE TEMPLATES (Is model repeating patterns?)")
    print("=" * 70)
    templates = analyze_response_templates(results)

    for cat, data in templates["common_patterns"].items():
        print(f"\n{cat}:")
        print(f"  Total responses: {data['total_responses']}")
        print(f"  Unique prefixes: {data['unique_prefixes']}")
        print(
            f"  Template usage rate: {(1 - data['unique_prefixes']/data['total_responses']):.1%}"
        )
        print("\n  Top repeated patterns:")
        for item in data["top_5"][:3]:
            print(
                f"    '{item['prefix']}...' → {item['count']} times ({item['percentage']:.1f}%)"
            )

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if diversity["overall"].get("high_similarity_rate", 0) > 0.5:
        issues.append("❌ High response similarity (>50% nearly identical)")

    if bias.get("cat1_duration", {}).get("is_constant"):
        issues.append("❌ Duration ratings are constant (model not discriminating)")

    if not bias.get("cat3_overall", {}).get("has_down", False):
        issues.append("❌ Model NEVER detects decreases (strong UP/LONGER bias)")

    baseline_acc = baseline.get("baseline_detection", {}).get("accuracy", 1.0)
    if baseline_acc < 0.5:
        issues.append(f"❌ Poor baseline detection ({baseline_acc:.1%})")

    if issues:
        print("\n🔴 CRITICAL ISSUES DETECTED:\n")
        for issue in issues:
            print(f"  {issue}")
        print(
            "\n⚠️  Model appears to be following prompt patterns rather than judging audio!"
        )
        print(
            "    Recommendation: Try two-stage pipeline (Music Flamingo describe → GPT-4o judge)"
        )
    else:
        print("\n✅ Model shows reasonable audio sensitivity")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
