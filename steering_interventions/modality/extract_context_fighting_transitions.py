"""Extract and rank context-fighting transitions from key analysis results.

This script identifies the most successful cases where steering overcame conditioning:
1. Minor conditioning → Major generated (positive alpha, low degradation)
2. Major conditioning → Minor generated (negative alpha, low degradation)

These represent cases where the model successfully "fought" the conditioning context.
"""

import argparse
import json
import pathlib
from typing import Dict, List


def extract_context_fighting_transitions(results_file: pathlib.Path) -> Dict:
    """Extract transitions where steering successfully fought conditioning.

    Args:
        results_file: Path to key_transition_analysis.json

    Returns:
        Dictionary with ranked context-fighting transitions
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    all_results = data["results"]

    # Filter for high-confidence generations
    confident = [r for r in all_results if r["generated_confidence"] >= 0.5]

    # Case 1: Minor conditioning → Major generated (positive alpha)
    minor_to_major = [
        r
        for r in confident
        if r["conditioning_mode"] == "minor"
        and r["generated_mode"] == "major"
        and r["alpha"] > 0
        and "degradation" in r
    ]

    # Case 2: Major conditioning → Minor generated (negative alpha)
    major_to_minor = [
        r
        for r in confident
        if r["conditioning_mode"] == "major"
        and r["generated_mode"] == "minor"
        and r["alpha"] < 0
        and "degradation" in r
    ]

    # Rank by success score: high confidence, low degradation, strong alpha
    def calculate_success_score(result: Dict) -> float:
        """Calculate success score for context-fighting transition.

        Higher score = better success at fighting context
        """
        # Components:
        # 1. Generated confidence (0-1) -> weight heavily
        confidence_score = result["generated_confidence"] * 100

        # 2. Low degradation (invert so lower is better) -> weight moderately
        # Typical degradation ranges 0-100, cap at 100
        degradation = min(result["degradation"]["total_degradation"], 100)
        quality_score = (100 - degradation) * 0.8

        # 3. Alpha strength (how strong the steering was) -> weight lightly
        alpha_strength = abs(result["alpha"]) * 10

        total_score = confidence_score + quality_score + alpha_strength

        return total_score

    # Calculate scores and rank
    for result in minor_to_major:
        result["success_score"] = calculate_success_score(result)

    for result in major_to_minor:
        result["success_score"] = calculate_success_score(result)

    # Sort by success score
    minor_to_major_ranked = sorted(
        minor_to_major, key=lambda x: x["success_score"], reverse=True
    )
    major_to_minor_ranked = sorted(
        major_to_minor, key=lambda x: x["success_score"], reverse=True
    )

    # Create summary statistics
    summary = {
        "minor_to_major": {
            "total_count": len(minor_to_major),
            "avg_success_score": (
                sum(r["success_score"] for r in minor_to_major) / len(minor_to_major)
                if minor_to_major
                else 0
            ),
            "avg_degradation": (
                sum(r["degradation"]["total_degradation"] for r in minor_to_major)
                / len(minor_to_major)
                if minor_to_major
                else 0
            ),
            "avg_confidence": (
                sum(r["generated_confidence"] for r in minor_to_major)
                / len(minor_to_major)
                if minor_to_major
                else 0
            ),
            "alpha_distribution": {},
        },
        "major_to_minor": {
            "total_count": len(major_to_minor),
            "avg_success_score": (
                sum(r["success_score"] for r in major_to_minor) / len(major_to_minor)
                if major_to_minor
                else 0
            ),
            "avg_degradation": (
                sum(r["degradation"]["total_degradation"] for r in major_to_minor)
                / len(major_to_minor)
                if major_to_minor
                else 0
            ),
            "avg_confidence": (
                sum(r["generated_confidence"] for r in major_to_minor)
                / len(major_to_minor)
                if major_to_minor
                else 0
            ),
            "alpha_distribution": {},
        },
    }

    # Alpha distribution for minor→major
    for r in minor_to_major:
        alpha = r["alpha"]
        if alpha not in summary["minor_to_major"]["alpha_distribution"]:
            summary["minor_to_major"]["alpha_distribution"][alpha] = {
                "count": 0,
                "avg_success_score": 0,
                "scores": [],
            }
        summary["minor_to_major"]["alpha_distribution"][alpha]["count"] += 1
        summary["minor_to_major"]["alpha_distribution"][alpha]["scores"].append(
            r["success_score"]
        )

    # Calculate averages
    for alpha, data in summary["minor_to_major"]["alpha_distribution"].items():
        data["avg_success_score"] = sum(data["scores"]) / len(data["scores"])
        del data["scores"]

    # Alpha distribution for major→minor
    for r in major_to_minor:
        alpha = r["alpha"]
        if alpha not in summary["major_to_minor"]["alpha_distribution"]:
            summary["major_to_minor"]["alpha_distribution"][alpha] = {
                "count": 0,
                "avg_success_score": 0,
                "scores": [],
            }
        summary["major_to_minor"]["alpha_distribution"][alpha]["count"] += 1
        summary["major_to_minor"]["alpha_distribution"][alpha]["scores"].append(
            r["success_score"]
        )

    # Calculate averages
    for alpha, data in summary["major_to_minor"]["alpha_distribution"].items():
        data["avg_success_score"] = sum(data["scores"]) / len(data["scores"])
        del data["scores"]

    return {
        "summary": summary,
        "minor_to_major_ranked": minor_to_major_ranked,
        "major_to_minor_ranked": major_to_minor_ranked,
    }


def create_listening_list(
    ranked_results: List[Dict], top_n: int = 20, output_dir: pathlib.Path = None
) -> List[Dict]:
    """Create a concise listening list with file paths.

    Args:
        ranked_results: Ranked results list
        top_n: Number of top results to include
        output_dir: Base output directory for constructing file paths

    Returns:
        List of listening recommendations
    """
    listening_list = []

    for i, r in enumerate(ranked_results[:top_n], 1):
        # Construct file paths
        category = r["conditioning_category"]
        alpha = r["alpha"]
        song_name = r["song_name"]

        file_paths = {}
        if output_dir is not None:
            base_path = output_dir / category / f"alpha_{alpha}" / song_name
            file_paths = {
                "midi_file": str(base_path.with_suffix(".mid")),
                "wav_file": str(base_path.with_suffix(".wav")),
                "npy_file": str(base_path.with_suffix(".npy")),
            }

        listening_list.append(
            {
                "rank": i,
                "song_name": song_name,
                "transition": r["key_transition"],
                "alpha": r["alpha"],
                "success_score": round(r["success_score"], 2),
                "generated_confidence": round(r["generated_confidence"], 3),
                "degradation": round(r["degradation"]["total_degradation"], 2),
                "quality_metrics": {
                    "pitch_class_entropy": round(
                        r["quality_metrics"].get("pitch_class_entropy", 0), 3
                    ),
                    "scale_consistency": round(
                        r["quality_metrics"].get("scale_consistency", 0), 1
                    ),
                    "groove_consistency": round(
                        r["quality_metrics"].get("groove_consistency", 0), 1
                    ),
                },
                "file_paths": file_paths,
            }
        )

    return listening_list


def print_summary(analysis: Dict):
    """Print formatted summary of context-fighting transitions."""
    summary = analysis["summary"]

    print("\n" + "=" * 80)
    print("CONTEXT-FIGHTING TRANSITIONS ANALYSIS")
    print("=" * 80)

    print("\n### MINOR → MAJOR (Positive Alpha) ###")
    print(
        f"Total successful transitions: {summary['minor_to_major']['total_count']}"
    )
    if summary["minor_to_major"]["total_count"] > 0:
        print(
            f"Average success score: {summary['minor_to_major']['avg_success_score']:.2f}"
        )
        print(
            f"Average confidence: {summary['minor_to_major']['avg_confidence']:.3f}"
        )
        print(
            f"Average degradation: {summary['minor_to_major']['avg_degradation']:.2f}"
        )

        print("\nSuccess by alpha value:")
        for alpha in sorted(summary["minor_to_major"]["alpha_distribution"].keys()):
            data = summary["minor_to_major"]["alpha_distribution"][alpha]
            print(
                f"  α = {alpha:>+5.1f}: {data['count']:>3} transitions, "
                f"avg score = {data['avg_success_score']:>6.2f}"
            )

        print(f"\nTop 10 Minor → Major transitions:")
        print(
            f"  {'Rank':<6} {'Song':<25} {'Transition':<35} {'α':<7} "
            f"{'Score':<8} {'Conf':<6} {'Degrad':<8}"
        )
        print("  " + "-" * 95)
        for i, r in enumerate(analysis["minor_to_major_ranked"][:10], 1):
            print(
                f"  {i:<6} {r['song_name'][:24]:<25} {r['key_transition']:<35} "
                f"{r['alpha']:>+5.1f}  {r['success_score']:>7.2f} "
                f"{r['generated_confidence']:>5.3f} {r['degradation']['total_degradation']:>7.2f}"
            )

    print("\n### MAJOR → MINOR (Negative Alpha) ###")
    print(
        f"Total successful transitions: {summary['major_to_minor']['total_count']}"
    )
    if summary["major_to_minor"]["total_count"] > 0:
        print(
            f"Average success score: {summary['major_to_minor']['avg_success_score']:.2f}"
        )
        print(
            f"Average confidence: {summary['major_to_minor']['avg_confidence']:.3f}"
        )
        print(
            f"Average degradation: {summary['major_to_minor']['avg_degradation']:.2f}"
        )

        print("\nSuccess by alpha value:")
        for alpha in sorted(summary["major_to_minor"]["alpha_distribution"].keys()):
            data = summary["major_to_minor"]["alpha_distribution"][alpha]
            print(
                f"  α = {alpha:>+5.1f}: {data['count']:>3} transitions, "
                f"avg score = {data['avg_success_score']:>6.2f}"
            )

        print(f"\nTop 10 Major → Minor transitions:")
        print(
            f"  {'Rank':<6} {'Song':<25} {'Transition':<35} {'α':<7} "
            f"{'Score':<8} {'Conf':<6} {'Degrad':<8}"
        )
        print("  " + "-" * 95)
        for i, r in enumerate(analysis["major_to_minor_ranked"][:10], 1):
            print(
                f"  {i:<6} {r['song_name'][:24]:<25} {r['key_transition']:<35} "
                f"{r['alpha']:>+5.1f}  {r['success_score']:>7.2f} "
                f"{r['generated_confidence']:>5.3f} {r['degradation']['total_degradation']:>7.2f}"
            )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract context-fighting transitions from key analysis"
    )
    parser.add_argument(
        "--results_file",
        type=pathlib.Path,
        default=None,
        help="Path to key_transition_analysis.json",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Base output directory for file paths",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top results to save in listening list",
    )

    args = parser.parse_args()

    # Default paths
    if args.results_file is None:
        args.results_file = pathlib.Path(
            "steering_interventions/modality/outputs/key_transition_analysis/key_transition_analysis.json"
        )

    if args.output_dir is None:
        args.output_dir = pathlib.Path(
            "steering_interventions/modality/outputs/key_transition_analysis"
        )

    if not args.results_file.exists():
        print(f"ERROR: Results file not found: {args.results_file}")
        print("Run key_analysis_extended.py first to generate results.")
        return

    # Extract and analyze
    print(f"Loading results from: {args.results_file}")
    analysis = extract_context_fighting_transitions(args.results_file)

    # Print summary
    print_summary(analysis)

    # Create listening lists
    print("\n\nCreating listening lists...")

    minor_to_major_list = create_listening_list(
        analysis["minor_to_major_ranked"], args.top_n, args.output_dir
    )
    major_to_minor_list = create_listening_list(
        analysis["major_to_minor_ranked"], args.top_n, args.output_dir
    )

    # Save to JSON
    output_file = args.results_file.parent / "context_fighting_transitions.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": analysis["summary"],
                "listening_lists": {
                    "minor_to_major_top_20": minor_to_major_list,
                    "major_to_minor_top_20": major_to_minor_list,
                },
                "full_results": {
                    "minor_to_major_all": analysis["minor_to_major_ranked"],
                    "major_to_minor_all": analysis["major_to_minor_ranked"],
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved detailed results to: {output_file}")

    # Save compact listening lists
    listening_file = args.results_file.parent / "listening_list_context_fighting.json"
    with open(listening_file, "w") as f:
        json.dump(
            {
                "instructions": "These are the best examples of steering successfully overriding conditioning context",
                "minor_to_major": {
                    "description": "Minor conditioning → Major generation (positive alpha)",
                    "top_20": minor_to_major_list,
                },
                "major_to_minor": {
                    "description": "Major conditioning → Minor generation (negative alpha)",
                    "top_20": major_to_minor_list,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved listening list to: {listening_file}")

    # Print file path examples
    if minor_to_major_list and minor_to_major_list[0].get("file_paths"):
        print("\n### Example file paths (Minor → Major, Rank #1) ###")
        print(f"WAV:  {minor_to_major_list[0]['file_paths'].get('wav_file', 'N/A')}")
        print(f"MIDI: {minor_to_major_list[0]['file_paths'].get('midi_file', 'N/A')}")

    if major_to_minor_list and major_to_minor_list[0].get("file_paths"):
        print("\n### Example file paths (Major → Minor, Rank #1) ###")
        print(f"WAV:  {major_to_minor_list[0]['file_paths'].get('wav_file', 'N/A')}")
        print(f"MIDI: {major_to_minor_list[0]['file_paths'].get('midi_file', 'N/A')}")

    print("\n" + "=" * 80)
    print("DONE - Check the JSON files for complete results with file paths")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
