"""Analyze already-generated evaluation results without re-generating."""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict

import numpy as np
import scipy.stats as stats

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config


def extract_velocities(notes: np.ndarray) -> np.ndarray:
    """Extract velocity values from note sequence.

    Args:
        notes: Note sequence array (n, d) where d >= 4

    Returns:
        Array of velocity values
    """
    # Assuming notes format: [type, beat, position, pitch, duration, instrument]
    # Velocity might be in the pitch column for note-on events
    # Or we need to check the actual format

    # Filter for note events (typically type 0 or 1)
    note_events = notes[notes[:, 0] <= 1]

    if len(note_events) == 0:
        return np.array([])

    # Velocity is typically stored in pitch column (index 3)
    # or there might be a separate velocity column
    velocities = note_events[:, 3]  # pitch column

    # Filter valid MIDI velocity range [0, 127]
    velocities = velocities[(velocities >= 0) & (velocities <= 127)]

    return velocities


def analyze_sample(sample_path: pathlib.Path) -> Dict:
    """Analyze a single generated sample.

    Args:
        sample_path: Path to .npy file

    Returns:
        Dictionary with statistics
    """
    notes = np.load(sample_path)
    velocities = extract_velocities(notes)

    if len(velocities) == 0:
        return {
            "n_notes": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }

    return {
        "n_notes": len(velocities),
        "mean": float(np.mean(velocities)),
        "std": float(np.std(velocities)),
        "min": int(np.min(velocities)),
        "max": int(np.max(velocities)),
    }


def analyze_results(results: Dict, concept: str) -> Dict:
    """Analyze evaluation results."""
    analysis = {
        "concept": concept,
        "alpha_values": {},
        "statistics": {},
        "comparisons": {},
    }

    alpha_values = sorted(results.keys())

    # Statistics per alpha
    for alpha in alpha_values:
        mean_values = [r["mean"] for r in results[alpha] if r["n_notes"] > 0]

        if mean_values:
            analysis["alpha_values"][str(alpha)] = {
                "n_samples": len(mean_values),
                "mean": float(np.mean(mean_values)),
                "std": float(np.std(mean_values)),
                "min": float(np.min(mean_values)),
                "max": float(np.max(mean_values)),
            }

    # Correlation analysis
    alphas = []
    mean_velocities = []

    for alpha in alpha_values:
        for result in results[alpha]:
            if result["n_notes"] > 0:
                alphas.append(alpha)
                mean_velocities.append(result["mean"])

    if len(alphas) > 2:
        correlation, p_value = stats.pearsonr(alphas, mean_velocities)

        # Handle NaN values from constant inputs
        if np.isnan(correlation) or np.isnan(p_value):
            analysis["statistics"]["correlation"] = {
                "pearson_r": None,
                "p_value": None,
                "significant": False,
                "interpretation": "Cannot compute correlation - all values are constant",
            }
        else:
            analysis["statistics"]["correlation"] = {
                "pearson_r": float(correlation),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "interpretation": (
                    f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} "
                    f"{'positive' if correlation > 0 else 'negative'} correlation "
                    f"({'significant' if p_value < 0.05 else 'not significant'})"
                ),
            }

    return analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze existing evaluation results")
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to analyze"
    )
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "evaluation",
        help="Directory with evaluation results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Find alpha directories
    results = {}

    for alpha_dir in sorted(args.results_dir.glob("alpha_*")):
        alpha_str = alpha_dir.name.replace("alpha_", "")
        try:
            alpha = float(alpha_str)
        except ValueError:
            logging.warning(f"Invalid alpha directory: {alpha_dir}")
            continue

        logging.info(f"Analyzing alpha={alpha}...")
        results[alpha] = []

        # Analyze all samples for this alpha
        for sample_file in sorted(alpha_dir.glob("*.npy")):
            sample_result = analyze_sample(sample_file)
            results[alpha].append(sample_result)

            if sample_result["n_notes"] > 0:
                logging.debug(f"  {sample_file.name}: mean={sample_result['mean']:.2f}")

        # Log summary for this alpha
        mean_values = [r["mean"] for r in results[alpha] if r["n_notes"] > 0]
        if mean_values:
            logging.info(
                f"Alpha {alpha}: Mean velocity = {np.mean(mean_values):.2f} ± {np.std(mean_values):.2f}"
            )

    # Analyze results
    logging.info("\nAnalyzing results...")
    analysis = analyze_results(results, args.concept)

    # Save analysis
    analysis_file = args.results_dir / f"{args.concept}_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logging.info(f"Saved analysis to: {analysis_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "correlation" in analysis["statistics"]:
        corr = analysis["statistics"]["correlation"]
        print(f"\nCorrelation: {corr['interpretation']}")
        if corr["pearson_r"] is not None:
            print(f"  Pearson r = {corr['pearson_r']:.4f}, p = {corr['p_value']:.4f}")

    print("\nResults by alpha:")
    for alpha_str, stats_dict in sorted(analysis["alpha_values"].items()):
        alpha = float(alpha_str)
        print(f"  α = {alpha:5.1f}: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
