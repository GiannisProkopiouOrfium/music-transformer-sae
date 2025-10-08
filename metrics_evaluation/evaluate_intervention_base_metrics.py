#!/usr/bin/env python3
"""
Base Metrics Evaluation for Feature Interventions

This script evaluates generated musical sequences from feature interventions using
the standard MusPy metrics (pitch_class_entropy, scale_consistency, groove_consistency).

It processes all intervention conditions (baseline, additions, ablation) and creates
comprehensive evaluation reports with paper benchmark comparisons and controlled
baseline documentation.

CONTROLLED GENERATION CONTEXT:
- Temperature: 0.1 (deterministic sampling for intervention comparison)
- Random noise: Added for controlled creativity
- Shared prefix: Consistent musical foundation across conditions
- Optimized for intervention effect detection vs general music generation
"""

import argparse
import logging
import pathlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import muspy
import numpy as np
import torch
import tqdm

# Add parent directory and mmt directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "mmt"))

# Import representation directly (it will find utils in the same directory)
import representation

# Import our benchmarks module
try:
    from benchmarks import (
        PAPER_BENCHMARKS,
        CONTROLLED_GENERATION_PARAMS,
        compute_confidence_interval,
        calculate_relative_performance,
    )
except ImportError as e:
    print(f"Warning: Could not import benchmarks module: {e}")
    print("Running without benchmark comparisons...")
    PAPER_BENCHMARKS = {}
    CONTROLLED_GENERATION_PARAMS = {}


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for the evaluation."""
    log_file = output_dir / "base_metrics_evaluation.log"

    logger = logging.getLogger("base_metrics_evaluation")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_tensor_sequence(tensor_file: Path) -> np.ndarray:
    """Load sequence from tensor file."""
    data = torch.load(tensor_file, map_location="cpu")

    if "generated" in data:
        sequence = data["generated"]
    elif "sequence" in data:
        sequence = data["sequence"]
    else:
        # Try to find the sequence tensor in the data
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                sequence = value
                break
        else:
            raise ValueError(f"Could not find sequence tensor in {tensor_file}")

    # Convert to numpy and remove batch dimension if present
    seq_np = sequence.numpy()
    if len(seq_np.shape) == 3 and seq_np.shape[0] == 1:
        seq_np = seq_np[0]  # Remove batch dimension

    return seq_np


def evaluate_sequence_base_metrics(
    sequence: np.ndarray, encoding: Dict
) -> Dict[str, float]:
    """Evaluate a sequence using base MusPy metrics."""
    try:
        # Convert to MusPy Music object
        music = representation.decode(sequence, encoding)

        # Trim the music to reasonable length for evaluation
        if music.resolution:
            music.trim(music.resolution * 64)

        # Check if music has tracks
        if not music.tracks:
            return {
                "pitch_class_entropy": np.nan,
                "scale_consistency": np.nan,
                "groove_consistency": np.nan,
                "error": "No tracks in generated music",
            }

        # Compute base metrics
        metrics = {}

        try:
            metrics["pitch_class_entropy"] = muspy.pitch_class_entropy(music)
        except Exception as e:
            metrics["pitch_class_entropy"] = np.nan
            metrics["pitch_class_entropy_error"] = str(e)

        try:
            # Convert to percentage to match paper benchmarks (0-1 -> 0-100)
            metrics["scale_consistency"] = muspy.scale_consistency(music) * 100.0
        except Exception as e:
            metrics["scale_consistency"] = np.nan
            metrics["scale_consistency_error"] = str(e)

        try:
            # Convert to percentage to match paper benchmarks (0-1 -> 0-100)
            metrics["groove_consistency"] = muspy.groove_consistency(
                music, 4 * music.resolution
            ) * 100.0
        except Exception as e:
            metrics["groove_consistency"] = np.nan
            metrics["groove_consistency_error"] = str(e)

        return metrics

    except Exception as e:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
            "error": f"Failed to decode sequence: {str(e)}",
        }


def find_intervention_directories(base_dir: Path) -> List[Path]:
    """Find all intervention directories in the base directory."""
    intervention_dirs = []

    # Look for layer directories
    for layer_dir in base_dir.glob("layer*"):
        if layer_dir.is_dir():
            # Look for feature directories within each layer
            for feature_dir in layer_dir.glob("feature*"):
                if feature_dir.is_dir():
                    intervention_dirs.append(feature_dir)

    return sorted(intervention_dirs)


def extract_condition_info(filename: str) -> Dict[str, Any]:
    """Extract condition information from filename."""
    # Expected patterns: baseline_*, add_+1.0_*, add_-2.0_*, ablation_*

    if filename.startswith("baseline_"):
        return {
            "condition_type": "baseline",
            "strength": 0.0,
            "condition_name": "baseline",
        }
    elif filename.startswith("add_"):
        # Extract strength from add_+1.0_ or add_-2.0_ pattern
        strength_part = filename.split("_")[1]  # Gets "+1.0" or "-2.0"
        try:
            strength = float(strength_part)
            return {
                "condition_type": "addition",
                "strength": strength,
                "condition_name": f"add_{strength:+.1f}",
            }
        except ValueError:
            return {
                "condition_type": "unknown",
                "strength": 0.0,
                "condition_name": filename,
            }
    elif filename.startswith("ablation_"):
        return {
            "condition_type": "ablation",
            "strength": 0.0,
            "condition_name": "ablation",
        }
    else:
        return {
            "condition_type": "unknown",
            "strength": 0.0,
            "condition_name": filename,
        }


def evaluate_intervention_directory(
    intervention_dir: Path, encoding: Dict, logger: logging.Logger
) -> Dict:
    """Evaluate all conditions in an intervention directory."""

    # Extract layer and feature info from directory name
    parts = intervention_dir.name.split("_", 1)
    feature_id = parts[0].replace("feature", "")
    feature_name = parts[1] if len(parts) > 1 else "unknown"
    layer = intervention_dir.parent.name.replace("layer", "")

    logger.info(f"🔍 Evaluating Layer {layer}, Feature {feature_id} ({feature_name})")

    results = {
        "layer": layer,
        "feature_id": feature_id,
        "feature_name": feature_name,
        "intervention_dir": str(intervention_dir),
        "conditions": {},
        "summary": {},
    }

    # Find all tensor files
    tensor_files = list(intervention_dir.glob("*.pt"))

    if not tensor_files:
        logger.warning(f"No tensor files found in {intervention_dir}")
        return results

    logger.info(f"Found {len(tensor_files)} tensor files")

    # Evaluate each condition
    for tensor_file in tensor_files:
        condition_info = extract_condition_info(tensor_file.stem)
        condition_name = condition_info["condition_name"]

        logger.info(f"  Evaluating condition: {condition_name}")

        try:
            # Load sequence
            sequence = load_tensor_sequence(tensor_file)

            # Evaluate metrics
            metrics = evaluate_sequence_base_metrics(sequence, encoding)

            # Store results
            results["conditions"][condition_name] = {
                "condition_info": condition_info,
                "tensor_file": str(tensor_file),
                "sequence_length": sequence.shape[0] if len(sequence.shape) >= 1 else 0,
                "metrics": metrics,
            }

            # Log metrics
            if "error" not in metrics:
                logger.info(
                    f"    ✅ pitch_class_entropy: {metrics['pitch_class_entropy']:.4f}"
                )
                logger.info(
                    f"    ✅ scale_consistency: {metrics['scale_consistency']:.4f}"
                )
                logger.info(
                    f"    ✅ groove_consistency: {metrics['groove_consistency']:.4f}"
                )
            else:
                logger.warning(f"    ⚠️  Error: {metrics['error']}")

        except Exception as e:
            logger.error(f"    ❌ Failed to evaluate {tensor_file}: {e}")
            results["conditions"][condition_name] = {
                "condition_info": condition_info,
                "tensor_file": str(tensor_file),
                "error": str(e),
            }

    # Compute summary statistics
    valid_conditions = {
        k: v
        for k, v in results["conditions"].items()
        if "metrics" in v and "error" not in v["metrics"]
    }

    if valid_conditions:
        # Compute mean and std for each metric
        for metric_name in [
            "pitch_class_entropy",
            "scale_consistency",
            "groove_consistency",
        ]:
            values = [
                v["metrics"][metric_name]
                for v in valid_conditions.values()
                if not np.isnan(v["metrics"][metric_name])
            ]

            if values:
                results["summary"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            else:
                results["summary"][metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "count": 0,
                }

    logger.info(f"✅ Completed evaluation for Layer {layer}, Feature {feature_id}")
    return results


def create_overall_summary(all_results: List[Dict]) -> Dict:
    """Create overall summary across all features and layers."""

    summary = {
        "total_features_evaluated": len(all_results),
        "features_by_layer": defaultdict(int),
        "overall_metrics": {},
        "layer_summaries": defaultdict(lambda: {"features": [], "metrics": {}}),
        "condition_summaries": defaultdict(lambda: {"count": 0, "metrics": {}}),
    }

    # Collect all valid metrics across all features and conditions
    all_metrics = defaultdict(list)
    layer_metrics = defaultdict(lambda: defaultdict(list))
    condition_metrics = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        layer = result["layer"]
        summary["features_by_layer"][layer] += 1
        summary["layer_summaries"][layer]["features"].append(result["feature_id"])

        for condition_name, condition_data in result["conditions"].items():
            if "metrics" in condition_data and "error" not in condition_data["metrics"]:
                metrics = condition_data["metrics"]

                for metric_name, metric_value in metrics.items():
                    if not isinstance(metric_value, str) and not np.isnan(metric_value):
                        all_metrics[metric_name].append(metric_value)
                        layer_metrics[layer][metric_name].append(metric_value)
                        condition_metrics[condition_name][metric_name].append(
                            metric_value
                        )

                        summary["condition_summaries"][condition_name]["count"] += 1

    # Compute overall statistics with confidence intervals
    for metric_name, values in all_metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            # Calculate confidence interval if we have the benchmarks module
            ci_lower, ci_upper, std_err = np.nan, np.nan, np.nan
            try:
                if "compute_confidence_interval" in globals():
                    ci_lower, ci_upper, std_err = compute_confidence_interval(values)
            except Exception:
                pass

            summary["overall_metrics"][metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "std_err": float(std_err) if not np.isnan(std_err) else None,
                "ci_lower": float(ci_lower) if not np.isnan(ci_lower) else None,
                "ci_upper": float(ci_upper) if not np.isnan(ci_upper) else None,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    # Compute layer summaries
    for layer, metrics_dict in layer_metrics.items():
        for metric_name, values in metrics_dict.items():
            if values:
                summary["layer_summaries"][layer]["metrics"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

    # Compute condition summaries
    for condition_name, metrics_dict in condition_metrics.items():
        for metric_name, values in metrics_dict.items():
            if values:
                summary["condition_summaries"][condition_name]["metrics"][
                    metric_name
                ] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

    return dict(summary)


def add_benchmark_comparisons(summary: Dict) -> Dict:
    """Add benchmark comparisons to the summary."""

    if not PAPER_BENCHMARKS:
        return summary

    # Add controlled generation context
    summary["controlled_generation_context"] = CONTROLLED_GENERATION_PARAMS
    summary["paper_benchmarks"] = PAPER_BENCHMARKS

    # Add benchmark comparisons for baseline condition
    if "baseline" in summary["condition_summaries"]:
        baseline_metrics = summary["condition_summaries"]["baseline"]["metrics"]
        summary["benchmark_comparisons"] = {}

        for metric_name in [
            "pitch_class_entropy",
            "scale_consistency",
            "groove_consistency",
        ]:
            if metric_name in baseline_metrics:
                baseline_mean = baseline_metrics[metric_name]["mean"]
                summary["benchmark_comparisons"][metric_name] = {}

                # Compare to each paper benchmark
                for benchmark_name in ["ground_truth", "original_mmt"]:
                    try:
                        comparison = calculate_relative_performance(
                            baseline_mean, benchmark_name, metric_name
                        )
                        summary["benchmark_comparisons"][metric_name][
                            benchmark_name
                        ] = comparison
                    except Exception:
                        continue

    return summary


def main():
    """Main evaluation function."""

    parser = argparse.ArgumentParser(
        description="Evaluate intervention results using base MusPy metrics"
    )
    parser.add_argument(
        "--interventions-dir",
        type=Path,
        default="batch_extractions_interventions/interventions",
        help="Directory containing intervention results",
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default="data/sod/processed/notes/encoding.json",
        help="Path to encoding JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        help="Specific layers to evaluate (e.g., --layers 1 3 5)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        help="Specific feature IDs to evaluate (e.g., --features 325 471)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_dir)

    logger.info("🎼 STARTING BASE METRICS EVALUATION FOR INTERVENTIONS")
    logger.info("=" * 70)
    logger.info(f"Interventions directory: {args.interventions_dir}")
    logger.info(f"Encoding path: {args.encoding_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load encoding
    logger.info("📄 Loading encoding...")
    try:
        encoding = representation.load_encoding(args.encoding_path)
        logger.info(f"✅ Loaded encoding from {args.encoding_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load encoding: {e}")
        return False

    # Find intervention directories
    logger.info("🔍 Finding intervention directories...")
    intervention_dirs = find_intervention_directories(args.interventions_dir)

    if not intervention_dirs:
        logger.error(
            f"❌ No intervention directories found in {args.interventions_dir}"
        )
        return False

    logger.info(f"Found {len(intervention_dirs)} intervention directories")

    # Filter by layers and features if specified
    if args.layers or args.features:
        filtered_dirs = []
        for dir_path in intervention_dirs:
            layer = dir_path.parent.name.replace("layer", "")
            feature_id = dir_path.name.split("_")[0].replace("feature", "")

            layer_match = not args.layers or layer in args.layers
            feature_match = not args.features or feature_id in args.features

            if layer_match and feature_match:
                filtered_dirs.append(dir_path)

        intervention_dirs = filtered_dirs
        logger.info(
            f"Filtered to {len(intervention_dirs)} directories based on layer/feature criteria"
        )

    # Evaluate each intervention directory
    logger.info("\n🎯 STARTING EVALUATION")
    all_results = []

    for i, intervention_dir in enumerate(
        tqdm.tqdm(intervention_dirs, desc="Evaluating interventions")
    ):
        logger.info(f"\n📍 [{i+1}/{len(intervention_dirs)}] {intervention_dir.name}")

        try:
            result = evaluate_intervention_directory(intervention_dir, encoding, logger)
            all_results.append(result)

            # Save individual result
            result_file = (
                args.output_dir
                / f"{intervention_dir.parent.name}_{intervention_dir.name}_base_metrics.json"
            )
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {intervention_dir}: {e}")

    # Create overall summary
    logger.info("\n📊 CREATING OVERALL SUMMARY")
    overall_summary = create_overall_summary(all_results)

    # Add benchmark comparisons
    logger.info("📊 ADDING BENCHMARK COMPARISONS")
    overall_summary = add_benchmark_comparisons(overall_summary)

    # Save comprehensive results
    comprehensive_results = {
        "evaluation_info": {
            "interventions_dir": str(args.interventions_dir),
            "encoding_path": str(args.encoding_path),
            "total_interventions_evaluated": len(all_results),
            "metrics_evaluated": [
                "pitch_class_entropy",
                "scale_consistency",
                "groove_consistency",
            ],
        },
        "overall_summary": overall_summary,
        "detailed_results": all_results,
    }

    results_file = args.output_dir / "base_metrics_evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 BASE METRICS EVALUATION COMPLETE!")
    logger.info(f"✅ Evaluated {len(all_results)} intervention sets")
    logger.info(f"📊 Results saved to: {results_file}")

    logger.info("\n📈 OVERALL METRICS SUMMARY:")
    for metric_name, stats in overall_summary["overall_metrics"].items():
        logger.info(
            f"  {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f} (n={stats['count']})"
        )

    logger.info("\n📋 BY LAYER:")
    for layer, layer_data in overall_summary["layer_summaries"].items():
        logger.info(f"  Layer {layer}: {len(layer_data['features'])} features")
        for metric_name, stats in layer_data["metrics"].items():
            logger.info(
                f"    {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
            )

    logger.info("\n📋 BY CONDITION:")
    for condition_name, condition_data in overall_summary[
        "condition_summaries"
    ].items():
        logger.info(f"  {condition_name}: {condition_data['count']} samples")
        for metric_name, stats in condition_data["metrics"].items():
            logger.info(
                f"    {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
            )

    # Log benchmark comparisons if available
    if "benchmark_comparisons" in overall_summary:
        logger.info("\n🎯 BENCHMARK COMPARISONS (Controlled Baseline vs Paper):")
        for metric_name, benchmarks in overall_summary["benchmark_comparisons"].items():
            logger.info(f"  {metric_name}:")
            for benchmark_name, comparison in benchmarks.items():
                if "relative_change_percent" in comparison:
                    logger.info(
                        f"    vs {benchmark_name}: {comparison['relative_change_percent']:+.1f}% ({comparison['performance_vs_benchmark']})"
                    )

    logger.info(f"\n📁 Detailed logs: {args.output_dir}/base_metrics_evaluation.log")
    logger.info("🎵 Ready for extended metrics evaluation and comparative analysis!")
    logger.info("\n📖 CONTROLLED GENERATION CONTEXT:")
    logger.info(
        "  Temperature: 0.1 (deterministic sampling for intervention comparison)"
    )
    logger.info("  Random noise: Added for controlled creativity")
    logger.info("  Shared prefix: Consistent musical foundation across conditions")
    logger.info(
        "  Purpose: Optimized for intervention effect detection vs general music generation"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
