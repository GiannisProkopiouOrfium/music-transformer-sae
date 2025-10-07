#!/usr/bin/env python3
"""
Comparative Analysis for Feature Interventions

This script analyzes the differences between baseline and intervention conditions
to assess model performance preservation and intervention effectiveness.

It provides:
- Baseline vs intervention comparisons
- Model performance degradation analysis
- Intervention strength effect analysis
- Statistical significance testing
- Visualization-ready summaries
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for the comparative analysis."""
    log_file = output_dir / "comparative_analysis.log"

    logger = logging.getLogger("comparative_analysis")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

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


def extract_metric_values(condition_data: Dict, metric_name: str) -> List[float]:
    """Extract metric values from condition data, handling nested structures."""
    values = []

    # Check in base metrics
    if "metrics" in condition_data and metric_name in condition_data["metrics"]:
        value = condition_data["metrics"][metric_name]
        if not np.isnan(value):
            values.append(float(value))

    # Check in extended metrics
    if "extended_metrics" in condition_data:
        extended = condition_data["extended_metrics"]

        # Handle nested metric categories
        for category_name, category_metrics in extended.items():
            if isinstance(category_metrics, dict):
                full_metric_name = f"{category_name}.{metric_name}"
                if metric_name in category_metrics:
                    value = category_metrics[metric_name]
                    if not isinstance(value, str) and not np.isnan(value):
                        values.append(float(value))

                # Also check for exact match
                if full_metric_name in extended:
                    value = extended[full_metric_name]
                    if not isinstance(value, str) and not np.isnan(value):
                        values.append(float(value))

    return values


def compute_intervention_effect(
    baseline_values: List[float], intervention_values: List[float]
) -> Dict[str, float]:
    """Compute intervention effect statistics."""
    if not baseline_values or not intervention_values:
        return {
            "effect_size": 0.0,
            "percent_change": 0.0,
            "p_value": 1.0,
            "significant": False,
            "baseline_mean": np.nan,
            "intervention_mean": np.nan,
        }

    baseline_array = np.array(baseline_values)
    intervention_array = np.array(intervention_values)

    baseline_mean = np.mean(baseline_array)
    intervention_mean = np.mean(intervention_array)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (
            (len(baseline_array) - 1) * np.var(baseline_array, ddof=1)
            + (len(intervention_array) - 1) * np.var(intervention_array, ddof=1)
        )
        / (len(baseline_array) + len(intervention_array) - 2)
    )

    if pooled_std > 0:
        effect_size = abs(intervention_mean - baseline_mean) / pooled_std
    else:
        effect_size = 0.0

    # Percent change
    if baseline_mean != 0:
        percent_change = ((intervention_mean - baseline_mean) / baseline_mean) * 100
    else:
        percent_change = 0.0

    # Statistical significance (t-test)
    try:
        if len(baseline_values) > 1 and len(intervention_values) > 1:
            t_stat, p_value = stats.ttest_ind(baseline_values, intervention_values)
            significant = p_value < 0.05
        else:
            p_value = 1.0
            significant = False
    except Exception:
        p_value = 1.0
        significant = False

    return {
        "effect_size": float(effect_size),
        "percent_change": float(percent_change),
        "p_value": float(p_value),
        "significant": bool(significant),
        "baseline_mean": float(baseline_mean),
        "intervention_mean": float(intervention_mean),
        "baseline_std": float(np.std(baseline_array)),
        "intervention_std": float(np.std(intervention_array)),
    }


def analyze_feature_interventions(results: Dict) -> Dict:
    """Analyze intervention effects for each feature."""

    feature_analyses = {}

    # Define metrics to analyze
    base_metrics = ["pitch_class_entropy", "scale_consistency", "groove_consistency"]
    extended_metrics = [
        "note_density",
        "notes_per_beat",
        "total_notes",
        "pitch_range",
        "mean_pitch",
        "pitch_std",
        "rhythmic_regularity",
        "tempo_stability",
        "harmonic_consistency",
        "chord_progression_quality",
        "structural_coherence",
        "melodic_coherence",
    ]

    all_metrics = base_metrics + extended_metrics

    for result in results["detailed_results"]:
        layer = result["layer"]
        feature_id = result["feature_id"]
        feature_name = result["feature_name"]

        feature_key = f"layer{layer}_feature{feature_id}_{feature_name}"

        # Get baseline values for each metric
        baseline_metrics = {}
        if "baseline" in result["conditions"]:
            baseline_condition = result["conditions"]["baseline"]
            for metric_name in all_metrics:
                baseline_metrics[metric_name] = extract_metric_values(
                    baseline_condition, metric_name
                )

        # Analyze each intervention condition
        condition_analyses = {}

        for condition_name, condition_data in result["conditions"].items():
            if condition_name == "baseline":
                continue

            condition_analysis = {
                "condition_info": condition_data.get("condition_info", {}),
                "metric_comparisons": {},
            }

            # Compare each metric to baseline
            for metric_name in all_metrics:
                intervention_values = extract_metric_values(condition_data, metric_name)
                baseline_values = baseline_metrics.get(metric_name, [])

                if baseline_values and intervention_values:
                    effect_stats = compute_intervention_effect(
                        baseline_values, intervention_values
                    )
                    condition_analysis["metric_comparisons"][metric_name] = effect_stats

            condition_analyses[condition_name] = condition_analysis

        feature_analyses[feature_key] = {
            "layer": layer,
            "feature_id": feature_id,
            "feature_name": feature_name,
            "baseline_metrics": baseline_metrics,
            "condition_analyses": condition_analyses,
        }

    return feature_analyses


def analyze_strength_effects(feature_analyses: Dict) -> Dict:
    """Analyze effects across different intervention strengths."""

    strength_effects = defaultdict(lambda: defaultdict(list))

    for feature_key, feature_data in feature_analyses.items():
        for condition_name, condition_analysis in feature_data[
            "condition_analyses"
        ].items():
            condition_info = condition_analysis["condition_info"]

            if condition_info.get("condition_type") == "addition":
                strength = condition_info.get("strength", 0.0)

                for metric_name, effect_stats in condition_analysis[
                    "metric_comparisons"
                ].items():
                    strength_effects[metric_name][strength].append(
                        effect_stats["percent_change"]
                    )

    # Compute strength effect summaries
    strength_summaries = {}
    for metric_name, strength_data in strength_effects.items():
        metric_summary = {}
        for strength, changes in strength_data.items():
            if changes:
                metric_summary[strength] = {
                    "mean_change": float(np.mean(changes)),
                    "std_change": float(np.std(changes)),
                    "count": len(changes),
                    "changes": changes,
                }
        strength_summaries[metric_name] = metric_summary

    return strength_summaries


def analyze_layer_effects(feature_analyses: Dict) -> Dict:
    """Analyze intervention effects by layer."""

    layer_effects = defaultdict(lambda: defaultdict(list))

    for feature_key, feature_data in feature_analyses.items():
        layer = feature_data["layer"]

        for condition_name, condition_analysis in feature_data[
            "condition_analyses"
        ].items():
            for metric_name, effect_stats in condition_analysis[
                "metric_comparisons"
            ].items():
                layer_effects[layer][metric_name].append(effect_stats["percent_change"])

    # Compute layer effect summaries
    layer_summaries = {}
    for layer, metrics_data in layer_effects.items():
        layer_summary = {}
        for metric_name, changes in metrics_data.items():
            if changes:
                layer_summary[metric_name] = {
                    "mean_change": float(np.mean(changes)),
                    "std_change": float(np.std(changes)),
                    "count": len(changes),
                    "effect_magnitude": float(np.mean(np.abs(changes))),
                }
        layer_summaries[layer] = layer_summary

    return layer_summaries


def assess_model_performance_preservation(feature_analyses: Dict) -> Dict:
    """Assess how well the model performance is preserved during interventions."""

    performance_metrics = [
        "pitch_class_entropy",
        "scale_consistency",
        "groove_consistency",
    ]

    preservation_analysis = {
        "overall_preservation": {},
        "by_condition_type": defaultdict(lambda: defaultdict(list)),
        "by_layer": defaultdict(lambda: defaultdict(list)),
        "degradation_summary": {},
    }

    all_degradations = defaultdict(list)

    for feature_key, feature_data in feature_analyses.items():
        layer = feature_data["layer"]

        for condition_name, condition_analysis in feature_data[
            "condition_analyses"
        ].items():
            condition_type = condition_analysis["condition_info"].get(
                "condition_type", "unknown"
            )

            for metric_name in performance_metrics:
                if metric_name in condition_analysis["metric_comparisons"]:
                    effect_stats = condition_analysis["metric_comparisons"][metric_name]

                    # Performance degradation (larger changes = worse preservation)
                    degradation = abs(effect_stats["percent_change"])

                    all_degradations[metric_name].append(degradation)
                    preservation_analysis["by_condition_type"][condition_type][
                        metric_name
                    ].append(degradation)
                    preservation_analysis["by_layer"][layer][metric_name].append(
                        degradation
                    )

    # Overall preservation statistics
    for metric_name, degradations in all_degradations.items():
        if degradations:
            preservation_analysis["overall_preservation"][metric_name] = {
                "mean_degradation": float(np.mean(degradations)),
                "std_degradation": float(np.std(degradations)),
                "max_degradation": float(np.max(degradations)),
                "preservation_score": float(
                    100.0 / (1.0 + np.mean(degradations))
                ),  # Higher = better preservation
                "count": len(degradations),
            }

    # Degradation summary by condition type
    for condition_type, metrics_data in preservation_analysis[
        "by_condition_type"
    ].items():
        condition_summary = {}
        for metric_name, degradations in metrics_data.items():
            if degradations:
                condition_summary[metric_name] = {
                    "mean_degradation": float(np.mean(degradations)),
                    "preservation_score": float(100.0 / (1.0 + np.mean(degradations))),
                    "count": len(degradations),
                }
        preservation_analysis["degradation_summary"][condition_type] = condition_summary

    return preservation_analysis


def generate_summary_recommendations(comparative_analysis: Dict) -> List[str]:
    """Generate recommendations based on the comparative analysis."""

    recommendations = []

    # Model performance preservation
    performance_analysis = comparative_analysis.get(
        "model_performance_preservation", {}
    )
    overall_preservation = performance_analysis.get("overall_preservation", {})

    avg_preservation = (
        np.mean(
            [
                metrics["preservation_score"]
                for metrics in overall_preservation.values()
                if "preservation_score" in metrics
            ]
        )
        if overall_preservation
        else 50.0
    )

    if avg_preservation > 80:
        recommendations.append(
            "✅ Excellent model performance preservation across interventions"
        )
    elif avg_preservation > 60:
        recommendations.append(
            "⚠️ Good model performance preservation with some degradation"
        )
    else:
        recommendations.append("❌ Significant model performance degradation detected")

    # Strength effects
    strength_analysis = comparative_analysis.get("strength_effects", {})

    if strength_analysis:
        # Find metrics with strong strength correlations
        strong_effects = []
        for metric_name, strength_data in strength_analysis.items():
            strengths = sorted([float(s) for s in strength_data.keys() if s != 0])
            if len(strengths) >= 2:
                changes = [
                    strength_data[s]["mean_change"]
                    for s in strengths
                    if s in strength_data
                ]
                if len(changes) >= 2:
                    # Check for monotonic relationship
                    correlation = (
                        np.corrcoef(strengths, changes)[0, 1]
                        if len(strengths) == len(changes)
                        else 0
                    )
                    if abs(correlation) > 0.7:
                        strong_effects.append(f"{metric_name} (r={correlation:.2f})")

        if strong_effects:
            recommendations.append(
                f"📈 Strong strength-effect relationships found: {', '.join(strong_effects)}"
            )
        else:
            recommendations.append(
                "📊 Intervention strength effects are not strongly dose-dependent"
            )

    # Layer differences
    layer_analysis = comparative_analysis.get("layer_effects", {})

    if layer_analysis:
        layer_effect_magnitudes = {}
        for layer, metrics_data in layer_analysis.items():
            avg_magnitude = (
                np.mean(
                    [
                        metrics["effect_magnitude"]
                        for metrics in metrics_data.values()
                        if "effect_magnitude" in metrics
                    ]
                )
                if metrics_data
                else 0.0
            )
            layer_effect_magnitudes[layer] = avg_magnitude

        if layer_effect_magnitudes:
            max_layer = max(layer_effect_magnitudes.items(), key=lambda x: x[1])
            min_layer = min(layer_effect_magnitudes.items(), key=lambda x: x[1])

            recommendations.append(
                f"🎯 Layer {max_layer[0]} shows strongest intervention effects ({max_layer[1]:.1f}% avg change)"
            )
            recommendations.append(
                f"🎯 Layer {min_layer[0]} shows weakest intervention effects ({min_layer[1]:.1f}% avg change)"
            )

    # Feature-specific insights
    feature_analysis = comparative_analysis.get("feature_analyses", {})

    if feature_analysis:
        # Find features with most significant effects
        significant_features = []
        for feature_key, feature_data in feature_analysis.items():
            significant_count = 0
            total_comparisons = 0

            for condition_analysis in feature_data["condition_analyses"].values():
                for effect_stats in condition_analysis["metric_comparisons"].values():
                    total_comparisons += 1
                    if effect_stats.get("significant", False):
                        significant_count += 1

            if total_comparisons > 0:
                significance_rate = significant_count / total_comparisons
                if significance_rate > 0.5:
                    feature_name = feature_data["feature_name"]
                    significant_features.append(
                        f"{feature_name} ({significance_rate:.1%} significant)"
                    )

        if significant_features:
            recommendations.append(
                f"🔍 Most effective interventions: {', '.join(significant_features)}"
            )
        else:
            recommendations.append(
                "🔍 Few statistically significant intervention effects detected"
            )

    return recommendations


def main():
    """Main comparative analysis function."""

    parser = argparse.ArgumentParser(
        description="Perform comparative analysis of intervention effects"
    )
    parser.add_argument(
        "--extended-results",
        type=Path,
        default="../evaluation_results/extended_metrics_evaluation_results.json",
        help="Path to extended metrics evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="../evaluation_results",
        help="Output directory for comparative analysis results",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_dir)

    logger.info("🎼 STARTING COMPARATIVE ANALYSIS OF INTERVENTION EFFECTS")
    logger.info("=" * 70)
    logger.info(f"Extended results file: {args.extended_results}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load extended results
    logger.info("📄 Loading extended evaluation results...")
    try:
        with open(args.extended_results, "r") as f:
            extended_results = json.load(f)
        logger.info(f"✅ Loaded extended results from {args.extended_results}")
    except Exception as e:
        logger.error(f"❌ Failed to load extended results: {e}")
        return False

    # Perform comparative analyses
    logger.info("🎯 ANALYZING FEATURE INTERVENTIONS")
    feature_analyses = analyze_feature_interventions(extended_results)
    logger.info(f"✅ Analyzed {len(feature_analyses)} feature intervention sets")

    logger.info("📊 ANALYZING STRENGTH EFFECTS")
    strength_effects = analyze_strength_effects(feature_analyses)
    logger.info(f"✅ Analyzed strength effects for {len(strength_effects)} metrics")

    logger.info("🔍 ANALYZING LAYER EFFECTS")
    layer_effects = analyze_layer_effects(feature_analyses)
    logger.info(f"✅ Analyzed layer effects for {len(layer_effects)} layers")

    logger.info("⚖️ ASSESSING MODEL PERFORMANCE PRESERVATION")
    model_performance = assess_model_performance_preservation(feature_analyses)
    logger.info("✅ Completed model performance preservation analysis")

    # Generate recommendations
    logger.info("💡 GENERATING RECOMMENDATIONS")

    comparative_analysis = {
        "analysis_info": {
            "source_file": str(args.extended_results),
            "total_features_analyzed": len(feature_analyses),
            "analysis_types": [
                "feature_interventions",
                "strength_effects",
                "layer_effects",
                "model_performance_preservation",
            ],
        },
        "feature_analyses": feature_analyses,
        "strength_effects": strength_effects,
        "layer_effects": layer_effects,
        "model_performance_preservation": model_performance,
    }

    recommendations = generate_summary_recommendations(comparative_analysis)
    comparative_analysis["recommendations"] = recommendations

    # Save comparative analysis results
    comparative_results_file = args.output_dir / "comparative_analysis_results.json"
    with open(comparative_results_file, "w") as f:
        json.dump(comparative_analysis, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 COMPARATIVE ANALYSIS COMPLETE!")
    logger.info(f"📊 Results saved to: {comparative_results_file}")

    logger.info("\n💡 KEY RECOMMENDATIONS:")
    for rec in recommendations:
        logger.info(f"  {rec}")

    logger.info("\n📈 MODEL PERFORMANCE PRESERVATION:")
    for metric_name, stats in model_performance["overall_preservation"].items():
        logger.info(
            f"  {metric_name}: {stats['preservation_score']:.1f}% preservation score"
        )

    logger.info("\n🎯 LAYER EFFECT SUMMARY:")
    for layer, metrics_data in layer_effects.items():
        avg_effect = np.mean([m["effect_magnitude"] for m in metrics_data.values()])
        logger.info(f"  Layer {layer}: {avg_effect:.1f}% average effect magnitude")

    logger.info(f"\n📁 Detailed logs: {args.output_dir}/comparative_analysis.log")
    logger.info("🎵 Comparative analysis ready for interpretation!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
