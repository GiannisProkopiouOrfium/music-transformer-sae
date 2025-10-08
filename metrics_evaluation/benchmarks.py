#!/usr/bin/env python3
"""
Reference Benchmarks for Music Generation Evaluation

This module contains benchmark results from the original MMT paper and related work
to enable proper comparison and assessment of intervention effects vs model degradation.

Original paper reference:
"Multi-Track Music Transformer: Learning Long-term Dependencies Across Musical Tracks"

Table 3. Objective evaluation results. Mean values and 95% confidence intervals are reported.
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats

# Original paper benchmarks (Table 3)
PAPER_BENCHMARKS = {
    "ground_truth": {
        "pitch_class_entropy": {
            "mean": 2.974,
            "ci_lower": 2.974 - 0.018,  # 95% CI
            "ci_upper": 2.974 + 0.018,
            "std_err": 0.018,
            "description": "Ground truth reference from original paper",
        },
        "scale_consistency": {
            "mean": 92.26,
            "ci_lower": 92.26 - 1.25,
            "ci_upper": 92.26 + 1.25,
            "std_err": 1.25,
            "description": "Ground truth scale consistency (%)",
        },
        "groove_consistency": {
            "mean": 93.05,
            "ci_lower": 93.05 - 1.00,
            "ci_upper": 93.05 + 1.00,
            "std_err": 1.00,
            "description": "Ground truth groove consistency (%)",
        },
    },
    "mmm": {
        "pitch_class_entropy": {
            "mean": 2.884,
            "ci_lower": 2.884 - 0.023,
            "ci_upper": 2.884 + 0.023,
            "std_err": 0.023,
            "description": "MMM baseline from original paper",
        },
        "scale_consistency": {
            "mean": 93.13,
            "ci_lower": 93.13 - 0.49,
            "ci_upper": 93.13 + 0.49,
            "std_err": 0.49,
            "description": "MMM scale consistency (%)",
        },
        "groove_consistency": {
            "mean": 91.90,
            "ci_lower": 91.90 - 0.64,
            "ci_upper": 91.90 + 0.64,
            "std_err": 0.64,
            "description": "MMM groove consistency (%)",
        },
    },
    "remi_plus": {
        "pitch_class_entropy": {
            "mean": 2.897,
            "ci_lower": 2.897 - 0.019,
            "ci_upper": 2.897 + 0.019,
            "std_err": 0.019,
            "description": "REMI+ baseline from original paper",
        },
        "scale_consistency": {
            "mean": 93.12,
            "ci_lower": 93.12 - 0.51,
            "ci_upper": 93.12 + 0.51,
            "std_err": 0.51,
            "description": "REMI+ scale consistency (%)",
        },
        "groove_consistency": {
            "mean": 92.90,
            "ci_lower": 92.90 - 0.49,
            "ci_upper": 92.90 + 0.49,
            "std_err": 0.49,
            "description": "REMI+ groove consistency (%)",
        },
    },
    "original_mmt": {
        "pitch_class_entropy": {
            "mean": 2.802,
            "ci_lower": 2.802 - 0.025,
            "ci_upper": 2.802 + 0.025,
            "std_err": 0.025,
            "description": "Original MMT from paper (our baseline reference)",
        },
        "scale_consistency": {
            "mean": 94.74,
            "ci_lower": 94.74 - 0.42,
            "ci_upper": 94.74 + 0.42,
            "std_err": 0.42,
            "description": "Original MMT scale consistency (%)",
        },
        "groove_consistency": {
            "mean": 92.09,
            "ci_lower": 92.09 - 0.49,
            "ci_upper": 92.09 + 0.49,
            "std_err": 0.49,
            "description": "Original MMT groove consistency (%)",
        },
    },
}

# Controlled generation parameters documentation
CONTROLLED_GENERATION_PARAMS = {
    "temperature": 0.1,
    "random_noise": True,
    "shared_prefix": True,
    "generation_strategy": "controlled_creativity",
    "rationale": {
        "temperature_0.1": "Ensures consistent, deterministic sampling for intervention comparison",
        "random_noise": "Adds controlled creativity while maintaining reproducibility",
        "shared_prefix": "Provides consistent musical foundation across all conditions",
        "benefits": [
            "Reduced baseline variance for clearer intervention signal detection",
            "Controlled experimental conditions for precise before/after comparison",
            "Minimized generation artifacts that could mask intervention effects",
            "Optimal setup for intervention research vs general music generation",
        ],
    },
    "comparison_notes": {
        "vs_original_paper": "Original paper used standard temperature=1.0 for general music quality evaluation",
        "our_approach": "Optimized for intervention effect detection with controlled baseline",
        "expected_differences": {
            "pitch_entropy": "Lower due to temperature 0.1 (more deterministic sampling)",
            "scale_consistency": "Higher due to shared prefix (consistent harmonic foundation)",
            "groove_consistency": "Higher due to shared prefix (consistent rhythmic patterns)",
        },
    },
}


def compute_confidence_interval(
    values: list, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a list of values.

    Returns:
        tuple: (lower_bound, upper_bound, standard_error)
    """
    if len(values) < 2:
        return np.nan, np.nan, np.nan

    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_err = stats.sem(values_array)  # Standard error of the mean

    # Calculate confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, len(values) - 1)
    margin_error = t_critical * std_err

    return mean_val - margin_error, mean_val + margin_error, std_err


def format_metric_with_ci(
    mean: float, ci_lower: float, ci_upper: float, is_percentage: bool = False
) -> str:
    """Format metric with confidence interval in paper style."""
    if is_percentage:
        return f"{mean:.2f}% ± {(ci_upper - ci_lower)/2:.2f}"
    else:
        return f"{mean:.3f} ± {(ci_upper - ci_lower)/2:.3f}"


def calculate_relative_performance(
    observed_mean: float, benchmark_name: str, metric_name: str
) -> Dict[str, float]:
    """Calculate relative performance vs paper benchmarks."""

    if (
        benchmark_name not in PAPER_BENCHMARKS
        or metric_name not in PAPER_BENCHMARKS[benchmark_name]
    ):
        return {"error": f"Benchmark {benchmark_name}.{metric_name} not found"}

    benchmark = PAPER_BENCHMARKS[benchmark_name][metric_name]
    benchmark_mean = benchmark["mean"]

    # Convert percentage metrics to 0-100 scale if needed
    if (
        metric_name in ["scale_consistency", "groove_consistency"]
        and benchmark_mean < 1.0
    ):
        benchmark_mean *= 100

    # Calculate relative performance
    if benchmark_mean != 0:
        relative_change = (observed_mean - benchmark_mean) / benchmark_mean * 100
    else:
        relative_change = 0.0

    # Performance assessment - all metrics: higher is generally better
    performance_quality = "better" if relative_change > 0 else "degraded"

    return {
        "observed_mean": observed_mean,
        "benchmark_mean": benchmark_mean,
        "absolute_difference": observed_mean - benchmark_mean,
        "relative_change_percent": relative_change,
        "performance_vs_benchmark": performance_quality,
        "benchmark_description": benchmark["description"],
    }


def calculate_cohens_d(group1_values: list, group2_values: list) -> float:
    """Calculate Cohen's d effect size between two groups."""
    if len(group1_values) < 2 or len(group2_values) < 2:
        return np.nan

    group1 = np.array(group1_values)
    group2 = np.array(group2_values)

    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return 0.0

    cohens_d = (mean1 - mean2) / pooled_std
    return float(cohens_d)


def interpret_cohens_d(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)

    if np.isnan(abs_d):
        return "Cannot compute"
    elif abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


def assess_statistical_significance_vs_benchmark(
    observed_values: list, benchmark_name: str, metric_name: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Assess statistical significance of observed values vs paper benchmark.

    Uses one-sample t-test against the benchmark mean.
    """

    if (
        benchmark_name not in PAPER_BENCHMARKS
        or metric_name not in PAPER_BENCHMARKS[benchmark_name]
    ):
        return {"error": f"Benchmark {benchmark_name}.{metric_name} not found"}

    if len(observed_values) < 2:
        return {"error": "Need at least 2 observations for statistical test"}

    benchmark = PAPER_BENCHMARKS[benchmark_name][metric_name]
    benchmark_mean = benchmark["mean"]

    # Convert percentage metrics if needed
    if (
        metric_name in ["scale_consistency", "groove_consistency"]
        and benchmark_mean < 1.0
    ):
        benchmark_mean *= 100

    # One-sample t-test
    try:
        t_stat, p_value = stats.ttest_1samp(observed_values, benchmark_mean)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "significance_level": alpha,
            "observed_mean": float(np.mean(observed_values)),
            "benchmark_mean": benchmark_mean,
            "interpretation": f"{'Significantly different' if p_value < alpha else 'Not significantly different'} from {benchmark_name} benchmark",
        }

    except Exception as e:
        return {"error": f"Statistical test failed: {str(e)}"}


def _extract_baseline_values(
    results_dict: Dict, controlled_baseline_name: str, key_metrics: list
) -> Dict:
    """Extract baseline values from results dictionary."""
    baseline_values = {}
    if controlled_baseline_name in results_dict:
        baseline_condition = results_dict[controlled_baseline_name]
        for metric in key_metrics:
            if (
                "metrics" in baseline_condition
                and metric in baseline_condition["metrics"]
            ):
                baseline_values[metric] = baseline_condition["metrics"][metric]
    return baseline_values


def _compare_baseline_to_benchmarks(baseline_values: Dict, key_metrics: list) -> Dict:
    """Compare controlled baseline to paper benchmarks."""
    comparison_results = {}

    for metric_name in key_metrics:
        if metric_name in baseline_values:
            baseline_val = baseline_values[metric_name]
            comparison_results[metric_name] = {}

            # Compare to each paper benchmark
            for benchmark_name in ["ground_truth", "original_mmt", "mmm", "remi_plus"]:
                comparison = calculate_relative_performance(
                    baseline_val, benchmark_name, metric_name
                )
                comparison_results[metric_name][benchmark_name] = comparison

    return comparison_results


def _calculate_single_metric_effect(
    baseline_val: float, intervention_val: float
) -> Dict:
    """Calculate intervention effect for a single metric."""
    percent_change = (
        (intervention_val - baseline_val) / baseline_val * 100
        if baseline_val != 0
        else 0.0
    )
    cohens_d = calculate_cohens_d([baseline_val], [intervention_val])

    return {
        "intervention_effects": {
            "baseline_value": baseline_val,
            "intervention_value": intervention_val,
            "absolute_change": intervention_val - baseline_val,
            "percent_change": percent_change,
            "effect_interpretation": (
                "improvement" if percent_change > 0 else "degradation"
            ),
        },
        "effect_sizes": {
            "cohens_d": cohens_d,
            "effect_size_interpretation": interpret_cohens_d(cohens_d),
        },
    }


def _analyze_intervention_effects(
    results_dict: Dict,
    controlled_baseline_name: str,
    baseline_values: Dict,
    key_metrics: list,
) -> Dict:
    """Analyze intervention effects relative to controlled baseline."""
    intervention_analysis = {}

    for condition_name, condition_data in results_dict.items():
        if (
            condition_name == controlled_baseline_name
            or "metrics" not in condition_data
        ):
            continue

        condition_analysis = {
            "condition_info": condition_data.get("condition_info", {}),
            "intervention_effects": {},
            "effect_sizes": {},
        }

        for metric_name in key_metrics:
            if (
                metric_name in baseline_values
                and metric_name in condition_data["metrics"]
                and not np.isnan(condition_data["metrics"][metric_name])
            ):

                baseline_val = baseline_values[metric_name]
                intervention_val = condition_data["metrics"][metric_name]

                # Calculate effects for this metric
                effects = _calculate_single_metric_effect(
                    baseline_val, intervention_val
                )
                condition_analysis["intervention_effects"][metric_name] = effects[
                    "intervention_effects"
                ]
                condition_analysis["effect_sizes"][metric_name] = effects[
                    "effect_sizes"
                ]

        intervention_analysis[condition_name] = condition_analysis

    return intervention_analysis


def create_benchmark_comparison_summary(
    results_dict: Dict, controlled_baseline_name: str = "baseline"
) -> Dict[str, Any]:
    """
    Create comprehensive benchmark comparison summary.

    Args:
        results_dict: Dictionary with condition results including metrics
        controlled_baseline_name: Name of the controlled baseline condition
    """

    key_metrics = ["pitch_class_entropy", "scale_consistency", "groove_consistency"]

    # Extract baseline values
    baseline_values = _extract_baseline_values(
        results_dict, controlled_baseline_name, key_metrics
    )

    # Compare baseline to benchmarks
    baseline_comparisons = _compare_baseline_to_benchmarks(baseline_values, key_metrics)

    # Analyze intervention effects
    intervention_effects = _analyze_intervention_effects(
        results_dict, controlled_baseline_name, baseline_values, key_metrics
    )

    return {
        "controlled_generation_context": CONTROLLED_GENERATION_PARAMS,
        "paper_benchmarks": PAPER_BENCHMARKS,
        "benchmark_comparisons": {},
        "controlled_baseline_performance": baseline_comparisons,
        "intervention_effects_analysis": intervention_effects,
    }


def get_benchmark_reference(benchmark_name: str, metric_name: str) -> Dict[str, Any]:
    """Get specific benchmark reference."""
    if (
        benchmark_name in PAPER_BENCHMARKS
        and metric_name in PAPER_BENCHMARKS[benchmark_name]
    ):
        return PAPER_BENCHMARKS[benchmark_name][metric_name].copy()
    else:
        return {"error": f"Benchmark {benchmark_name}.{metric_name} not found"}


def format_benchmark_table() -> str:
    """Format benchmark table in paper style for documentation."""
    table = "\nReference Benchmarks (from original MMT paper Table 3)\n"
    table += "=" * 70 + "\n"
    table += f"{'Model':<15} {'Pitch Entropy':<15} {'Scale Cons. (%)':<15} {'Groove Cons. (%)':<15}\n"
    table += "-" * 70 + "\n"

    for model_name, model_data in PAPER_BENCHMARKS.items():
        model_display = model_name.replace("_", " ").title()
        pce = format_metric_with_ci(**model_data["pitch_class_entropy"])
        sc = format_metric_with_ci(
            **model_data["scale_consistency"], is_percentage=True
        )
        gc = format_metric_with_ci(
            **model_data["groove_consistency"], is_percentage=True
        )

        table += f"{model_display:<15} {pce:<15} {sc:<15} {gc:<15}\n"

    table += "=" * 70 + "\n"
    return table


# Export key functions and constants
__all__ = [
    "PAPER_BENCHMARKS",
    "CONTROLLED_GENERATION_PARAMS",
    "compute_confidence_interval",
    "format_metric_with_ci",
    "calculate_relative_performance",
    "calculate_cohens_d",
    "interpret_cohens_d",
    "assess_statistical_significance_vs_benchmark",
    "create_benchmark_comparison_summary",
    "get_benchmark_reference",
    "format_benchmark_table",
]
