#!/usr/bin/env python3
"""Step 3: Analyze Music Flamingo Evaluation Results.

This script:
1. Loads results.json from step 2
2. Calculates metrics for each category:
   - Category I: Directional success rate + Spearman correlation
   - Category II: Dual directional success + Spearman r (both concepts)
   - Category III: Override Success Rate (OSR)
   - Category IV: Dual-OSR
3. Performs statistical tests (Wilcoxon, Cohen's d, FDR correction)
4. Generates LaTeX tables and figures
5. Saves comprehensive analysis report

Usage:
    python steering_interventions/flamingo_eval/3_analyze_results.py \\
        --config steering_interventions/flamingo_eval/config.yaml
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats import spearmanr, wilcoxon


def load_config(config_path: pathlib.Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.
    
    Args:
        group1: First group values
        group2: Second group values
    
    Returns:
        Cohen's d statistic
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
    
    Returns:
        (reject_list, corrected_p_values)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Calculate critical values
    critical_values = (np.arange(1, n+1) / n) * alpha
    
    # Find largest i where p_i <= critical_value_i
    reject = sorted_p <= critical_values
    
    # Unsort
    reject_unsorted = np.zeros(n, dtype=bool)
    reject_unsorted[sorted_idx] = reject
    
    # Corrected p-values
    corrected_p = sorted_p * n / np.arange(1, n+1)
    corrected_p = np.minimum.accumulate(corrected_p[::-1])[::-1]
    corrected_p_unsorted = np.zeros(n)
    corrected_p_unsorted[sorted_idx] = corrected_p
    
    return reject_unsorted.tolist(), corrected_p_unsorted.tolist()


def analyze_category_1(results: List[dict], config: dict) -> dict:
    """Analyze Category I: Unconditional Single Steering.
    
    Metrics:
    - Directional success rate: % where sign(α) matches rating direction
    - Spearman correlation: α vs. rating
    
    Returns:
        Analysis dictionary
    """
    logging.info("\n" + "="*70)
    logging.info("CATEGORY I: Unconditional Single Steering Analysis")
    logging.info("="*70)
    
    analysis = {"pitch": {}, "duration": {}}
    
    for concept in ["pitch", "duration"]:
        concept_results = [r for r in results if r.get("concept") == concept]
        
        if not concept_results:
            logging.warning(f"No results for {concept}")
            continue
        
        logging.info(f"\n{concept.upper()} Analysis:")
        
        # Extract alphas and ratings
        alphas = []
        ratings = []
        baseline_rating = None
        
        for r in concept_results:
            alpha = r["alpha"]
            rating = r.get("final_rating")
            
            if rating is None:
                logging.warning(f"Missing rating for alpha={alpha}")
                continue
            
            alphas.append(alpha)
            ratings.append(rating)
            
            if alpha == 0.0:
                baseline_rating = rating
        
        alphas = np.array(alphas)
        ratings = np.array(ratings)
        
        # Directional success rate
        if baseline_rating is not None:
            directional_success = []
            
            for alpha, rating in zip(alphas, ratings):
                if alpha == 0.0:
                    continue  # Skip baseline
                
                # Expected direction
                expected_up = alpha > 0
                actual_up = rating > baseline_rating
                
                # Success if direction matches
                success = (expected_up == actual_up)
                directional_success.append(success)
            
            success_rate = np.mean(directional_success) if directional_success else 0.0
        else:
            success_rate = None
        
        # Spearman correlation
        if len(alphas) > 2:
            rho, p_value = spearmanr(alphas, ratings)
        else:
            rho, p_value = None, None
        
        # Wilcoxon test (positive vs negative alphas)
        pos_ratings = ratings[alphas > 0]
        neg_ratings = ratings[alphas < 0]
        
        if len(pos_ratings) > 0 and len(neg_ratings) > 0:
            wilcoxon_stat, wilcoxon_p = wilcoxon(pos_ratings, neg_ratings)
            cohens_d = calculate_cohens_d(pos_ratings, neg_ratings)
        else:
            wilcoxon_stat, wilcoxon_p, cohens_d = None, None, None
        
        analysis[concept] = {
            "n_samples": len(concept_results),
            "directional_success_rate": success_rate,
            "spearman_rho": rho,
            "spearman_p": p_value,
            "baseline_rating": baseline_rating,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "cohens_d": cohens_d,
            "alpha_values": alphas.tolist(),
            "rating_values": ratings.tolist(),
        }
        
        # Log results
        logging.info(f"  Samples: {len(concept_results)}")
        if success_rate is not None:
            logging.info(f"  Directional success rate: {success_rate:.1%}")
        if rho is not None:
            logging.info(f"  Spearman ρ: {rho:.3f} (p={p_value:.4f})")
        if cohens_d is not None:
            logging.info(f"  Cohen's d: {cohens_d:.3f}")
    
    return analysis


def analyze_category_2(results: List[dict], config: dict) -> dict:
    """Analyze Category II: Unconditional Dual Steering.
    
    Metrics:
    - Dual directional success: % where both concepts match expected direction
    - Spearman correlation for pitch and duration separately
    
    Returns:
        Analysis dictionary
    """
    logging.info("\n" + "="*70)
    logging.info("CATEGORY II: Unconditional Dual Steering Analysis")
    logging.info("="*70)
    
    # Get baseline ratings
    baseline = next((r for r in results if r.get("scenario") == "baseline"), None)
    
    if baseline:
        baseline_pitch = baseline.get("final_pitch_rating")
        baseline_duration = baseline.get("final_duration_rating")
        logging.info(f"Baseline: pitch={baseline_pitch}, duration={baseline_duration}")
    else:
        baseline_pitch = None
        baseline_duration = None
        logging.warning("Baseline not found")
    
    # Group by scenario
    by_scenario = defaultdict(list)
    for r in results:
        scenario = r.get("scenario")
        if scenario and scenario != "baseline":
            by_scenario[scenario].append(r)
    
    scenario_analysis = {}
    
    for scenario, scenario_results in by_scenario.items():
        logging.info(f"\n{scenario.upper()}:")
        
        # Extract data
        alpha_pitches = []
        alpha_durations = []
        pitch_ratings = []
        duration_ratings = []
        dual_success = []
        
        for r in scenario_results:
            alpha_p = r.get("alpha_pitch")
            alpha_d = r.get("alpha_duration")
            pitch_rating = r.get("final_pitch_rating")
            duration_rating = r.get("final_duration_rating")
            
            if None in [alpha_p, alpha_d, pitch_rating, duration_rating]:
                continue
            
            alpha_pitches.append(alpha_p)
            alpha_durations.append(alpha_d)
            pitch_ratings.append(pitch_rating)
            duration_ratings.append(duration_rating)
            
            # Dual directional success
            if baseline_pitch is not None and baseline_duration is not None:
                pitch_success = (alpha_p > 0) == (pitch_rating > baseline_pitch)
                duration_success = (alpha_d > 0) == (duration_rating > baseline_duration)
                dual_success.append(pitch_success and duration_success)
        
        alpha_pitches = np.array(alpha_pitches)
        alpha_durations = np.array(alpha_durations)
        pitch_ratings = np.array(pitch_ratings)
        duration_ratings = np.array(duration_ratings)
        
        # Dual success rate
        dual_success_rate = np.mean(dual_success) if dual_success else None
        
        # Spearman correlations
        if len(alpha_pitches) > 2:
            rho_pitch, p_pitch = spearmanr(alpha_pitches, pitch_ratings)
            rho_duration, p_duration = spearmanr(alpha_durations, duration_ratings)
        else:
            rho_pitch, p_pitch = None, None
            rho_duration, p_duration = None, None
        
        scenario_analysis[scenario] = {
            "n_samples": len(scenario_results),
            "dual_success_rate": dual_success_rate,
            "pitch_spearman_rho": rho_pitch,
            "pitch_spearman_p": p_pitch,
            "duration_spearman_rho": rho_duration,
            "duration_spearman_p": p_duration,
            "alpha_pitch_values": alpha_pitches.tolist(),
            "alpha_duration_values": alpha_durations.tolist(),
            "pitch_rating_values": pitch_ratings.tolist(),
            "duration_rating_values": duration_ratings.tolist(),
        }
        
        logging.info(f"  Samples: {len(scenario_results)}")
        if dual_success_rate is not None:
            logging.info(f"  Dual success rate: {dual_success_rate:.1%}")
        if rho_pitch is not None:
            logging.info(f"  Pitch Spearman ρ: {rho_pitch:.3f} (p={p_pitch:.4f})")
        if rho_duration is not None:
            logging.info(f"  Duration Spearman ρ: {rho_duration:.3f} (p={p_duration:.4f})")
    
    return {
        "baseline_pitch_rating": baseline_pitch,
        "baseline_duration_rating": baseline_duration,
        "scenarios": scenario_analysis,
    }


def analyze_category_3(results: List[dict], config: dict) -> dict:
    """Analyze Category III: Conditional Single Steering.
    
    Metrics:
    - Override Success Rate (OSR): % where steering direction detected correctly
    
    Returns:
        Analysis dictionary
    """
    logging.info("\n" + "="*70)
    logging.info("CATEGORY III: Conditional Single Steering Analysis")
    logging.info("="*70)
    
    analysis = {}
    
    # Group by concept and direction
    for concept in ["pitch", "duration"]:
        concept_analysis = {}
        
        for direction in ["high_to_low", "low_to_high"]:
            dir_results = [
                r for r in results
                if r.get("concept") == concept and r.get("override_direction") == direction
            ]
            
            if not dir_results:
                continue
            
            logging.info(f"\n{concept.upper()} - {direction}:")
            
            # Expected change direction
            if direction == "high_to_low":
                expected_changes = {"pitch": "DOWN", "duration": "SHORTER"}
            else:  # low_to_high
                expected_changes = {"pitch": "UP", "duration": "LONGER"}
            
            expected = expected_changes[concept]
            
            # Group by alpha
            by_alpha = defaultdict(list)
            for r in dir_results:
                alpha = r.get("alpha")
                detected_direction = r.get("final_direction")
                
                if alpha is not None and detected_direction:
                    by_alpha[alpha].append(detected_direction == expected)
            
            # Calculate OSR per alpha
            osr_by_alpha = {}
            for alpha, successes in sorted(by_alpha.items()):
                osr = np.mean(successes)
                osr_by_alpha[alpha] = osr
                logging.info(f"  α={alpha:+.1f}: OSR={osr:.1%} ({sum(successes)}/{len(successes)})")
            
            # Overall OSR
            all_successes = [s for successes in by_alpha.values() for s in successes]
            overall_osr = np.mean(all_successes) if all_successes else 0.0
            
            # Alpha response correlation (stronger α → higher OSR?)
            if len(by_alpha) > 2:
                alphas = np.array(sorted(by_alpha.keys()))
                osrs = np.array([osr_by_alpha[a] for a in alphas])
                abs_alphas = np.abs(alphas[alphas != 0.0]) if any(alphas != 0.0) else alphas
                abs_osrs = osrs[alphas != 0.0] if any(alphas != 0.0) else osrs
                
                if len(abs_alphas) > 2:
                    rho, p_value = spearmanr(abs_alphas, abs_osrs)
                else:
                    rho, p_value = None, None
            else:
                rho, p_value = None, None
            
            concept_analysis[direction] = {
                "n_samples": len(dir_results),
                "overall_osr": overall_osr,
                "osr_by_alpha": osr_by_alpha,
                "expected_direction": expected,
                "alpha_response_rho": rho,
                "alpha_response_p": p_value,
            }
            
            logging.info(f"  Overall OSR: {overall_osr:.1%}")
            if rho is not None:
                logging.info(f"  Alpha-OSR correlation: ρ={rho:.3f} (p={p_value:.4f})")
        
        analysis[concept] = concept_analysis
    
    return analysis


def analyze_category_4(results: List[dict], config: dict) -> dict:
    """Analyze Category IV: Conditional Dual Steering.
    
    Metrics:
    - Dual-OSR: % where both pitch and duration changes detected correctly
    - Individual OSR for pitch and duration
    
    Returns:
        Analysis dictionary
    """
    logging.info("\n" + "="*70)
    logging.info("CATEGORY IV: Conditional Dual Steering Analysis")
    logging.info("="*70)
    
    # Expected directions per scenario
    expected_directions = {
        "highpitchshortdurationtolowlong": {
            "pitch": "DOWN",
            "duration": "LONGER"
        },
        "highpitchlongdurationtolowshort": {
            "pitch": "DOWN",
            "duration": "SHORTER"
        },
        "lowpitchshortdurationtohighlong": {
            "pitch": "UP",
            "duration": "LONGER"
        },
        "lowpitchlongdurationtohighshort": {
            "pitch": "UP",
            "duration": "SHORTER"
        },
    }
    
    # Group by scenario
    by_scenario = defaultdict(list)
    for r in results:
        scenario = r.get("scenario")
        if scenario and not r.get("is_baseline", False):
            by_scenario[scenario].append(r)
    
    scenario_analysis = {}
    
    for scenario, scenario_results in by_scenario.items():
        logging.info(f"\n{scenario.upper()}:")
        
        expected = expected_directions.get(scenario, {})
        expected_pitch = expected.get("pitch")
        expected_duration = expected.get("duration")
        
        # Calculate success rates
        pitch_successes = []
        duration_successes = []
        dual_successes = []
        
        for r in scenario_results:
            pitch_dir = r.get("final_pitch_direction")
            duration_dir = r.get("final_duration_direction")
            
            if pitch_dir and expected_pitch:
                pitch_success = (pitch_dir == expected_pitch)
                pitch_successes.append(pitch_success)
            
            if duration_dir and expected_duration:
                duration_success = (duration_dir == expected_duration)
                duration_successes.append(duration_success)
            
            if pitch_dir and duration_dir and expected_pitch and expected_duration:
                dual_success = (pitch_dir == expected_pitch) and (duration_dir == expected_duration)
                dual_successes.append(dual_success)
        
        pitch_osr = np.mean(pitch_successes) if pitch_successes else 0.0
        duration_osr = np.mean(duration_successes) if duration_successes else 0.0
        dual_osr = np.mean(dual_successes) if dual_successes else 0.0
        
        scenario_analysis[scenario] = {
            "n_samples": len(scenario_results),
            "expected_pitch_direction": expected_pitch,
            "expected_duration_direction": expected_duration,
            "pitch_osr": pitch_osr,
            "duration_osr": duration_osr,
            "dual_osr": dual_osr,
            "pitch_success_count": sum(pitch_successes),
            "duration_success_count": sum(duration_successes),
            "dual_success_count": sum(dual_successes),
        }
        
        logging.info(f"  Samples: {len(scenario_results)}")
        logging.info(f"  Pitch OSR: {pitch_osr:.1%} ({sum(pitch_successes)}/{len(pitch_successes)})")
        logging.info(f"  Duration OSR: {duration_osr:.1%} ({sum(duration_successes)}/{len(duration_successes)})")
        logging.info(f"  Dual OSR: {dual_osr:.1%} ({sum(dual_successes)}/{len(dual_successes)})")
    
    return scenario_analysis


def generate_latex_tables(analysis: dict, output_dir: pathlib.Path):
    """Generate LaTeX tables for paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Category I - Unconditional Single
    with open(output_dir / "table_category1.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Category I: Unconditional Single Steering Results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Concept & N & Dir. Success & Spearman $\\rho$ & $p$-value \\\\\n")
        f.write("\\hline\n")
        
        for concept in ["pitch", "duration"]:
            data = analysis["category_1"].get(concept, {})
            n = data.get("n_samples", 0)
            success = data.get("directional_success_rate")
            rho = data.get("spearman_rho")
            p = data.get("spearman_p")
            
            success_str = f"{success:.1%}" if success is not None else "---"
            rho_str = f"{rho:.3f}" if rho is not None else "---"
            p_str = f"{p:.4f}" if p is not None else "---"
            
            f.write(f"{concept.capitalize()} & {n} & {success_str} & {rho_str} & {p_str} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logging.info("Generated table_category1.tex")
    
    # Table 2: Category II - Unconditional Dual
    with open(output_dir / "table_category2.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Category II: Unconditional Dual Steering Results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Scenario & N & Dual Success & Pitch $\\rho$ & Duration $\\rho$ \\\\\n")
        f.write("\\hline\n")
        
        scenarios = analysis["category_2"].get("scenarios", {})
        for scenario, data in sorted(scenarios.items()):
            n = data.get("n_samples", 0)
            dual_success = data.get("dual_success_rate")
            rho_p = data.get("pitch_spearman_rho")
            rho_d = data.get("duration_spearman_rho")
            
            success_str = f"{dual_success:.1%}" if dual_success is not None else "---"
            rho_p_str = f"{rho_p:.3f}" if rho_p is not None else "---"
            rho_d_str = f"{rho_d:.3f}" if rho_d is not None else "---"
            
            scenario_name = scenario.replace("_", " ").title()
            f.write(f"{scenario_name} & {n} & {success_str} & {rho_p_str} & {rho_d_str} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logging.info("Generated table_category2.tex")
    
    # Table 3: Category III - Conditional Single (OSR)
    # Table 4: Category IV - Conditional Dual (Dual-OSR)
    # ... (similar patterns)


def generate_figures(analysis: dict, output_dir: pathlib.Path, config: dict):
    """Generate analysis figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Category I - Alpha response curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, concept in enumerate(["pitch", "duration"]):
        data = analysis["category_1"].get(concept, {})
        alphas = data.get("alpha_values", [])
        ratings = data.get("rating_values", [])
        
        if alphas and ratings:
            axes[idx].scatter(alphas, ratings, s=100, alpha=0.6)
            axes[idx].plot(alphas, ratings, 'o-', alpha=0.3)
            axes[idx].axhline(y=3, color='gray', linestyle='--', label='Neutral')
            axes[idx].set_xlabel('Alpha')
            axes[idx].set_ylabel('Music Flamingo Rating (1-5)')
            axes[idx].set_title(f'{concept.capitalize()} Control')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure_category1_response_curves.pdf")
    plt.close()
    
    logging.info("Generated figure_category1_response_curves.pdf")
    
    # Additional figures for other categories...


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Music Flamingo evaluation results"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/flamingo_eval/config.yaml"),
        help="Path to config.yaml",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    log_level = getattr(logging, config["logging"]["level"])
    handlers = [logging.StreamHandler()]
    
    if config["logging"]["save_to_file"]:
        log_file = pathlib.Path(config["logging"]["log_file"])
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    
    logging.info("="*70)
    logging.info("Music Flamingo Evaluation - Results Analysis")
    logging.info("="*70)
    
    # Load results
    output_root = pathlib.Path(config["paths"]["output_root"])
    results_path = output_root / "results.json"
    
    logging.info(f"Loading results from {results_path}")
    with open(results_path, "r") as f:
        all_results = json.load(f)
    
    logging.info(f"Total results: {len(all_results)}")
    
    # Group by category
    by_category = {
        "category_1": [r for r in all_results if r.get("category") == "unconditional_single"],
        "category_2": [r for r in all_results if r.get("category") == "unconditional_dual"],
        "category_3": [r for r in all_results if r.get("category") == "conditional_single"],
        "category_4": [r for r in all_results if r.get("category") == "conditional_dual"],
    }
    
    # Analyze each category
    analysis = {}
    
    if by_category["category_1"]:
        analysis["category_1"] = analyze_category_1(by_category["category_1"], config)
    
    if by_category["category_2"]:
        analysis["category_2"] = analyze_category_2(by_category["category_2"], config)
    
    if by_category["category_3"]:
        analysis["category_3"] = analyze_category_3(by_category["category_3"], config)
    
    if by_category["category_4"]:
        analysis["category_4"] = analyze_category_4(by_category["category_4"], config)
    
    # Save analysis
    analysis_path = output_root / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logging.info(f"\nAnalysis saved to {analysis_path}")
    
    # Generate LaTeX tables
    if config["output"]["generate_latex"]:
        tables_dir = output_root / "latex_tables"
        generate_latex_tables(analysis, tables_dir)
    
    # Generate figures
    if config["output"]["generate_figures"]:
        figures_dir = output_root / "figures"
        generate_figures(analysis, figures_dir, config)
    
    logging.info("\n" + "="*70)
    logging.info("Analysis complete!")
    logging.info(f"Results: {analysis_path}")
    if config["output"]["generate_latex"]:
        logging.info(f"LaTeX tables: {output_root / 'latex_tables'}")
    if config["output"]["generate_figures"]:
        logging.info(f"Figures: {output_root / 'figures'}")
    logging.info("="*70)


if __name__ == "__main__":
    main()
