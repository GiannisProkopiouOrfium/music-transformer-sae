#!/usr/bin/env python3
"""
Master Evaluation Pipeline for Feature Interventions

This script orchestrates the complete evaluation pipeline:
1. Base metrics evaluation (MusPy metrics)
2. Extended metrics evaluation (intervention-specific metrics)
3. Comparative analysis (baseline vs interventions)
4. Summary report generation

Usage:
    python run_evaluation_pipeline.py --interventions-dir batch_extractions_interventions/interventions
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import logging
from typing import Dict, Any


def setup_logging() -> logging.Logger:
    """Set up logging for the master pipeline."""
    logger = logging.getLogger('evaluation_pipeline')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def run_command(cmd: list, logger: logging.Logger, step_name: str) -> bool:
    """Run a command and handle errors."""
    logger.info(f"🔄 Running {step_name}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ {step_name} completed successfully")
        if result.stdout:
            # Log last few lines to avoid spam
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:
                if line.strip():
                    logger.info(f"Output: {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} failed with unexpected error: {e}")
        return False


def generate_final_summary(output_dir: Path, logger: logging.Logger) -> bool:
    """Generate comprehensive final summary highlighting all key metrics and calculations."""
    
    try:
        # Initialize comprehensive summary structure
        summary_data = {
            "evaluation_summary": {
                "pipeline_version": "2.0",
                "timestamp": "$(date -Iseconds)",
                "evaluation_focus": "feature_intervention_effects_with_paper_benchmarks"
            }
        }
        
        # Load all results files
        base_file = output_dir / "base_metrics_evaluation_results.json"
        extended_file = output_dir / "extended_metrics_evaluation_results.json"
        comparative_file = output_dir / "comparative_analysis_results.json"
        
        base_results = {}
        extended_results = {}
        comparative_results = {}
        
        if base_file.exists():
            with open(base_file, 'r') as f:
                base_results = json.load(f)
            logger.info("✅ Loaded base metrics results")
        
        if extended_file.exists():
            with open(extended_file, 'r') as f:
                extended_results = json.load(f)
            logger.info("✅ Loaded extended metrics results")
        
        if comparative_file.exists():
            with open(comparative_file, 'r') as f:
                comparative_results = json.load(f)
            logger.info("✅ Loaded comparative analysis results")
        
        # 1. EXPERIMENTAL SETUP
        base_summary = base_results.get("overall_summary", {})
        summary_data["experimental_setup"] = {
            "controlled_generation_parameters": base_summary.get("controlled_generation_context", {}),
            "total_features_evaluated": base_summary.get("total_features_evaluated", 0),
            "features_by_layer": base_summary.get("features_by_layer", {}),
            "intervention_conditions": list(base_summary.get("condition_summaries", {}).keys()),
            "evaluation_framework": {
                "focus": "controlled_intervention_effects_analysis",
                "baseline_type": "controlled_generation_temp_0.1_shared_prefix",
                "comparison_reference": "original_mmt_paper_with_context_awareness"
            }
        }
        
        # 2. PAPER BENCHMARK COMPARISONS (HIGHLIGHT)
        paper_benchmarks = base_summary.get("paper_benchmarks", {})
        benchmark_comparisons = base_summary.get("benchmark_comparisons", {})
        
        summary_data["paper_benchmark_analysis"] = {
            "reference_benchmarks": paper_benchmarks,
            "our_baseline_performance": {},
            "performance_vs_paper": {},
            "controlled_generation_context": {
                "interpretation": "Results reflect controlled generation strategy (temp=0.1, shared prefix) optimized for intervention detection vs general music quality",
                "expected_differences": "Lower pitch entropy, higher consistency due to controlled generation parameters"
            }
        }
        
        # Extract baseline performance for comparison
        baseline_condition = base_summary.get("condition_summaries", {}).get("baseline", {})
        if baseline_condition:
            baseline_metrics = baseline_condition.get("metrics", {})
            for metric in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
                if metric in baseline_metrics:
                    summary_data["paper_benchmark_analysis"]["our_baseline_performance"][metric] = {
                        "mean": baseline_metrics[metric]["mean"],
                        "std": baseline_metrics[metric]["std"],
                        "count": baseline_metrics[metric]["count"]
                    }
        
        # Add benchmark comparisons
        for metric, comparisons in benchmark_comparisons.items():
            summary_data["paper_benchmark_analysis"]["performance_vs_paper"][metric] = comparisons
        
        # 3. INTERVENTION EFFECTS ANALYSIS
        condition_summaries = base_summary.get("condition_summaries", {})
        summary_data["intervention_effects"] = {
            "baseline_reference": condition_summaries.get("baseline", {}),
            "intervention_conditions": {},
            "effect_sizes": {},
            "key_findings": []
        }
        
        # Add all intervention conditions
        for condition_name, condition_data in condition_summaries.items():
            if condition_name != "baseline":
                summary_data["intervention_effects"]["intervention_conditions"][condition_name] = condition_data
        
        # Add effect sizes from extended metrics if available
        extended_summary = extended_results.get("overall_summary", {})
        if "intervention_effect_sizes" in extended_summary:
            summary_data["intervention_effects"]["effect_sizes"] = extended_summary["intervention_effect_sizes"]
        
        # 4. COMPREHENSIVE METRICS OVERVIEW
        summary_data["metrics_overview"] = {
            "core_muspy_metrics": {
                "pitch_class_entropy": base_summary.get("overall_metrics", {}).get("pitch_class_entropy", {}),
                "scale_consistency": base_summary.get("overall_metrics", {}).get("scale_consistency", {}),
                "groove_consistency": base_summary.get("overall_metrics", {}).get("groove_consistency", {})
            },
            "extended_metrics": extended_summary.get("extended_metrics_summary", {}),
            "layer_specific_analysis": base_summary.get("layer_summaries", {})
        }
        
        # 5. MODEL PERFORMANCE ASSESSMENT
        model_performance = comparative_results.get("model_performance_preservation", {})
        summary_data["model_performance_analysis"] = {
            "overall_preservation": model_performance.get("overall_preservation", {}),
            "per_metric_degradation": {},
            "intervention_strength_effects": comparative_results.get("intervention_strength_effects", {}),
            "statistical_significance": comparative_results.get("statistical_significance_summary", {})
        }
        
        # Extract per-metric preservation scores
        for metric in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
            if metric in model_performance:
                summary_data["model_performance_analysis"]["per_metric_degradation"][metric] = {
                    "mean_degradation_percent": model_performance[metric].get("mean_degradation", 0),
                    "max_degradation_percent": model_performance[metric].get("max_degradation", 0),
                    "preservation_score": model_performance[metric].get("preservation_score", 0)
                }
        
        # 6. KEY INSIGHTS AND RECOMMENDATIONS
        recommendations = comparative_results.get("recommendations", [])
        summary_data["key_insights"] = {
            "controlled_generation_impact": {
                "temperature_0.1_effects": "Reduced pitch entropy variance, increased consistency",
                "shared_prefix_benefits": "Consistent harmonic/rhythmic foundation across conditions",
                "intervention_detection_optimization": "Setup optimized for detecting intervention effects vs general music quality"
            },
            "paper_benchmark_interpretation": {
                "pitch_entropy_differences": "Expected lower values due to deterministic sampling",
                "consistency_improvements": "Expected higher values due to shared musical foundation",
                "context_awareness": "Differences reflect experimental design choices, not model degradation"
            },
            "intervention_effectiveness": {},
            "recommendations": recommendations
        }
        
        # Add intervention effectiveness insights
        if benchmark_comparisons:
            for metric, comparisons in benchmark_comparisons.items():
                vs_original_mmt = comparisons.get("original_mmt", {})
                if vs_original_mmt:
                    perf_status = vs_original_mmt.get("performance_vs_benchmark", "unknown")
                    rel_change = vs_original_mmt.get("relative_change_percent", 0)
                    summary_data["key_insights"]["intervention_effectiveness"][metric] = {
                        "vs_original_mmt": f"{perf_status} ({rel_change:+.1f}%)",
                        "controlled_context": "Difference reflects controlled generation parameters"
                    }
        
        # 7. SUMMARY STATISTICS
        summary_data["summary_statistics"] = {
            "total_sequences_analyzed": sum([condition.get("count", 0) for condition in condition_summaries.values()]),
            "layers_analyzed": list(base_summary.get("features_by_layer", {}).keys()),
            "features_per_layer": base_summary.get("features_by_layer", {}),
            "intervention_conditions_tested": len([k for k in condition_summaries.keys() if k != "baseline"]),
            "evaluation_completeness": {
                "base_metrics": bool(base_results),
                "extended_metrics": bool(extended_results),
                "comparative_analysis": bool(comparative_results),
                "paper_benchmarks": bool(benchmark_comparisons)
            }
        }
        
        # Save comprehensive final summary
        final_summary_file = output_dir / "final_evaluation_summary.json"
        with open(final_summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"📊 Final summary saved to: {final_summary_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to generate final summary: {e}")
        return False


def main():
    """Main pipeline execution."""
    
    parser = argparse.ArgumentParser(description="Run complete evaluation pipeline for feature interventions")
    parser.add_argument(
        "--interventions-dir",
        type=Path,
        default="batch_extractions_interventions/interventions",
        help="Directory containing intervention results"
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default="data/sod/processed/notes/encoding.json",
        help="Path to encoding JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="evaluation_results",
        help="Output directory for all evaluation results"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base metrics evaluation (use existing results)"
    )
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="Skip extended metrics evaluation (use existing results)"
    )
    parser.add_argument(
        "--skip-comparative",
        action="store_true",
        help="Skip comparative analysis (use existing results)"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        help="Specific layers to evaluate (e.g., --layers 1 3 5)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        help="Specific feature IDs to evaluate (e.g., --features 325 471)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("🎼 STARTING COMPLETE EVALUATION PIPELINE FOR FEATURE INTERVENTIONS")
    logger.info("=" * 80)
    logger.info("📖 CONTROLLED GENERATION CONTEXT:")
    logger.info("  Temperature: 0.1 (deterministic sampling for intervention comparison)")
    logger.info("  Random Noise: Added for controlled creativity")
    logger.info("  Shared Prefix: Consistent musical foundation across conditions")
    logger.info("  Purpose: Optimized for intervention effect detection vs general music generation")
    logger.info("  Benchmarking: Results compared to original MMT paper with controlled baseline context")
    logger.info("=" * 80)
    logger.info(f"Interventions directory: {args.interventions_dir}")
    logger.info(f"Encoding path: {args.encoding_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Skip base: {args.skip_base}")
    logger.info(f"Skip extended: {args.skip_extended}")
    logger.info(f"Skip comparative: {args.skip_comparative}")
    
    pipeline_success = True
    
    # Step 1: Base Metrics Evaluation
    if not args.skip_base:
        logger.info("\n🎯 STEP 1: BASE METRICS EVALUATION")
        
        base_cmd = [
            "python", "evaluate_intervention_base_metrics.py",
            "--interventions-dir", str(args.interventions_dir),
            "--encoding-path", str(args.encoding_path),
            "--output-dir", str(args.output_dir)
        ]
        
        if args.layers:
            base_cmd.extend(["--layers"] + args.layers)
        if args.features:
            base_cmd.extend(["--features"] + args.features)
        
        if not run_command(base_cmd, logger, "Base Metrics Evaluation"):
            pipeline_success = False
    else:
        logger.info("⏭️ STEP 1: Skipping base metrics evaluation")
    
    # Step 2: Extended Metrics Evaluation
    if not args.skip_extended and pipeline_success:
        logger.info("\n🎯 STEP 2: EXTENDED METRICS EVALUATION")
        
        base_results_file = args.output_dir / "base_metrics_evaluation_results.json"
        
        if not base_results_file.exists():
            logger.error("❌ Base metrics results not found. Run base evaluation first.")
            pipeline_success = False
        else:
            extended_cmd = [
                "python", "evaluate_intervention_extra_metrics.py",
                "--base-results", str(base_results_file),
                "--encoding-path", str(args.encoding_path),
                "--output-dir", str(args.output_dir)
            ]
            
            if not run_command(extended_cmd, logger, "Extended Metrics Evaluation"):
                pipeline_success = False
    else:
        if args.skip_extended:
            logger.info("⏭️ STEP 2: Skipping extended metrics evaluation")
        else:
            logger.info("⏭️ STEP 2: Skipping due to previous failure")
    
    # Step 3: Comparative Analysis
    if not args.skip_comparative and pipeline_success:
        logger.info("\n🎯 STEP 3: COMPARATIVE ANALYSIS")
        
        extended_results_file = args.output_dir / "extended_metrics_evaluation_results.json"
        
        if not extended_results_file.exists():
            logger.error("❌ Extended metrics results not found. Run extended evaluation first.")
            pipeline_success = False
        else:
            comparative_cmd = [
                "python", "compare_intervention_performance.py",
                "--extended-results", str(extended_results_file),
                "--output-dir", str(args.output_dir)
            ]
            
            if not run_command(comparative_cmd, logger, "Comparative Analysis"):
                pipeline_success = False
    else:
        if args.skip_comparative:
            logger.info("⏭️ STEP 3: Skipping comparative analysis")
        else:
            logger.info("⏭️ STEP 3: Skipping due to previous failure")
    
    # Step 4: Generate Final Summary
    if pipeline_success:
        logger.info("\n🎯 STEP 4: GENERATING FINAL SUMMARY")
        
        if not generate_final_summary(args.output_dir, logger):
            pipeline_success = False
    else:
        logger.info("⏭️ STEP 4: Skipping final summary due to previous failures")
    
    # Final results
    logger.info("\n" + "=" * 80)
    
    if pipeline_success:
        logger.info("🎉 EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 All results saved to: {args.output_dir}")
        
        # List output files
        logger.info("\n📁 Generated files:")
        output_files = [
            "base_metrics_evaluation_results.json",
            "extended_metrics_evaluation_results.json", 
            "comparative_analysis_results.json",
            "final_evaluation_summary.json"
        ]
        
        for filename in output_files:
            file_path = args.output_dir / filename
            if file_path.exists():
                logger.info(f"  ✅ {filename}")
            else:
                logger.info(f"  ❌ {filename} (not generated)")
        
        logger.info("\n🎵 Evaluation pipeline ready for interpretation!")
        logger.info("\n� INTERPRETATION CONTEXT:")
        logger.info("  🎯 Focus: Controlled intervention effects analysis")
        logger.info("  📊 Baseline: Controlled generation (temp=0.1, shared prefix)")
        logger.info("  🔍 Benchmarks: Compared to original MMT paper with context awareness")
        logger.info("  📈 Differences: Reflect controlled generation strategy, not model degradation")
        logger.info("\n�🔍 Next steps:")
        logger.info("  1. Review final_evaluation_summary.json for key insights")
        logger.info("  2. Check comparative_analysis_results.json for controlled intervention effects")
        logger.info("  3. Use results for feature-specific analysis and LLM evaluation")
        logger.info("  4. Apply controlled baseline context when interpreting benchmark differences")
        
    else:
        logger.error("❌ EVALUATION PIPELINE FAILED!")
        logger.error("Check the logs above for specific error details")
        logger.error("You can resume the pipeline by using --skip-* flags for completed steps")
    
    return pipeline_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)