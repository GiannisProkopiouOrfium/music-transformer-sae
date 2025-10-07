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
    logger = logging.getLogger("evaluation_pipeline")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def run_command(cmd: list, logger: logging.Logger, step_name: str) -> bool:
    """Run a command and handle errors."""
    logger.info(f"🔄 Running {step_name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"✅ {step_name} completed successfully")
        if result.stdout:
            # Log last few lines to avoid spam
            output_lines = result.stdout.strip().split("\n")
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
    """Generate final summary report combining all evaluation results."""

    try:
        summary_data = {
            "evaluation_pipeline_summary": {
                "pipeline_version": "1.0",
                "evaluation_steps": [
                    "base_metrics_evaluation",
                    "extended_metrics_evaluation",
                    "comparative_analysis",
                ],
            }
        }

        # Load base metrics results
        base_file = output_dir / "base_metrics_evaluation_results.json"
        if base_file.exists():
            with open(base_file, "r") as f:
                base_results = json.load(f)
            summary_data["base_metrics_summary"] = base_results.get(
                "overall_summary", {}
            )
            logger.info("✅ Loaded base metrics results")
        else:
            logger.warning("⚠️ Base metrics results not found")

        # Load extended metrics results
        extended_file = output_dir / "extended_metrics_evaluation_results.json"
        if extended_file.exists():
            with open(extended_file, "r") as f:
                extended_results = json.load(f)
            summary_data["extended_metrics_summary"] = extended_results.get(
                "overall_summary", {}
            ).get("extended_metrics_summary", {})
            logger.info("✅ Loaded extended metrics results")
        else:
            logger.warning("⚠️ Extended metrics results not found")

        # Load comparative analysis results
        comparative_file = output_dir / "comparative_analysis_results.json"
        if comparative_file.exists():
            with open(comparative_file, "r") as f:
                comparative_results = json.load(f)
            summary_data["comparative_analysis_summary"] = {
                "model_performance_preservation": comparative_results.get(
                    "model_performance_preservation", {}
                ).get("overall_preservation", {}),
                "recommendations": comparative_results.get("recommendations", []),
            }
            logger.info("✅ Loaded comparative analysis results")
        else:
            logger.warning("⚠️ Comparative analysis results not found")

        # Save final summary
        final_summary_file = output_dir / "final_evaluation_summary.json"
        with open(final_summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"📊 Final summary saved to: {final_summary_file}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to generate final summary: {e}")
        return False


def main():
    """Main pipeline execution."""

    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline for feature interventions"
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
        help="Output directory for all evaluation results",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base metrics evaluation (use existing results)",
    )
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="Skip extended metrics evaluation (use existing results)",
    )
    parser.add_argument(
        "--skip-comparative",
        action="store_true",
        help="Skip comparative analysis (use existing results)",
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

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Set up logging
    logger = setup_logging()

    logger.info("🎼 STARTING COMPLETE EVALUATION PIPELINE FOR FEATURE INTERVENTIONS")
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
            "python",
            str(script_dir / "evaluate_intervention_base_metrics.py"),
            "--interventions-dir",
            str(args.interventions_dir),
            "--encoding-path",
            str(args.encoding_path),
            "--output-dir",
            str(args.output_dir),
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
            logger.error(
                "❌ Base metrics results not found. Run base evaluation first."
            )
            pipeline_success = False
        else:
            extended_cmd = [
                "python",
                str(script_dir / "evaluate_intervention_extra_metrics.py"),
                "--base-results",
                str(base_results_file),
                "--encoding-path",
                str(args.encoding_path),
                "--output-dir",
                str(args.output_dir),
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

        extended_results_file = (
            args.output_dir / "extended_metrics_evaluation_results.json"
        )

        if not extended_results_file.exists():
            logger.error(
                "❌ Extended metrics results not found. Run extended evaluation first."
            )
            pipeline_success = False
        else:
            comparative_cmd = [
                "python",
                str(script_dir / "compare_intervention_performance.py"),
                "--extended-results",
                str(extended_results_file),
                "--output-dir",
                str(args.output_dir),
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
            "final_evaluation_summary.json",
        ]

        for filename in output_files:
            file_path = args.output_dir / filename
            if file_path.exists():
                logger.info(f"  ✅ {filename}")
            else:
                logger.info(f"  ❌ {filename} (not generated)")

        logger.info("\n🎵 Evaluation pipeline ready for interpretation!")
        logger.info("\n🔍 Next steps:")
        logger.info("  1. Review final_evaluation_summary.json for key insights")
        logger.info(
            "  2. Check comparative_analysis_results.json for detailed comparisons"
        )
        logger.info("  3. Use results for feature-specific analysis and LLM evaluation")

    else:
        logger.error("❌ EVALUATION PIPELINE FAILED!")
        logger.error("Check the logs above for specific error details")
        logger.error(
            "You can resume the pipeline by using --skip-* flags for completed steps"
        )

    return pipeline_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
