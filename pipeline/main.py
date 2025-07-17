"""Main pipeline script for cross-platform SAE analysis."""

import argparse
import logging
import sys
from pathlib import Path

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.sae_pipeline import UnifiedSAEPipeline


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _print_stage_completion(stage_name: str, results: dict, output_dir: str):
    """Print stage completion information."""
    print(f"\n{'='*60}")
    print(f"🎯 {stage_name} COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

    if "activations_file" in results:
        print(f"📁 Activations: {results['activations_file']}")
        print(f"📊 Shape: {results.get('shape', 'Unknown')}")
        print(f"📊 Data type: {results.get('data_type', 'Unknown')}")

    if "model_file" in results:
        print(f"📁 Model: {results['model_file']}")
        print(f"🏗️  Architecture: {results.get('architecture', 'Unknown')}")

    if "analysis_file" in results:
        print(f"📁 Analysis: {results['analysis_file']}")
        print(f"📊 Features: {results.get('total_features', 'Unknown')}")
        print(f"📊 Sparsity: {results.get('sparsity_percent', 'Unknown'):.1f}%")

    print(f"📂 Output directory: {output_dir}")


def _print_pipeline_completion(results: dict, output_dir: str, experiment_name: str):
    """Print full pipeline completion information."""
    print(f"\n{'='*60}")
    print(f"🎉 FULL PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"🔬 Experiment: {experiment_name}")
    print(f"📂 Results: {output_dir}")

    if isinstance(results, dict):
        if "total_features" in results:
            print(f"📊 Discovered Features: {results.get('total_features', 'Unknown')}")
        if "sparsity_percent" in results:
            print(f"📊 Sparsity: {results.get('sparsity_percent', 'Unknown'):.1f}%")
        if "explained_variance" in results:
            print(
                f"📊 Explained Variance: {results.get('explained_variance', 'Unknown'):.1%}"
            )

    print(f"\n🚀 Next steps:")
    print(f"   • View visualizations: {output_dir}/{experiment_name}/sae_analysis.png")
    print(
        f"   • Check detailed results: {output_dir}/{experiment_name}/analysis_results.h5"
    )


def main():
    """Main pipeline execution - unified entry point for all platforms."""
    parser = argparse.ArgumentParser(
        description="Unified SAE Pipeline for Music Transformer Interpretability"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--experiment-name", required=True, help="Name for the experiment"
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "train", "analyze", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("🎵 Starting Music Transformer SAE Analysis Pipeline")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Stage: {args.stage}")

    try:
        # Create pipeline (auto-detects platform and optimizes)
        pipeline = UnifiedSAEPipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

        # Run specified stage(s)
        if args.stage == "extract":
            logger.info("🔄 Running activation extraction...")
            results = pipeline.extract_activations()
            logger.info(f"✅ Extraction completed")
            _print_stage_completion("EXTRACTION", results, args.output_dir)

        elif args.stage == "train":
            logger.info("🔄 Running SAE training...")
            results = pipeline.train_sae()
            logger.info(f"✅ Training completed")
            _print_stage_completion("TRAINING", results, args.output_dir)

        elif args.stage == "analyze":
            logger.info("🔄 Running SAE analysis...")
            results = pipeline.analyze_sae()
            logger.info(f"✅ Analysis completed")
            _print_stage_completion("ANALYSIS", results, args.output_dir)

        else:  # "all"
            logger.info("🔄 Running full pipeline...")
            results = pipeline.run_full_pipeline()
            logger.info(f"✅ Full pipeline completed")
            _print_pipeline_completion(results, args.output_dir, args.experiment_name)

        logger.info("🎉 Pipeline execution successful!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
