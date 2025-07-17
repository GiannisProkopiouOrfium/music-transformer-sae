#!/usr/bin/env python3
"""
Standalone activation extraction module for SAE pipeline.

This module can be run independently to extract activations from music transformers.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.sae_pipeline import UnifiedSAEPipeline


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    """Extract activations from music transformer."""
    parser = argparse.ArgumentParser(
        description="Extract activations from music transformer layers"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for extracted activations"
    )
    parser.add_argument(
        "--experiment-name",
        default="extract_experiment",
        help="Name for the experiment",
    )
    parser.add_argument(
        "--layer", type=int, help="Target layer to extract (overrides config)"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum samples to process (overrides config)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force use of synthetic activations for testing",
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

    logger.info("🎵 Music Transformer Activation Extraction")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")

    try:
        # Load and modify config if needed
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Override config with CLI arguments
        if args.layer is not None:
            config["extraction"]["target_layer"] = args.layer
            logger.info(f"Overriding target layer: {args.layer}")

        if args.max_samples is not None:
            config["extraction"]["max_samples"] = args.max_samples
            logger.info(f"Overriding max samples: {args.max_samples}")

        if args.synthetic:
            config["platform"]["synthetic_fallback"] = True
            logger.info("Forcing synthetic activations")

        # Create pipeline
        pipeline = UnifiedSAEPipeline(
            config=config, output_dir=args.output, experiment_name=args.experiment_name
        )

        # Extract activations
        logger.info("🔄 Starting activation extraction...")
        results = pipeline.extract_activations()

        # Print results
        logger.info("✅ Extraction completed successfully!")
        logger.info(f"📁 Activations saved to: {results['file_path']}")
        logger.info(f"📊 Shape: {results['activations_shape']}")
        logger.info(f"📊 Metadata: {results['metadata']}")

        # Usage instructions
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("=" * 60)
        print("Train SAE:")
        print(
            f"  python pipeline/train.py --activations {results['file_path']} --output {args.output}/sae"
        )
        print("\nFull pipeline:")
        print(
            f"  python pipeline/main.py --config {args.config} --output-dir {args.output} --experiment-name {args.experiment_name} --stage train"
        )

    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
