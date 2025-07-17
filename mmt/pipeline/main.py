"""
Main Pipeline Runner

Unified interface to run the complete SAE pipeline or individual components.
"""

import argparse
import logging
import pathlib
import json
import sys
from typing import Optional, Dict, Any

# Add parent directory to path to import mmt modules
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from .config import PipelineConfig
from .extract import extract_activations_pipeline
from .train import train_sae_pipeline

# Import analyze conditionally to avoid import errors
try:
    from .analyze import analyze_sae_pipeline

    ANALYZE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Analysis module not available: {e}")
    ANALYZE_AVAILABLE = False
    analyze_sae_pipeline = None


def setup_logging(config: PipelineConfig):
    """Setup logging configuration."""
    log_level = getattr(logging, config.log_level.upper())

    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    handlers = [logging.StreamHandler(sys.stdout)]

    if config.save_logs and config.output_dir:
        log_file = config.output_dir / "pipeline.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        handlers=handlers,
        force=True,  # Force reconfiguration
    )

    # Ensure the root logger level is set correctly
    logging.getLogger().setLevel(log_level)


def load_config_file(config_path: pathlib.Path) -> PipelineConfig:
    """Load configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return PipelineConfig.from_dict(config_dict)


def save_config_file(config: PipelineConfig, output_path: pathlib.Path):
    """Save configuration to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def create_default_configs():
    """Create default configuration files."""
    configs_dir = pathlib.Path("mmt/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Default config
    default_config = PipelineConfig()
    save_config_file(default_config, configs_dir / "default.json")

    # Small config for testing
    small_config = PipelineConfig()
    small_config.extraction.n_samples = 50
    small_config.extraction.layers = [2]
    small_config.sae.hidden_dim = 1024
    small_config.sae.num_epochs = 2
    save_config_file(small_config, configs_dir / "small.json")

    # Large config for production
    large_config = PipelineConfig()
    large_config.extraction.n_samples = 2000
    large_config.extraction.layers = [2, 3, 4]
    large_config.sae.hidden_dim = 4096
    large_config.sae.num_epochs = 10
    save_config_file(large_config, configs_dir / "large.json")

    logging.info(f"Created default config files in: {configs_dir}")


def run_pipeline(config: PipelineConfig) -> Dict[str, pathlib.Path]:
    """
    Run the complete SAE pipeline or individual components.

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary with paths to results from each stage
    """
    results = {}

    # Setup logging
    setup_logging(config)

    logging.info("=== STARTING SAE PIPELINE ===")
    logging.info(f"Configuration: {config.config_name}")
    logging.info(f"Dataset: {config.model.dataset}")
    logging.info(f"Representation: {config.model.representation}")
    logging.info(f"Output directory: {config.output_dir}")

    # Save pipeline config
    if config.output_dir:
        config_path = config.output_dir / "pipeline_config.json"
        save_config_file(config, config_path)
        logging.info(f"Saved pipeline config: {config_path}")

    # Stage 1: Extract activations
    if config.run_extraction:
        try:
            logging.info("\\n" + "=" * 50)
            logging.info("STAGE 1: ACTIVATION EXTRACTION")
            logging.info("=" * 50)

            activations_path = extract_activations_pipeline(config)
            results["activations"] = activations_path

            logging.info(f"✓ Activations extracted: {activations_path}")

        except Exception as e:
            logging.error(f"✗ Activation extraction failed: {e}")
            raise
    else:
        # Look for existing activations
        layer_str = "_".join(map(str, config.extraction.layers))
        activations_path = (
            config.output_dir / "activations" / f"activations_layers_{layer_str}.h5"
        )
        if not activations_path.exists():
            raise FileNotFoundError(
                f"Activations not found and extraction disabled: {activations_path}"
            )
        results["activations"] = activations_path
        logging.info(f"Using existing activations: {activations_path}")

    # Stage 2: Train SAE
    if config.run_training:
        try:
            logging.info("\\n" + "=" * 50)
            logging.info("STAGE 2: SAE TRAINING")
            logging.info("=" * 50)

            model_path = train_sae_pipeline(config, results["activations"])
            results["model"] = model_path

            logging.info(f"✓ SAE trained: {model_path}")

        except Exception as e:
            logging.error(f"✗ SAE training failed: {e}")
            raise
    else:
        # Look for existing model
        model_path = (
            config.output_dir / "sae_models" / f"sae_layer_{config.sae.hidden_dim}d.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"SAE model not found and training disabled: {model_path}"
            )
        results["model"] = model_path
        logging.info(f"Using existing SAE model: {model_path}")

    # Stage 3: Analyze results
    if config.run_analysis:
        if not ANALYZE_AVAILABLE:
            logging.warning("Analysis module not available, skipping analysis stage")
            logging.info(
                "Install additional dependencies for analysis: matplotlib, seaborn"
            )
        else:
            try:
                logging.info("\\n" + "=" * 50)
                logging.info("STAGE 3: SAE ANALYSIS")
                logging.info("=" * 50)

                analysis_dir = analyze_sae_pipeline(
                    config, results["model"], results["activations"]
                )
                results["analysis"] = analysis_dir

                logging.info(f"✓ Analysis complete: {analysis_dir}")

            except Exception as e:
                logging.error(f"✗ Analysis failed: {e}")
                raise

    logging.info("\\n" + "=" * 50)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 50)

    for stage, path in results.items():
        logging.info(f"  {stage}: {path}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SAE Pipeline for Music Transformer Interpretability"
    )

    # Config file
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, help="Path to configuration JSON file"
    )

    # Quick options (override config)
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["sod", "lmd", "lmd_full", "snd"],
        help="Dataset to use",
    )

    parser.add_argument(
        "-r",
        "--representation",
        choices=["ape", "rpe", "npe", "mmm", "remi"],
        help="Model representation to use",
    )

    parser.add_argument(
        "-o", "--output_dir", type=pathlib.Path, help="Output directory"
    )

    parser.add_argument(
        "-g", "--gpu", type=int, default=-1, help="GPU device (-1 for CPU)"
    )

    # Pipeline control
    parser.add_argument(
        "--extract-only", action="store_true", help="Only run activation extraction"
    )

    parser.add_argument(
        "--train-only", action="store_true", help="Only run SAE training"
    )

    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis")

    parser.add_argument(
        "--skip-extraction", action="store_true", help="Skip activation extraction"
    )

    parser.add_argument(
        "--skip-training", action="store_true", help="Skip SAE training"
    )

    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis")

    # Quick parameters
    parser.add_argument(
        "--layers", type=int, nargs="+", help="Transformer layers to extract"
    )

    parser.add_argument(
        "--n-samples", type=int, help="Number of samples for extraction"
    )

    parser.add_argument("--hidden-dim", type=int, help="SAE hidden dimension")

    parser.add_argument("--epochs", type=int, help="Number of training epochs")

    # Utility actions
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create default configuration files and exit",
    )

    parser.add_argument(
        "--list-configs", action="store_true", help="List available configuration files"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle utility actions
    if args.create_configs:
        create_default_configs()
        return

    if args.list_configs:
        configs_dir = pathlib.Path("mmt/configs")
        if configs_dir.exists():
            config_files = list(configs_dir.glob("*.json"))
            print("Available configuration files:")
            for config_file in config_files:
                print(f"  {config_file.stem}")
        else:
            print("No configuration directory found. Run --create-configs first.")
        return

    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = PipelineConfig()

    # Apply command line overrides
    if args.dataset:
        config.model.dataset = args.dataset
    if args.representation:
        config.model.representation = args.representation
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.gpu is not None:
        config.model.gpu = args.gpu
    if args.layers:
        config.extraction.layers = args.layers
    if args.n_samples:
        config.extraction.n_samples = args.n_samples
    if args.hidden_dim:
        config.sae.hidden_dim = args.hidden_dim
    if args.epochs:
        config.sae.num_epochs = args.epochs

    # Apply pipeline control
    if args.extract_only:
        config.run_extraction = True
        config.run_training = False
        config.run_analysis = False
    elif args.train_only:
        config.run_extraction = False
        config.run_training = True
        config.run_analysis = False
    elif args.analyze_only:
        config.run_extraction = False
        config.run_training = False
        config.run_analysis = True
    else:
        # Apply skip flags
        if args.skip_extraction:
            config.run_extraction = False
        if args.skip_training:
            config.run_training = False
        if args.skip_analysis:
            config.run_analysis = False

    # Update paths based on dataset/representation
    config.__post_init__()

    # Run pipeline
    try:
        results = run_pipeline(config)

        print("\\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        for stage, path in results.items():
            print(f"{stage.upper()}: {path}")

    except Exception as e:
        print(f"\\nPIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
