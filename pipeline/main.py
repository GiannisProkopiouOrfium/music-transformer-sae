"""Main pipeline script for AWS EC2 SAE analysis."""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.gpu import GPUManager, log_system_info
from utils.memory import MemoryOptimizer
from utils.storage import create_storage_directory
from pipeline.extract import run_extraction_pipeline
from pipeline.train import run_training_pipeline
from pipeline.analyze import run_analysis_pipeline


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup comprehensive logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_sections = ["model", "extraction", "training", "analysis"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    model_config = config["model"]
    required_model_keys = ["checkpoint_path", "layer_idx"]
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model config key: {key}")

    # Validate extraction config
    extraction_config = config["extraction"]
    if "output_file" not in extraction_config:
        raise ValueError("Missing required extraction config key: output_file")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="SAE Analysis Pipeline for AWS EC2")
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "train", "analyze", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--experiment-name", required=True, help="Name for the experiment"
    )
    parser.add_argument(
        "--output-dir", default="./experiments", help="Output directory"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing experiment"
    )

    args = parser.parse_args()

    # Create experiment directory
    experiment_dirs = create_storage_directory(args.output_dir, args.experiment_name)

    # Setup logging
    log_file = Path(experiment_dirs["logs"]) / "pipeline.log"
    setup_logging(args.log_level, str(log_file))
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting SAE analysis pipeline for experiment: {args.experiment_name}"
    )
    logger.info(f"Stage: {args.stage}")

    # Log system information
    log_system_info()

    try:
        # Load and validate configuration
        config = load_config(args.config)
        validate_config(config)
        logger.info(f"Loaded configuration from {args.config}")

        # Initialize system managers
        gpu_manager = GPUManager()
        memory_optimizer = MemoryOptimizer()

        # Log system capabilities
        gpu_info = gpu_manager.monitor_usage()
        memory_info = memory_optimizer.get_memory_info()

        logger.info(f"GPU Available: {gpu_info.get('gpu_available', False)}")
        if gpu_info.get("gpu_available"):
            logger.info(f"GPU Memory: {gpu_info['memory_total']:.1f} GB")
        logger.info(f"System Memory: {memory_info['total_gb']:.1f} GB")

        # Update config with experiment directories
        config["experiment"] = {
            "name": args.experiment_name,
            "directories": experiment_dirs,
            "resume": args.resume,
        }

        # Run pipeline stages
        if args.stage in ["extract", "all"]:
            logger.info("Starting activation extraction...")
            extraction_results = run_extraction_pipeline(config)
            logger.info(f"Extraction completed: {extraction_results}")

        if args.stage in ["train", "all"]:
            logger.info("Starting SAE training...")
            training_results = run_training_pipeline(config)
            logger.info(f"Training completed: {training_results}")

        if args.stage in ["analyze", "all"]:
            logger.info("Starting SAE analysis...")
            analysis_results = run_analysis_pipeline(config)
            logger.info(f"Analysis completed: {analysis_results}")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
