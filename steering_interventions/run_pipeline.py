"""End-to-end pipeline for steering interventions.

This script runs the complete pipeline:
1. Data curation
2. Activation extraction
3. Steering vector calculation
4. Evaluation
"""

import argparse
import logging
import pathlib
import subprocess
import sys

import config


def run_command(cmd: list, description: str):
    """Run a command and log the output.
    
    Args:
        cmd: Command to run
        description: Description for logging
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"{description}")
    logging.info(f"{'='*60}")
    logging.info(f"Command: {' '.join(str(c) for c in cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    logging.info(f"✓ {description} complete")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run complete steering intervention pipeline")
    parser.add_argument(
        "--concept",
        type=str,
        default="velocity",
        help="Concept to use"
    )
    parser.add_argument(
        "--n_beats",
        type=int,
        default=config.SEGMENT_N_BEATS,
        help="Segment length in beats"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for activation extraction"
    )
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=config.N_GENERATION_SAMPLES,
        help="Number of evaluation samples per alpha"
    )
    parser.add_argument(
        "--skip_curation",
        action="store_true",
        help="Skip data curation step"
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip activation extraction step"
    )
    parser.add_argument(
        "--skip_calculation",
        action="store_true",
        help="Skip steering vector calculation step"
    )
    parser.add_argument(
        "--only_evaluation",
        action="store_true",
        help="Only run evaluation (skip all preprocessing)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    script_dir = pathlib.Path(__file__).parent
    
    # Step 1: Data Curation
    if not args.skip_curation and not args.only_evaluation:
        run_command(
            [
                sys.executable,
                str(script_dir / "data_curator.py"),
                "--concept", args.concept,
                "--n_beats", str(args.n_beats),
            ],
            "STEP 1: Data Curation"
        )
    
    # Step 2: Activation Extraction
    if not args.skip_extraction and not args.only_evaluation:
        cmd = [
            sys.executable,
            str(script_dir / "activation_extractor.py"),
            "--concept", args.concept,
            "--batch_size", str(args.batch_size),
        ]
        
        if args.gpu is not None:
            cmd.extend(["--gpu", str(args.gpu)])
        
        run_command(cmd, "STEP 2: Activation Extraction")
    
    # Step 3: Steering Vector Calculation
    if not args.skip_calculation and not args.only_evaluation:
        run_command(
            [
                sys.executable,
                str(script_dir / "steering_vector_calculator.py"),
                "--concept", args.concept,
                "--visualize",
            ],
            "STEP 3: Steering Vector Calculation"
        )
    
    # Step 4: Evaluation
    cmd = [
        sys.executable,
        str(script_dir / "evaluator.py"),
        "--concept", args.concept,
        "--n_samples", str(args.n_eval_samples),
    ]
    
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    
    run_command(cmd, "STEP 4: Evaluation")
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("PIPELINE COMPLETE!")
    logging.info(f"{'='*60}")
    logging.info(f"\nResults saved in: {config.OUTPUT_DIR}")
    logging.info(f"\nCheck the following files:")
    logging.info(f"  - {config.OUTPUT_DIR / 'evaluation' / f'{args.concept}_analysis.json'}")
    logging.info(f"  - {config.OUTPUT_DIR / 'evaluation' / f'{args.concept}_plot.png'}")
    logging.info(f"  - {config.OUTPUT_DIR / 'evaluation' / 'samples' / 'alpha_*' / '*.mid'}")
    logging.info("")


if __name__ == "__main__":
    main()
