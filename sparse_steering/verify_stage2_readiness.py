"""Verify readiness for Stage 2: SAS Vector Generation.

This script checks that all prerequisites are met:
1. Stage 1 complete: Trained SAE models exist
2. Curated datasets exist for chosen concepts
3. Model and encoding files are accessible
"""

import argparse
import logging
import pathlib
import sys

from config_sas import (
    SAE_CHECKPOINT_DIR,
    SAS_VECTORS_DIR,
    NUM_LAYERS,
)


def check_sae_checkpoints(checkpoint_dir: pathlib.Path) -> bool:
    """Check that all SAE models are trained."""
    logging.info("Checking SAE checkpoints...")

    missing = []
    for layer_idx in range(NUM_LAYERS):
        checkpoint_path = checkpoint_dir / f"sae_layer_{layer_idx}_best.pt"
        if not checkpoint_path.exists():
            missing.append(layer_idx)

    if missing:
        logging.error(f"Missing SAE checkpoints for layers: {missing}")
        logging.error(f"Please run train_sae.py first to train these layers")
        return False

    logging.info(f"✓ All {NUM_LAYERS} SAE checkpoints found")
    return True


def check_curated_datasets(datasets_dir: pathlib.Path, concepts: list) -> bool:
    """Check that curated datasets exist for concepts."""
    logging.info("Checking curated datasets...")

    missing = []
    for concept in concepts:
        for group in ["high", "low"]:
            dataset_path = datasets_dir / f"{group}_{concept}_segments.json"
            if not dataset_path.exists():
                missing.append(f"{group}_{concept}")

    if missing:
        logging.error(f"Missing curated datasets: {missing}")
        logging.error(f"Expected location: {datasets_dir}")
        logging.error("")
        logging.error("To generate curated datasets, run:")
        logging.error("  cd steering_interventions")
        logging.error("  python data_curator.py --concept pitch --segment_length 8")
        logging.error("  python data_curator.py --concept duration --segment_length 8")
        logging.error("")
        logging.error("Or run the full pipeline:")
        logging.error("  cd steering_interventions")
        logging.error("  bash run_pipeline.sh")
        return False

    logging.info(f"✓ All curated datasets found for concepts: {concepts}")

    # Log dataset sizes
    for concept in concepts:
        for group in ["high", "low"]:
            dataset_path = datasets_dir / f"{group}_{concept}_segments.json"
            import json

            with open(dataset_path, "r") as f:
                segments = json.load(f)
            logging.info(f"  {group}_{concept}: {len(segments)} segments")

    return True


def check_model_files(
    model_checkpoint: pathlib.Path,
    train_args_path: pathlib.Path,
    encoding_path: pathlib.Path,
) -> bool:
    """Check that model and configuration files exist."""
    logging.info("Checking model files...")

    files = {
        "Model checkpoint": model_checkpoint,
        "Training args": train_args_path,
        "Encoding": encoding_path,
    }

    missing = []
    for name, path in files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        logging.error("Missing model files:")
        for m in missing:
            logging.error(f"  {m}")
        return False

    logging.info("✓ All model files found")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify readiness for Stage 2")
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["pitch", "duration"],
        help="Concepts to check (default: pitch duration)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=SAE_CHECKPOINT_DIR,
        help="Directory with trained SAE checkpoints",
    )
    parser.add_argument(
        "--datasets_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/outputs/datasets"),
        help="Directory with curated datasets",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/ape/checkpoints/best_model.pt"),
        help="MMT model checkpoint",
    )
    parser.add_argument(
        "--notes_dir",
        type=pathlib.Path,
        default=pathlib.Path("data/sod/processed/notes"),
        help="Directory with .npy note files",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("=" * 60)
    logging.info("Stage 2 Readiness Check")
    logging.info("=" * 60)

    # Derive paths
    train_args_path = args.model_checkpoint.parent.parent / "train-args.json"
    encoding_path = args.notes_dir.parent / "encoding.json"

    # Run checks
    checks = [
        ("SAE Checkpoints", lambda: check_sae_checkpoints(args.checkpoint_dir)),
        (
            "Curated Datasets",
            lambda: check_curated_datasets(args.datasets_dir, args.concepts),
        ),
        (
            "Model Files",
            lambda: check_model_files(
                args.model_checkpoint, train_args_path, encoding_path
            ),
        ),
    ]

    results = []
    for name, check_fn in checks:
        logging.info("")
        success = check_fn()
        results.append((name, success))

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logging.info(f"{status}: {name}")
        if not success:
            all_passed = False

    logging.info("=" * 60)

    if all_passed:
        logging.info("")
        logging.info("✓ All checks passed! Ready for Stage 2.")
        logging.info("")
        logging.info("Next steps:")
        logging.info("  1. python encode_concept_activations.py --gpu 0")
        logging.info("  2. python compute_sas_vectors.py")
        return 0
    else:
        logging.error("")
        logging.error("✗ Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
