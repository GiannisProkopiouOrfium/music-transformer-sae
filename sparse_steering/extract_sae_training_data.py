"""Extract random activations from MMT for SAE training.

This script samples random 16-beat segments from the dataset and extracts
their activations to train the Sparse Autoencoder. Unlike concept-based
extraction, this samples uniformly from all available music to learn a
general sparse representation.
"""

import argparse
import json
import logging
import pathlib
import random
import sys
from typing import Dict, List

import h5py
import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "steering_interventions"))

# Import from steering_interventions (sibling folder)
from activation_extractor import (
    ActivationExtractor,
    extract_activations_for_segments,
    save_activations,
)

# Import from mmt
import representation
import utils
import music_x_transformers
import config as mmt_config

# Import from sparse_steering
from config_sas import (
    N_TRAINING_SEGMENTS,
    QUICK_TEST_SEGMENTS,
    SAE_TRAINING_DATA_DIR,
    print_config,
)


def load_all_available_files(notes_dir: pathlib.Path) -> List[str]:
    """Load list of all available .npy files in the dataset.

    Args:
        notes_dir: Directory containing .npy files

    Returns:
        List of file identifiers (e.g., "Musicalion-1234")
    """
    file_list = []

    # Iterate through subdirectories
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        # Find all .npy files in this subfolder
        for npy_file in subfolder.glob("*.npy"):
            # Extract file identifier (without .npy extension)
            file_id = npy_file.stem
            file_list.append(file_id)

    logging.info(f"Found {len(file_list)} available files in {notes_dir}")
    return file_list


def sample_random_segments(
    file_list: List[str],
    n_segments: int,
    n_beats: int = 16,
    max_beat: int = 256,
    seed: int = 42,
) -> List[Dict]:
    """Sample random segments from available files.

    Args:
        file_list: List of available file identifiers
        n_segments: Number of segments to sample
        n_beats: Number of beats per segment
        max_beat: Maximum beat value in dataset
        seed: Random seed for reproducibility

    Returns:
        List of segment dictionaries compatible with activation_extractor
    """
    random.seed(seed)
    np.random.seed(seed)

    segments = []

    for i in range(n_segments):
        # Randomly select a file
        file_id = random.choice(file_list)

        # Randomly select a starting beat
        # Ensure we don't go beyond max_beat
        max_start_beat = max(0, max_beat - n_beats)
        start_beat = random.randint(0, max_start_beat)

        # Create segment dictionary (compatible with activation_extractor format)
        segment = {
            "file_name": file_id,
            "segment_idx": i,
            "start_beat": start_beat,
            "n_beats": n_beats,
        }

        segments.append(segment)

    logging.info(f"Sampled {len(segments)} random segments")
    logging.info(f"  Beats per segment: {n_beats}")
    logging.info(f"  Random seed: {seed}")

    return segments


def extract_random_activations(
    n_segments: int,
    notes_dir: pathlib.Path,
    model: torch.nn.Module,
    num_layers: int,
    encoding: Dict,
    device: torch.device,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    max_beat: int = 256,
    n_beats: int = 16,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Extract activations from random segments.

    Args:
        n_segments: Number of random segments to extract (guaranteed minimum)
        notes_dir: Directory with .npy files
        model: Pretrained MMT model
        num_layers: Number of layers in model
        encoding: Encoding dictionary
        device: Device to run on
        batch_size: Batch size for processing
        max_seq_len: Maximum sequence length
        max_beat: Maximum beat value
        n_beats: Beats per segment
        seed: Random seed

    Returns:
        Dictionary mapping layer_idx -> activations array (at least n_segments)
    """
    # Get all available files
    file_list = load_all_available_files(notes_dir)

    if len(file_list) == 0:
        logging.error(f"No .npy files found in {notes_dir}")
        return {}

    # Oversample to account for empty segments (50% extra)
    # We'll keep sampling until we get enough valid segments
    oversample_factor = 1.5
    initial_sample_size = int(n_segments * oversample_factor)

    logging.info(
        f"Target: {n_segments} valid segments (oversampling to {initial_sample_size} initially)"
    )

    # Sample random segments
    segments = sample_random_segments(
        file_list,
        initial_sample_size,
        n_beats=n_beats,
        max_beat=max_beat,
        seed=seed,
    )

    # Extract activations using existing infrastructure
    activations = extract_activations_for_segments(
        segments,
        model,
        num_layers,
        encoding,
        notes_dir,
        device,
        batch_size,
        max_seq_len,
        max_beat,
    )

    # Check if we need more segments
    if len(activations) > 0:
        first_layer_count = activations[0].shape[0]
        attempts = 1

        while first_layer_count < n_segments and attempts < 5:
            needed = n_segments - first_layer_count
            logging.info(
                f"Got {first_layer_count}/{n_segments} segments, sampling {needed} more..."
            )

            # Sample more segments (use different seed to avoid duplicates)
            more_segments = sample_random_segments(
                file_list,
                needed,
                n_beats=n_beats,
                max_beat=max_beat,
                seed=seed + attempts * 1000,
            )

            # Extract activations for additional segments
            more_activations = extract_activations_for_segments(
                more_segments,
                model,
                num_layers,
                encoding,
                notes_dir,
                device,
                batch_size,
                max_seq_len,
                max_beat,
            )

            # Concatenate with existing activations
            if len(more_activations) > 0:
                for layer_idx in activations:
                    activations[layer_idx] = np.concatenate(
                        [activations[layer_idx], more_activations[layer_idx]], axis=0
                    )
                first_layer_count = activations[0].shape[0]

            attempts += 1

        # Trim to exact count if we oversampled
        if first_layer_count > n_segments:
            logging.info(f"Trimming from {first_layer_count} to {n_segments} segments")
            for layer_idx in activations:
                activations[layer_idx] = activations[layer_idx][:n_segments]

    return activations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract random activations for SAE training"
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        default=None,
        help=f"Number of segments to extract (default: {N_TRAINING_SEGMENTS})",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help=f"Quick test mode with {QUICK_TEST_SEGMENTS} segments",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=SAE_TRAINING_DATA_DIR,
        help="Output directory for training data",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Model checkpoint path (default: best_model.pt)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--n_beats",
        type=int,
        default=16,
        help="Number of beats per segment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (e.g., 0, 1). If not specified, uses CPU.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Determine number of segments
    if args.quick_test:
        n_segments = QUICK_TEST_SEGMENTS
        logging.info(f"Quick test mode: extracting {n_segments} segments")
    elif args.n_segments is not None:
        n_segments = args.n_segments
    else:
        n_segments = N_TRAINING_SEGMENTS

    # Print configuration
    print_config(quick_test=args.quick_test)

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logging.warning(
                f"CUDA/MPS not available, falling back to CPU (requested GPU {args.gpu})"
            )
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU specified)")

    # Load model configuration
    train_args_file = mmt_config.MODEL_DIR / "train-args.json"
    if not train_args_file.exists():
        logging.error(f"Training args not found: {train_args_file}")
        return

    train_args = utils.load_json(train_args_file)
    logging.info(f"Loaded training args from: {train_args_file}")

    # Load encoding
    encoding_file = mmt_config.NOTES_DIR / "encoding.json"
    encoding = representation.load_encoding(encoding_file)
    logging.info(f"Loaded encoding from: {encoding_file}")

    # Create model
    logging.info("Creating model...")
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    # Load checkpoint
    if args.checkpoint is None:
        checkpoint_path = mmt_config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f"Loaded checkpoint from: {checkpoint_path}")

    # Get number of layers
    decoder_wrapper = model.decoder
    transformer = decoder_wrapper.net
    attn_layers = transformer.attn_layers
    num_layers = len(attn_layers.layers)

    logging.info(
        f"Model has {train_args['layers']} transformer blocks = {num_layers} layer modules"
    )

    # Extract random activations
    logging.info("=" * 60)
    logging.info("Extracting RANDOM activations for SAE training")
    logging.info("=" * 60)

    activations = extract_random_activations(
        n_segments=n_segments,
        notes_dir=mmt_config.NOTES_DIR,
        model=model,
        num_layers=num_layers,
        encoding=encoding,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        n_beats=args.n_beats,
        seed=args.seed,
    )

    # Save activations
    mode_suffix = "quick_test" if args.quick_test else "full"
    output_path = args.output_dir / f"random_activations_{mode_suffix}.h5"

    metadata = {
        "type": "random",
        "n_segments": n_segments,
        "n_beats": args.n_beats,
        "seed": args.seed,
        "quick_test": args.quick_test,
        "num_layers": num_layers,
        "model_dim": train_args["dim"],
    }

    save_activations(activations, output_path, metadata)

    # Print summary statistics
    logging.info("=" * 60)
    logging.info("Extraction Summary")
    logging.info("=" * 60)
    for layer_idx, layer_acts in activations.items():
        logging.info(
            f"Layer {layer_idx}: {layer_acts.shape[0]} samples, "
            f"dim={layer_acts.shape[1]}"
        )

    logging.info("=" * 60)
    logging.info(f"✓ Random activation extraction complete!")
    logging.info(f"✓ Saved to: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
