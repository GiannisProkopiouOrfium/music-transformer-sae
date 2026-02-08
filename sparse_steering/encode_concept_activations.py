"""Encode concept activations through trained SAE encoders.

This script loads high/low activations for concepts (pitch, duration) from the
DiffMean steering infrastructure and encodes them through trained SAE encoders
to produce sparse feature representations.

Stage 2, Part 1 of SAS implementation.
"""

import argparse
import logging
import pathlib
import sys
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "steering_interventions"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

from activation_extractor import extract_activations_for_segments
import representation
import utils
import music_x_transformers
import config as mmt_config

from sae_model import SparseAutoencoder
from config_sas import (
    SAE_CHECKPOINT_DIR,
    SAS_VECTORS_DIR,
    HIDDEN_DIM,
    SPARSE_DIM,
    NUM_LAYERS,
)


def load_curated_dataset(dataset_path: pathlib.Path) -> list:
    """Load curated dataset from JSON file.

    Args:
        dataset_path: Path to JSON file with segment list

    Returns:
        List of segment dictionaries with 'notes_file', 'start_beat', 'n_beats' fields
    """
    import json

    with open(dataset_path, "r") as f:
        segments = json.load(f)

    return segments


def get_adaptive_k(layer_idx: int) -> int:
    """Get adaptive K for layer."""
    if layer_idx == 0:
        return 32
    elif layer_idx < 4:
        return 64
    elif layer_idx < 8:
        return 96
    else:
        return 128


def load_sae(
    checkpoint_path: pathlib.Path, k: int, device: torch.device
) -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    sae = SparseAutoencoder(
        input_dim=HIDDEN_DIM,
        sparse_dim=SPARSE_DIM,
        k=k,
        tied_weights=True,
        normalize_input=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Backward compatibility: add _normalization_fitted if missing
    if "_normalization_fitted" not in state_dict:
        if "input_mean" in state_dict:
            has_normalization = state_dict["input_mean"].abs().sum() > 0
            state_dict["_normalization_fitted"] = torch.tensor(
                1 if has_normalization else 0
            )
        else:
            state_dict["_normalization_fitted"] = torch.tensor(0)

    sae.load_state_dict(state_dict)
    sae.eval()
    sae = sae.to(device)

    return sae


def load_all_saes(checkpoint_dir: pathlib.Path, device: torch.device) -> dict:
    """Load all 12 trained SAE models."""
    saes = {}

    logging.info("Loading trained SAE models...")
    for layer_idx in range(NUM_LAYERS):
        k = get_adaptive_k(layer_idx)
        checkpoint_path = checkpoint_dir / f"sae_layer_{layer_idx}_best.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

        saes[layer_idx] = load_sae(checkpoint_path, k, device)
        logging.info(f"  Layer {layer_idx}: K={k}")

    logging.info(f"✓ Loaded {NUM_LAYERS} SAE models")
    return saes


def extract_concept_activations(
    concept: str,
    group: str,
    model: torch.nn.Module,
    num_layers: int,
    encoding: dict,
    notes_dir: pathlib.Path,
    device: torch.device,
    datasets_dir: pathlib.Path,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    max_beat: int = 256,
) -> dict:
    """Extract activations for a concept group (e.g., high pitch)."""

    # Load curated dataset for this concept and group
    logging.info(f"Loading curated dataset: {group}_{concept}")
    dataset_path = datasets_dir / f"{group}_{concept}_segments.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Curated dataset not found: {dataset_path}\n"
            f"Please run steering_interventions/data_curator.py first to generate datasets."
        )

    segments = load_curated_dataset(dataset_path)
    logging.info(f"Loaded {len(segments)} segments for {concept}_{group}")

    # Extract activations
    logging.info(f"Extracting activations...")
    activations = extract_activations_for_segments(
        segments=segments,
        model=model,
        num_layers=num_layers,
        encoding=encoding,
        notes_dir=notes_dir,
        device=device,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_beat=max_beat,
    )

    if len(activations) == 0:
        raise ValueError(f"No activations extracted for {concept}_{group}")

    # Log shapes
    for layer_idx, acts in activations.items():
        logging.info(f"  Layer {layer_idx}: {acts.shape}")

    return activations


def encode_activations_with_saes(
    activations: dict,
    saes: dict,
    device: torch.device,
) -> dict:
    """Encode dense activations through SAE encoders to get sparse features."""

    sparse_activations = {}

    logging.info("Encoding activations through SAEs...")
    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in activations:
            logging.warning(f"  Layer {layer_idx}: No activations, skipping")
            continue

        # Get dense activations
        dense_acts = torch.from_numpy(activations[layer_idx]).float().to(device)

        # Encode through SAE
        with torch.no_grad():
            sparse_acts = saes[layer_idx].encode(dense_acts)

        # Move to CPU and convert to numpy
        sparse_activations[layer_idx] = sparse_acts.cpu().numpy()

        # Log sparsity statistics
        l0 = (sparse_acts != 0).float().sum(dim=-1).mean().item()
        logging.info(
            f"  Layer {layer_idx}: "
            f"{dense_acts.shape} → {sparse_acts.shape}, "
            f"L0={l0:.1f}"
        )

    return sparse_activations


def main():
    parser = argparse.ArgumentParser(
        description="Encode concept activations through trained SAEs"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
        help="Concepts to encode (default: average_pitch average_duration)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=SAE_CHECKPOINT_DIR,
        help="Directory with trained SAE checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=SAS_VECTORS_DIR,
        help="Output directory for encoded activations",
    )
    parser.add_argument(
        "--notes_dir",
        type=pathlib.Path,
        default=None,
        help="Directory with .npy note files",
    )
    parser.add_argument(
        "--datasets_dir",
        type=pathlib.Path,
        default=None,
        help="Directory with curated datasets (high_*_segments.json, low_*_segments.json)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=pathlib.Path,
        default=None,
        help="MMT model checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (e.g., 0, 1). If not specified, uses CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for activation extraction",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Using CUDA device: GPU {args.gpu}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Set defaults
    if args.notes_dir is None:
        args.notes_dir = pathlib.Path("data/sod/processed/notes")

    if args.datasets_dir is None:
        args.datasets_dir = pathlib.Path("steering_interventions/outputs/datasets")

    if args.model_checkpoint is None:
        args.model_checkpoint = pathlib.Path("exp/sod/ape/checkpoints/best_model.pt")

    # Load MMT model
    logging.info("Loading MMT model...")
    train_args_path = args.model_checkpoint.parent.parent / "train-args.json"
    encoding_path = args.notes_dir / "encoding.json"

    train_args = utils.load_json(train_args_path)
    encoding = utils.load_json(encoding_path)

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

    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    )
    model.eval()

    # Get the actual number of layer modules (not transformer blocks)
    # Each transformer block has 2 modules: Attention + FeedForward
    # So actual layer count = len(model.decoder.net.attn_layers.layers)
    decoder_wrapper = model.decoder
    transformer = decoder_wrapper.net
    attn_layers = transformer.attn_layers
    num_layers = len(attn_layers.layers)
    
    logging.info(f"Model has {num_layers} layers")

    # Load all SAEs
    saes = load_all_saes(args.checkpoint_dir, device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each concept
    for concept in args.concepts:
        logging.info("=" * 60)
        logging.info(f"Processing concept: {concept}")
        logging.info("=" * 60)

        for group in ["high", "low"]:
            logging.info(f"\n{concept}_{group}:")
            logging.info("-" * 60)

            # Extract dense activations
            dense_activations = extract_concept_activations(
                concept=concept,
                group=group,
                model=model,
                num_layers=num_layers,
                encoding=encoding,
                notes_dir=args.notes_dir,
                datasets_dir=args.datasets_dir,
                device=device,
                batch_size=args.batch_size,
                max_seq_len=train_args["max_seq_len"],
                max_beat=train_args["max_beat"],
            )

            # Encode through SAEs
            sparse_activations = encode_activations_with_saes(
                activations=dense_activations,
                saes=saes,
                device=device,
            )

            # Save sparse activations
            output_path = args.output_dir / f"{concept}_{group}_sparse.pt"
            torch.save(sparse_activations, output_path)
            logging.info(f"✓ Saved to: {output_path}")

    logging.info("\n" + "=" * 60)
    logging.info("✓ Concept activation encoding complete!")
    logging.info(f"✓ Saved to: {args.output_dir}")
    logging.info("=" * 60)
    logging.info("\nNext: Run compute_sas_vectors.py to generate steering vectors")


if __name__ == "__main__":
    main()
