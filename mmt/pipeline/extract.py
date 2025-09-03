"""
Activation Extraction Pipeline

Extracts transformer activations from music data for SAE training.
"""

import logging
import pathlib
import torch
import torch.utils.data
from typing import Tuple
import sys

# Add parent directory to path to import mmt modules
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from activation_extractor import ActivationExtractor
import music_x_transformers
import representation
import dataset
import utils
from .config import PipelineConfig


def setup_device(gpu: int) -> torch.device:
    """Setup compute device with proper fallbacks."""
    if gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
            logging.info(f"Using CUDA device: {device}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            logging.warning("GPU requested but not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    return device


def load_model_and_encoding(
    config: PipelineConfig, device: torch.device
) -> Tuple[music_x_transformers.MusicXTransformer, dict, dict]:
    """Load the trained model and encoding."""
    # Determine paths
    if config.input_dir:
        encoding_path = config.input_dir / "encoding.json"
    else:
        encoding_path = pathlib.Path(
            f"data/{config.model.dataset}/processed/notes/encoding.json"
        )

    if config.output_dir:
        train_args_path = config.output_dir / "train-args.json"
        checkpoint_dir = config.output_dir / "checkpoints"
    else:
        exp_dir = pathlib.Path(
            f"exp/{config.model.dataset}/{config.model.representation}"
        )
        train_args_path = exp_dir / "train-args.json"
        checkpoint_dir = exp_dir / "checkpoints"

    # Load encoding
    if not encoding_path.exists():
        raise FileNotFoundError(f"Encoding not found: {encoding_path}")
    encoding = representation.load_encoding(encoding_path)
    logging.info(f"Loaded encoding from {encoding_path}")

    # Load training args
    if not train_args_path.exists():
        raise FileNotFoundError(f"Training args not found: {train_args_path}")
    train_args = utils.load_json(train_args_path)
    logging.info(f"Loaded training args from {train_args_path}")

    # Create model
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
    if config.model.model_steps is None:
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_path = checkpoint_dir / f"model_{config.model.model_steps}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info(f"Loaded model checkpoint from {checkpoint_path}")

    return model, encoding, train_args


def create_data_loader(config: PipelineConfig, encoding: dict, train_args: dict):
    """Create data loader for activation extraction."""
    # Determine test names path
    dataset_paths = [
        pathlib.Path(f"data/{config.model.dataset}/processed/test-names.txt"),
        config.input_dir / "test-names.txt" if config.input_dir else None,
        config.input_dir / "../test-names.txt" if config.input_dir else None,
    ]

    test_names_path = None
    for path in dataset_paths:
        if path and path.exists():
            test_names_path = path
            break

    if test_names_path is None:
        raise FileNotFoundError(
            f"Could not find test-names.txt in any of: {dataset_paths}"
        )

    logging.info(f"Loading test names from: {test_names_path}")

    # Create dataset
    notes_dir = (
        config.input_dir
        if config.input_dir
        else pathlib.Path(f"data/{config.model.dataset}/processed/notes")
    )

    test_dataset = dataset.MusicDataset(
        test_names_path,  # This expects the path, not the list
        notes_dir,
        encoding,
        max_seq_len=config.extraction.max_seq_len,
        max_beat=train_args["max_beat"],
        use_csv=False,
    )

    # Limit to n_samples
    if len(test_dataset) > config.extraction.n_samples:
        indices = torch.randperm(len(test_dataset))[: config.extraction.n_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    logging.info(f"Using {len(test_dataset)} samples for extraction")

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.extraction.batch_size,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=dataset.MusicDataset.collate,
    )

    return test_loader


def extract_activations_from_dataset(model, test_loader, device, n_samples: int):
    """Extract activations from dataset examples."""
    logging.info("Extracting activations from dataset...")

    with torch.no_grad():
        sample_count = 0
        batch_count = 0

        for batch in test_loader:
            if sample_count >= n_samples:
                break

            # Move batch to device
            seq = batch["seq"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass - triggers hooks
            _ = model(seq, mask=mask)

            sample_count += seq.shape[0]
            batch_count += 1

            if batch_count % 10 == 0:
                logging.info(f"Processed {batch_count} batches, {sample_count} samples")


def extract_activations_pipeline(config: PipelineConfig) -> pathlib.Path:
    """
    Run the activation extraction pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Path to saved activations file
    """
    logging.info("=== STARTING ACTIVATION EXTRACTION ===")

    # Setup device
    device = setup_device(config.model.gpu)

    # Load model and encoding
    model, encoding, train_args = load_model_and_encoding(config, device)

    # Validate layer indices
    total_transformer_layers = len(model.decoder.net.attn_layers.layers) // 2
    for layer_idx in config.extraction.layers:
        if layer_idx >= total_transformer_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of bounds. Valid range: 0-{total_transformer_layers-1}"
            )

    logging.info(f"Model has {total_transformer_layers} transformer layers")
    logging.info(f"Extracting from layers: {config.extraction.layers}")

    # Create data loader
    test_loader = create_data_loader(config, encoding, train_args)

    # Setup output directory
    output_dir = config.output_dir or pathlib.Path(
        f"exp/{config.model.dataset}/{config.model.representation}"
    )
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    # Extract activations
    with ActivationExtractor(
        model, config.extraction.layers, config.extraction.max_seq_len
    ) as extractor:
        extract_activations_from_dataset(
            model, test_loader, device, config.extraction.n_samples
        )

        # Check collection
        total_activations = sum(len(acts) for acts in extractor.activations.values())
        if total_activations == 0:
            raise RuntimeError(
                "No activations were collected! Check hook registration."
            )

        logging.info(f"Collected {total_activations} activation batches")

        # Save activations
        layer_str = "_".join(map(str, config.extraction.layers))
        save_path = activation_dir / f"activations_layers_{layer_str}.h5"
        extractor.save_activations(save_path)

        # Log statistics
        for layer_idx in config.extraction.layers:
            if layer_idx in extractor.activations:
                flattened = extractor.get_flattened_activations(layer_idx)
                logging.info(
                    f"Layer {layer_idx}: {flattened.shape[0]} tokens, {flattened.shape[1]} dimensions"
                )

    logging.info(f"=== EXTRACTION COMPLETE: {save_path} ===")
    return save_path


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig()
    config.model.dataset = "sod"
    config.model.representation = "ape"
    config.extraction.layers = [3]
    config.extraction.n_samples = 100

    save_path = extract_activations_pipeline(config)
    print(f"Activations saved to: {save_path}")
