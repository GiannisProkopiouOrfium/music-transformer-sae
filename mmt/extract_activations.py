import argparse
import logging
import pathlib
import pprint
import sys
import h5py

import torch
import torch.utils.data
import tqdm
import numpy as np

import dataset
import music_x_transformers
import representation
import utils
from activation_extractor import ActivationExtractor


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract activations from MusicXTransformer"
    )

    # Data arguments
    parser.add_argument("-d", "--dataset", type=str, help="dataset key")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="output directory")
    parser.add_argument(
        "-ns", "--n_samples", type=int, default=100, help="number of samples"
    )

    # Model arguments
    parser.add_argument("-s", "--model_steps", type=int, help="model steps to load")
    parser.add_argument("-g", "--gpu", type=int, help="gpu number (use -1 for CPU)")

    # Activation extraction arguments
    parser.add_argument(
        "-l", "--layers", type=int, nargs="+", help="layer indices to extract from"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="maximum sequence length"
    )
    parser.add_argument("--use_csv", action="store_true", help="use CSV files")

    # Generation/sampling arguments (for optional generation-based extraction)
    parser.add_argument(
        "--seq_len", default=1024, type=int, help="sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )

    # Processing arguments
    parser.add_argument("-j", "--jobs", type=int, default=1, help="number of workers")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )

    return parser.parse_args(args=args, namespace=namespace)


def extract_activations_from_generation(
    model, extractor, encoding, device, n_samples=100, args=None
):
    """Extract activations during unconditional generation."""
    logging.info("Extracting activations from generation...")

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    # Use arguments from command line if provided
    seq_len = args.seq_len if args else 512
    temperature = args.temperature[0] if args and args.temperature else 1.0
    filter_threshold = (
        args.filter_threshold[0] if args and args.filter_threshold else 0.9
    )
    filter_fn = args.filter[0] if args and args.filter else "top_k"

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_samples), desc="Generating samples"):
            # Create start tokens for unconditional generation
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos

            # Generate sample - this will trigger our hooks
            generated = model.generate(
                tgt_start,
                seq_len=seq_len,
                eos_token=eos,
                temperature=temperature,
                filter_logits_fn=filter_fn,
                filter_thres=filter_threshold,
                monotonicity_dim=("type", "beat"),
            )


def extract_activations_from_dataset(
    model, extractor, test_loader, device, n_samples=100
):
    """Extract activations from dataset examples."""
    logging.info("Extracting activations from dataset...")

    with torch.no_grad():
        sample_count = 0
        for batch in tqdm.tqdm(test_loader, desc="Processing batches"):
            if sample_count >= n_samples:
                break

            # Move batch to device
            seq = batch["seq"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass - this will trigger our hooks
            _ = model(seq, mask=mask)

            sample_count += seq.shape[0]


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/{args.dataset}")

    # Set up logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save arguments
    logging.info(f"Saved arguments to {args.out_dir / 'extract-activations-args.json'}")
    utils.save_args(args.out_dir / "extract-activations-args.json", args)

    # Load training configurations
    logging.info(f"Loading training arguments from: {args.out_dir / 'train-args.json'}")
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Create activation directory
    activation_dir = args.out_dir / "activations"
    activation_dir.mkdir(exist_ok=True)

    # Get device - handle both GPU and CPU cases
    if args.gpu is not None and args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using GPU device: {device}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            logging.warning("Neither CUDA nor MPS available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    # Load encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create test dataset - Fix the path issue
    logging.info("Creating the data loader...")

    # Try different possible paths for test names
    possible_test_paths = [
        args.in_dir / "test-names.txt",
        args.in_dir / "../test-names.txt",
        pathlib.Path(f"data/{args.dataset}/processed/test-names.txt"),
    ]

    test_names_path = None
    for path in possible_test_paths:
        if path.exists():
            test_names_path = path
            logging.info(f"Found test names at: {path}")
            break

    if test_names_path is None:
        raise FileNotFoundError(
            f"Could not find test-names.txt in any of: {possible_test_paths}"
        )

    test_names = utils.load_txt(test_names_path)[: args.n_samples]
    logging.info(f"Loaded {len(test_names)} test names")

    test_dataset = dataset.MusicDataset(
        pathlib.Path(f"data/sod/processed/test-names.txt"),  # test_names,
        "data/sod/processed/notes/",
        encoding,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )

    logging.info(f"Created dataset with {len(test_dataset)} samples")

    # Adjust batch size based on device
    batch_size = 2 if device.type == "cpu" else 4
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=args.jobs if device.type == "cpu" else min(args.jobs, 2),
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create model
    logging.info("Creating the model...")
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
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"

    # Check if checkpoint exists
    if not checkpoint_filename.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_filename}")

    # Load model weights with proper device mapping
    logging.info(f"Loading model weights from: {checkpoint_filename}")
    checkpoint = torch.load(
        checkpoint_filename, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint)
    logging.info("Model weights loaded successfully")
    model.eval()

    # Set up activation extractor - FIX: Use the actual number of transformer layers
    if args.layers is None:
        # Calculate total transformer layers correctly
        # Each transformer layer consists of 2 layer groups (attention + FFN)
        total_model_layers = len(model.decoder.net.attn_layers.layers)
        total_transformer_layers = total_model_layers // 2

        # Default to middle transformer layer (0-indexed)
        middle_layer = total_transformer_layers // 2
        layer_indices = [middle_layer]
        logging.info(
            f"Using default middle transformer layer: {middle_layer} (out of 0-{total_transformer_layers-1})"
        )
    else:
        layer_indices = args.layers
        total_transformer_layers = len(model.decoder.net.attn_layers.layers) // 2

        # Validate layer indices
        for layer_idx in layer_indices:
            if layer_idx >= total_transformer_layers:
                raise ValueError(
                    f"Layer index {layer_idx} is out of bounds. Valid range: 0-{total_transformer_layers-1}"
                )

        logging.info(f"Extracting from transformer layers: {layer_indices}")

    # Add debugging: print model structure
    logging.info("Model structure:")
    total_model_layers = len(model.decoder.net.attn_layers.layers)
    total_transformer_layers = total_model_layers // 2
    logging.info(f"  Total model layer groups: {total_model_layers}")
    logging.info(f"  Total transformer layers: {total_transformer_layers}")
    logging.info(f"  Extracting from transformer layers: {layer_indices}")

    with ActivationExtractor(model, layer_indices, args.max_seq_len) as extractor:
        # Extract activations from dataset
        extract_activations_from_dataset(
            model, extractor, test_loader, device, args.n_samples
        )

        # Check if we actually collected any activations
        total_activations = sum(len(acts) for acts in extractor.activations.values())
        logging.info(f"Collected {total_activations} activation batches")

        if total_activations == 0:
            logging.error("No activations were collected! Check the hook registration.")
            return

        # Extract activations from generation (optional)
        # Uncomment this line if you want to also extract from generated samples
        # extract_activations_from_generation(model, extractor, encoding, device, n_samples=10, args=args)

        # Save activations
        save_path = (
            activation_dir
            / f"activations_layers_{'_'.join(map(str, layer_indices))}.h5"
        )
        extractor.save_activations(save_path)

        # Print statistics
        for layer_idx in layer_indices:
            if layer_idx in extractor.activations:
                flattened = extractor.get_flattened_activations(layer_idx)
                logging.info(
                    f"Layer {layer_idx}: {flattened.shape[0]} tokens, {flattened.shape[1]} dimensions"
                )
            else:
                logging.warning(f"No activations found for layer {layer_idx}")


if __name__ == "__main__":
    main()
