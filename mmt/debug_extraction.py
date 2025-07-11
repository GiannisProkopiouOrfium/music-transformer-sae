import torch
import pathlib
import logging
import sys
import utils
import representation
import dataset
import music_x_transformers
from activation_extractor import ActivationExtractor

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def debug_extraction():
    """Debug the activation extraction process step by step."""

    # Load minimal configuration
    out_dir = pathlib.Path("exp/sod/ape")
    in_dir = pathlib.Path("data/sod/processed/notes")

    # Load training args
    train_args = utils.load_json(out_dir / "train-args.json")
    logging.info(f"Loaded train args: {train_args}")

    # Load encoding
    encoding = representation.load_encoding(in_dir / "encoding.json")
    logging.info(f"Loaded encoding")

    # Use CPU to avoid CUDA issues
    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Create a minimal test dataset (just 1 sample)
    test_names_path = pathlib.Path("data/sod/processed/test-names.txt")
    if not test_names_path.exists():
        logging.error(f"Test names not found at {test_names_path}")
        return

    test_names = utils.load_txt(test_names_path)[:1]  # Just 1 sample
    logging.info(f"Using test sample: {test_names[0]}")

    test_dataset = dataset.MusicDataset(
        pathlib.Path(f"data/sod/processed/test-names.txt"),  # test_names,
        "data/sod/processed/notes/",
        encoding,
        max_seq_len=256,
        max_beat=train_args["max_beat"],
        use_csv=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.MusicDataset.collate,
    )

    logging.info(f"Created dataset with {len(test_dataset)} samples")

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

    # Load weights
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Model loaded and set to eval mode")

    # Print model structure
    logging.info("Model structure:")
    logging.info(f"  Total layers: {len(model.decoder.net.attn_layers.layers)}")
    for i, layer in enumerate(model.decoder.net.attn_layers.layers):
        logging.info(f"  Layer {i}: {type(layer)}")

    # Set up extractor for middle layer
    total_layers = len(model.decoder.net.attn_layers.layers)
    middle_layer = total_layers // 2
    logging.info(f"Using middle layer: {middle_layer}")

    with ActivationExtractor(model, [middle_layer], 1024) as extractor:
        # Process one batch
        logging.info("Processing test batch...")

        with torch.no_grad():
            for batch in test_loader:
                logging.info(f"Batch keys: {batch.keys()}")

                seq = batch["seq"].to(device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)

                logging.info(f"Input seq shape: {seq.shape}")
                logging.info(
                    f"Input mask shape: {mask.shape if mask is not None else 'None'}"
                )

                # Forward pass
                logging.info("Starting forward pass...")
                output = model(seq, mask=mask)
                logging.info(
                    f"Forward pass complete, output shape: {output.shape if hasattr(output, 'shape') else type(output)}"
                )

                break  # Just process one batch

        # Check what we collected
        logging.info("Checking collected activations...")
        for layer_idx in extractor.layer_indices:
            if layer_idx in extractor.activations:
                num_batches = len(extractor.activations[layer_idx])
                logging.info(f"Layer {layer_idx}: {num_batches} activation batches")
                if num_batches > 0:
                    first_batch = extractor.activations[layer_idx][0]
                    logging.info(f"  First batch shape: {first_batch.shape}")
            else:
                logging.info(f"Layer {layer_idx}: No activations collected")


if __name__ == "__main__":
    debug_extraction()
