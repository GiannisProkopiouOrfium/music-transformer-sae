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


def test_fixed_extraction():
    """Test the fixed activation extraction."""

    # Load configuration
    out_dir = pathlib.Path("exp/sod/ape")
    in_dir = pathlib.Path("data/sod/processed/notes")

    train_args = utils.load_json(out_dir / "train-args.json")
    encoding = representation.load_encoding(in_dir / "encoding.json")
    device = torch.device("cpu")

    # Create test dataset
    test_names_path = pathlib.Path("data/sod/processed/test-names.txt")
    test_names = utils.load_txt(test_names_path)[:1]

    test_dataset = dataset.MusicDataset(
        pathlib.Path(f"data/sod/processed/test-names.txt"),  # test_names,
        "data/sod/processed/notes/",
        encoding,
        max_seq_len=256,  # Smaller for testing
        max_beat=train_args["max_beat"],
        use_csv=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.MusicDataset.collate,
    )

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

    # Load weights
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    # Test extraction from middle transformer layer (layer 2 out of 0-5)
    middle_layer = 2

    with ActivationExtractor(model, [middle_layer], 1024) as extractor:
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)

                logging.info(f"Processing batch with seq shape: {seq.shape}")

                # Forward pass
                output = model(seq, mask=mask)
                logging.info(f"Forward pass complete")
                break

        # Check results
        logging.info("=== EXTRACTION RESULTS ===")
        for layer_idx in extractor.layer_indices:
            if (
                layer_idx in extractor.activations
                and len(extractor.activations[layer_idx]) > 0
            ):
                first_batch = extractor.activations[layer_idx][0]
                logging.info(
                    f"SUCCESS: Layer {layer_idx} collected activation with shape {first_batch.shape}"
                )

                # Try to get flattened activations
                try:
                    flattened = extractor.get_flattened_activations(layer_idx)
                    logging.info(f"SUCCESS: Flattened to shape {flattened.shape}")

                    # Test save (create a temp file)
                    temp_path = pathlib.Path("test_activations.h5")
                    extractor.save_activations(temp_path)

                    if temp_path.exists():
                        logging.info(f"SUCCESS: Saved activations to {temp_path}")
                        # Clean up
                        temp_path.unlink()
                    else:
                        logging.error("FAILED: File was not created")

                except Exception as e:
                    logging.error(f"FAILED: Error processing activations: {e}")
            else:
                logging.error(f"FAILED: No activations collected for layer {layer_idx}")


if __name__ == "__main__":
    test_fixed_extraction()
