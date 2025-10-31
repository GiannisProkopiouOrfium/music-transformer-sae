"""Test baseline generation quality without steering.

This script tests if the model can generate diverse outputs without any steering.
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from generated tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
        return pitches
    except Exception as e:
        logging.error(f"Error extracting pitches: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Test baseline generation quality")
    parser.add_argument("--gpu", type=int, default=None, help="GPU number to use")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Using CUDA device: GPU {args.gpu}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load model
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

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

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    all_pitches = []

    logging.info(f"Generating {args.n_samples} samples without steering...")

    for i in range(args.n_samples):
        # Create start tokens
        start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
        start_tokens[:, 0, 0] = sos

        # Generate WITHOUT steering (baseline model behavior)
        # Use model.generate DIRECTLY without any hooks
        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                args.seq_len,
                eos_token=eos,
                temperature=1.0,  # Standard temperature
                filter_logits_fn="top_k",
                filter_thres=0.9,
                monotonicity_dim=("type", "beat"),
            )

        # Combine start and generated
        full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]

        # Extract pitches
        pitches = extract_pitches_from_tokens(full_seq, encoding)

        if pitches:
            all_pitches.extend(pitches)
            logging.info(
                f"Sample {i}: {len(pitches)} notes, "
                f"mean={np.mean(pitches):.1f}, "
                f"std={np.std(pitches):.1f}, "
                f"min={np.min(pitches)}, "
                f"max={np.max(pitches)}"
            )
        else:
            logging.warning(f"Sample {i}: No notes extracted!")

    if all_pitches:
        logging.info("\n" + "=" * 60)
        logging.info("OVERALL STATISTICS (without steering)")
        logging.info("=" * 60)
        logging.info(f"Total notes: {len(all_pitches)}")
        logging.info(f"Mean pitch: {np.mean(all_pitches):.2f}")
        logging.info(f"Std pitch: {np.std(all_pitches):.2f}")
        logging.info(f"Min pitch: {np.min(all_pitches)}")
        logging.info(f"Max pitch: {np.max(all_pitches)}")
        logging.info(f"Unique pitches: {len(set(all_pitches))}")

        if np.std(all_pitches) < 1.0:
            logging.error("\n⚠️  WARNING: Very low pitch diversity!")
            logging.error("The model is generating mostly the same pitch.")
            logging.error(
                "This suggests a problem with the model or generation parameters."
            )
        else:
            logging.info("\n✓ Model generates diverse pitches")
    else:
        logging.error("No notes extracted from any sample!")


if __name__ == "__main__":
    main()
