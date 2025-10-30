"""Test script to verify steering interventions work correctly.

This script:
1. Loads the model and steering vectors
2. Generates 3 samples: baseline (alpha=0), high velocity (alpha=+2), low velocity (alpha=-2)
3. Extracts velocities from generated MIDI
4. Compares statistics to verify steering effect
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
from steered_generator import SteeredGenerator, load_steering_vectors


def extract_velocities_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract velocity values from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        List of velocity values
    """
    try:
        music = representation.decode(tokens, encoding)

        velocities = []
        for track in music.tracks:
            for note in track.notes:
                velocities.append(note.velocity)

        return velocities
    except Exception as e:
        logging.error(f"Error extracting velocities: {e}")
        return []


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        List of pitch values
    """
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


def test_steering(
    model,
    steering_vectors,
    encoding,
    device,
    alphas=None,
    seq_len=512,
    n_samples=3,
):
    """Test steering with different alpha values.

    Args:
        model: The model
        steering_vectors: Steering vectors dict
        encoding: Encoding dictionary
        device: Device to use
        alphas: List of alpha values to test
        seq_len: Generation length
        n_samples: Number of samples per alpha

    Returns:
        Dictionary with results
    """
    if alphas is None:
        alphas = [-2.0, 0.0, 2.0]

    results = {}

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    generator = SteeredGenerator(model, steering_vectors, encoding)

    for alpha in alphas:
        logging.info(f"Testing alpha={alpha}")

        alpha_velocities = []
        alpha_pitches = []

        for i in range(n_samples):
            # Create start tokens
            start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start_tokens[:, 0, 0] = sos

            # Generate with steering
            generated = generator.generate(
                start_tokens,
                seq_len,
                alpha=alpha,
                target_layers=None,
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Combine start and generated
            full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]

            # Extract velocities and pitches
            velocities = extract_velocities_from_tokens(full_seq, encoding)
            pitches = extract_pitches_from_tokens(full_seq, encoding)

            if velocities:
                alpha_velocities.extend(velocities)
            if pitches:
                alpha_pitches.extend(pitches)
                logging.info(
                    f"  Sample {i}: {len(pitches)} notes, "
                    f"mean pitch={np.mean(pitches):.1f}, "
                    f"std={np.std(pitches):.1f}, "
                    f"range=[{np.min(pitches)}, {np.max(pitches)}]"
                )
            else:
                logging.warning(
                    f"  Sample {i}: No notes extracted! Generation may have failed."
                )

        if alpha_pitches:
            results[alpha] = {
                "velocities": alpha_velocities,
                "pitches": alpha_pitches,
                "velocity_mean": np.mean(alpha_velocities) if alpha_velocities else 0.0,
                "velocity_std": np.std(alpha_velocities) if alpha_velocities else 0.0,
                "pitch_mean": np.mean(alpha_pitches),
                "pitch_std": np.std(alpha_pitches),
                "pitch_min": np.min(alpha_pitches),
                "pitch_max": np.max(alpha_pitches),
                "n_notes": len(alpha_pitches),
            }
        else:
            results[alpha] = {
                "velocities": [],
                "pitches": [],
                "velocity_mean": 0.0,
                "velocity_std": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0,
                "pitch_max": 0,
                "n_notes": 0,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Test steering interventions")
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to use"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None, help="Model checkpoint path"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-1.0,0.0,1.0",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--n_samples", type=int, default=3, help="Number of samples per alpha"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (0, 1, etc.)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA/MPS not available, using CPU")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = (
            config.OUTPUT_DIR
            / "steering_vectors"
            / f"{args.concept}_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

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

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    # Run test
    logging.info("=" * 60)
    logging.info("Testing steering interventions")
    logging.info("=" * 60)

    results = test_steering(
        model,
        steering_vectors,
        encoding,
        device,
        alphas=alphas,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
    )

    # Print results
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS SUMMARY - PITCH ANALYSIS")
    logging.info("=" * 60)

    for alpha in sorted(results.keys()):
        stats = results[alpha]
        logging.info(
            f"Alpha {alpha:+5.1f}: "
            f"pitch_mean={stats['pitch_mean']:6.2f}, "
            f"pitch_std={stats['pitch_std']:5.2f}, "
            f"range=[{stats['pitch_min']:3d}, {stats['pitch_max']:3d}], "
            f"n={stats['n_notes']:4d} notes"
        )

    # Verify steering effect
    if 0.0 in results and -2.0 in results and 2.0 in results:
        baseline_mean = results[0.0]["pitch_mean"]
        low_mean = results[-2.0]["pitch_mean"]
        high_mean = results[2.0]["pitch_mean"]

        logging.info("\n" + "=" * 60)
        logging.info("STEERING EFFECT VERIFICATION - PITCH")
        logging.info("=" * 60)
        logging.info(f"Baseline (alpha=0):     {baseline_mean:.2f}")
        logging.info(f"Low pitch (alpha=-2):   {low_mean:.2f}")
        logging.info(f"High pitch (alpha=+2):  {high_mean:.2f}")
        logging.info(f"Low vs Baseline:        {low_mean - baseline_mean:+.2f}")
        logging.info(f"High vs Baseline:       {high_mean - baseline_mean:+.2f}")

        if high_mean > baseline_mean and low_mean < baseline_mean:
            logging.info("\n✓ SUCCESS: Pitch steering works as expected!")
        elif high_mean == baseline_mean == low_mean:
            logging.warning("\n✗ WARNING: No steering effect detected (all equal)")
        else:
            logging.warning(
                "\n? WARNING: Unexpected steering pattern (check if inverted)"
            )
    else:
        logging.info("\nNot all alphas tested, skipping verification")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
