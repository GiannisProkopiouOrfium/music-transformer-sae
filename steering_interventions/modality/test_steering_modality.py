"""Test script to verify modality steering interventions work correctly.

This script:
1. Loads the model and modality steering vectors
2. Generates samples with different alpha values
3. Detects modality (major/minor) in generated outputs
4. Verifies: α<0 → more major, α>0 → more minor
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors

# Import music21
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)


def extract_modality_from_tokens(tokens: np.ndarray, encoding: dict) -> tuple:
    """Extract modality (major/minor) from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        (mode, confidence): ("major"/"minor"/"unknown", correlation_coefficient)
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        # Extract pitches from tokens
        for token in tokens:
            if token[0] == note_type:
                pitch = token[3]  # pitch is at index 3
                pitches.append(pitch)

        if len(pitches) < 10:
            return ("unknown", 0.0)

        # Create music21 stream and analyze key
        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")
        return (key.mode, key.correlationCoefficient)

    except Exception as e:
        logging.warning(f"Modality detection failed: {e}")
        return ("unknown", 0.0)


def test_steering(
    model,
    steering_vectors,
    encoding,
    device,
    alphas=None,
    target_layers=None,
    seq_len=512,
    n_samples=10,
):
    """Test modality steering with different alpha values.

    Args:
        model: The model
        steering_vectors: Steering vectors dict
        encoding: Encoding dictionary
        device: Device to use
        alphas: List of alpha values to test
        target_layers: List of layer indices to apply steering (None = all)
        seq_len: Generation length
        n_samples: Number of samples per alpha

    Returns:
        Dictionary with results
    """
    if alphas is None:
        alphas = [-1.0, 0.0, 1.0]

    results = {}

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    generator = SteeredGenerator(model, steering_vectors, encoding)

    for alpha in alphas:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing α = {alpha}")
        logging.info(f"{'='*60}")

        modality_results = []

        for i in range(n_samples):
            # Create start tokens
            start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start_tokens[:, 0, 0] = sos

            # Generate with steering
            generated = generator.generate(
                start_tokens,
                seq_len,
                alpha=alpha,
                target_layers=target_layers,
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Combine start and generated
            full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]

            # Extract modality
            mode, confidence = extract_modality_from_tokens(full_seq, encoding)
            modality_results.append((mode, confidence))

            logging.info(f"  Sample {i+1:2d}: {mode:7s} (confidence: {confidence:.3f})")

        # Summarize results for this alpha
        major_count = sum(1 for m, c in modality_results if m == "major" and c >= 0.5)
        minor_count = sum(1 for m, c in modality_results if m == "minor" and c >= 0.5)
        unknown_count = sum(1 for m, c in modality_results if c < 0.5)
        total_confident = major_count + minor_count

        # Calculate average confidence
        confident_results = [(m, c) for m, c in modality_results if c >= 0.5]
        avg_confidence = (
            np.mean([c for _, c in confident_results]) if confident_results else 0.0
        )

        results[alpha] = {
            "modality_results": modality_results,
            "major_count": major_count,
            "minor_count": minor_count,
            "unknown_count": unknown_count,
            "total_confident": total_confident,
            "major_percentage": (
                100 * major_count / total_confident if total_confident > 0 else 0.0
            ),
            "minor_percentage": (
                100 * minor_count / total_confident if total_confident > 0 else 0.0
            ),
            "avg_confidence": avg_confidence,
            "n_samples": n_samples,
        }

        # Print summary
        logging.info(f"\nSummary for α = {alpha}:")
        logging.info(
            f"  Major:   {major_count:2d}/{total_confident} ({results[alpha]['major_percentage']:.1f}%)"
        )
        logging.info(
            f"  Minor:   {minor_count:2d}/{total_confident} ({results[alpha]['minor_percentage']:.1f}%)"
        )
        logging.info(f"  Unknown: {unknown_count:2d} (low confidence)")
        logging.info(f"  Avg confidence: {avg_confidence:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test modality steering interventions")
    parser.add_argument(
        "--concept", type=str, default="modality", help="Concept (should be modality)"
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
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to apply steering (default: all layers)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of samples per alpha"
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
        steering_path = pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    if not steering_path.exists():
        logging.error(f"Steering vectors not found: {steering_path}")
        logging.error("Run steering vector calculation first!")
        sys.exit(1)

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

    # Parse target layers if specified
    target_layers = None
    if args.target_layers is not None:
        target_layers = [int(l.strip()) for l in args.target_layers.split(",")]
        logging.info(f"Will apply steering to layers: {target_layers}")
    else:
        logging.info("Will apply steering to all layers")

    # Run test
    logging.info("=" * 60)
    logging.info("TESTING MODALITY STEERING INTERVENTIONS")
    logging.info("=" * 60)

    results = test_steering(
        model,
        steering_vectors,
        encoding,
        device,
        alphas=alphas,
        target_layers=target_layers,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
    )

    # Print overall results summary
    logging.info("\n" + "=" * 60)
    logging.info("OVERALL RESULTS SUMMARY")
    logging.info("=" * 60)

    for alpha in sorted(results.keys()):
        stats = results[alpha]
        logging.info(
            f"α = {alpha:+5.1f}: "
            f"{stats['major_percentage']:5.1f}% major, "
            f"{stats['minor_percentage']:5.1f}% minor "
            f"(conf: {stats['avg_confidence']:.3f}, "
            f"n={stats['total_confident']})"
        )

    # Verify steering effect
    if -1.0 in results and 0.0 in results and 1.0 in results:
        neg_major_pct = results[-1.0]["major_percentage"]
        baseline_major_pct = results[0.0]["major_percentage"]
        pos_major_pct = results[1.0]["major_percentage"]

        logging.info("\n" + "=" * 60)
        logging.info("STEERING EFFECT VERIFICATION")
        logging.info("=" * 60)
        logging.info(f"α = -1.0: {neg_major_pct:.1f}% major (steering towards major)")
        logging.info(f"α =  0.0: {baseline_major_pct:.1f}% major (baseline)")
        logging.info(f"α = +1.0: {pos_major_pct:.1f}% major (steering towards minor)")
        logging.info(f"\nShift from baseline:")
        logging.info(f"  α = -1.0: {neg_major_pct - baseline_major_pct:+.1f}% major")
        logging.info(f"  α = +1.0: {pos_major_pct - baseline_major_pct:+.1f}% major")

        # Check if steering works as expected
        if neg_major_pct > baseline_major_pct and pos_major_pct < baseline_major_pct:
            logging.info("\n✓ SUCCESS: Modality steering works as expected!")
            logging.info("  Negative α increases major, positive α increases minor.")
        elif neg_major_pct < baseline_major_pct and pos_major_pct > baseline_major_pct:
            logging.warning(
                "\n? WARNING: Steering direction is inverted (major ↔ minor flipped)"
            )
        elif neg_major_pct == baseline_major_pct == pos_major_pct:
            logging.warning("\n✗ WARNING: No steering effect detected (all equal)")
        else:
            logging.warning(
                "\n? WARNING: Partial or inconsistent steering effect detected"
            )
    else:
        logging.info("\nNot all test alphas present, skipping verification")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
