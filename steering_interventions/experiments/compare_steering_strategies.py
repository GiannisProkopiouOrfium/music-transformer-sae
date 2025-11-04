"""Compare All-to-All vs One-to-All Steering Strategies.

This script compares two steering approaches:
1. All-to-All: Each layer receives its own steering vector
2. One-to-All: A single "best" layer's vector is applied to all layers

Usage:
    python experiments/compare_steering_strategies.py \\
        --concept average_pitch \\
        --best_layers 3,6,9 \\
        --n_samples 50 \\
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

import muspy
import numpy as np
import torch
import tqdm

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import utils
import music_x_transformers
import representation
from steered_generator import SteeredGenerator, load_steering_vectors


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "compare_strategies.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare steering strategies")
    parser.add_argument(
        "--concept", type=str, required=True, help="Concept name (e.g., average_pitch)"
    )
    parser.add_argument(
        "--best_layers",
        type=str,
        default="3,6,9",
        help="Best layers to test for One-to-All (comma-separated)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Alpha values to test (comma-separated)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50, help="Number of samples per configuration"
    )
    parser.add_argument(
        "--output_len", type=int, default=512, help="Length of generated sequences"
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default=pathlib.Path("../exp/sod/model_best.pt"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--steering_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_vectors"),
        help="Directory containing steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results"),
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    return parser.parse_args()


def measure_pitch_from_tokens(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Measure pitch statistics from tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)

        if not pitches:
            return {"mean": 0.0, "std": 0.0, "n_notes": 0, "unique_pitches": 0}

        return {
            "mean": float(np.mean(pitches)),
            "std": float(np.std(pitches)),
            "min": int(np.min(pitches)),
            "max": int(np.max(pitches)),
            "n_notes": len(pitches),
            "unique_pitches": len(set(pitches)),
        }
    except Exception as e:
        logging.error(f"Error measuring pitch: {e}")
        return {
            "mean": 0.0,
            "std": 0.0,
            "n_notes": 0,
            "unique_pitches": 0,
            "error": str(e),
        }


def evaluate_quality_metrics(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Evaluate objective quality metrics."""
    try:
        music = representation.decode(tokens, encoding)
        music.trim(music.resolution * 64)

        if not music.tracks:
            return {
                "pitch_class_entropy": np.nan,
                "scale_consistency": np.nan,
                "groove_consistency": np.nan,
            }

        return {
            "pitch_class_entropy": muspy.pitch_class_entropy(music),
            "scale_consistency": muspy.scale_consistency(music),
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution),
        }
    except Exception as e:
        logging.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
            "error": str(e),
        }


def generate_with_strategy(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    strategy: str,
    best_layer: int,
    alpha: float,
    n_samples: int,
    output_len: int,
    device: torch.device,
) -> List[np.ndarray]:
    """Generate samples with specified strategy."""
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    # Prepare steering vectors based on strategy
    if strategy == "all_to_all":
        steering_dict = steering_vectors
        target_layers = None  # Use all layers
    elif strategy == "one_to_all":
        # Use only the best layer's vector for ALL layers
        best_vector = steering_vectors[best_layer]
        steering_dict = {layer: best_vector for layer in range(12)}
        target_layers = None  # Apply to all
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create generator
    generator = SteeredGenerator(model, steering_dict, encoding)

    samples = []
    for _ in range(n_samples):
        # Start token
        tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
        tgt_start[:, 0, 0] = sos

        # Generate
        generated = generator.generate(
            tgt_start,
            output_len,
            alpha=alpha,
            target_layers=target_layers,
            eos_token=eos,
            temperature=config.GENERATION_TEMPERATURE,
            filter_logits_fn=config.GENERATION_FILTER,
            filter_thres=config.GENERATION_FILTER_THRESHOLD,
            monotonicity_dim=("type", "beat"),
        )

        # Combine
        full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]
        samples.append(full_seq)

    return samples


def evaluate_samples(samples: List[np.ndarray], encoding: Dict) -> Dict:
    """Evaluate a list of samples."""
    pitch_stats = [measure_pitch_from_tokens(s, encoding) for s in samples]
    quality_metrics = [evaluate_quality_metrics(s, encoding) for s in samples]

    return {
        "pitch": {
            "mean": float(np.nanmean([p["mean"] for p in pitch_stats])),
            "std": float(np.nanstd([p["mean"] for p in pitch_stats])),
            "diversity": float(np.nanmean([p["unique_pitches"] for p in pitch_stats])),
        },
        "quality": {
            "pitch_class_entropy": float(
                np.nanmean([q["pitch_class_entropy"] for q in quality_metrics])
            ),
            "scale_consistency": float(
                np.nanmean([q["scale_consistency"] for q in quality_metrics])
            ),
            "groove_consistency": float(
                np.nanmean([q["groove_consistency"] for q in quality_metrics])
            ),
        },
        "n_samples": len(samples),
    }


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("STEERING STRATEGY COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Concept: {args.concept}")
    logging.info(f"Best layers for One-to-All: {args.best_layers}")
    logging.info(f"Alphas: {args.alphas}")
    logging.info(f"Samples per config: {args.n_samples}")

    # Parse parameters
    best_layers = [int(l) for l in args.best_layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    # Setup device
    device = torch.device(f"cuda:{args.gpu}")
    logging.info(f"Using device: {device}")

    # Load encoding
    logging.info(f"Loading encoding from {args.encoding_path}")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")

    # Load model
    logging.info(f"Loading model from {args.model_path}")
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

    # Load steering vectors
    logging.info(f"Loading steering vectors from {args.steering_dir}")
    steering_path = (
        config.OUTPUT_DIR / "steering_vectors" / f"{args.concept}_steering_vectors.pt"
    )
    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Storage for results
    results = {
        "all_to_all": {},
        "one_to_all": {layer: {} for layer in best_layers},
    }

    # Test All-to-All strategy
    logging.info("\n" + "=" * 80)
    logging.info("STRATEGY: ALL-TO-ALL")
    logging.info("=" * 80)

    for alpha in tqdm.tqdm(alphas, desc="All-to-All"):
        logging.info(f"\nAlpha: {alpha}")

        samples = generate_with_strategy(
            model,
            steering_vectors,
            encoding,
            "all_to_all",
            None,
            alpha,
            args.n_samples,
            args.output_len,
            device,
        )

        eval_results = evaluate_samples(samples, encoding)
        results["all_to_all"][alpha] = eval_results

        logging.info(f"  Mean pitch: {eval_results['pitch']['mean']:.2f}")
        logging.info(
            f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
        )

    # Test One-to-All strategy for each best layer
    for best_layer in best_layers:
        logging.info("\n" + "=" * 80)
        logging.info(f"STRATEGY: ONE-TO-ALL (Layer {best_layer})")
        logging.info("=" * 80)

        for alpha in tqdm.tqdm(alphas, desc=f"One-to-All (L{best_layer})"):
            logging.info(f"\nAlpha: {alpha}")

            samples = generate_with_strategy(
                model,
                steering_vectors,
                encoding,
                "one_to_all",
                best_layer,
                alpha,
                args.n_samples,
                args.output_len,
                device,
            )

            eval_results = evaluate_samples(samples, encoding)
            results["one_to_all"][best_layer][alpha] = eval_results

            logging.info(f"  Mean pitch: {eval_results['pitch']['mean']:.2f}")
            logging.info(
                f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
            )

    # Save results
    output_file = args.output_dir / f"strategy_comparison_{args.concept}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "concept": args.concept,
                "best_layers": best_layers,
                "alphas": alphas,
                "n_samples": args.n_samples,
                "results": results,
            },
            f,
            indent=2,
        )
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to {output_file}")

    # Print summary comparison
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY COMPARISON")
    logging.info("=" * 80)

    for alpha in alphas:
        if alpha == 0.0:
            continue

        logging.info(f"\nAlpha = {alpha:+.1f}:")

        # All-to-All
        ata_pitch = results["all_to_all"][alpha]["pitch"]["mean"]
        ata_entropy = results["all_to_all"][alpha]["quality"]["pitch_class_entropy"]
        logging.info(
            f"  All-to-All:        pitch={ata_pitch:6.2f}, entropy={ata_entropy:.3f}"
        )

        # One-to-All for each layer
        for layer in best_layers:
            ota_pitch = results["one_to_all"][layer][alpha]["pitch"]["mean"]
            ota_entropy = results["one_to_all"][layer][alpha]["quality"][
                "pitch_class_entropy"
            ]
            logging.info(
                f"  One-to-All (L{layer:2d}): pitch={ota_pitch:6.2f}, entropy={ota_entropy:.3f}"
            )

    logging.info("\nComparison complete!")


if __name__ == "__main__":
    main()
