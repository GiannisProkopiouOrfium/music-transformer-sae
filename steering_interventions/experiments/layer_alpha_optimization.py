"""Layer and Alpha Optimization Experiment.

This script performs a grid search over layer groups and alpha values to find
the optimal configuration that:
1. Achieves target pitch manipulation (positive α → higher, negative α → lower)
2. Maintains model quality (pitch_class_entropy, scale_consistency, groove_consistency)

Usage:
    python experiments/layer_alpha_optimization.py \\
        --concept average_pitch \\
        --n_samples 50 \\
        --output_len 512 \\
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List

import muspy
import numpy as np
import torch
import tqdm

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors


# Layer groups to test
LAYER_GROUPS = {
    "single_0": [0],
    "single_1": [1],
    "single_2": [2],
    "single_3": [3],
    "single_4": [4],
    "single_5": [5],
    "single_6": [6],
    "single_7": [7],
    "single_8": [8],
    "single_9": [9],
    "single_10": [10],
    "single_11": [11],
    "early": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "late": [8, 9, 10, 11],
    "all": list(range(12)),
}

# Alpha values to test
ALPHA_VALUES = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "layer_alpha_optimization.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Layer and alpha optimization")
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept to evaluate (e.g., average_pitch)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples per configuration",
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
            return {"mean": 0.0, "std": 0.0, "n_notes": 0}

        return {
            "mean": float(np.mean(pitches)),
            "std": float(np.std(pitches)),
            "min": int(np.min(pitches)),
            "max": int(np.max(pitches)),
            "n_notes": len(pitches),
        }
    except Exception as e:
        logging.error(f"Error measuring pitch: {e}")
        return {"mean": 0.0, "std": 0.0, "n_notes": 0, "error": str(e)}


def evaluate_quality_metrics(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Evaluate objective quality metrics (from evaluate.py)."""
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


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline."""

    entropy_diff = abs(metrics["pitch_class_entropy"] - baseline["pitch_class_entropy"])
    scale_diff = max(0, baseline["scale_consistency"] - metrics["scale_consistency"])
    groove_diff = max(0, baseline["groove_consistency"] - metrics["groove_consistency"])

    total_degradation = entropy_diff + scale_diff + groove_diff

    return {
        "entropy_diff": float(entropy_diff),
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": float(total_degradation),
    }


def generate_samples(
    generator: SteeredGenerator,
    encoding: Dict,
    n_samples: int,
    output_len: int,
    alpha: float,
    target_layers: List[int],
    device: torch.device,
) -> List[np.ndarray]:
    """Generate samples with steering."""
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

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

        # Combine start + generated
        full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]
        samples.append(full_seq)

    return samples


def evaluate_configuration(
    generator: SteeredGenerator,
    encoding: Dict,
    layer_group: List[int],
    alpha: float,
    n_samples: int,
    output_len: int,
    device: torch.device,
) -> Dict:
    """Evaluate a single (layer_group, alpha) configuration."""
    # Generate samples
    samples = generate_samples(
        generator, encoding, n_samples, output_len, alpha, layer_group, device
    )

    # Measure pitch statistics
    pitch_stats = []
    for sample in samples:
        pitch_stats.append(measure_pitch_from_tokens(sample, encoding))

    # Measure quality metrics
    quality_metrics = []
    for sample in samples:
        quality_metrics.append(evaluate_quality_metrics(sample, encoding))

    # Aggregate results
    mean_pitch = np.nanmean([s["mean"] for s in pitch_stats])
    std_pitch = np.nanstd([s["mean"] for s in pitch_stats])

    mean_entropy = np.nanmean([m["pitch_class_entropy"] for m in quality_metrics])
    mean_scale = np.nanmean([m["scale_consistency"] for m in quality_metrics])
    mean_groove = np.nanmean([m["groove_consistency"] for m in quality_metrics])

    return {
        "mean_pitch": float(mean_pitch),
        "std_pitch": float(std_pitch),
        "pitch_class_entropy": float(mean_entropy),
        "scale_consistency": float(mean_scale),
        "groove_consistency": float(mean_groove),
        "n_samples": n_samples,
    }


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("LAYER AND ALPHA OPTIMIZATION EXPERIMENT")
    logging.info("=" * 80)
    logging.info(f"Concept: {args.concept}")
    logging.info(f"Samples per config: {args.n_samples}")
    logging.info(f"Output length: {args.output_len}")
    logging.info(f"Layer groups: {len(LAYER_GROUPS)}")
    logging.info(f"Alpha values: {len(ALPHA_VALUES)}")
    logging.info(f"Total configurations: {len(LAYER_GROUPS) * len(ALPHA_VALUES)}")

    # Setup device
    device = torch.device(f"cuda:{args.gpu}")
    logging.info(f"Using device: {device}")

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

    # Load steering vectors
    logging.info(f"Loading steering vectors from {args.steering_dir}")
    steering_vectors = load_steering_vectors(args.steering_dir, args.concept, device)

    # Create generator
    generator = SteeredGenerator(model, steering_vectors, encoding)

    # Storage for results
    results = defaultdict(lambda: defaultdict(dict))
    baseline_metrics = None

    # Run grid search
    total_configs = len(LAYER_GROUPS) * len(ALPHA_VALUES)
    with tqdm.tqdm(total=total_configs, desc="Evaluating configurations") as pbar:
        for layer_name, layer_group in LAYER_GROUPS.items():
            for alpha in ALPHA_VALUES:
                logging.info(f"\n{'='*60}")
                logging.info(f"Layer group: {layer_name} {layer_group}")
                logging.info(f"Alpha: {alpha}")
                logging.info(f"{'='*60}")

                # Evaluate configuration
                config_results = evaluate_configuration(
                    generator,
                    encoding,
                    layer_group,
                    alpha,
                    args.n_samples,
                    args.output_len,
                    device,
                )

                # Store results
                results[layer_name][alpha] = config_results

                # Store baseline (alpha=0, all layers) for degradation calculation
                if layer_name == "all" and alpha == 0.0:
                    baseline_metrics = {
                        "pitch_class_entropy": config_results["pitch_class_entropy"],
                        "scale_consistency": config_results["scale_consistency"],
                        "groove_consistency": config_results["groove_consistency"],
                    }
                    logging.info(f"Baseline metrics: {baseline_metrics}")

                logging.info(f"Mean pitch: {config_results['mean_pitch']:.2f}")
                logging.info(
                    f"Pitch class entropy: {config_results['pitch_class_entropy']:.3f}"
                )
                logging.info(
                    f"Scale consistency: {config_results['scale_consistency']:.2f}%"
                )
                logging.info(
                    f"Groove consistency: {config_results['groove_consistency']:.2f}%"
                )

                pbar.update(1)

    # Calculate degradation scores
    if baseline_metrics:
        for layer_name in results:
            for alpha in results[layer_name]:
                metrics = results[layer_name][alpha]
                degradation = calculate_degradation(metrics, baseline_metrics)
                results[layer_name][alpha]["degradation"] = degradation

    # Calculate pitch shifts from baseline
    for layer_name in results:
        if 0.0 in results[layer_name]:
            baseline_pitch = results[layer_name][0.0]["mean_pitch"]
            for alpha in results[layer_name]:
                pitch_shift = results[layer_name][alpha]["mean_pitch"] - baseline_pitch
                results[layer_name][alpha]["pitch_shift_from_baseline"] = float(
                    pitch_shift
                )

    # Save results
    output_file = args.output_dir / f"layer_alpha_grid_{args.concept}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "concept": args.concept,
                "n_samples": args.n_samples,
                "output_len": args.output_len,
                "layer_groups": {k: list(v) for k, v in LAYER_GROUPS.items()},
                "alpha_values": ALPHA_VALUES,
                "baseline_metrics": baseline_metrics,
                "results": results,
            },
            f,
            indent=2,
        )
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to {output_file}")
    logging.info(f"{'='*80}")

    # Print summary
    logging.info("\nBEST CONFIGURATIONS (by pitch control + quality):")
    logging.info("-" * 80)

    for layer_name in LAYER_GROUPS:
        for alpha in [a for a in ALPHA_VALUES if a != 0.0]:
            if (
                alpha in results[layer_name]
                and "degradation" in results[layer_name][alpha]
            ):
                r = results[layer_name][alpha]
                score = (
                    abs(r.get("pitch_shift_from_baseline", 0))
                    - r["degradation"]["total_degradation"]
                )
                logging.info(
                    f"{layer_name:12s} α={alpha:+.1f}: "
                    f"pitch_shift={r.get('pitch_shift_from_baseline', 0):+6.2f}, "
                    f"degradation={r['degradation']['total_degradation']:5.2f}, "
                    f"score={score:+6.2f}"
                )

    logging.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
