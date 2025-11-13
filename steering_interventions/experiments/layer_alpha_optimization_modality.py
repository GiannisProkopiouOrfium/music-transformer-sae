"""Layer and Alpha Optimization Experiment for Modality.

This script performs a grid search over layer groups and alpha values to find
the optimal configuration that:
1. Achieves target modality manipulation (positive α → major, negative α → minor)
2. Maintains model quality (pitch_class_entropy, scale_consistency, groove_consistency)

Usage:
    python experiments/layer_alpha_optimization_modality.py \\
        --n_samples 30 \\
        --output_len 512 \\
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

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

# Import music21 for modality detection
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)


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
ALPHA_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "layer_alpha_optimization_modality.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Layer and alpha optimization for modality"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of samples per configuration",
    )
    parser.add_argument(
        "--output_len", type=int, default=512, help="Length of generated sequences"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/experiments/results"),
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    return parser.parse_args()


def detect_modality_from_tokens(
    tokens: np.ndarray, encoding: Dict
) -> Tuple[str, float]:
    """Detect modality from tokens."""
    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        for token in tokens:
            if token[0] == note_type:
                pitch = token[3]
                pitches.append(pitch)

        if len(pitches) < 10:
            return ("unknown", 0.0)

        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")
        return (key.mode, key.correlationCoefficient)

    except Exception as e:
        logging.warning(f"Modality detection failed: {e}")
        return ("unknown", 0.0)


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

    # Detect modality for each sample
    modality_results = []
    for sample in samples:
        mode, confidence = detect_modality_from_tokens(sample, encoding)
        modality_results.append({"mode": mode, "confidence": confidence})

    # Measure quality metrics
    quality_metrics = []
    for sample in samples:
        quality_metrics.append(evaluate_quality_metrics(sample, encoding))

    # Calculate modality statistics
    confident_modes = [r for r in modality_results if r["confidence"] >= 0.5]

    if confident_modes:
        major_count = sum(1 for r in confident_modes if r["mode"] == "major")
        minor_count = sum(1 for r in confident_modes if r["mode"] == "minor")
        total = len(confident_modes)

        major_pct = 100 * major_count / total if total > 0 else 0
        minor_pct = 100 * minor_count / total if total > 0 else 0
        avg_confidence = np.mean([r["confidence"] for r in confident_modes])
    else:
        major_pct = 0
        minor_pct = 0
        avg_confidence = 0

    # Aggregate quality metrics
    mean_entropy = np.nanmean([m["pitch_class_entropy"] for m in quality_metrics])
    mean_scale = np.nanmean([m["scale_consistency"] for m in quality_metrics])
    mean_groove = np.nanmean([m["groove_consistency"] for m in quality_metrics])

    return {
        "major_percentage": float(major_pct),
        "minor_percentage": float(minor_pct),
        "avg_confidence": float(avg_confidence),
        "n_confident": len(confident_modes),
        "n_samples": n_samples,
        "pitch_class_entropy": float(mean_entropy),
        "scale_consistency": float(mean_scale),
        "groove_consistency": float(mean_groove),
    }


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("LAYER AND ALPHA OPTIMIZATION EXPERIMENT - MODALITY")
    logging.info("=" * 80)
    logging.info(f"Samples per config: {args.n_samples}")
    logging.info(f"Output length: {args.output_len}")
    logging.info(f"Layer groups: {len(LAYER_GROUPS)}")
    logging.info(f"Alpha values: {len(ALPHA_VALUES)}")
    logging.info(f"Total configurations: {len(LAYER_GROUPS) * len(ALPHA_VALUES)}")

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
    if args.steering_vectors is None:
        steering_path = pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    logging.info(f"Loading steering vectors from {steering_path}")
    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Create generator
    generator = SteeredGenerator(model, steering_vectors, encoding)

    # Storage for results
    results = defaultdict(lambda: defaultdict(dict))
    baseline_metrics = None
    baseline_major_pct = None

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

                # Store baseline (alpha=0, all layers)
                if layer_name == "all" and alpha == 0.0:
                    baseline_metrics = {
                        "pitch_class_entropy": config_results["pitch_class_entropy"],
                        "scale_consistency": config_results["scale_consistency"],
                        "groove_consistency": config_results["groove_consistency"],
                    }
                    baseline_major_pct = config_results["major_percentage"]
                    logging.info(f"Baseline metrics: {baseline_metrics}")
                    logging.info(f"Baseline major%: {baseline_major_pct:.1f}%")

                logging.info(
                    f"Major: {config_results['major_percentage']:.1f}%, Minor: {config_results['minor_percentage']:.1f}%"
                )
                logging.info(f"Avg confidence: {config_results['avg_confidence']:.3f}")
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

    # Calculate degradation scores and modality shifts
    if baseline_metrics and baseline_major_pct is not None:
        for layer_name in results:
            for alpha in results[layer_name]:
                metrics = results[layer_name][alpha]

                # Quality degradation
                degradation = calculate_degradation(metrics, baseline_metrics)
                results[layer_name][alpha]["degradation"] = degradation

                # Modality shift from baseline
                major_shift = metrics["major_percentage"] - baseline_major_pct
                results[layer_name][alpha]["major_shift_from_baseline"] = float(
                    major_shift
                )

    # Save results
    output_file = args.output_dir / "layer_alpha_grid_modality.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "concept": "modality",
                "n_samples": args.n_samples,
                "output_len": args.output_len,
                "layer_groups": {k: list(v) for k, v in LAYER_GROUPS.items()},
                "alpha_values": ALPHA_VALUES,
                "baseline_metrics": baseline_metrics,
                "baseline_major_pct": baseline_major_pct,
                "results": results,
            },
            f,
            indent=2,
        )
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to {output_file}")
    logging.info(f"{'='*80}")

    # Print summary
    logging.info("\nBEST CONFIGURATIONS (by modality control + quality):")
    logging.info("-" * 80)
    logging.info(f"Baseline: {baseline_major_pct:.1f}% major")
    logging.info("")

    for layer_name in LAYER_GROUPS:
        for alpha in [a for a in ALPHA_VALUES if a != 0.0]:
            if (
                alpha in results[layer_name]
                and "degradation" in results[layer_name][alpha]
            ):
                r = results[layer_name][alpha]
                shift = r.get("major_shift_from_baseline", 0)

                # Score: correct directional shift minus degradation
                # Positive α should increase major%, negative α should decrease major%
                if alpha > 0:
                    # Want positive shift (toward major)
                    score = shift - r["degradation"]["total_degradation"]
                else:
                    # Want negative shift (toward minor)
                    score = -shift - r["degradation"]["total_degradation"]

                logging.info(
                    f"{layer_name:12s} α={alpha:+.1f}: "
                    f"major={r['major_percentage']:5.1f}% (shift={shift:+6.1f}%), "
                    f"degradation={r['degradation']['total_degradation']:5.2f}, "
                    f"score={score:+6.2f}"
                )

    logging.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
