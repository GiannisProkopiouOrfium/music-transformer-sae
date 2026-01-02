#!/usr/bin/env python3
"""Phase 3: Dual-Steering Grid Search for Pitch + Duration.

Tests all three composition strategies (direct, gram_schmidt_duration, gram_schmidt_pitch)
with a symmetric alpha grid to map the complete steering landscape.

Test matrix:
- Strategies: direct, gram_schmidt_duration, gram_schmidt_pitch (3 strategies)
- Alpha pitch: [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] (13 values)
- Alpha duration: [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] (13 values)
- Total configs: 3 * 13 * 13 = 507 configs
- Samples per config: 5
- Total generations: 507 * 5 = 2,535 samples (~8-10 hours on single GPU)

Output:
- Generated MIDI files with comprehensive metrics
- JSON with detailed metrics per config
- Heatmap visualizations (pitch, duration, quality)
- Pareto frontier and interaction analyses

Usage:
    python dual_steering/test_multi_steering.py \\
        --model_checkpoint path/to/model.ckpt \\
        --pitch_vectors outputs/steering_vectors/average_pitch_steering_vectors.pt \\
        --duration_vectors outputs/steering_vectors/average_duration_steering_vectors.pt \\
        --output_dir steering_interventions/dual_steering/outputs/phase3_grid_search \\
        --n_samples 5 \\
        --strategies direct gram_schmidt_duration gram_schmidt_pitch
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import muspy
import representation
import utils
from multi_steered_generator import MultiSteeringGenerator
from vector_composition import VectorComposer

# Note: Duration measurement uses token[4] directly, no music21 needed

# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


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


def measure_duration_from_tokens(tokens: np.ndarray, encoding: dict) -> Dict:
    """Measure duration statistics from tokens.

    Duration is stored in token[4] (5th dimension) representing note length in ticks.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with duration statistics: {mean, std, min, max, values}
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        durations = []

        for token in tokens:
            if token[0] == note_type:  # Only extract from note tokens
                duration = token[4]  # Duration in ticks
                durations.append(duration)

        if len(durations) < 5:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "values": [],
            }

        return {
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
            "values": durations,
        }

    except Exception as e:
        logging.error(f"Error measuring duration: {e}")
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "values": [],
        }


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> Dict:
    """Evaluate objective quality metrics.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with quality metrics
    """
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
            "scale_consistency": muspy.scale_consistency(music) * 100,
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution)
            * 100,
        }
    except Exception as e:
        logging.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline.

    Args:
        metrics: Current quality metrics
        baseline: Baseline quality metrics (GROUND_TRUTH_METRICS)

    Returns:
        Degradation scores
    """
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


def generate_and_evaluate(
    model,
    composer: VectorComposer,
    strategy: str,
    alpha_pitch: float,
    alpha_duration: float,
    encoding: dict,
    device: torch.device,
    n_samples: int = 5,
    seq_len: int = 512,
) -> Dict:
    """Generate samples and evaluate for one configuration.

    Args:
        model: Transformer model
        composer: VectorComposer instance
        strategy: Composition strategy ("direct", "gram_schmidt_duration", or "gram_schmidt_pitch")
        alpha_pitch: Pitch scaling factor
        alpha_duration: Duration scaling factor
        encoding: Encoding dictionary
        device: Torch device
        n_samples: Number of samples to generate
        seq_len: Target sequence length

    Returns:
        Dictionary with aggregated metrics
    """
    # Create generator directly with composer (avoid reloading vectors)
    from multi_steered_generator import MultiSteeringGenerator

    generator = MultiSteeringGenerator(
        model=model,
        composer=composer,
        alpha_pitch=alpha_pitch,
        alpha_duration=alpha_duration,
        strategy=strategy,
        intervention_position="last",
    )

    # Determine scenario folder based on alpha signs
    if alpha_pitch > 0 and alpha_duration > 0:
        scenario = "high_long"
    elif alpha_pitch > 0 and alpha_duration < 0:
        scenario = "high_short"
    elif alpha_pitch < 0 and alpha_duration < 0:
        scenario = "low_short"
    elif alpha_pitch < 0 and alpha_duration > 0:
        scenario = "low_long"
    else:
        scenario = "neutral"

    # Dictionary to store per-sample metrics
    sample_metrics = []

    # Generate samples
    all_pitch_means = []
    all_duration_stats = []  # List of duration stat dicts
    all_quality_metrics = []
    all_degradations = []
    valid_count = 0
    tokens_list = []

    for i in range(n_samples):
        # Create empty primer
        primer = torch.zeros((1, 1, 6), dtype=torch.long, device=device)

        # Generate
        try:
            output = generator.generate(
                primer,
                seq_len,
                eos_token=None,
                temperature=1.0,
                filter_logits_fn="top_k",
                filter_thres=0.9,
            )

            # Combine start and generated
            full_seq = torch.cat((primer, output), 1).cpu().numpy()[0]

            # Save npy file with the necessary data
            save_dir = pathlib.Path(f"flamingo_exp/dual/unconditional/{scenario}")
            save_dir.mkdir(parents=True, exist_ok=True)
            sample_filepath = (
                save_dir / f"sample_alpha_p{alpha_pitch}_d{alpha_duration}_num_{i}.npy"
            )
            np.save(sample_filepath, full_seq)

            # Convert to numpy
            tokens = output[0].cpu().numpy()
            tokens_list.append(tokens)

            # Extract pitches
            pitches = extract_pitches_from_tokens(tokens, encoding)

            if len(pitches) >= 10:
                valid_count += 1

                # Pitch statistics
                pitch_mean = float(np.mean(pitches))
                all_pitch_means.append(pitch_mean)

                # Measure duration statistics
                duration_stats = measure_duration_from_tokens(tokens, encoding)
                all_duration_stats.append(duration_stats)

                # Evaluate quality metrics
                quality = evaluate_quality_metrics(tokens, encoding)
                all_quality_metrics.append(quality)

                # Calculate degradation from ground truth
                if not np.isnan(quality["pitch_class_entropy"]):
                    degradation = calculate_degradation(quality, GROUND_TRUTH_METRICS)
                    all_degradations.append(degradation)

                # Store per-sample metrics
                sample_metrics.append(
                    {
                        "filepath": str(sample_filepath),
                        "strategy": strategy,
                        "alpha_pitch": alpha_pitch,
                        "alpha_duration": alpha_duration,
                        "sample_num": i,
                        "mean_pitch": pitch_mean,
                        "mean_duration": duration_stats["mean"],
                        "duration_std": duration_stats["std"],
                        "n_notes": len(pitches),
                        "pitch_class_entropy": quality.get(
                            "pitch_class_entropy", np.nan
                        ),
                        "scale_consistency": quality.get("scale_consistency", np.nan),
                        "groove_consistency": quality.get("groove_consistency", np.nan),
                    }
                )

        except Exception as e:
            logging.error(f"Generation failed for sample {i}: {e}")
            continue

    # Save sample metrics to JSON file
    if sample_metrics:
        metrics_file = (
            save_dir / f"sample_metrics_p{alpha_pitch}_d{alpha_duration}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(sample_metrics, f, indent=2)
        logging.info(f"Saved sample metrics to {metrics_file}")

    # Aggregate metrics
    if valid_count == 0:
        return {
            "strategy": strategy,
            "alpha_pitch": alpha_pitch,
            "alpha_duration": alpha_duration,
            "valid_samples": 0,
            "pitch_control": {"mean": 0.0, "std": 0.0},
            "duration_control": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "quality_metrics": {
                "pitch_class_entropy": {"mean": np.nan, "std": np.nan},
                "scale_consistency": {"mean": np.nan, "std": np.nan},
                "groove_consistency": {"mean": np.nan, "std": np.nan},
            },
            "degradation": {
                "entropy_diff": {"mean": np.nan, "std": np.nan},
                "scale_diff": {"mean": np.nan, "std": np.nan},
                "groove_diff": {"mean": np.nan, "std": np.nan},
                "total_degradation": {"mean": np.nan, "std": np.nan},
            },
            "success_rate": 0.0,
        }

    # Aggregate duration statistics across all samples
    if all_duration_stats:
        duration_means = [d["mean"] for d in all_duration_stats if d["mean"] > 0]
        duration_stds = [d["std"] for d in all_duration_stats if d["std"] > 0]
        duration_mins = [d["min"] for d in all_duration_stats if d["min"] > 0]
        duration_maxs = [d["max"] for d in all_duration_stats if d["max"] > 0]

        duration_summary = {
            "mean": float(np.mean(duration_means)) if duration_means else 0.0,
            "std": float(np.mean(duration_stds)) if duration_stds else 0.0,
            "min": float(np.mean(duration_mins)) if duration_mins else 0.0,
            "max": float(np.mean(duration_maxs)) if duration_maxs else 0.0,
        }
    else:
        duration_summary = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    # Aggregate quality metrics
    quality_summary = {}
    for metric in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        values = [
            q[metric]
            for q in all_quality_metrics
            if not np.isnan(q.get(metric, np.nan))
        ]
        quality_summary[metric] = {
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
        }

    # Aggregate degradation
    degradation_summary = {}
    for metric in ["entropy_diff", "scale_diff", "groove_diff", "total_degradation"]:
        values = [d[metric] for d in all_degradations if not np.isnan(d[metric])]
        degradation_summary[metric] = {
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
        }

    return {
        "strategy": strategy,
        "alpha_pitch": alpha_pitch,
        "alpha_duration": alpha_duration,
        "valid_samples": valid_count,
        "pitch_control": {
            "mean": float(np.mean(all_pitch_means)),
            "std": float(np.std(all_pitch_means)),
        },
        "duration_control": duration_summary,
        "quality_metrics": quality_summary,
        "degradation": degradation_summary,
        "success_rate": valid_count / n_samples,
        "tokens": [t.tolist() for t in tokens_list],  # For later analysis
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Test dual-steering strategies"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to model checkpoint (defaults to best_model.pt)",
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "outputs/steering_vectors/average_pitch_steering_vectors.pt"
        ),
        help="Path to pitch steering vectors",
    )
    parser.add_argument(
        "--duration_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "outputs/steering_vectors/average_duration_steering_vectors.pt"
        ),
        help="Path to duration steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase3_grid_search"
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Samples per configuration"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--alphas_pitch",
        type=float,
        nargs="+",
        default=[
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
        ],
        help="Pitch alpha values to test",
    )
    parser.add_argument(
        "--alphas_duration",
        type=float,
        nargs="+",
        default=[
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
        ],
        help="Duration alpha values to test",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["direct", "gram_schmidt_duration", "gram_schmidt_pitch"],
        help="Strategies to test",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (0, 1, etc.)",
    )

    args = parser.parse_args()

    # Create output directory first (needed for log file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "phase3_grid_search.log"),
            logging.StreamHandler(),
        ],
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

    # Load model
    logging.info(f"Loading model from {args.model_checkpoint}")
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

    if args.model_checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.model_checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Create composer
    logging.info("Creating vector composer")
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(
        str(args.pitch_vectors), str(args.duration_vectors)
    )

    # Generate test matrix
    test_configs = []
    for strategy in args.strategies:
        for alpha_pitch in args.alphas_pitch:
            for alpha_duration in args.alphas_duration:
                test_configs.append((strategy, alpha_pitch, alpha_duration))

    logging.info(f"Testing {len(test_configs)} configurations")
    logging.info(
        f"Total generations: {len(test_configs)} configs × {args.n_samples} samples = {len(test_configs) * args.n_samples}"
    )

    # Run experiments
    results = []
    start_time = time.time()

    for strategy, alpha_pitch, alpha_duration in tqdm(
        test_configs, desc="Testing configurations"
    ):
        logging.info(
            f"\nTesting: strategy={strategy}, α_pitch={alpha_pitch}, α_duration={alpha_duration}"
        )

        result = generate_and_evaluate(
            model=model,
            composer=composer,
            strategy=strategy,
            alpha_pitch=alpha_pitch,
            alpha_duration=alpha_duration,
            encoding=encoding,
            device=device,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
        )

        results.append(result)

        # Log immediate results
        logging.info(
            f"  Valid samples: {result['valid_samples']}/{args.n_samples} "
            f"({result['success_rate']*100:.1f}%)"
        )
        if result["valid_samples"] > 0:
            logging.info(
                f"  Pitch mean: {result['pitch_control']['mean']:.2f} ± {result['pitch_control']['std']:.2f}"
            )
            logging.info(
                f"  Duration mean: {result['duration_control']['mean']:.1f} ticks (std: {result['duration_control']['std']:.1f})"
            )
            logging.info(
                f"  Quality degradation: {result['degradation']['total_degradation']['mean']:.2f}"
            )

    elapsed_time = time.time() - start_time

    # Save results
    results_file = args.output_dir / "phase3_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "n_samples": args.n_samples,
                    "seq_len": args.seq_len,
                    "alphas_pitch": args.alphas_pitch,
                    "alphas_duration": args.alphas_duration,
                    "strategies": args.strategies,
                    "total_configs": len(test_configs),
                    "elapsed_time_seconds": elapsed_time,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    logging.info(f"\nSaved results to: {results_file}")
    logging.info(
        f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/len(test_configs):.1f}s per config)"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 3: PITCH+DURATION GRID SEARCH COMPLETE")
    print("=" * 80)
    print(
        f"\nTested {len(test_configs)} configurations in {elapsed_time/60:.1f} minutes"
    )
    print(f"\nResults saved to: {results_file}")
    print("\nNext step: Analyze results to identify best strategy and alpha ranges")
    print("=" * 80)


if __name__ == "__main__":
    main()
