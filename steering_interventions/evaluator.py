"""Step E: Evaluation of Steering Interventions.

This module:
1. Generates music with various alpha values
2. Measures actual metrics in generated outputs
3. Compares against ground truth and baseline (alpha=0)
4. Provides statistical validation
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors


def measure_velocity_from_tokens(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Measure velocity statistics from token sequence.

    Args:
        tokens: Token sequence (shape: seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with velocity statistics
    """
    # Decode to get the music object
    try:
        music = representation.decode(tokens, encoding)

        # Extract all velocities
        velocities = []
        for track in music.tracks:
            for note in track.notes:
                velocities.append(note.velocity)

        if not velocities:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "median": 0.0,
                "n_notes": 0,
            }

        return {
            "mean": float(np.mean(velocities)),
            "std": float(np.std(velocities)),
            "min": int(np.min(velocities)),
            "max": int(np.max(velocities)),
            "median": float(np.median(velocities)),
            "n_notes": len(velocities),
        }

    except Exception as e:
        logging.error(f"Error decoding tokens: {e}")
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "n_notes": 0,
            "error": str(e),
        }


def generate_and_evaluate(
    model: nn.Module,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    device: torch.device,
    n_samples: int,
    seq_len: int,
    alpha_values: List[float],
    target_layers: List[int] = None,
    prompt_tokens: torch.Tensor = None,
    output_dir: pathlib.Path = None,
) -> Dict[float, List[Dict]]:
    """Generate samples with different alpha values and evaluate them.

    Args:
        model: The model
        steering_vectors: Steering vectors
        encoding: Encoding dictionary
        device: Device to use
        n_samples: Number of samples per alpha
        seq_len: Generation length
        alpha_values: List of alpha values to test
        target_layers: Layers to intervene on
        prompt_tokens: Optional prompt for continuation (None = unconditioned)
        output_dir: Optional directory to save generated samples

    Returns:
        Dictionary mapping alpha -> list of metric dictionaries
    """
    generator = SteeredGenerator(model, steering_vectors, encoding)

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    results = {}

    for alpha in alpha_values:
        logging.info(f"\n{'='*60}")
        logging.info(f"Generating with alpha={alpha}")
        logging.info(f"{'='*60}")

        alpha_results = []

        # Create output directory for this alpha
        if output_dir is not None:
            alpha_dir = output_dir / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # Create start tokens
            if prompt_tokens is not None:
                # Use provided prompt for continuation
                tgt_start = prompt_tokens.clone().to(device)
            else:
                # Unconditioned generation
                tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
                tgt_start[:, 0, 0] = sos

            # Generate with steering
            generated = generator.generate(
                tgt_start,
                seq_len,
                alpha=alpha,
                target_layers=target_layers,
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Combine with start tokens
            full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]

            # Measure metrics
            metrics = measure_velocity_from_tokens(full_seq, encoding)
            metrics["alpha"] = alpha
            metrics["sample_idx"] = i

            alpha_results.append(metrics)

            # Save if output directory provided
            if output_dir is not None:
                # Save as NPY
                np.save(alpha_dir / f"sample_{i}.npy", full_seq)

                # Save as MIDI
                try:
                    music = representation.decode(full_seq, encoding)
                    music.write(alpha_dir / f"sample_{i}.mid")
                except Exception as e:
                    logging.error(f"Error saving MIDI: {e}")

            if (i + 1) % 10 == 0:
                logging.info(f"Generated {i+1}/{n_samples} samples")

        results[alpha] = alpha_results

        # Log summary for this alpha
        mean_velocities = [r["mean"] for r in alpha_results if r["n_notes"] > 0]
        if mean_velocities:
            logging.info(
                f"Alpha {alpha}: Mean velocity = {np.mean(mean_velocities):.2f} ± {np.std(mean_velocities):.2f}"
            )

    return results


def analyze_results(
    results: Dict[float, List[Dict]], concept: str = "velocity"
) -> Dict:
    """Analyze evaluation results and compute statistics.

    Args:
        results: Dictionary mapping alpha -> list of metric dicts
        concept: Concept being evaluated

    Returns:
        Analysis dictionary with statistics
    """
    analysis = {
        "concept": concept,
        "alpha_values": sorted(results.keys()),
        "summary": {},
        "statistics": {},
    }

    # Compute summary statistics for each alpha
    for alpha, alpha_results in results.items():
        valid_results = [r for r in alpha_results if r["n_notes"] > 0]

        if not valid_results:
            continue

        mean_values = [r["mean"] for r in valid_results]

        analysis["summary"][f"alpha_{alpha}"] = {
            "n_samples": len(valid_results),
            "mean_velocity": {
                "mean": float(np.mean(mean_values)),
                "std": float(np.std(mean_values)),
                "min": float(np.min(mean_values)),
                "max": float(np.max(mean_values)),
                "median": float(np.median(mean_values)),
            },
            "total_notes": sum(r["n_notes"] for r in valid_results),
        }

    # Statistical tests
    alpha_values = sorted(results.keys())

    # 1. Correlation between alpha and mean velocity
    alphas = []
    mean_velocities = []

    for alpha in alpha_values:
        for result in results[alpha]:
            if result["n_notes"] > 0:
                alphas.append(alpha)
                mean_velocities.append(result["mean"])

    if len(alphas) > 2:
        correlation, p_value = stats.pearsonr(alphas, mean_velocities)

        analysis["statistics"]["correlation"] = {
            "pearson_r": float(correlation),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "interpretation": (
                f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} "
                f"{'positive' if correlation > 0 else 'negative'} correlation "
                f"({'significant' if p_value < 0.05 else 'not significant'})"
            ),
        }

    # 2. Compare positive vs negative alpha
    positive_alphas = [a for a in alpha_values if a > 0]
    negative_alphas = [a for a in alpha_values if a < 0]
    zero_alpha = [a for a in alpha_values if a == 0]

    if positive_alphas and zero_alpha:
        pos_velocities = [
            r["mean"] for a in positive_alphas for r in results[a] if r["n_notes"] > 0
        ]
        zero_velocities = [
            r["mean"] for a in zero_alpha for r in results[a] if r["n_notes"] > 0
        ]

        if pos_velocities and zero_velocities:
            t_stat, p_val = stats.ttest_ind(pos_velocities, zero_velocities)

            analysis["statistics"]["positive_vs_baseline"] = {
                "mean_positive": float(np.mean(pos_velocities)),
                "mean_baseline": float(np.mean(zero_velocities)),
                "difference": float(np.mean(pos_velocities) - np.mean(zero_velocities)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
            }

    if negative_alphas and zero_alpha:
        neg_velocities = [
            r["mean"] for a in negative_alphas for r in results[a] if r["n_notes"] > 0
        ]
        zero_velocities = [
            r["mean"] for a in zero_alpha for r in results[a] if r["n_notes"] > 0
        ]

        if neg_velocities and zero_velocities:
            t_stat, p_val = stats.ttest_ind(neg_velocities, zero_velocities)

            analysis["statistics"]["negative_vs_baseline"] = {
                "mean_negative": float(np.mean(neg_velocities)),
                "mean_baseline": float(np.mean(zero_velocities)),
                "difference": float(np.mean(neg_velocities) - np.mean(zero_velocities)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
            }

    return analysis


def plot_results(
    results: Dict[float, List[Dict]],
    output_path: pathlib.Path,
    concept: str = "velocity",
):
    """Plot evaluation results.

    Args:
        results: Dictionary mapping alpha -> list of metric dicts
        output_path: Path to save plot
        concept: Concept being evaluated
    """
    alpha_values = sorted(results.keys())

    # Collect mean velocities for each alpha
    means = []
    stds = []

    for alpha in alpha_values:
        valid_results = [r for r in results[alpha] if r["n_notes"] > 0]
        if valid_results:
            mean_vals = [r["mean"] for r in valid_results]
            means.append(np.mean(mean_vals))
            stds.append(np.std(mean_vals))
        else:
            means.append(0)
            stds.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        alpha_values, means, yerr=stds, marker="o", capsize=5, capthick=2, linewidth=2
    )
    ax.axhline(
        y=means[alpha_values.index(0.0)] if 0.0 in alpha_values else 0,
        color="r",
        linestyle="--",
        label="Baseline (α=0)",
    )
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel(f"Mean {concept.capitalize()}", fontsize=12)
    ax.set_title(
        f"Effect of Steering on {concept.capitalize()}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved plot to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate steering interventions")
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to evaluate"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None, help="Model checkpoint"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=config.N_GENERATION_SAMPLES,
        help="Samples per alpha value",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=config.GENERATION_SEQ_LEN,
        help="Generation length",
    )
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=config.ALPHA_VALUES,
        help="Alpha values to test",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (e.g., 0, 1). If not specified, uses CPU. Automatically detects CUDA or MPS.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.gpu is not None:
        # User specified a GPU
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logging.warning(
                f"CUDA/MPS not available, falling back to CPU (requested GPU {args.gpu})"
            )
    else:
        # No GPU specified, use CPU
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU specified)")

    logging.info(f"Using device: {device}")

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

    logging.info("Model loaded successfully")

    # Generate and evaluate
    logging.info(
        f"\nGenerating {args.n_samples} samples for each of {len(args.alpha_values)} alpha values"
    )

    results = generate_and_evaluate(
        model,
        steering_vectors,
        encoding,
        device,
        args.n_samples,
        args.seq_len,
        args.alpha_values,
        target_layers=config.TARGET_LAYERS,
        prompt_tokens=None,
        output_dir=args.output_dir / "samples",
    )

    # Analyze results
    logging.info("\nAnalyzing results...")
    analysis = analyze_results(results, args.concept)

    # Save analysis
    analysis_file = args.output_dir / f"{args.concept}_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logging.info(f"Saved analysis to: {analysis_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "correlation" in analysis["statistics"]:
        corr = analysis["statistics"]["correlation"]
        print(f"\nCorrelation: {corr['interpretation']}")
        print(f"  Pearson r = {corr['pearson_r']:.4f}, p = {corr['p_value']:.4f}")

    if "positive_vs_baseline" in analysis["statistics"]:
        pos = analysis["statistics"]["positive_vs_baseline"]
        print(f"\nPositive α vs Baseline:")
        print(f"  Mean positive: {pos['mean_positive']:.2f}")
        print(f"  Mean baseline: {pos['mean_baseline']:.2f}")
        print(
            f"  Difference: {pos['difference']:.2f} ({'significant' if pos['significant'] else 'not significant'})"
        )

    if "negative_vs_baseline" in analysis["statistics"]:
        neg = analysis["statistics"]["negative_vs_baseline"]
        print(f"\nNegative α vs Baseline:")
        print(f"  Mean negative: {neg['mean_negative']:.2f}")
        print(f"  Mean baseline: {neg['mean_baseline']:.2f}")
        print(
            f"  Difference: {neg['difference']:.2f} ({'significant' if neg['significant'] else 'not significant'})"
        )

    print("=" * 60 + "\n")

    # Plot results
    plot_path = args.output_dir / f"{args.concept}_plot.png"
    plot_results(results, plot_path, args.concept)

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    import torch.nn as nn

    main()
