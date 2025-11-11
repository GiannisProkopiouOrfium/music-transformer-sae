"""Evaluation of Modality Steering Interventions.

This module:
1. Generates music with various alpha values
2. Detects modality (major/minor) in generated outputs
3. Performs statistical analysis using chi-square tests
4. Validates steering effectiveness
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

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


def generate_and_evaluate(
    model: nn.Module,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    device: torch.device,
    n_samples: int,
    seq_len: int,
    alpha_values: List[float],
    target_layers: List[int] = None,
    output_dir: pathlib.Path = None,
) -> Dict[float, List[Dict]]:
    """Generate samples with different alpha values and evaluate modality.

    Args:
        model: The model
        steering_vectors: Steering vectors
        encoding: Encoding dictionary
        device: Device to use
        n_samples: Number of samples per alpha
        seq_len: Generation length
        alpha_values: List of alpha values to test
        target_layers: Layers to intervene on
        output_dir: Optional directory to save generated samples

    Returns:
        Dictionary mapping alpha -> list of result dictionaries
    """
    generator = SteeredGenerator(model, steering_vectors, encoding)

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    results = {}

    for alpha in alpha_values:
        logging.info(f"\n{'='*60}")
        logging.info(f"Generating with α = {alpha}")
        logging.info(f"{'='*60}")

        alpha_results = []

        if output_dir is not None:
            alpha_dir = output_dir / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # Create start tokens
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

            full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]

            # Detect modality
            mode, confidence = extract_modality_from_tokens(full_seq, encoding)

            result = {
                "alpha": alpha,
                "sample_idx": i,
                "mode": mode,
                "confidence": confidence,
                "is_major": mode == "major",
                "is_minor": mode == "minor",
                "is_confident": confidence >= 0.5,
            }

            alpha_results.append(result)

            # Save if output directory provided
            if output_dir is not None:
                np.save(alpha_dir / f"sample_{i}.npy", full_seq)

                try:
                    music = representation.decode(full_seq, encoding)
                    music.write(str(alpha_dir / f"sample_{i}.mid"))
                    # music.write_audio(str(alpha_dir / f"sample_{i}.wav"))
                except Exception as e:
                    logging.error(f"Error saving MIDI: {e}")

            if (i + 1) % 10 == 0:
                logging.info(f"Generated {i+1}/{n_samples} samples")

        results[alpha] = alpha_results

        # Log summary
        confident = [r for r in alpha_results if r["is_confident"]]
        if confident:
            major_pct = 100 * sum(r["is_major"] for r in confident) / len(confident)
            logging.info(
                f"α = {alpha}: {major_pct:.1f}% major, {100-major_pct:.1f}% minor (n={len(confident)})"
            )

    return results


def analyze_results(results: Dict[float, List[Dict]]) -> Dict:
    """Analyze evaluation results with categorical statistics.

    Args:
        results: Dictionary mapping alpha -> list of result dicts

    Returns:
        Analysis dictionary with statistics
    """
    analysis = {
        "concept": "modality",
        "alpha_values": sorted(results.keys()),
        "summary": {},
        "statistics": {},
    }

    # Compute summary for each alpha
    for alpha, alpha_results in results.items():
        confident = [r for r in alpha_results if r["is_confident"]]
        all_results = alpha_results

        if confident:
            major_count = sum(r["is_major"] for r in confident)
            minor_count = sum(r["is_minor"] for r in confident)
            total = len(confident)

            analysis["summary"][f"alpha_{alpha}"] = {
                "n_samples": len(all_results),
                "n_confident": total,
                "n_low_confidence": len(all_results) - total,
                "major_count": major_count,
                "minor_count": minor_count,
                "major_percentage": 100 * major_count / total,
                "minor_percentage": 100 * minor_count / total,
                "avg_confidence": float(np.mean([r["confidence"] for r in confident])),
            }

    # Statistical tests
    alpha_values = sorted(results.keys())

    # 1. Chi-square test: Test if modality distribution differs across alphas
    if len(alpha_values) >= 2:
        # Build contingency table [major_counts, minor_counts] for each alpha
        contingency_table = []
        alpha_labels = []

        for alpha in alpha_values:
            confident = [r for r in results[alpha] if r["is_confident"]]
            if len(confident) > 0:
                major_count = sum(r["is_major"] for r in confident)
                minor_count = sum(r["is_minor"] for r in confident)
                contingency_table.append([major_count, minor_count])
                alpha_labels.append(alpha)

        if len(contingency_table) >= 2:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            analysis["statistics"]["chi_square_overall"] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "significant": bool(p_value < 0.05),
                "interpretation": (
                    f"Modality distribution {'differs significantly' if p_value < 0.05 else 'does not differ significantly'} "
                    f"across alpha values (p={'<0.001' if p_value < 0.001 else f'={p_value:.4f}'})"
                ),
            }

    # 2. Proportion test: Compare negative vs baseline
    if any(a < 0 for a in alpha_values) and 0.0 in alpha_values:
        negative_alphas = [a for a in alpha_values if a < 0]

        for neg_alpha in negative_alphas:
            neg_confident = [r for r in results[neg_alpha] if r["is_confident"]]
            zero_confident = [r for r in results[0.0] if r["is_confident"]]

            if len(neg_confident) > 0 and len(zero_confident) > 0:
                neg_major = sum(r["is_major"] for r in neg_confident)
                zero_major = sum(r["is_major"] for r in zero_confident)

                # Two-proportion z-test
                n1, n2 = len(neg_confident), len(zero_confident)
                p1, p2 = neg_major / n1, zero_major / n2
                p_pooled = (neg_major + zero_major) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

                if se > 0:
                    z_stat = (p1 - p2) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                    analysis["statistics"][f"negative_{neg_alpha}_vs_baseline"] = {
                        "alpha": neg_alpha,
                        "neg_major_pct": 100 * p1,
                        "baseline_major_pct": 100 * p2,
                        "difference": 100 * (p1 - p2),
                        "z_statistic": float(z_stat),
                        "p_value": float(p_val),
                        "significant": bool(p_val < 0.05),
                    }

    # 3. Proportion test: Compare positive vs baseline
    if any(a > 0 for a in alpha_values) and 0.0 in alpha_values:
        positive_alphas = [a for a in alpha_values if a > 0]

        for pos_alpha in positive_alphas:
            pos_confident = [r for r in results[pos_alpha] if r["is_confident"]]
            zero_confident = [r for r in results[0.0] if r["is_confident"]]

            if len(pos_confident) > 0 and len(zero_confident) > 0:
                pos_major = sum(r["is_major"] for r in pos_confident)
                zero_major = sum(r["is_major"] for r in zero_confident)

                n1, n2 = len(pos_confident), len(zero_confident)
                p1, p2 = pos_major / n1, zero_major / n2
                p_pooled = (pos_major + zero_major) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

                if se > 0:
                    z_stat = (p1 - p2) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                    analysis["statistics"][f"positive_{pos_alpha}_vs_baseline"] = {
                        "alpha": pos_alpha,
                        "pos_major_pct": 100 * p1,
                        "baseline_major_pct": 100 * p2,
                        "difference": 100 * (p1 - p2),
                        "z_statistic": float(z_stat),
                        "p_value": float(p_val),
                        "significant": bool(p_val < 0.05),
                    }

    return analysis


def plot_results(results: Dict[float, List[Dict]], output_path: pathlib.Path):
    """Plot evaluation results.

    Args:
        results: Dictionary mapping alpha -> list of result dicts
        output_path: Path to save plot
    """
    alpha_values = sorted(results.keys())

    major_percentages = []
    minor_percentages = []

    for alpha in alpha_values:
        confident = [r for r in results[alpha] if r["is_confident"]]
        if confident:
            major_pct = 100 * sum(r["is_major"] for r in confident) / len(confident)
            major_percentages.append(major_pct)
            minor_percentages.append(100 - major_pct)
        else:
            major_percentages.append(0)
            minor_percentages.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        alpha_values,
        major_percentages,
        marker="o",
        linewidth=2,
        label="Major",
        color="blue",
    )
    ax.plot(
        alpha_values,
        minor_percentages,
        marker="s",
        linewidth=2,
        label="Minor",
        color="red",
    )
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% (balanced)")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(
        "Effect of Steering on Modality (Major vs Minor)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved plot to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate modality steering interventions"
    )
    parser.add_argument(
        "--concept", type=str, default="modality", help="Concept (should be modality)"
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
        "--n_samples", type=int, default=30, help="Samples per alpha value"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--alpha_values",
        type=str,
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number to use")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.output_dir is None:
        args.output_dir = pathlib.Path(
            "steering_interventions/modality/outputs/evaluation"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Parse alpha values
    alpha_values = [float(a.strip()) for a in args.alpha_values.split(",")]

    # Generate and evaluate
    logging.info(
        f"\nGenerating {args.n_samples} samples for each of {len(alpha_values)} alpha values"
    )

    results = generate_and_evaluate(
        model,
        steering_vectors,
        encoding,
        device,
        args.n_samples,
        args.seq_len,
        alpha_values,
        target_layers=config.TARGET_LAYERS,
        output_dir=args.output_dir / "samples",
    )

    # Analyze results
    logging.info("\nAnalyzing results...")
    analysis = analyze_results(results)

    # Save analysis
    analysis_file = args.output_dir / "modality_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logging.info(f"Saved analysis to: {analysis_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for alpha in sorted(analysis["summary"].keys()):
        summary = analysis["summary"][alpha]
        print(
            f"\n{alpha}: {summary['major_percentage']:.1f}% major, {summary['minor_percentage']:.1f}% minor"
        )
        print(f"  Confident: {summary['n_confident']}/{summary['n_samples']}")

    if "chi_square_overall" in analysis["statistics"]:
        chi2 = analysis["statistics"]["chi_square_overall"]
        print(f"\nChi-square test: {chi2['interpretation']}")

    # Print proportion tests
    for key in analysis["statistics"]:
        if "vs_baseline" in key:
            stat = analysis["statistics"][key]
            print(f"\n{key}:")

            # Determine which percentage to display (negative or positive alpha)
            if "neg_major_pct" in stat:
                alpha_major_pct = stat["neg_major_pct"]
            else:
                alpha_major_pct = stat["pos_major_pct"]

            print(
                f"  {alpha_major_pct:.1f}% vs {stat['baseline_major_pct']:.1f}% major"
            )
            print(
                f"  Difference: {stat['difference']:+.1f}% ({'significant' if stat['significant'] else 'not significant'})"
            )

    print("=" * 60 + "\n")

    # Plot results
    plot_path = args.output_dir / "modality_plot.png"
    plot_results(results, plot_path)

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
