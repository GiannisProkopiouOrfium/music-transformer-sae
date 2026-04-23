"""Grid Search for PID Gain Tuning.

Mirrors PID paper Figure 6: heatmaps of attribute metrics across
(Kp, Ki), (Kp, Kd) and (Ki, Kd) axes.

Produces:
- JSON results with full grid
- Heatmap plots for each pair of gains
"""

import argparse
import json
import itertools
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_vector_calculator import (
    load_diffmean_vectors,
    compute_pid_vectors,
)
from pid_steering.pid_steered_generator import PIDSteeredGenerator, load_model
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)


def compute_degradation(metrics: Dict, concept: str) -> float:
    """Compute degradation score: quality loss relative to ground truth.

    Lower is better. Measures how much steering hurts musical quality
    (PC entropy + scale consistency + groove consistency if available).
    """
    import math

    gt = config_pid.GROUND_TRUTH
    pc_dev = (
        abs(metrics.get("pitch_class_entropy_mean", 0) - gt["pitch_class_entropy"])
        / gt["pitch_class_entropy"]
    )
    sc_dev = (
        abs(metrics.get("scale_consistency_mean", 0) - gt["scale_consistency"])
        / gt["scale_consistency"]
    )
    deviations = [pc_dev, sc_dev]

    gc_val = metrics.get("groove_consistency_mean", float("nan"))
    if not math.isnan(gc_val):
        gc_dev = abs(gc_val - gt["groove_consistency"]) / gt["groove_consistency"]
        deviations.append(gc_dev)

    return sum(deviations) / len(deviations) * 100


def compute_steering_effectiveness(metrics: Dict, concept: str, alpha: float) -> float:
    """Measure how much the attribute shifted in the desired direction.

    Higher is better. Returns the absolute attribute value (for positive alpha)
    or negative of it (for negative alpha), so maximizing = stronger steering.
    """
    if "pitch" in concept:
        val = metrics.get("average_pitch_mean", 0)
    else:
        val = metrics.get("average_duration_mean", 0)
    return val if alpha > 0 else -val


def run_grid_search(
    generator: PIDSteeredGenerator,
    dm_vectors: Dict[int, torch.Tensor],
    Kp_range: List[float],
    Ki_range: List[float],
    Kd_range: List[float],
    alphas: List[float],
    n_samples: int,
    seq_len: int,
    device: torch.device,
    concept: str = "average_pitch",
    encoding=None,
) -> Dict[str, Dict]:
    """Run 3D grid search over (Kp, Ki, Kd), testing both steering directions.

    For each gain combo, generates samples at each alpha and averages the
    composite score across directions. This ensures gains generalize.

    Returns:
        Dict with keys "grid" (full results), "best" (best combo).
    """
    results = []
    best_score = None
    best_combo = None

    total = len(Kp_range) * len(Ki_range) * len(Kd_range)
    step = 0

    for Kp, Ki, Kd in itertools.product(Kp_range, Ki_range, Kd_range):
        step += 1

        pid_vectors = compute_pid_vectors(
            dm_vectors,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=config_pid.SPATIAL_PID_DEFAULTS["max_I"],
            dim=config_pid.MODEL_DIM,
        )

        # Test each alpha direction
        alpha_results = {}
        total_degradation = 0.0
        total_effectiveness = 0.0
        total_score = 0.0

        for alpha in alphas:
            sequences = generator.generate_samples(
                pid_vectors=pid_vectors,
                alpha=alpha,
                n_samples=n_samples,
                seq_len=seq_len,
            )

            metrics_list = [compute_generation_metrics(s, encoding) for s in sequences]
            avg = {}
            for key in metrics_list[0]:
                values = [m[key] for m in metrics_list]
                avg[f"{key}_mean"] = float(np.nanmean(values))
                avg[f"{key}_std"] = float(np.nanstd(values))

            degradation = compute_degradation(avg, concept)
            effectiveness = compute_steering_effectiveness(avg, concept, alpha)
            score = effectiveness / (1.0 + degradation / 100.0)

            alpha_results[str(alpha)] = {
                "metrics": avg,
                "degradation": degradation,
                "effectiveness": effectiveness,
                "score": score,
            }
            total_degradation += degradation
            total_effectiveness += effectiveness
            total_score += score

        n_alphas = len(alphas)
        avg_degradation = total_degradation / n_alphas
        avg_effectiveness = total_effectiveness / n_alphas
        avg_score = total_score / n_alphas

        entry = {
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "per_alpha": alpha_results,
            "degradation": avg_degradation,
            "effectiveness": avg_effectiveness,
            "score": avg_score,
        }
        results.append(entry)

        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_combo = entry

        if step % 10 == 0 or step == total:
            logger.info(
                f"Grid search: {step}/{total} "
                f"(Kp={Kp:.2f}, Ki={Ki:.3f}, Kd={Kd:.3f}) "
                f"→ avg_eff={avg_effectiveness:.2f}, avg_degrad={avg_degradation:.1f}%, "
                f"avg_score={avg_score:.2f}"
            )

    return {"grid": results, "best": best_combo}


def plot_gain_heatmaps(
    results: List[Dict],
    concept: str,
    output_dir: pathlib.Path,
):
    """Plot 2D heatmap slices through the 3D gain grid.

    Produces (Kp, Ki), (Kp, Kd), (Ki, Kd) heatmaps for both
    degradation and effectiveness, averaging over the third dimension.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    pairs = [
        ("Kp", "Ki", "Kd"),
        ("Kp", "Kd", "Ki"),
        ("Ki", "Kd", "Kp"),
    ]

    for plot_key in ["degradation", "score"]:
        for x_key, y_key, avg_key in pairs:
            x_vals = sorted(set(r[x_key] for r in results))
            y_vals = sorted(set(r[y_key] for r in results))

            heatmap = np.zeros((len(y_vals), len(x_vals)))
            counts = np.zeros((len(y_vals), len(x_vals)))

            for r in results:
                xi = x_vals.index(r[x_key])
                yi = y_vals.index(r[y_key])
                heatmap[yi, xi] += r.get(plot_key, 0)
                counts[yi, xi] += 1

            counts[counts == 0] = 1
            heatmap /= counts

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            )
            ax.set_xlabel(x_key, fontsize=12)
            ax.set_ylabel(y_key, fontsize=12)
            label = (
                "Degradation %" if plot_key == "degradation" else "Score (eff/degrad)"
            )
            ax.set_title(f"Gain Ablation: {label}\n{concept}", fontsize=13)
            plt.colorbar(im, ax=ax, label=label)

            plot_path = (
                output_dir / f"gain_heatmap_{concept}_{plot_key}_{x_key}_{y_key}.png"
            )
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved heatmap to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over PID gains (Kp, Ki, Kd)"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-1.0, 1.0],
        help="Alpha values to test at each gain combo (both directions recommended)",
    )
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument(
        "--Kp_range",
        type=float,
        nargs="+",
        default=config_pid.GRID_SEARCH["Kp"],
    )
    parser.add_argument(
        "--Ki_range",
        type=float,
        nargs="+",
        default=config_pid.GRID_SEARCH["Ki"],
    )
    parser.add_argument(
        "--Kd_range",
        type=float,
        nargs="+",
        default=config_pid.GRID_SEARCH["Kd"],
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "grid_search",
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )
    generator = PIDSteeredGenerator(model, encoding)

    dm_path = config_pid.STEERING_VECTORS_DIR / f"{args.concept}_steering_vectors.pt"
    dm_vectors = load_diffmean_vectors(dm_path)

    total_combos = len(args.Kp_range) * len(args.Ki_range) * len(args.Kd_range)
    logger.info(
        f"Starting grid search: "
        f"{len(args.Kp_range)}×{len(args.Ki_range)}×{len(args.Kd_range)} = "
        f"{total_combos} combos × {len(args.alphas)} alphas × {args.n_samples} samples"
    )

    results = run_grid_search(
        generator,
        dm_vectors,
        args.Kp_range,
        args.Ki_range,
        args.Kd_range,
        args.alphas,
        args.n_samples,
        args.seq_len,
        device,
        concept=args.concept,
        encoding=encoding,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"grid_search_{args.concept}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"Saved grid search results to {results_path}")

    # Plot heatmaps
    plot_gain_heatmaps(
        results["grid"],
        args.concept,
        args.output_dir,
    )

    # Print best
    best = results["best"]
    alphas_str = ", ".join(f"{a}" for a in args.alphas)
    print(f"\n{'='*60}")
    print(f"Best gains for {args.concept} (alphas=[{alphas_str}]):")
    print(f"  Kp={best['Kp']:.2f}, Ki={best['Ki']:.3f}, Kd={best['Kd']:.3f}")
    print(f"  Avg Effectiveness: {best['effectiveness']:.2f}")
    print(f"  Avg Degradation: {best['degradation']:.1f}%")
    print(f"  Avg Score: {best['score']:.2f}")
    for alpha_str, ar in best.get("per_alpha", {}).items():
        print(
            f"    α={alpha_str}: eff={ar['effectiveness']:.2f}, degrad={ar['degradation']:.1f}%"
        )
    print(f"{'='*60}")

    # Print top 5
    sorted_results = sorted(results["grid"], key=lambda r: r["score"], reverse=True)
    print(f"\nTop 5 gain combinations:")
    print(f"{'Kp':<6} {'Ki':<8} {'Kd':<8} {'Eff':<10} {'Degrad%':<10} {'Score':<10}")
    print("-" * 52)
    for r in sorted_results[:5]:
        print(
            f"{r['Kp']:<6.2f} {r['Ki']:<8.3f} {r['Kd']:<8.3f} "
            f"{r['effectiveness']:<10.2f} {r['degradation']:<10.1f} {r['score']:<10.2f}"
        )


if __name__ == "__main__":
    main()
