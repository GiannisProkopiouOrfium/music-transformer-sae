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


def run_grid_search(
    generator: PIDSteeredGenerator,
    dm_vectors: Dict[int, torch.Tensor],
    Kp_range: List[float],
    Ki_range: List[float],
    Kd_range: List[float],
    alpha: float,
    n_samples: int,
    seq_len: int,
    device: torch.device,
    metric_key: str = "average_pitch_mean",
) -> Dict[str, Dict]:
    """Run 3D grid search over (Kp, Ki, Kd).

    Returns:
        Dict with keys "grid" (full results), "best" (best combo).
    """
    results = []
    best_val = None
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

        sequences = generator.generate_samples(
            pid_vectors=pid_vectors,
            alpha=alpha,
            n_samples=n_samples,
            seq_len=seq_len,
        )

        metrics_list = [compute_generation_metrics(s) for s in sequences]
        avg = {}
        for key in metrics_list[0]:
            values = [m[key] for m in metrics_list]
            avg[f"{key}_mean"] = float(np.mean(values))
            avg[f"{key}_std"] = float(np.std(values))

        entry = {
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "metrics": avg,
        }
        results.append(entry)

        val = avg.get(metric_key, 0)
        if best_val is None or val > best_val:
            best_val = val
            best_combo = entry

        if step % 10 == 0:
            logger.info(
                f"Grid search: {step}/{total} "
                f"(Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}) "
                f"→ {metric_key}={val:.3f}"
            )

    return {"grid": results, "best": best_combo}


def plot_gain_heatmaps(
    results: List[Dict],
    metric_key: str,
    concept: str,
    output_dir: pathlib.Path,
):
    """Plot 2D heatmap slices through the 3D gain grid.

    Produces (Kp, Ki), (Kp, Kd), (Ki, Kd) heatmaps averaging over
    the third dimension.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    pairs = [
        ("Kp", "Ki", "Kd"),
        ("Kp", "Kd", "Ki"),
        ("Ki", "Kd", "Kp"),
    ]

    for x_key, y_key, avg_key in pairs:
        # Extract unique values
        x_vals = sorted(set(r[x_key] for r in results))
        y_vals = sorted(set(r[y_key] for r in results))

        # Build 2D matrix by averaging over the third dimension
        heatmap = np.zeros((len(y_vals), len(x_vals)))
        counts = np.zeros((len(y_vals), len(x_vals)))

        for r in results:
            xi = x_vals.index(r[x_key])
            yi = y_vals.index(r[y_key])
            heatmap[yi, xi] += r["metrics"].get(metric_key, 0)
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
        ax.set_title(f"Gain Ablation: {metric_key}\n{concept}", fontsize=13)
        plt.colorbar(im, ax=ax, label=metric_key)

        plot_path = output_dir / f"gain_heatmap_{concept}_{x_key}_{y_key}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved heatmap to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over PID gains (Kp, Ki, Kd)"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument(
        "--metric_key",
        type=str,
        default="average_pitch_mean",
        help="Metric to optimize in the grid search",
    )
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

    logger.info(
        f"Starting grid search: "
        f"{len(args.Kp_range)}×{len(args.Ki_range)}×{len(args.Kd_range)} = "
        f"{len(args.Kp_range)*len(args.Ki_range)*len(args.Kd_range)} combos"
    )

    results = run_grid_search(
        generator,
        dm_vectors,
        args.Kp_range,
        args.Ki_range,
        args.Kd_range,
        args.alpha,
        args.n_samples,
        args.seq_len,
        device,
        args.metric_key,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"grid_search_{args.concept}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"Saved grid search results to {results_path}")

    # Plot heatmaps
    plot_gain_heatmaps(
        results["grid"],
        args.metric_key,
        args.concept,
        args.output_dir,
    )

    # Print best
    best = results["best"]
    print(
        f"\nBest gains: Kp={best['Kp']:.2f}, Ki={best['Ki']:.2f}, "
        f"Kd={best['Kd']:.2f}"
    )
    print(f"Best {args.metric_key}: {best['metrics'].get(args.metric_key, 0):.4f}")


if __name__ == "__main__":
    main()
