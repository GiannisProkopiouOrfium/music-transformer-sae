"""Hook Ablation Study.

Compares different sublayer subsets for PID steering:
- all_12: All 12 sublayers (attention + FF interleaved)
- attention_only: 6 attention sublayers (even indices: 0,2,4,6,8,10)
- feedforward_only: 6 FF sublayers (odd indices: 1,3,5,7,9,11)
- deep_only: Last 4 sublayers (8,9,10,11)
- mid_deep: Middle-to-deep sublayers (4,5,6,7,8,9,10,11)

This investigates whether PID's integral + derivative terms can
compensate for reduced sublayer coverage.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

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


def run_hook_ablation(
    generator: PIDSteeredGenerator,
    dm_vectors: Dict[int, torch.Tensor],
    hook_configs: Dict[str, List[int]],
    Kp: float,
    Ki: float,
    Kd: float,
    max_I: float,
    alpha: float,
    n_samples: int,
    seq_len: int,
    device: torch.device,
    encoding=None,
) -> Dict[str, Dict]:
    """Run ablation over different sublayer subsets.

    For each config, computes PID vectors only at the included layers,
    generates samples, and computes metrics.
    """
    results = {}

    for config_name, target_layers in hook_configs.items():
        logger.info(f"\nAblation: {config_name} — layers {target_layers}")

        # Compute PID vectors for all layers
        full_pid_vectors = compute_pid_vectors(
            dm_vectors,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            dim=config_pid.MODEL_DIM,
        )

        # Filter to only target layers
        filtered_vectors = {
            k: v for k, v in full_pid_vectors.items() if k in target_layers
        }

        # Generate
        sequences = generator.generate_samples(
            pid_vectors=filtered_vectors,
            alpha=alpha,
            n_samples=n_samples,
            seq_len=seq_len,
        )

        # Metrics
        metrics_list = [compute_generation_metrics(s, encoding) for s in sequences]
        avg = {}
        for key in metrics_list[0]:
            values = [m[key] for m in metrics_list]
            avg[f"{key}_mean"] = float(np.nanmean(values))
            avg[f"{key}_std"] = float(np.nanstd(values))

        results[config_name] = {
            "layers": target_layers,
            "n_layers": len(target_layers),
            "metrics": avg,
        }

        logger.info(
            f"  → avg_pitch={avg.get('average_pitch_mean', 0):.2f}, "
            f"pc_entropy={avg.get('pitch_class_entropy_mean', 0):.3f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Hook ablation study for PID steering")
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "hook_ablation",
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

    results = run_hook_ablation(
        generator,
        dm_vectors,
        config_pid.HOOK_CONFIGS,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        alpha=args.alpha,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        device=device,
        encoding=encoding,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"hook_ablation_{args.concept}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved to {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"Hook Ablation Results — {args.concept}")
    print("=" * 80)
    print(
        f"{'Config':<20} {'Layers':<8} {'Avg Pitch':<12} {'PC Entropy':<12} "
        f"{'Scale Cons':<12}"
    )
    print("-" * 70)
    for name, data in results.items():
        m = data["metrics"]
        print(
            f"{name:<20} {data['n_layers']:<8} "
            f"{m.get('average_pitch_mean', 0):<12.2f} "
            f"{m.get('pitch_class_entropy_mean', 0):<12.3f} "
            f"{m.get('scale_consistency_mean', 0):<12.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
