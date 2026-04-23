"""Non-Sequential PID Vector Calculator.

Computes PID-enhanced steering vectors offline from pre-computed DiffMean
vectors. This is a zero-cost enhancement: vectors are pre-computed once and
then used identically to standard DiffMean vectors during inference.

The PID computation treats the layer index k as the discrete time step
and applies the standard PID control law:

    u(k) = Kp * r(k) + Ki * Σ_{j<k} r(j) + Kd * (r(k) - r(k-1))

where r(k) is the DiffMean vector at sublayer k.

Variants:
- P-only (Ki=0, Kd=0): Equivalent to standard DiffMean (baseline)
- PI (Kd=0): Removes steady-state error via integral accumulation
- PID (full): Adds derivative damping to reduce overshoot
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_controller import SpatialPIDController
import config_pid

logger = logging.getLogger(__name__)


def compute_pid_vectors(
    diffmean_vectors: Dict[int, torch.Tensor],
    Kp: float = 1.0,
    Ki: float = 0.10,
    Kd: float = 0.05,
    max_I: float = 5.0,
    dim: int = 512,
) -> Dict[int, torch.Tensor]:
    """Compute PID-enhanced steering vectors from DiffMean vectors.

    This is a non-sequential (offline) computation. The DiffMean vectors
    r(k) are treated as pre-computed error signals, and the PID controller
    accumulates across sublayer indices to produce enhanced vectors.

    Args:
        diffmean_vectors: Dict mapping sublayer_idx -> DiffMean vector (dim,).
        Kp: Proportional gain.
        Ki: Integral gain (0 = P-only baseline).
        Kd: Derivative gain (0 = PI-only).
        max_I: Anti-windup bound for integral accumulator.
        dim: Activation dimension.

    Returns:
        Dict mapping sublayer_idx -> PID-enhanced steering vector (dim,).
    """
    controller = SpatialPIDController(Kp=Kp, Ki=Ki, Kd=Kd, max_I=max_I, dim=dim)
    controller.reset()

    pid_vectors = {}
    layer_indices = sorted(diffmean_vectors.keys())

    for layer_idx in layer_indices:
        r_k = diffmean_vectors[layer_idx]
        if isinstance(r_k, np.ndarray):
            r_k = torch.from_numpy(r_k).float()

        u_k = controller.compute(r_k)
        pid_vectors[layer_idx] = u_k.clone()

    return pid_vectors


def compute_pid_variants(
    diffmean_vectors: Dict[int, torch.Tensor],
    Kp: float = 1.0,
    Ki: float = 0.10,
    Kd: float = 0.05,
    max_I: float = 5.0,
    dim: int = 512,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute P-only, PI, and PID variants for comparison.

    Args:
        diffmean_vectors: Dict mapping sublayer_idx -> DiffMean vector.
        Kp, Ki, Kd, max_I, dim: PID parameters.

    Returns:
        Dict with keys "p_only", "pi", "pid" mapping to per-layer vectors.
    """
    variants = {}

    # P-only (equivalent to standard DiffMean with scaling)
    variants["p_only"] = compute_pid_vectors(
        diffmean_vectors, Kp=Kp, Ki=0.0, Kd=0.0, max_I=max_I, dim=dim
    )

    # PI (integral removes steady-state error)
    variants["pi"] = compute_pid_vectors(
        diffmean_vectors, Kp=Kp, Ki=Ki, Kd=0.0, max_I=max_I, dim=dim
    )

    # Full PID
    variants["pid"] = compute_pid_vectors(
        diffmean_vectors, Kp=Kp, Ki=Ki, Kd=Kd, max_I=max_I, dim=dim
    )

    return variants


def load_diffmean_vectors(path: pathlib.Path) -> Dict[int, torch.Tensor]:
    """Load pre-computed DiffMean steering vectors.

    Args:
        path: Path to .pt file saved by steering_vector_calculator.py.

    Returns:
        Dict mapping sublayer_idx -> steering vector tensor.
    """
    data = torch.load(path, map_location="cpu")
    if "steering_vectors" in data:
        return data["steering_vectors"]
    return data


def save_pid_vectors(
    pid_vectors: Dict[int, torch.Tensor],
    output_path: pathlib.Path,
    metadata: Dict,
):
    """Save PID-enhanced steering vectors.

    Args:
        pid_vectors: Dict mapping sublayer_idx -> PID steering vector.
        output_path: Output .pt file path.
        metadata: Metadata dict (PID gains, source concept, etc.).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"steering_vectors": pid_vectors, "metadata": metadata},
        output_path,
    )
    logger.info(f"Saved PID vectors to: {output_path}")

    # Save statistics JSON
    stats = {}
    for layer_idx, vec in pid_vectors.items():
        v = vec.numpy() if isinstance(vec, torch.Tensor) else vec
        stats[f"layer_{layer_idx}"] = {
            "norm": float(np.linalg.norm(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "statistics": stats}, f, indent=2)
    logger.info(f"Saved statistics to: {json_path}")


def compare_vectors(
    original: Dict[int, torch.Tensor],
    pid: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """Compare original DiffMean vectors with PID-enhanced vectors.

    Returns:
        Dict with comparison statistics (cosine similarity, norm ratio, etc.).
    """
    layer_indices = sorted(set(original.keys()) & set(pid.keys()))
    cosine_sims = []
    norm_ratios = []

    for k in layer_indices:
        o = original[k].float()
        p = pid[k].float()
        cos_sim = torch.nn.functional.cosine_similarity(o.unsqueeze(0), p.unsqueeze(0))
        cosine_sims.append(cos_sim.item())
        norm_ratios.append(p.norm().item() / (o.norm().item() + 1e-8))

    return {
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "min_cosine_similarity": float(np.min(cosine_sims)),
        "mean_norm_ratio": float(np.mean(norm_ratios)),
        "max_norm_ratio": float(np.max(norm_ratios)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute PID-enhanced steering vectors from DiffMean vectors"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="average_pitch",
        help="Concept name (average_pitch or average_duration)",
    )
    parser.add_argument(
        "--input_path",
        type=pathlib.Path,
        default=None,
        help="Path to DiffMean vectors .pt file",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_VECTORS_DIR,
        help="Output directory for PID vectors",
    )
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
        "--all_variants",
        action="store_true",
        help="Compute P-only, PI, and PID variants",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Resolve input path
    input_path = args.input_path
    if input_path is None:
        input_path = (
            config_pid.STEERING_VECTORS_DIR / f"{args.concept}_steering_vectors.pt"
        )

    if not input_path.exists():
        logger.error(f"DiffMean vectors not found: {input_path}")
        logger.error("Run steering_vector_calculator.py first!")
        return

    # Load DiffMean vectors
    logger.info(f"Loading DiffMean vectors from: {input_path}")
    diffmean_vectors = load_diffmean_vectors(input_path)
    logger.info(f"Loaded vectors for {len(diffmean_vectors)} sublayers")

    metadata_base = {
        "concept": args.concept,
        "source": str(input_path),
        "mode": "non_sequential",
    }

    if args.all_variants:
        variants = compute_pid_variants(
            diffmean_vectors,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
        )
        for variant_name, vectors in variants.items():
            out_path = args.output_dir / f"{args.concept}_{variant_name}_vectors.pt"
            metadata = {**metadata_base, "variant": variant_name}
            if variant_name == "p_only":
                metadata.update({"Kp": args.Kp, "Ki": 0.0, "Kd": 0.0})
            elif variant_name == "pi":
                metadata.update({"Kp": args.Kp, "Ki": args.Ki, "Kd": 0.0})
            else:
                metadata.update({"Kp": args.Kp, "Ki": args.Ki, "Kd": args.Kd})
            metadata["max_I"] = args.max_I
            save_pid_vectors(vectors, out_path, metadata)

            comp = compare_vectors(diffmean_vectors, vectors)
            logger.info(
                f"{variant_name}: cosine_sim={comp['mean_cosine_similarity']:.4f}, "
                f"norm_ratio={comp['mean_norm_ratio']:.4f}"
            )
    else:
        pid_vectors = compute_pid_vectors(
            diffmean_vectors,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
        )
        out_path = args.output_dir / f"{args.concept}_pid_vectors.pt"
        metadata = {
            **metadata_base,
            "variant": "pid",
            "Kp": args.Kp,
            "Ki": args.Ki,
            "Kd": args.Kd,
            "max_I": args.max_I,
        }
        save_pid_vectors(pid_vectors, out_path, metadata)

        comp = compare_vectors(diffmean_vectors, pid_vectors)
        logger.info(
            f"PID vs DiffMean: cosine_sim={comp['mean_cosine_similarity']:.4f}, "
            f"norm_ratio={comp['mean_norm_ratio']:.4f}"
        )

    logger.info("PID vector computation complete!")


if __name__ == "__main__":
    main()
