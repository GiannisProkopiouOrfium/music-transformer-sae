"""FMD Evaluation for PID Steering.

Computes Fréchet Music Distance (FMD) via CLaMP2 embeddings
for PID-steered vs baseline generations. This mirrors the
existing FMD pipeline in metrics_evaluation/evaluate_fmd.py.

Produces FMD scores for:
- Unconditioned baseline
- P-only (standard DiffMean)
- PI
- PID
at various alpha values.
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
    compute_pid_variants,
)
from pid_steering.pid_steered_generator import PIDSteeredGenerator, load_model
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)


def sequences_to_midi(
    sequences: List[np.ndarray], output_dir: pathlib.Path, encoding: dict
):
    """Convert token sequences to MIDI files for FMD computation."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
    import representation

    output_dir.mkdir(parents=True, exist_ok=True)
    midi_paths = []

    for i, seq in enumerate(sequences):
        try:
            midi_path = output_dir / f"sample_{i}.mid"
            music = representation.decode(seq, encoding)
            if music is not None:
                music.write(str(midi_path))
                midi_paths.append(midi_path)
        except Exception as e:
            logger.warning(f"Failed to decode sample {i}: {e}")

    return midi_paths


def compute_fmd(generated_dir: pathlib.Path, reference_dir: pathlib.Path) -> float:
    """Compute FMD between generated and reference MIDI sets.

    Delegates to the existing FMD evaluation pipeline.
    """
    try:
        sys.path.insert(
            0, str(pathlib.Path(__file__).parent.parent / "metrics_evaluation")
        )
        from evaluate_fmd import compute_fmd_from_dirs

        return compute_fmd_from_dirs(str(generated_dir), str(reference_dir))
    except ImportError:
        logger.warning("evaluate_fmd not available. Computing basic FMD proxy...")
        return _compute_fmd_proxy(generated_dir, reference_dir)


def _compute_fmd_proxy(gen_dir: pathlib.Path, ref_dir: pathlib.Path) -> float:
    """Lightweight FMD proxy using token-level statistics.

    Used when CLaMP2 is not available.
    """
    gen_metrics = []
    ref_metrics = []

    for midi_path in gen_dir.glob("*.mid"):
        # Just use the metrics we already computed as a proxy
        pass

    logger.warning("FMD proxy not fully implemented — use evaluate_fmd.py on EC2")
    return -1.0


def main():
    parser = argparse.ArgumentParser(description="FMD evaluation for PID steering")
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
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
        "--reference_dir",
        type=pathlib.Path,
        default=config_pid.PROJECT_ROOT / "data" / "sod",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "fmd",
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
    variants = compute_pid_variants(
        dm_vectors,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        dim=config_pid.MODEL_DIM,
    )

    results = {}

    for method_name, method_vectors in variants.items():
        method_results = {}

        for alpha in args.alpha_grid:
            logger.info(f"FMD eval: {method_name}, alpha={alpha}")

            # Generate sequences
            sequences = generator.generate_samples(
                pid_vectors=method_vectors,
                alpha=alpha,
                n_samples=args.n_samples,
                seq_len=args.seq_len,
            )

            # Convert to MIDI
            midi_dir = args.output_dir / f"midi_{method_name}_alpha{alpha}"
            midi_paths = sequences_to_midi(sequences, midi_dir, encoding)
            logger.info(f"  Converted {len(midi_paths)} sequences to MIDI")

            # Compute FMD
            if midi_paths:
                fmd = compute_fmd(midi_dir, args.reference_dir)
                method_results[str(alpha)] = {
                    "fmd": fmd,
                    "n_valid_sequences": len(midi_paths),
                }
                logger.info(f"  FMD = {fmd:.4f}")
            else:
                method_results[str(alpha)] = {"fmd": -1.0, "n_valid_sequences": 0}

        results[method_name] = method_results

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"fmd_results_{args.concept}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved FMD results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"FMD Results — {args.concept}")
    print("=" * 60)
    print(f"{'Method':<10} {'Alpha':<8} {'FMD':<12} {'Valid Seqs':<12}")
    print("-" * 45)
    for method, alphas in results.items():
        for alpha_str, data in alphas.items():
            print(
                f"{method:<10} {alpha_str:<8} {data['fmd']:<12.4f} "
                f"{data['n_valid_sequences']:<12}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
