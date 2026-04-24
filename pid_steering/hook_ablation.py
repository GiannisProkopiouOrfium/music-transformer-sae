"""Hook Ablation Study — Conditioned Generation.

Compares different sublayer subsets for PID steering using conditioned
generation (16-beat prefix, 512-token continuation) to match the DM/SAS
evaluator methodology.

Layer configurations tested:
- all_12: All 12 sublayers (attention + FF interleaved)
- attention_only: 6 attention sublayers (even indices: 0,2,4,6,8,10)
- feedforward_only: 6 FF sublayers (odd indices: 1,3,5,7,9,11)
- deep_only: Last 4 sublayers (8,9,10,11)
- mid_deep: Middle-to-deep sublayers (4,5,6,7,8,9,10,11)

Investigates:
1. Which sublayer types matter most for steering (attention vs feedforward)
2. Whether PID's integral+derivative terms compensate for reduced coverage
3. Whether the answer differs across concepts (pitch vs duration) and alphas
"""

import argparse
import json
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
from pid_steering.conditioned_pid_evaluator import (
    load_song_tokens,
    extract_conditioning_prefix,
    find_extreme_songs,
    evaluate_quality_metrics,
    measure_concept_from_tokens,
    calculate_degradation,
)

import config_pid
import representation

logger = logging.getLogger(__name__)


def run_conditioned_hook_ablation(
    generator: PIDSteeredGenerator,
    dm_vectors: Dict[int, torch.Tensor],
    hook_configs: Dict[str, List[int]],
    encoding: Dict,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, float]],
    category: str,
    alpha: float,
    concept: str,
    Kp: float,
    Ki: float,
    Kd: float,
    max_I: float,
    conditioning_beats: int,
    continuation_len: int,
) -> Dict[str, Dict]:
    """Run ablation over sublayer subsets using conditioned generation."""
    eos = encoding["type_code_map"]["end-of-song"]
    gen_kwargs = {
        "eos_token": eos,
        "temperature": config_pid.TEMPERATURE,
        "filter_logits_fn": "top_k",
        "filter_thres": config_pid.FILTER_THRESHOLD,
        "monotonicity_dim": ("type", "beat"),
    }

    results = {}

    for config_name, target_layers in hook_configs.items():
        logger.info(f"\nAblation: {config_name} — layers {target_layers}")

        # Compute full PID vectors, then filter to target layers
        full_pid_vectors = compute_pid_vectors(
            dm_vectors,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            dim=config_pid.MODEL_DIM,
        )
        filtered_vectors = {
            k: v for k, v in full_pid_vectors.items() if k in target_layers
        }

        attr_key = "pitch_mean" if "pitch" in concept else "duration_mean"
        changes, degradations, attr_vals = [], [], []

        for filepath, initial_value in song_list:
            tokens = load_song_tokens(filepath, encoding)
            conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(
                device
            )

            generator.apply_precomputed_steering(filtered_vectors, alpha)
            with torch.no_grad():
                generated = generator.model.generate(
                    conditioning, continuation_len, **gen_kwargs
                )
            generator.remove_steering()

            full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]
            gen_only = generated.cpu().numpy()[0]

            gen_metrics = measure_concept_from_tokens(gen_only, encoding)
            quality = evaluate_quality_metrics(full_seq, encoding)
            degradation = calculate_degradation(quality)

            changes.append(gen_metrics[attr_key] - initial_value)
            degradations.append(degradation["total_degradation"])
            attr_vals.append(gen_metrics[attr_key])

        results[config_name] = {
            "layers": target_layers,
            "n_layers": len(target_layers),
            "n_songs": len(song_list),
            "attr_change_mean": float(np.mean(changes)),
            "attr_change_std": float(np.std(changes)),
            "attr_value_mean": float(np.mean(attr_vals)),
            "degradation_mean": float(np.mean(degradations)),
            "degradation_std": float(np.std(degradations)),
        }

        logger.info(
            f"  → change={np.mean(changes):+.1f}±{np.std(changes):.1f}, "
            f"degrad={np.mean(degradations):.2f}±{np.std(degradations):.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hook ablation study for PID steering (conditioned)"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="Alpha magnitudes to test (applied as +α on low, -α on high)",
    )
    parser.add_argument("--n_songs", type=int, default=5)
    parser.add_argument(
        "--conditioning_beats",
        type=int,
        default=config_pid.CONDITIONING_BEATS,
    )
    parser.add_argument(
        "--continuation_len",
        type=int,
        default=config_pid.CONTINUATION_LEN,
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

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
    all_results = {}

    for concept in args.concepts:
        logger.info(f"\n{'='*70}\nHook Ablation — {concept}\n{'='*70}")

        dm_path = config_pid.STEERING_VECTORS_DIR / f"{concept}_steering_vectors.pt"
        dm_vectors = load_diffmean_vectors(dm_path)

        low_songs, high_songs = find_extreme_songs(
            notes_dir,
            encoding,
            concept,
            n_songs=args.n_songs,
            conditioning_beats=args.conditioning_beats,
        )

        concept_results = {}

        for alpha in args.alphas:
            # Low songs: steer UP (+α)
            logger.info(f"\n--- Low {concept}, α=+{alpha} ---")
            low_results = run_conditioned_hook_ablation(
                generator,
                dm_vectors,
                config_pid.HOOK_CONFIGS,
                encoding,
                device,
                low_songs,
                "low",
                alpha=alpha,
                concept=concept,
                Kp=args.Kp,
                Ki=args.Ki,
                Kd=args.Kd,
                max_I=args.max_I,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
            )

            # High songs: steer DOWN (-α)
            logger.info(f"\n--- High {concept}, α=-{alpha} ---")
            high_results = run_conditioned_hook_ablation(
                generator,
                dm_vectors,
                config_pid.HOOK_CONFIGS,
                encoding,
                device,
                high_songs,
                "high",
                alpha=-alpha,
                concept=concept,
                Kp=args.Kp,
                Ki=args.Ki,
                Kd=args.Kd,
                max_I=args.max_I,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
            )

            # Merge low + high per hook config
            merged = {}
            for config_name in config_pid.HOOK_CONFIGS:
                lo = low_results[config_name]
                hi = high_results[config_name]
                merged[config_name] = {
                    "layers": lo["layers"],
                    "n_layers": lo["n_layers"],
                    "n_songs": lo["n_songs"] + hi["n_songs"],
                    "low": lo,
                    "high": hi,
                    "abs_change_mean": float(
                        (abs(lo["attr_change_mean"]) + abs(hi["attr_change_mean"])) / 2
                    ),
                    "degradation_mean": float(
                        (lo["degradation_mean"] + hi["degradation_mean"]) / 2
                    ),
                }

            concept_results[f"alpha_{alpha}"] = merged

        all_results[concept] = concept_results

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "hook_ablation_conditioned.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {results_path}")

    # Print summary
    for concept, concept_data in all_results.items():
        for alpha_key, configs in concept_data.items():
            print(f"\n{'='*80}")
            print(f"Hook Ablation — {concept} | {alpha_key}")
            print(f"{'='*80}")
            print(f"{'Config':<20} {'Layers':<8} {'|Δ Attr|':<12} {'Degradation':<14}")
            print("-" * 54)
            for name, data in configs.items():
                print(
                    f"{name:<20} {data['n_layers']:<8} "
                    f"{data['abs_change_mean']:<12.1f} "
                    f"{data['degradation_mean']:<14.2f}"
                )
            print(f"{'='*80}")


if __name__ == "__main__":
    main()
