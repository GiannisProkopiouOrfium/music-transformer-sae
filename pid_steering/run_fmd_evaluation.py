"""FMD Evaluation for PID Steering — Conditioned Generation.

Computes Fréchet Music Distance (FMD) via CLaMP2 embeddings
for PID-steered conditioned generations. Mirrors the existing
FMD pipeline in metrics_evaluation/evaluate_fmd.py.

Produces FMD scores for:
- baseline (no steering, conditioned)
- P-only (standard DiffMean)
- PI
- PID
at various alpha values, for both pitch and duration concepts.

Uses 16-beat conditioning prefix and 512-token continuations
to match the SAS/DM evaluator methodology.
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
    compute_pid_variants,
)
from pid_steering.pid_steered_generator import PIDSteeredGenerator, load_model
from pid_steering.conditioned_pid_evaluator import (
    load_song_tokens,
    extract_conditioning_prefix,
    find_extreme_songs,
)

import config_pid
import representation

logger = logging.getLogger(__name__)


def generate_conditioned_midis(
    generator: PIDSteeredGenerator,
    variants: Dict[str, Dict[int, torch.Tensor]],
    encoding: Dict,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, float]],
    alpha: float,
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path,
) -> Dict[str, pathlib.Path]:
    """Generate conditioned MIDI files for each method variant.

    Returns dict mapping method_name -> directory of MIDI files.
    """
    eos = encoding["type_code_map"]["end-of-song"]
    gen_kwargs = {
        "eos_token": eos,
        "temperature": config_pid.TEMPERATURE,
        "filter_logits_fn": "top_k",
        "filter_thres": config_pid.FILTER_THRESHOLD,
        "monotonicity_dim": ("type", "beat"),
    }

    method_configs = {"baseline": None}
    method_configs.update(variants)

    midi_dirs = {}

    for method_name, vectors in method_configs.items():
        method_alpha = alpha if method_name != "baseline" else 0.0
        midi_dir = output_dir / method_name
        midi_dir.mkdir(parents=True, exist_ok=True)
        n_valid = 0

        for i, (filepath, _) in enumerate(song_list):
            tokens = load_song_tokens(filepath, encoding)
            conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(
                device
            )

            if vectors is not None and method_alpha != 0.0:
                generator.apply_precomputed_steering(vectors, method_alpha)
            else:
                generator.remove_steering()

            with torch.no_grad():
                generated = generator.model.generate(
                    conditioning, continuation_len, **gen_kwargs
                )
            generator.remove_steering()

            full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]

            try:
                midi_path = midi_dir / f"sample_{i}.mid"
                music = representation.decode(full_seq, encoding)
                if music is not None:
                    music.write(str(midi_path))
                    n_valid += 1
            except Exception as e:
                logger.warning(f"MIDI export failed for {method_name} sample {i}: {e}")

        logger.info(f"  {method_name}: {n_valid}/{len(song_list)} valid MIDIs")
        midi_dirs[method_name] = midi_dir

    return midi_dirs


def compute_fmd(generated_dir: pathlib.Path, reference_dir: pathlib.Path) -> float:
    """Compute FMD between generated and reference MIDI sets."""
    try:
        sys.path.insert(
            0, str(pathlib.Path(__file__).parent.parent / "metrics_evaluation")
        )
        from evaluate_fmd import compute_fmd_from_dirs

        return compute_fmd_from_dirs(str(generated_dir), str(reference_dir))
    except ImportError:
        logger.warning("evaluate_fmd not available — returning -1.0")
        return -1.0


def main():
    parser = argparse.ArgumentParser(
        description="FMD evaluation for PID steering (conditioned)"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument("--n_songs", type=int, default=10)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="Alpha magnitudes (applied as +α on low, -α on high)",
    )
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
        "--Kp",
        type=float,
        default=None,
        help="Override proportional gain (default: per-concept from grid search)",
    )
    parser.add_argument(
        "--Ki",
        type=float,
        default=None,
        help="Override integral gain (default: per-concept from grid search)",
    )
    parser.add_argument(
        "--Kd",
        type=float,
        default=None,
        help="Override derivative gain (default: per-concept from grid search)",
    )
    parser.add_argument(
        "--max_I",
        type=float,
        default=None,
        help="Override integral clamp (default: per-concept from grid search)",
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

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
    all_results = {}

    for concept in args.concepts:
        logger.info(f"\n{'='*70}\nFMD Evaluation — {concept}\n{'='*70}")

        # Per-concept PID gains (data-driven from grid search)
        gains = config_pid.get_gains(concept)
        Kp = args.Kp if args.Kp is not None else gains["Kp"]
        Ki = args.Ki if args.Ki is not None else gains["Ki"]
        Kd = args.Kd if args.Kd is not None else gains["Kd"]
        max_I = args.max_I if args.max_I is not None else gains["max_I"]
        logger.info(f"PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}, max_I={max_I}")

        dm_path = config_pid.STEERING_VECTORS_DIR / f"{concept}_steering_vectors.pt"
        dm_vectors = load_diffmean_vectors(dm_path)
        variants = compute_pid_variants(
            dm_vectors,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            dim=config_pid.MODEL_DIM,
        )

        low_songs, high_songs = find_extreme_songs(
            notes_dir,
            encoding,
            concept,
            n_songs=args.n_songs,
            conditioning_beats=args.conditioning_beats,
        )

        concept_results = {}

        for alpha in args.alphas:
            alpha_key = f"alpha_{alpha}"

            # Low songs: steer UP (+α)
            logger.info(f"\n--- Low {concept}, α=+{alpha} ---")
            low_midi_dirs = generate_conditioned_midis(
                generator,
                variants,
                encoding,
                device,
                low_songs,
                alpha=alpha,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
                output_dir=args.output_dir / concept / f"low_alpha{alpha}",
            )

            # High songs: steer DOWN (-α)
            logger.info(f"\n--- High {concept}, α=-{alpha} ---")
            high_midi_dirs = generate_conditioned_midis(
                generator,
                variants,
                encoding,
                device,
                high_songs,
                alpha=-alpha,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
                output_dir=args.output_dir / concept / f"high_alpha{alpha}",
            )

            # Compute FMD for each method (pool low+high MIDIs)
            fmd_results = {}
            for method in ["baseline", "p_only", "pi", "pid"]:
                # Merge low and high MIDI dirs into a combined dir
                combined_dir = args.output_dir / concept / f"combined_{alpha}" / method
                combined_dir.mkdir(parents=True, exist_ok=True)

                n_files = 0
                for src_dir in [low_midi_dirs.get(method), high_midi_dirs.get(method)]:
                    if src_dir and src_dir.exists():
                        for midi_file in src_dir.glob("*.mid"):
                            dst = (
                                combined_dir / f"{src_dir.parent.name}_{midi_file.name}"
                            )
                            if not dst.exists():
                                import shutil

                                shutil.copy2(midi_file, dst)
                            n_files += 1

                fmd = compute_fmd(combined_dir, args.reference_dir)
                fmd_results[method] = {"fmd": fmd, "n_files": n_files}
                logger.info(f"  {method}: FMD={fmd:.4f} ({n_files} files)")

            concept_results[alpha_key] = fmd_results

        all_results[concept] = concept_results

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "fmd_pid_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    for concept, concept_data in all_results.items():
        for alpha_key, methods in concept_data.items():
            print(f"\n{'='*60}")
            print(f"FMD — {concept} | {alpha_key}")
            print(f"{'='*60}")
            print(f"{'Method':<12} {'FMD':<12} {'N Files':<10}")
            print("-" * 34)
            for method, data in methods.items():
                print(f"{method:<12} {data['fmd']:<12.4f} {data['n_files']:<10}")
            print(f"{'='*60}")

    logger.info(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
