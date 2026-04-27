"""FMD Evaluation for Temporal PID SAS Steering.

Generates MIDI files from temporal PID, static smooth, and baseline,
then computes Fréchet Music Distance against the SOD reference corpus.

This complements the spatial PID FMD evaluation (run_fmd_evaluation.py)
by evaluating the temporal (token-level) SAS intervention.
"""

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.temporal_pid_sas_generator import (
    TemporalPIDSASGenerator,
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_temporal_pid_sas import generate_static_smooth
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid
import representation

logger = logging.getLogger(__name__)


def sequences_to_midis(sequences, encoding, output_dir: pathlib.Path) -> int:
    """Convert numpy token sequences to MIDI files.

    Returns number of successfully written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    for i, seq in enumerate(sequences):
        try:
            music = representation.decode(seq, encoding)
            if music is not None:
                midi_path = output_dir / f"sample_{i:03d}.mid"
                music.write(str(midi_path))
                n_valid += 1
        except Exception as e:
            logger.warning(f"MIDI export failed for sample {i}: {e}")
    return n_valid


def main():
    parser = argparse.ArgumentParser(description="FMD evaluation for temporal PID SAS")
    parser.add_argument(
        "--concept",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=40,
        help="Samples per method (40+ recommended for FMD)",
    )
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--reference_dir",
        type=pathlib.Path,
        default=None,
        help="Reference MIDI directory for FMD. Auto-detected if not set.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "temporal_fmd",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation, reuse existing MIDIs",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set GPU before importing CLaMP2 (it reads CUDA_VISIBLE_DEVICES at init)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve reference directory
    reference_dir = args.reference_dir
    if reference_dir is None:
        existing_ref = (
            config_pid.PROJECT_ROOT
            / "exp"
            / "sod"
            / "sparse_steering"
            / "fmd_workspace"
            / "reference_sod"
        )
        if existing_ref.exists() and len(list(existing_ref.glob("*.mid"))) >= 2:
            reference_dir = existing_ref
        else:
            logger.error("No reference MIDI directory found. Specify --reference_dir.")
            sys.exit(1)

    ref_count = len(list(reference_dir.glob("*.mid")))
    logger.info(f"Reference: {reference_dir} ({ref_count} MIDIs)")

    # Lazy-load FMD metric
    fmd_metric = None

    def get_fmd():
        nonlocal fmd_metric
        if fmd_metric is None:
            from frechet_music_distance import FrechetMusicDistance

            fmd_metric = FrechetMusicDistance(
                feature_extractor="clamp2",
                gaussian_estimator="mle",
                verbose=True,
            )
            logger.info("Initialized FrechetMusicDistance (CLaMP2)")
        return fmd_metric

    # Load model + SAE (shared across concepts)
    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )
    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    all_results = {}

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Temporal FMD Evaluation — {concept}")
        logger.info(f"{'='*70}")

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

        concept_dir = args.output_dir / concept
        methods = {}

        if not args.skip_generation:
            # 1. Temporal PID
            logger.info("Generating temporal PID samples...")
            pid_gen = TemporalPIDSASGenerator(
                model,
                encoding,
                sae_model,
                sas_vector,
                args.target_layer,
            )
            pid_seqs, _ = pid_gen.generate_with_temporal_pid(
                n_samples=args.n_samples,
                seq_len=args.seq_len,
                target_magnitude=args.target_magnitude,
                Kp=args.Kp,
                Ki=args.Ki,
                Kd=args.Kd,
                max_I=args.max_I,
                lambda_max=args.lambda_max,
                n_ramp_beats=args.n_ramp_steps,
                device=device,
            )
            pid_midi_dir = concept_dir / "temporal_pid"
            n_pid = sequences_to_midis(pid_seqs, encoding, pid_midi_dir)
            logger.info(f"  temporal_pid: {n_pid}/{args.n_samples} valid MIDIs")
            methods["temporal_pid"] = pid_midi_dir

            # 2. Static smooth
            logger.info("Generating static smooth samples...")
            static_seqs, _ = generate_static_smooth(
                model,
                encoding,
                sae_model,
                sas_vector,
                args.target_layer,
                lambda_max=args.lambda_max,
                n_ramp_steps=args.n_ramp_steps,
                n_samples=args.n_samples,
                seq_len=args.seq_len,
                device=device,
            )
            static_midi_dir = concept_dir / "static_smooth"
            n_static = sequences_to_midis(static_seqs, encoding, static_midi_dir)
            logger.info(f"  static_smooth: {n_static}/{args.n_samples} valid MIDIs")
            methods["static_smooth"] = static_midi_dir

            # 3. Baseline
            logger.info("Generating baseline samples...")
            baseline_seqs, _ = generate_static_smooth(
                model,
                encoding,
                sae_model,
                sas_vector,
                args.target_layer,
                lambda_max=0.0,
                n_ramp_steps=0,
                n_samples=args.n_samples,
                seq_len=args.seq_len,
                device=device,
            )
            baseline_midi_dir = concept_dir / "baseline"
            n_base = sequences_to_midis(baseline_seqs, encoding, baseline_midi_dir)
            logger.info(f"  baseline: {n_base}/{args.n_samples} valid MIDIs")
            methods["baseline"] = baseline_midi_dir
        else:
            # Reuse existing MIDIs
            for method in ["temporal_pid", "static_smooth", "baseline"]:
                d = concept_dir / method
                if d.exists() and len(list(d.glob("*.mid"))) > 0:
                    methods[method] = d
            logger.info(f"Reusing existing MIDIs: {list(methods.keys())}")

        # Compute FMD for each method
        concept_fmd = {}
        for method_name, midi_dir in methods.items():
            n_midis = len(list(midi_dir.glob("*.mid")))
            if n_midis < 2:
                logger.warning(f"  {method_name}: only {n_midis} MIDIs, skipping FMD")
                concept_fmd[method_name] = -1.0
                continue

            try:
                metric = get_fmd()
                score = metric.score(
                    reference_path=str(reference_dir),
                    test_path=str(midi_dir),
                )
                concept_fmd[method_name] = float(score)
                logger.info(f"  {method_name}: FMD = {score:.4f} ({n_midis} MIDIs)")
            except Exception as e:
                logger.error(f"  {method_name}: FMD failed: {e}")
                concept_fmd[method_name] = -1.0

        all_results[concept] = concept_fmd

        # Print summary
        print(f"\n{'='*50}")
        print(f"FMD Results — {concept}")
        print(f"{'='*50}")
        print(f"{'Method':<20} {'FMD':>10}")
        print("-" * 32)
        for method_name, fmd_val in concept_fmd.items():
            print(f"{method_name:<20} {fmd_val:>10.4f}")
        print(f"{'='*50}")

    # Save all results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "temporal_fmd_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved FMD results to {results_path}")

    # Print combined summary
    print(f"\n{'='*60}")
    print("Combined FMD Summary")
    print(f"{'='*60}")
    print(f"{'Concept':<20} {'Method':<20} {'FMD':>10}")
    print("-" * 52)
    for concept, fmd_dict in all_results.items():
        for method_name, fmd_val in fmd_dict.items():
            print(f"{concept:<20} {method_name:<20} {fmd_val:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
