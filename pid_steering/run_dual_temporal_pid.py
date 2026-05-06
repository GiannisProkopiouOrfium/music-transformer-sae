"""Dual-Concept Temporal PID SAS Evaluation.

Steers BOTH pitch and duration simultaneously using two independent
PID controllers. Compares against:
- Static SAS dual steering (gs_ek2 from ISMIR paper)
- DiffMean dual steering (GS pitch priority from ISMIR paper)
- Unsteered baseline

Evaluates in unconditioned and conditioned settings, computing:
- Per-concept steering success and magnitude
- Cross-concept interference / correlation
- Quality degradation (entropy, scale, groove)
- Dual success rate

Usage:
    python pid_steering/run_dual_temporal_pid.py \
        --mode unconditioned --n_samples 20 --gpu 0

    python pid_steering/run_dual_temporal_pid.py \
        --mode conditioned --n_songs 10 --n_per_song 2 --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.dual_temporal_pid_hook import (
    DualTemporalPIDSASHook,
    gram_schmidt_orthogonalize,
)
from pid_steering.temporal_pid_sas_generator import (
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid
import representation

logger = logging.getLogger(__name__)

# Ground truth for degradation
GT = config_pid.GROUND_TRUTH


def compute_degradation(metrics: dict) -> float:
    """Compute quality degradation δ matching the ISMIR paper formula."""
    h = metrics.get("pitch_class_entropy", GT["pitch_class_entropy"])
    s = metrics.get("scale_consistency", GT["scale_consistency"])
    g = metrics.get("groove_consistency", GT["groove_consistency"])
    delta = abs(h - GT["pitch_class_entropy"])
    delta += max(0, GT["scale_consistency"] - s)
    delta += max(0, GT["groove_consistency"] - g)
    return delta


# ═══════════════════════════════════════════════════════════════════════
# Generation helpers
# ═══════════════════════════════════════════════════════════════════════


def generate_dual_pid(
    model, encoding, hook: DualTemporalPIDSASHook,
    target_layer: int, start_tokens: torch.Tensor, seq_len: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict]:
    """Generate a single sample with dual PID hook."""
    eos = encoding["type_code_map"]["end-of-song"]
    start_tokens = start_tokens.to(device)
    hook.reset()

    layers = model.decoder.net.attn_layers.layers
    layer_module = layers[target_layer]
    target_module = layer_module[1] if isinstance(layer_module, nn.ModuleList) and len(layer_module) > 1 else layer_module

    handle = target_module.register_forward_hook(
        lambda mod, inp, out, h=hook: h(mod, inp, out, target_layer)
    )
    try:
        with torch.no_grad():
            generated = model.generate(
                start_tokens, seq_len, eos_token=eos,
                temperature=config_pid.TEMPERATURE,
                filter_logits_fn="top_k",
                filter_thres=config_pid.FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
        full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
    finally:
        handle.remove()

    return full_seq, hook.get_diagnostics()


def generate_dual_static_sas(
    model, encoding, sae_model, pitch_vector, duration_vector,
    target_layer: int, lambda_pitch: float, lambda_duration: float,
    start_tokens: torch.Tensor, seq_len: int, device: torch.device,
    k_multiplier: float = 2.0,
) -> np.ndarray:
    """Generate with static dual SAS (gs_ek2 strategy)."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "sparse_steering"))
    from sparse_steering.dual_steering.sparse_vector_composer import SparseVectorComposer
    from sparse_steering.dual_steering.dual_steered_generator import ExpandedKDualSASSteeringHook

    # Compose vectors
    p_dict = {target_layer: pitch_vector if isinstance(pitch_vector, np.ndarray) else pitch_vector.numpy()}
    d_dict = {target_layer: duration_vector if isinstance(duration_vector, np.ndarray) else duration_vector.numpy()}
    composer = SparseVectorComposer(p_dict, d_dict)
    combined = composer.compose(lambda_pitch, lambda_duration, strategy="gram_schmidt_ek2")

    sae_models = {target_layer: sae_model}
    hook = ExpandedKDualSASSteeringHook(
        sae_models=sae_models,
        combined_vectors=combined,
        layers_to_steer=[target_layer],
        k_multiplier=k_multiplier,
    )

    eos = encoding["type_code_map"]["end-of-song"]
    start_tokens = start_tokens.to(device)

    layers = model.decoder.net.attn_layers.layers
    layer_module = layers[target_layer]
    target_module = layer_module[1] if isinstance(layer_module, nn.ModuleList) and len(layer_module) > 1 else layer_module

    handle = target_module.register_forward_hook(
        lambda mod, inp, out, h=hook: h(mod, inp, out, target_layer)
    )
    try:
        with torch.no_grad():
            generated = model.generate(
                start_tokens, seq_len, eos_token=eos,
                temperature=config_pid.TEMPERATURE,
                filter_logits_fn="top_k",
                filter_thres=config_pid.FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
        full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
    finally:
        handle.remove()

    return full_seq


def generate_dual_diffmean(
    model, encoding, pitch_vectors: Dict[int, torch.Tensor],
    duration_vectors: Dict[int, torch.Tensor],
    alpha_pitch: float, alpha_duration: float,
    start_tokens: torch.Tensor, seq_len: int, device: torch.device,
) -> np.ndarray:
    """Generate with DiffMean dual steering (GS pitch priority)."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "steering_interventions"))
    from steering_interventions.dual_steering.vector_composition import VectorComposer
    from steering_interventions.steered_generator import SteeredGenerator

    composer = VectorComposer(pitch_vectors, duration_vectors)
    combined = composer.compose(alpha_pitch, alpha_duration, strategy="gram_schmidt_pitch")

    generator = SteeredGenerator(model, combined, encoding)
    eos = encoding["type_code_map"]["end-of-song"]
    start_tokens = start_tokens.to(device)

    generated = generator.generate(
        start_tokens, seq_len, alpha=1.0,  # alpha baked into composed vector
        eos_token=eos, temperature=config_pid.TEMPERATURE,
        filter_logits_fn="top_k", filter_thres=config_pid.FILTER_THRESHOLD,
        monotonicity_dim=("type", "beat"),
    )
    full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
    return full_seq


def generate_baseline(
    model, encoding, start_tokens: torch.Tensor, seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Generate unsteered baseline."""
    eos = encoding["type_code_map"]["end-of-song"]
    start_tokens = start_tokens.to(device)
    with torch.no_grad():
        generated = model.generate(
            start_tokens, seq_len, eos_token=eos,
            temperature=config_pid.TEMPERATURE,
            filter_logits_fn="top_k",
            filter_thres=config_pid.FILTER_THRESHOLD,
            monotonicity_dim=("type", "beat"),
        )
    full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
    return full_seq


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════


def evaluate_method(sequences: List[np.ndarray], encoding: dict) -> dict:
    """Compute mean metrics across sequences."""
    all_metrics = [compute_generation_metrics(s, encoding) for s in sequences]
    if not all_metrics:
        return {}
    summary = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if key in m]
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))
    # Compute mean degradation
    degradations = [compute_degradation(m) for m in all_metrics]
    summary["degradation_mean"] = float(np.nanmean(degradations))
    summary["degradation_std"] = float(np.nanstd(degradations))
    return summary


def compute_dual_success(metrics_list: List[dict], baseline_metrics: List[dict],
                         pitch_dir: str, dur_dir: str) -> float:
    """Compute dual success rate: fraction where BOTH attributes shift correctly."""
    n_success = 0
    for m, b in zip(metrics_list, baseline_metrics):
        pitch_ok = (m["average_pitch"] > b["average_pitch"]) if pitch_dir == "up" else (m["average_pitch"] < b["average_pitch"])
        dur_ok = (m["average_duration"] > b["average_duration"]) if dur_dir == "up" else (m["average_duration"] < b["average_duration"])
        if pitch_ok and dur_ok:
            n_success += 1
    return n_success / max(len(metrics_list), 1) * 100


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Dual temporal PID SAS evaluation")
    parser.add_argument("--mode", choices=["unconditioned", "conditioned"], default="unconditioned")
    parser.add_argument("--n_samples", type=int, default=20, help="Samples for unconditioned")
    parser.add_argument("--n_songs", type=int, default=10, help="Songs per scenario for conditioned")
    parser.add_argument("--n_per_song", type=int, default=2, help="Reps per song for conditioned")
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--continuation_len", type=int, default=config_pid.CONTINUATION_LEN)
    parser.add_argument("--conditioning_beats", type=int, default=config_pid.CONDITIONING_BEATS)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument("--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"])
    parser.add_argument("--Ki", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Ki"])
    parser.add_argument("--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"])
    parser.add_argument("--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"])
    # DiffMean alphas for dual steering
    parser.add_argument("--dm_alpha_pitch", type=float, default=1.0)
    parser.add_argument("--dm_alpha_duration", type=float, default=1.0)
    # SAS static lambda for dual steering
    parser.add_argument("--sas_lambda_pitch", type=float, default=0.75)
    parser.add_argument("--sas_lambda_duration", type=float, default=0.75)
    parser.add_argument("--output_dir", type=pathlib.Path,
                        default=config_pid.PID_EXPERIMENTS_DIR / "dual_temporal_pid")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if MIDI already exists on disk",
    )
    parser.add_argument(
        "--song_offset",
        type=int,
        default=0,
        help="Skip first N songs in each extreme group (to generate new ones)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT, config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH, device,
    )

    # Load SAE
    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    # Load SAS vectors
    pitch_sas = load_sas_vector(
        config_pid.SAS_VECTORS_DIR / "average_pitch_sas_vectors.pt",
        layer_idx=args.target_layer,
    )
    duration_sas = load_sas_vector(
        config_pid.SAS_VECTORS_DIR / "average_duration_sas_vectors.pt",
        layer_idx=args.target_layer,
    )

    # Load DiffMean vectors
    dm_pitch_path = config_pid.STEERING_VECTORS_DIR / "average_pitch_steering_vectors.pt"
    dm_dur_path = config_pid.STEERING_VECTORS_DIR / "average_duration_steering_vectors.pt"
    dm_pitch_vecs, dm_dur_vecs = None, None
    try:
        from steering_interventions.steered_generator import load_steering_vectors
        dm_pitch_vecs, _ = load_steering_vectors(dm_pitch_path)
        dm_dur_vecs, _ = load_steering_vectors(dm_dur_path)
        # Move to device
        dm_pitch_vecs = {k: v.to(device) for k, v in dm_pitch_vecs.items()}
        dm_dur_vecs = {k: v.to(device) for k, v in dm_dur_vecs.items()}
        logger.info("Loaded DiffMean vectors for comparison")
    except Exception as e:
        logger.warning(f"Could not load DiffMean vectors: {e}. Skipping DiffMean comparison.")

    if args.mode == "unconditioned":
        _run_unconditioned(model, encoding, sae_model, pitch_sas, duration_sas,
                           dm_pitch_vecs, dm_dur_vecs, args, device)
    else:
        _run_conditioned(model, encoding, sae_model, pitch_sas, duration_sas,
                         dm_pitch_vecs, dm_dur_vecs, args, device)


def _run_unconditioned(model, encoding, sae_model, pitch_sas, duration_sas,
                       dm_pitch_vecs, dm_dur_vecs, args, device):
    """Unconditioned dual steering comparison."""
    logger.info("="*70)
    logger.info("Dual Temporal PID — Unconditioned")
    logger.info("="*70)

    sos = encoding["type_code_map"]["start-of-song"]
    start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    start[:, 0, 0] = sos

    results = {}

    # 1. Dual PID (steer both UP)
    logger.info("Generating with Dual Temporal PID (both UP)...")
    hook = DualTemporalPIDSASHook(
        sae_model=sae_model,
        pitch_vector=pitch_sas,
        duration_vector=duration_sas,
        target_magnitude=args.target_magnitude,
        Kp=args.Kp, Ki=args.Ki, Kd=args.Kd, max_I=args.max_I,
        lambda_max=args.lambda_max, n_ramp_beats=args.n_ramp_steps,
        orthogonalize=True, k_multiplier=2.0,
    )

    pid_seqs, pid_diags = [], []
    for i in range(args.n_samples):
        seq, diag = generate_dual_pid(
            model, encoding, hook, args.target_layer,
            start, args.seq_len, device,
        )
        pid_seqs.append(seq)
        pid_diags.append(diag)
        if (i + 1) % 5 == 0:
            logger.info(f"  PID {i+1}/{args.n_samples} "
                        f"(avg λ_p={np.mean(diag['pitch_lambda_trajectory']):.3f}, "
                        f"λ_d={np.mean(diag['duration_lambda_trajectory']):.3f})")

    pid_metrics = evaluate_method(pid_seqs, encoding)
    pid_metrics["avg_lambda_pitch"] = float(np.mean([np.mean(d["pitch_lambda_trajectory"]) for d in pid_diags]))
    pid_metrics["avg_lambda_duration"] = float(np.mean([np.mean(d["duration_lambda_trajectory"]) for d in pid_diags]))
    results["dual_pid"] = pid_metrics

    # 2. Static SAS dual (gs_ek2)
    logger.info("Generating with Static SAS dual (gs_ek2)...")
    sas_seqs = []
    for i in range(args.n_samples):
        seq = generate_dual_static_sas(
            model, encoding, sae_model, pitch_sas, duration_sas,
            args.target_layer, args.sas_lambda_pitch, args.sas_lambda_duration,
            start, args.seq_len, device,
        )
        sas_seqs.append(seq)
        if (i + 1) % 5 == 0:
            logger.info(f"  SAS {i+1}/{args.n_samples}")
    results["static_sas_gs_ek2"] = evaluate_method(sas_seqs, encoding)

    # 3. DiffMean dual (GS pitch priority)
    if dm_pitch_vecs and dm_dur_vecs:
        logger.info("Generating with DiffMean dual (GS pitch priority)...")
        dm_seqs = []
        for i in range(args.n_samples):
            seq = generate_dual_diffmean(
                model, encoding, dm_pitch_vecs, dm_dur_vecs,
                args.dm_alpha_pitch, args.dm_alpha_duration,
                start, args.seq_len, device,
            )
            dm_seqs.append(seq)
            if (i + 1) % 5 == 0:
                logger.info(f"  DM {i+1}/{args.n_samples}")
        results["diffmean_gs_pitch"] = evaluate_method(dm_seqs, encoding)

    # 4. Baseline
    logger.info("Generating baseline...")
    base_seqs = []
    for i in range(args.n_samples):
        seq = generate_baseline(model, encoding, start, args.seq_len, device)
        base_seqs.append(seq)
    results["baseline"] = evaluate_method(base_seqs, encoding)

    # Dual success rates
    base_metrics_list = [compute_generation_metrics(s, encoding) for s in base_seqs]
    pid_metrics_list = [compute_generation_metrics(s, encoding) for s in pid_seqs]
    sas_metrics_list = [compute_generation_metrics(s, encoding) for s in sas_seqs]
    results["dual_pid"]["dual_success_rate"] = compute_dual_success(
        pid_metrics_list, base_metrics_list, "up", "up")
    results["static_sas_gs_ek2"]["dual_success_rate"] = compute_dual_success(
        sas_metrics_list, base_metrics_list, "up", "up")
    if "diffmean_gs_pitch" in results:
        dm_metrics_list = [compute_generation_metrics(s, encoding) for s in dm_seqs]
        results["diffmean_gs_pitch"]["dual_success_rate"] = compute_dual_success(
            dm_metrics_list, base_metrics_list, "up", "up")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "dual_unconditioned_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Save PID diagnostics
    with open(args.output_dir / "dual_pid_diagnostics.json", "w") as f:
        json.dump(pid_diags, f, indent=2, default=float)

    # Save MIDIs
    for method_name, seqs in [("dual_pid", pid_seqs), ("static_sas", sas_seqs), ("baseline", base_seqs)]:
        midi_dir = args.output_dir / "midis" / method_name
        midi_dir.mkdir(parents=True, exist_ok=True)
        for i, seq in enumerate(seqs):
            try:
                music = representation.decode(seq, encoding)
                if music:
                    music.write(str(midi_dir / f"sample_{i:03d}.mid"))
            except Exception:
                pass
    if dm_pitch_vecs and dm_dur_vecs:
        midi_dir = args.output_dir / "midis" / "diffmean"
        midi_dir.mkdir(parents=True, exist_ok=True)
        for i, seq in enumerate(dm_seqs):
            try:
                music = representation.decode(seq, encoding)
                if music:
                    music.write(str(midi_dir / f"sample_{i:03d}.mid"))
            except Exception:
                pass

    _print_summary(results, "Unconditioned Dual Steering")


def _run_conditioned(model, encoding, sae_model, pitch_sas, duration_sas,
                     dm_pitch_vecs, dm_dur_vecs, args, device):
    """Conditioned dual steering — 4 extreme scenarios from ISMIR paper."""
    from pid_steering.conditioned_pid_evaluator import (
        load_song_tokens, extract_conditioning_prefix, find_extreme_songs,
    )

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"

    # Find extreme songs for both concepts
    logger.info("Finding extreme songs...")
    pitch_low, pitch_high = find_extreme_songs(
        notes_dir, encoding, "average_pitch",
        n_songs=args.n_songs + args.song_offset, conditioning_beats=args.conditioning_beats,
    )
    dur_low, dur_high = find_extreme_songs(
        notes_dir, encoding, "average_duration",
        n_songs=args.n_songs + args.song_offset, conditioning_beats=args.conditioning_beats,
    )
    # Apply song offset to skip already-generated songs
    if args.song_offset > 0:
        pitch_low = pitch_low[args.song_offset:]
        pitch_high = pitch_high[args.song_offset:]
        dur_low = dur_low[args.song_offset:]
        dur_high = dur_high[args.song_offset:]
        logger.info(f"Song offset={args.song_offset}: skipping first {args.song_offset} songs per group")

    # 4 scenarios matching ISMIR paper Table 4
    scenarios = [
        {
            "name": "low_short_to_high_long",
            "pitch_songs": pitch_low, "dur_songs": dur_low,
            "pitch_dir": "up", "dur_dir": "up",
            "pitch_sign": 1.0, "dur_sign": 1.0,
            "dm_pitch_sign": 1.0, "dm_dur_sign": 1.0,
        },
        {
            "name": "high_long_to_low_short",
            "pitch_songs": pitch_high, "dur_songs": dur_high,
            "pitch_dir": "down", "dur_dir": "down",
            "pitch_sign": -1.0, "dur_sign": -1.0,
            "dm_pitch_sign": -1.0, "dm_dur_sign": -1.0,
        },
        {
            "name": "low_long_to_high_short",
            "pitch_songs": pitch_low, "dur_songs": dur_high,
            "pitch_dir": "up", "dur_dir": "down",
            "pitch_sign": 1.0, "dur_sign": -1.0,
            "dm_pitch_sign": 1.0, "dm_dur_sign": -1.0,
        },
        {
            "name": "high_short_to_low_long",
            "pitch_songs": pitch_high, "dur_songs": dur_low,
            "pitch_dir": "down", "dur_dir": "up",
            "pitch_sign": -1.0, "dur_sign": 1.0,
            "dm_pitch_sign": -1.0, "dm_dur_sign": 1.0,
        },
    ]

    all_results = {}

    for scenario in scenarios:
        name = scenario["name"]
        logger.info(f"\n{'='*70}")
        logger.info(f"Scenario: {name}")
        logger.info(f"{'='*70}")

        # Use minimum of available songs from each concept
        n_use = min(len(scenario["pitch_songs"]), len(scenario["dur_songs"]), args.n_songs)
        method_seqs = {"dual_pid": [], "static_sas": [], "baseline": []}
        if dm_pitch_vecs:
            method_seqs["diffmean"] = []
        pid_diags = []
        sample_ids = []

        directed_pitch = pitch_sas * scenario["pitch_sign"]
        directed_dur = duration_sas * scenario["dur_sign"]

        for song_idx in range(n_use):
            # Use pitch song for conditioning (could also interleave)
            filepath, _ = scenario["pitch_songs"][song_idx]
            song_stem = filepath.stem  # SOD piece name
            abs_idx = song_idx + args.song_offset
            tokens = load_song_tokens(filepath, encoding)
            cond = extract_conditioning_prefix(tokens, args.conditioning_beats)

            for rep_idx in range(args.n_per_song):
                sample_id = f"song{abs_idx}_{song_stem}_rep{rep_idx}"

                # Check skip_existing
                if args.skip_existing:
                    pid_path = args.output_dir / "conditioned" / name / "dual_pid" / f"{sample_id}.mid"
                    if pid_path.exists():
                        logger.info(f"  Skipping {sample_id} (already exists)")
                        continue

                # 1. Dual PID
                hook = DualTemporalPIDSASHook(
                    sae_model=sae_model,
                    pitch_vector=directed_pitch,
                    duration_vector=directed_dur,
                    target_magnitude=args.target_magnitude,
                    Kp=args.Kp, Ki=args.Ki, Kd=args.Kd, max_I=args.max_I,
                    lambda_max=args.lambda_max, n_ramp_beats=args.n_ramp_steps,
                    orthogonalize=True, k_multiplier=2.0,
                    skip_conditioning=True,
                )
                seq, diag = generate_dual_pid(
                    model, encoding, hook, args.target_layer,
                    cond, args.continuation_len, device,
                )
                method_seqs["dual_pid"].append(seq)
                pid_diags.append(diag)

                # 2. Static SAS (gs_ek2)
                seq_s = generate_dual_static_sas(
                    model, encoding, sae_model, directed_pitch, directed_dur,
                    args.target_layer,
                    args.sas_lambda_pitch, args.sas_lambda_duration,
                    cond, args.continuation_len, device,
                )
                method_seqs["static_sas"].append(seq_s)

                # 3. DiffMean
                if dm_pitch_vecs and dm_dur_vecs:
                    seq_dm = generate_dual_diffmean(
                        model, encoding, dm_pitch_vecs, dm_dur_vecs,
                        args.dm_alpha_pitch * scenario["dm_pitch_sign"],
                        args.dm_alpha_duration * scenario["dm_dur_sign"],
                        cond, args.continuation_len, device,
                    )
                    method_seqs["diffmean"].append(seq_dm)

                # 4. Baseline
                seq_b = generate_baseline(model, encoding, cond, args.continuation_len, device)
                method_seqs["baseline"].append(seq_b)
                sample_ids.append(sample_id)

            if (song_idx + 1) % 5 == 0:
                logger.info(f"  Processed {song_idx+1}/{n_use} songs")

        # Evaluate
        scenario_results = {}
        for method, seqs in method_seqs.items():
            scenario_results[method] = evaluate_method(seqs, encoding)

        # Dual success
        base_ml = [compute_generation_metrics(s, encoding) for s in method_seqs["baseline"]]
        for method in ["dual_pid", "static_sas"]:
            ml = [compute_generation_metrics(s, encoding) for s in method_seqs[method]]
            scenario_results[method]["dual_success_rate"] = compute_dual_success(
                ml, base_ml, scenario["pitch_dir"], scenario["dur_dir"])
        if "diffmean" in method_seqs:
            ml = [compute_generation_metrics(s, encoding) for s in method_seqs["diffmean"]]
            scenario_results["diffmean"]["dual_success_rate"] = compute_dual_success(
                ml, base_ml, scenario["pitch_dir"], scenario["dur_dir"])

        # PID diagnostics summary
        if pid_diags:
            scenario_results["dual_pid"]["avg_lambda_pitch"] = float(np.mean([
                np.mean(d["pitch_lambda_trajectory"]) for d in pid_diags if d["pitch_lambda_trajectory"]]))
            scenario_results["dual_pid"]["avg_lambda_duration"] = float(np.mean([
                np.mean(d["duration_lambda_trajectory"]) for d in pid_diags if d["duration_lambda_trajectory"]]))

        all_results[name] = scenario_results

        # Save MIDIs per scenario
        for method, seqs in method_seqs.items():
            midi_dir = args.output_dir / "conditioned" / name / method
            midi_dir.mkdir(parents=True, exist_ok=True)
            for i, seq in enumerate(seqs):
                sid = sample_ids[i] if i < len(sample_ids) else f"sample_{i:03d}"
                try:
                    music = representation.decode(seq, encoding)
                    if music:
                        music.write(str(midi_dir / f"{sid}.mid"))
                except Exception:
                    pass

        _print_summary(scenario_results, f"Conditioned: {name}")

    # Save all
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "dual_conditioned_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"Saved to {args.output_dir}")


def _print_summary(results: dict, title: str):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Method':<22} {'Pitch':<8} {'Dur':<8} {'Entropy':<9} "
          f"{'Scale%':<8} {'Groove%':<9} {'δ':<6} {'Dual%':<7}")
    print("-" * 80)
    for method, m in results.items():
        print(
            f"{method:<22} "
            f"{m.get('average_pitch_mean', 0):<8.2f} "
            f"{m.get('average_duration_mean', 0):<8.2f} "
            f"{m.get('pitch_class_entropy_mean', 0):<9.3f} "
            f"{m.get('scale_consistency_mean', 0):<8.1f} "
            f"{m.get('groove_consistency_mean', 0):<9.1f} "
            f"{m.get('degradation_mean', 0):<6.2f} "
            f"{m.get('dual_success_rate', '-')}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
