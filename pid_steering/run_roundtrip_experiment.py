"""Round-Trip Steering Experiment.

Demonstrates PID's unique temporal control capability: steer away from
a conditioned prefix, hold, then steer back toward original characteristics.
Static SAS cannot do this (fixed direction for entire generation).

Structure per sample:
    [16-beat prefix] → [steer AWAY 64 tokens] → [hold 64 tokens] → [steer BACK 64 tokens]

Evaluation:
    - Per-window attribute measurement (3 windows of 64 tokens)
    - Recovery metric: how close window 3 is to unsteered baseline window 3
    - Comparison vs unsteered baseline and one-way static SAS

Usage:
    python pid_steering/run_roundtrip_experiment.py \
        --concept all --n_songs 10 --n_per_song 2 --gpu 0
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

from pid_steering.roundtrip_pid_hook import RoundTripPIDSASHook
from pid_steering.temporal_pid_sas_generator import (
    TemporalPIDSASGenerator,
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
    get_target_feature_indices,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.conditioned_pid_evaluator import (
    load_song_tokens,
    extract_conditioning_prefix,
    find_extreme_songs,
)
from pid_steering.run_single_attribute import compute_generation_metrics
from pid_steering.run_conditioned_temporal_pid import (
    generate_static_smooth_conditioned,
    _save_midi,
)

import config_pid
import representation

logger = logging.getLogger(__name__)


# ─── Per-Window Metrics ────────────────────────────────────────────────────────


def compute_window_metrics(
    sequence: np.ndarray,
    prefix_len: int,
    window_sizes: List[int],
    encoding=None,
) -> List[Dict[str, float]]:
    """Compute metrics for each window of the continuation.

    Args:
        sequence: Full token sequence (prefix + continuation).
        prefix_len: Number of tokens in the conditioning prefix.
        window_sizes: List of token counts per window.
        encoding: Encoding dict for muspy metric computation.

    Returns:
        List of metric dicts, one per window.
    """
    continuation = sequence[prefix_len:]
    window_metrics = []
    offset = 0

    for w_size in window_sizes:
        end = min(offset + w_size, len(continuation))
        window_tokens = continuation[offset:end]

        if len(window_tokens) < 5:
            window_metrics.append(
                {
                    "n_notes": 0,
                    "average_pitch": float("nan"),
                    "average_duration": float("nan"),
                    "window_tokens": int(end - offset),
                }
            )
        else:
            # Compute metrics on just this window
            metrics = _compute_metrics_from_tokens(window_tokens)
            metrics["window_tokens"] = int(end - offset)
            window_metrics.append(metrics)

        offset = end

    return window_metrics


def _compute_metrics_from_tokens(tokens: np.ndarray) -> Dict[str, float]:
    """Compute basic metrics from a token array (no muspy needed)."""
    TYPE_IDX = 0
    PITCH_IDX = 3
    DURATION_IDX = 4

    note_mask = tokens[:, TYPE_IDX] == 3
    n_notes = int(note_mask.sum())

    if n_notes < 3:
        return {
            "n_notes": n_notes,
            "average_pitch": float("nan"),
            "average_duration": float("nan"),
            "pitch_class_entropy": float("nan"),
            "scale_consistency": float("nan"),
        }

    pitches = tokens[note_mask, PITCH_IDX].astype(float)
    durations = tokens[note_mask, DURATION_IDX].astype(float)

    # Pitch class entropy
    pitch_classes = pitches.astype(int) % 12
    counts = np.bincount(pitch_classes.astype(int), minlength=12).astype(float)
    counts = counts / counts.sum()
    counts = counts[counts > 0]
    pc_entropy = -np.sum(counts * np.log2(counts))

    # Scale consistency
    major_pattern = [0, 2, 4, 5, 7, 9, 11]
    best_fit = 0
    for root in range(12):
        scale = set((root + p) % 12 for p in major_pattern)
        notes_in_scale = sum(1 for p in pitches if (int(p) % 12) in scale)
        fit = notes_in_scale / len(pitches)
        best_fit = max(best_fit, fit)

    return {
        "n_notes": n_notes,
        "average_pitch": float(pitches.mean()),
        "average_duration": float(durations.mean()),
        "pitch_class_entropy": float(pc_entropy),
        "scale_consistency": float(best_fit * 100.0),
    }


# ─── Generation Functions ──────────────────────────────────────────────────────


def generate_roundtrip(
    model,
    encoding,
    sae_model,
    sas_vector_up: np.ndarray,
    target_layer: int,
    conditioning_prefix: torch.Tensor,
    phases: List[Dict],
    target_magnitude: float,
    Kp: float,
    Ki: float,
    Kd: float,
    max_I: float,
    lambda_max: float,
    device: torch.device,
) -> Tuple[np.ndarray, Dict]:
    """Generate a single round-trip conditioned sample.

    Args:
        model: MMT model.
        encoding: Encoding dict.
        sae_model: Trained SAE.
        sas_vector_up: SAS vector for the "up" direction.
        target_layer: Layer index for SAS hook.
        conditioning_prefix: Tensor (1, cond_len, 6).
        phases: Phase schedule for the round-trip hook.
        target_magnitude, Kp, Ki, Kd, max_I, lambda_max: PID params.
        device: Torch device.

    Returns:
        (full_sequence, diagnostics)
    """
    eos = encoding["type_code_map"]["end-of-song"]
    conditioning_prefix = conditioning_prefix.to(device)

    # Compute target feature indices for both directions
    target_indices_up = get_target_feature_indices(sas_vector_up, top_n=32)
    target_indices_down = get_target_feature_indices(-sas_vector_up, top_n=32)

    hook = RoundTripPIDSASHook(
        sae_model=sae_model,
        sas_vector_up=sas_vector_up,
        target_feature_indices_up=target_indices_up,
        target_feature_indices_down=target_indices_down,
        phases=phases,
        target_magnitude=target_magnitude,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        max_I=max_I,
        lambda_max=lambda_max,
        skip_conditioning=True,
    )
    hook.reset()

    total_continuation = hook.total_tokens

    # Register hook
    if hasattr(model, "decoder"):
        transformer = model.decoder.net
    else:
        transformer = model.net
    attn_layers = transformer.attn_layers
    layer_module = attn_layers.layers[target_layer]
    if isinstance(layer_module, nn.ModuleList) and len(layer_module) > 1:
        target_module = layer_module[1]
    else:
        target_module = layer_module

    handle = target_module.register_forward_hook(
        lambda mod, inp, out, h=hook: h(mod, inp, out, target_layer)
    )

    try:
        with torch.no_grad():
            generated = model.generate(
                conditioning_prefix,
                total_continuation,
                eos_token=eos,
                temperature=config_pid.TEMPERATURE,
                filter_logits_fn="top_k",
                filter_thres=config_pid.FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
        full_seq = torch.cat((conditioning_prefix, generated), 1).cpu().numpy()[0]
    finally:
        handle.remove()

    return full_seq, hook.get_diagnostics()


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Round-trip PID steering experiment")
    parser.add_argument(
        "--concept",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
        help="Concept(s). Use 'all' for both.",
    )
    parser.add_argument("--n_songs", type=int, default=10)
    parser.add_argument("--n_per_song", type=int, default=2)
    parser.add_argument(
        "--conditioning_beats", type=int, default=config_pid.CONDITIONING_BEATS
    )
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=2.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument(
        "--phase_tokens",
        type=int,
        default=64,
        help="Tokens per phase (steer, hold, return)",
    )
    parser.add_argument(
        "--ramp_steps", type=int, default=64, help="PID ramp steps within steer phases"
    )
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki",
        type=float,
        default=None,
        help="Ki (default: concept-specific from paper)",
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "roundtrip",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
    all_results = {}

    # Concept-specific Ki values from the paper
    KI_MAP = {"average_pitch": 0.05, "average_duration": 0.025}

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Round-Trip Experiment — {concept}")
        logger.info(f"{'='*70}")

        Ki = args.Ki if args.Ki is not None else KI_MAP.get(concept, 0.05)

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

        low_songs, high_songs = find_extreme_songs(
            notes_dir,
            encoding,
            concept,
            n_songs=args.n_songs,
            conditioning_beats=args.conditioning_beats,
        )

        # Two starting directions:
        # 1. Low prefix → steer UP → hold → steer DOWN (return)
        # 2. High prefix → steer DOWN → hold → steer UP (return)
        scenarios = [
            {
                "name": "low_up_then_down",
                "songs": low_songs,
                "away_direction": "up",
                "back_direction": "down",
            },
            {
                "name": "high_down_then_up",
                "songs": high_songs,
                "away_direction": "down",
                "back_direction": "up",
            },
        ]

        concept_results = {}

        for scenario in scenarios:
            scn_name = scenario["name"]
            songs = scenario["songs"]

            logger.info(
                f"\n--- {scn_name} ({len(songs)} songs × {args.n_per_song}) ---"
            )

            # Build phase schedule
            phases = [
                {
                    "tokens": args.phase_tokens,
                    "direction": scenario["away_direction"],
                    "ramp_steps": args.ramp_steps,
                },
                {"tokens": args.phase_tokens, "direction": "hold", "ramp_steps": 0},
                {
                    "tokens": args.phase_tokens,
                    "direction": scenario["back_direction"],
                    "ramp_steps": args.ramp_steps,
                },
            ]
            total_continuation = sum(p["tokens"] for p in phases)
            window_sizes = [p["tokens"] for p in phases]

            roundtrip_results = []
            baseline_results = []
            static_results = []
            diagnostics_list = []

            for song_idx, (filepath, init_val) in enumerate(songs):
                song_stem = filepath.stem
                tokens = load_song_tokens(filepath, encoding)
                cond = extract_conditioning_prefix(tokens, args.conditioning_beats)
                prefix_len = cond.shape[1]  # number of prefix tokens

                for rep in range(args.n_per_song):
                    sample_id = f"song{song_idx}_{song_stem}_rep{rep}"

                    if args.skip_existing:
                        pid_path = (
                            args.output_dir
                            / concept
                            / scn_name
                            / "roundtrip"
                            / f"{sample_id}.mid"
                        )
                        if pid_path.exists():
                            logger.info(f"  Skipping {sample_id}")
                            continue

                    # 1. Round-trip PID
                    seq_rt, diag = generate_roundtrip(
                        model,
                        encoding,
                        sae_model,
                        sas_vector,
                        args.target_layer,
                        cond,
                        phases,
                        args.target_magnitude,
                        args.Kp,
                        Ki,
                        args.Kd,
                        args.max_I,
                        args.lambda_max,
                        device,
                    )
                    rt_full = compute_generation_metrics(seq_rt, encoding)
                    rt_windows = compute_window_metrics(
                        seq_rt, prefix_len, window_sizes, encoding
                    )
                    rt_full["windows"] = rt_windows
                    rt_full["init_value"] = float(init_val)
                    rt_full["sample_id"] = sample_id
                    roundtrip_results.append(rt_full)
                    diagnostics_list.append(diag)

                    _save_midi(
                        seq_rt,
                        encoding,
                        args.output_dir / concept / scn_name / "roundtrip",
                        sample_id,
                    )

                    # 2. Unsteered baseline (same prefix, same length)
                    seq_bl = generate_static_smooth_conditioned(
                        model,
                        encoding,
                        sae_model,
                        sas_vector,
                        args.target_layer,
                        conditioning_prefix=cond,
                        continuation_len=total_continuation,
                        lambda_max=0.0,
                        n_ramp_steps=0,
                        device=device,
                    )
                    bl_full = compute_generation_metrics(seq_bl, encoding)
                    bl_windows = compute_window_metrics(
                        seq_bl, prefix_len, window_sizes, encoding
                    )
                    bl_full["windows"] = bl_windows
                    bl_full["init_value"] = float(init_val)
                    bl_full["sample_id"] = sample_id
                    baseline_results.append(bl_full)

                    _save_midi(
                        seq_bl,
                        encoding,
                        args.output_dir / concept / scn_name / "baseline",
                        sample_id,
                    )

                    # 3. One-way static (steer in "away" direction only, no return)
                    directed_vec = (
                        sas_vector
                        if scenario["away_direction"] == "up"
                        else -sas_vector
                    )
                    seq_st = generate_static_smooth_conditioned(
                        model,
                        encoding,
                        sae_model,
                        directed_vec,
                        args.target_layer,
                        conditioning_prefix=cond,
                        continuation_len=total_continuation,
                        lambda_max=args.lambda_max,
                        n_ramp_steps=args.ramp_steps,
                        device=device,
                    )
                    st_full = compute_generation_metrics(seq_st, encoding)
                    st_windows = compute_window_metrics(
                        seq_st, prefix_len, window_sizes, encoding
                    )
                    st_full["windows"] = st_windows
                    st_full["init_value"] = float(init_val)
                    st_full["sample_id"] = sample_id
                    static_results.append(st_full)

                    _save_midi(
                        seq_st,
                        encoding,
                        args.output_dir / concept / scn_name / "static_oneway",
                        sample_id,
                    )

                if (song_idx + 1) % 5 == 0:
                    logger.info(f"  Processed {song_idx + 1}/{len(songs)} songs")

            # ─── Analyze ────────────────────────────────────────────────
            scn_summary = _analyze_scenario(
                concept,
                scn_name,
                roundtrip_results,
                baseline_results,
                static_results,
                diagnostics_list,
                window_sizes,
            )
            concept_results[scn_name] = scn_summary

            # Save diagnostics
            diag_path = args.output_dir / concept / f"diagnostics_{scn_name}.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w") as f:
                json.dump(diagnostics_list, f, indent=2, default=float)

        all_results[concept] = concept_results

    # ─── Save all results ──────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "roundtrip_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nSaved results to {results_path}")

    # Print summary
    _print_summary(all_results)


def _analyze_scenario(
    concept,
    scn_name,
    roundtrip_results,
    baseline_results,
    static_results,
    diagnostics_list,
    window_sizes,
) -> Dict:
    """Compute per-scenario analysis."""
    target_attr = concept  # "average_pitch" or "average_duration"
    summary = {"n_samples": len(roundtrip_results)}

    # Per-window averages
    for method_name, results in [
        ("roundtrip", roundtrip_results),
        ("baseline", baseline_results),
        ("static_oneway", static_results),
    ]:
        window_avgs = []
        for w_idx in range(len(window_sizes)):
            vals = []
            for r in results:
                if "windows" in r and w_idx < len(r["windows"]):
                    v = r["windows"][w_idx].get(target_attr)
                    if v is not None and not np.isnan(v):
                        vals.append(v)
            window_avgs.append(
                {
                    f"{target_attr}_mean": (
                        float(np.mean(vals)) if vals else float("nan")
                    ),
                    f"{target_attr}_std": float(np.std(vals)) if vals else float("nan"),
                    "n_samples": len(vals),
                }
            )
        summary[method_name] = {"windows": window_avgs}

        # Full-piece metrics
        full_vals = [r.get(target_attr, float("nan")) for r in results]
        full_vals = [v for v in full_vals if not np.isnan(v)]
        summary[method_name]["full_mean"] = (
            float(np.mean(full_vals)) if full_vals else float("nan")
        )

    # Recovery metric: |roundtrip_window3 - baseline_window3| / |static_window3 - baseline_window3|
    rt_w3_vals = []
    bl_w3_vals = []
    st_w3_vals = []
    for rt, bl, st in zip(roundtrip_results, baseline_results, static_results):
        if (
            "windows" in rt
            and len(rt["windows"]) >= 3
            and "windows" in bl
            and len(bl["windows"]) >= 3
            and "windows" in st
            and len(st["windows"]) >= 3
        ):
            rt_v = rt["windows"][2].get(target_attr)
            bl_v = bl["windows"][2].get(target_attr)
            st_v = st["windows"][2].get(target_attr)
            if all(v is not None and not np.isnan(v) for v in [rt_v, bl_v, st_v]):
                rt_w3_vals.append(rt_v)
                bl_w3_vals.append(bl_v)
                st_w3_vals.append(st_v)

    if rt_w3_vals:
        rt_w3 = np.array(rt_w3_vals)
        bl_w3 = np.array(bl_w3_vals)
        st_w3 = np.array(st_w3_vals)

        # Recovery error: absolute distance from baseline in window 3
        recovery_error_rt = np.abs(rt_w3 - bl_w3)
        recovery_error_st = np.abs(st_w3 - bl_w3)

        summary["recovery"] = {
            "roundtrip_error_mean": float(recovery_error_rt.mean()),
            "roundtrip_error_std": float(recovery_error_rt.std()),
            "static_error_mean": float(recovery_error_st.mean()),
            "static_error_std": float(recovery_error_st.std()),
            # Recovery success: % within 1 std of baseline
            "roundtrip_success_pct": (
                float((recovery_error_rt < bl_w3.std()).mean() * 100)
                if bl_w3.std() > 0
                else float("nan")
            ),
        }

    # PID diagnostics summary
    if diagnostics_list:
        avg_lambdas = [
            np.mean(d["lambda_trajectory"])
            for d in diagnostics_list
            if d.get("lambda_trajectory")
        ]
        summary["pid_avg_lambda"] = float(np.mean(avg_lambdas)) if avg_lambdas else 0.0

    return summary


def _print_summary(all_results):
    """Print formatted summary of round-trip results."""
    for concept, scenarios in all_results.items():
        attr = concept.replace("average_", "")
        print(f"\n{'='*80}")
        print(f"ROUND-TRIP RESULTS — {concept}")
        print(f"{'='*80}")

        for scn_name, data in scenarios.items():
            print(f"\n--- {scn_name} (n={data.get('n_samples', '?')}) ---")
            print(f"{'Method':<18} {'Window 1':<14} {'Window 2':<14} {'Window 3':<14}")
            print(f"{'':18} {'(steer away)':<14} {'(hold)':<14} {'(steer back)':<14}")
            print("-" * 60)

            for method in ["roundtrip", "baseline", "static_oneway"]:
                mdata = data.get(method, {})
                windows = mdata.get("windows", [])
                vals = []
                for w in windows:
                    v = w.get(f"{concept}_mean", float("nan"))
                    vals.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
                while len(vals) < 3:
                    vals.append("N/A")
                print(f"{method:<18} {vals[0]:<14} {vals[1]:<14} {vals[2]:<14}")

            # Recovery
            rec = data.get("recovery", {})
            if rec:
                print(f"\nRecovery (Window 3 distance from baseline):")
                print(
                    f"  Round-trip: {rec.get('roundtrip_error_mean', 0):.2f} ± {rec.get('roundtrip_error_std', 0):.2f} {attr}"
                )
                print(
                    f"  Static:    {rec.get('static_error_mean', 0):.2f} ± {rec.get('static_error_std', 0):.2f} {attr}"
                )
                pct = rec.get("roundtrip_success_pct", float("nan"))
                if not np.isnan(pct):
                    print(f"  Recovery success (within 1σ): {pct:.0f}%")

            if "pid_avg_lambda" in data:
                print(f"  PID avg λ: {data['pid_avg_lambda']:.3f}")


if __name__ == "__main__":
    main()
