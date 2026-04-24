"""Audio Export for PID Steering — Conditioned Comparison.

Generates matched comparison sets using conditioned generation:
- baseline (no steering, same conditioning prefix)
- p_only (standard DiffMean)
- pi (PID without derivative)
- pid (full PID)

Uses first N beats as conditioning prefix (same as DM/SAS evaluators)
for meaningful before/after comparison.

Outputs per song: MIDI + WAV for each method, metrics summary,
and auto-selection of best PID demo examples.
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

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
    evaluate_quality_metrics,
    measure_concept_from_tokens,
    calculate_degradation,
)

import config_pid
import representation

logger = logging.getLogger(__name__)


def midi_to_wav(
    midi_path: pathlib.Path,
    wav_path: pathlib.Path,
    soundfont: Optional[str] = None,
) -> bool:
    """Convert MIDI to WAV using FluidSynth."""
    if soundfont is None:
        for sf in [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            pathlib.Path.home() / "soundfonts" / "FluidR3_GM.sf2",
        ]:
            if pathlib.Path(sf).exists():
                soundfont = str(sf)
                break
    if soundfont is None:
        return False
    try:
        result = subprocess.run(
            [
                "fluidsynth",
                "-ni",
                soundfont,
                str(midi_path),
                "-F",
                str(wav_path),
                "-r",
                "44100",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def export_conditioned_comparison(
    generator: PIDSteeredGenerator,
    pid_variants: Dict[str, Dict[int, torch.Tensor]],
    encoding: Dict,
    device: torch.device,
    song_list: List,
    category: str,
    alpha: float,
    concept: str,
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path,
    soundfont: Optional[str] = None,
) -> List[Dict]:
    """Generate and export conditioned comparison set."""
    eos = encoding["type_code_map"]["end-of-song"]
    gen_kwargs = {
        "eos_token": eos,
        "temperature": config_pid.TEMPERATURE,
        "filter_logits_fn": "top_k",
        "filter_thres": config_pid.FILTER_THRESHOLD,
        "monotonicity_dim": ("type", "beat"),
    }

    method_configs = {"baseline": None}
    method_configs.update(pid_variants)

    results = []

    for i, (filepath, initial_value) in enumerate(song_list):
        logger.info(f"\nSong {i+1}/{len(song_list)}: {filepath.name}")

        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(
            device
        )

        song_result = {
            "song_name": filepath.stem,
            "category": category,
            "initial_value": float(initial_value),
            "methods": {},
        }

        for method_name, vectors in method_configs.items():
            method_alpha = alpha if method_name != "baseline" else 0.0

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
            generated_only = generated.cpu().numpy()[0]

            gen_metrics = measure_concept_from_tokens(generated_only, encoding)
            quality = evaluate_quality_metrics(full_seq, encoding)
            degradation = calculate_degradation(quality)

            # Save MIDI + WAV
            save_dir = output_dir / concept / category / method_name
            save_dir.mkdir(parents=True, exist_ok=True)

            midi_ok, wav_ok = False, False
            midi_path = save_dir / f"{filepath.stem}_a{alpha}.mid"
            wav_path = save_dir / f"{filepath.stem}_a{alpha}.wav"

            try:
                music = representation.decode(full_seq, encoding)
                music.write(str(midi_path))
                midi_ok = True
                wav_ok = midi_to_wav(midi_path, wav_path, soundfont)
            except Exception as e:
                logger.warning(f"Export failed: {e}")

            attr_key = "pitch_mean" if "pitch" in concept else "duration_mean"
            song_result["methods"][method_name] = {
                "attr_value": gen_metrics[attr_key],
                "attr_change": gen_metrics[attr_key] - initial_value,
                "n_notes": gen_metrics["n_notes"],
                "quality": quality,
                "degradation": degradation["total_degradation"],
                "midi": midi_ok,
                "wav": wav_ok,
            }

            logger.info(
                f"  {method_name:8s}: {attr_key}={gen_metrics[attr_key]:.1f}, "
                f"change={gen_metrics[attr_key] - initial_value:+.1f}, "
                f"degrad={degradation['total_degradation']:.2f}"
            )

        results.append(song_result)

    return results


def select_best_demos(results: List[Dict], n_demos: int = 3) -> List[Dict]:
    """Select best demo songs where PID shows clear advantage over P-only."""
    scored = []
    for r in results:
        methods = r["methods"]
        if "p_only" not in methods or "pid" not in methods:
            continue
        p = methods["p_only"]
        pid = methods["pid"]
        change_adv = abs(pid["attr_change"]) - abs(p["attr_change"])
        degrad_adv = p["degradation"] - pid["degradation"]
        scored.append(
            {
                "song": r,
                "score": change_adv + degrad_adv,
                "change_advantage": change_adv,
                "degrad_advantage": degrad_adv,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_demos]


def main():
    parser = argparse.ArgumentParser(
        description="Export conditioned PID audio comparisons"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--n_songs", type=int, default=5)
    parser.add_argument("--conditioning_beats", type=int, default=4)
    parser.add_argument("--continuation_len", type=int, default=256)
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
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PROJECT_ROOT / "AUDIO EVAL" / "PID AUDIOS",
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

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
    low_songs, high_songs = find_extreme_songs(
        notes_dir,
        encoding,
        args.concept,
        n_songs=args.n_songs,
    )

    all_results = []

    # Low songs: steer UP
    logger.info(f"\n--- Low {args.concept} songs (α=+{args.alpha}) ---")
    low_results = export_conditioned_comparison(
        generator,
        variants,
        encoding,
        device,
        low_songs,
        "low",
        alpha=args.alpha,
        concept=args.concept,
        conditioning_beats=args.conditioning_beats,
        continuation_len=args.continuation_len,
        output_dir=args.output_dir,
        soundfont=args.soundfont,
    )
    all_results.extend(low_results)

    # High songs: steer DOWN
    logger.info(f"\n--- High {args.concept} songs (α=-{args.alpha}) ---")
    high_results = export_conditioned_comparison(
        generator,
        variants,
        encoding,
        device,
        high_songs,
        "high",
        alpha=-args.alpha,
        concept=args.concept,
        conditioning_beats=args.conditioning_beats,
        continuation_len=args.continuation_len,
        output_dir=args.output_dir,
        soundfont=args.soundfont,
    )
    all_results.extend(high_results)

    # Select best demos
    best_demos = select_best_demos(all_results)
    if best_demos:
        print(f"\n{'='*70}")
        print("Best PID demo songs (PID advantage over P-only):")
        print(f"{'='*70}")
        for demo in best_demos:
            s = demo["song"]
            print(
                f"  {s['song_name']} ({s['category']}): "
                f"change_adv={demo['change_advantage']:+.1f}, "
                f"degrad_adv={demo['degrad_advantage']:+.2f}"
            )

    # Save summary
    summary_path = args.output_dir / args.concept / "audio_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "results": all_results,
                "best_demos": (
                    [d["song"]["song_name"] for d in best_demos] if best_demos else []
                ),
            },
            f,
            indent=2,
            default=str,
        )

    # Print summary
    print(f"\n{'='*70}")
    print(f"Audio Export Summary — {args.concept}")
    print(f"{'='*70}")
    print(
        f"{'Method':<12} {'Songs':<8} {'Avg |Change|':<14} {'Avg Degrad':<12} {'MIDI':<6} {'WAV':<6}"
    )
    print("-" * 58)
    for method in ["baseline", "p_only", "pi", "pid"]:
        changes, degrads, midis, wavs = [], [], 0, 0
        for r in all_results:
            if method in r["methods"]:
                m = r["methods"][method]
                changes.append(abs(m["attr_change"]))
                degrads.append(m["degradation"])
                midis += int(m["midi"])
                wavs += int(m["wav"])
        if changes:
            print(
                f"{method:<12} {len(changes):<8} {np.mean(changes):<14.1f} "
                f"{np.mean(degrads):<12.2f} {midis:<6} {wavs:<6}"
            )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
