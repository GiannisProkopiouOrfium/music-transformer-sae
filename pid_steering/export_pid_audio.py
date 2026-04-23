"""Audio Export for PID Steering.

Generates MIDI and WAV audio files for qualitative evaluation.
Exports matched sets: baseline / P-only / PI / PID for side-by-side
listening comparison.

Requires FluidSynth 2.2.5 for audio rendering (available on EC2).
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
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid
import representation

logger = logging.getLogger(__name__)


def sequence_to_midi(
    seq: np.ndarray, output_path: pathlib.Path, encoding: dict
) -> bool:
    """Convert a token sequence to a MIDI file."""
    try:
        music = representation.decode(seq, encoding)
        if music is not None:
            music.write(str(output_path))
            return True
    except Exception as e:
        logger.warning(f"Failed to decode to MIDI: {e}")
    return False


def midi_to_wav(
    midi_path: pathlib.Path,
    wav_path: pathlib.Path,
    soundfont: Optional[str] = None,
    sample_rate: int = 44100,
) -> bool:
    """Convert MIDI to WAV using FluidSynth."""
    if soundfont is None:
        # Common soundfont locations
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
            pathlib.Path.home() / "soundfonts" / "FluidR3_GM.sf2",
        ]
        for sf in candidates:
            if pathlib.Path(sf).exists():
                soundfont = str(sf)
                break

    if soundfont is None:
        logger.warning("No soundfont found — skipping WAV conversion")
        return False

    try:
        cmd = [
            "fluidsynth",
            "-ni",
            soundfont,
            str(midi_path),
            "-F",
            str(wav_path),
            "-r",
            str(sample_rate),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            logger.warning(f"FluidSynth error: {result.stderr}")
    except FileNotFoundError:
        logger.warning("FluidSynth not found — skipping WAV conversion")
    except subprocess.TimeoutExpired:
        logger.warning("FluidSynth timed out")
    return False


def export_comparison_set(
    generator: PIDSteeredGenerator,
    dm_vectors: Dict[int, torch.Tensor],
    encoding: dict,
    concept: str,
    alpha: float,
    Kp: float,
    Ki: float,
    Kd: float,
    max_I: float,
    n_samples: int,
    seq_len: int,
    output_dir: pathlib.Path,
    soundfont: Optional[str] = None,
    device: torch.device = None,
) -> Dict:
    """Generate and export a full comparison set.

    For each sample index, generates:
    - baseline (alpha=0)
    - p_only
    - pi
    - pid
    and saves MIDI + WAV.
    """
    variants = compute_pid_variants(
        dm_vectors,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        max_I=max_I,
        dim=config_pid.MODEL_DIM,
    )

    method_configs = {
        "baseline": (None, 0.0),
        "p_only": (variants["p_only"], alpha),
        "pi": (variants["pi"], alpha),
        "pid": (variants["pid"], alpha),
    }

    results = {}

    for method_name, (vectors, method_alpha) in method_configs.items():
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {method_name} samples (alpha={method_alpha})...")

        sequences = generator.generate_samples(
            pid_vectors=vectors,
            alpha=method_alpha,
            n_samples=n_samples,
            seq_len=seq_len,
        )

        method_metrics = []
        n_midi = 0
        n_wav = 0

        for i, seq in enumerate(sequences):
            # Save raw tokens
            np.save(method_dir / f"tokens_{i}.npy", seq)

            # MIDI
            midi_path = method_dir / f"sample_{i}.mid"
            if sequence_to_midi(seq, midi_path, encoding):
                n_midi += 1

                # WAV
                wav_path = method_dir / f"sample_{i}.wav"
                if midi_to_wav(midi_path, wav_path, soundfont):
                    n_wav += 1

            # Metrics
            metrics = compute_generation_metrics(seq)
            method_metrics.append(metrics)

        # Average metrics
        avg = {}
        if method_metrics:
            for key in method_metrics[0]:
                values = [m[key] for m in method_metrics]
                avg[f"{key}_mean"] = float(np.mean(values))
                avg[f"{key}_std"] = float(np.std(values))

        results[method_name] = {
            "n_samples": n_samples,
            "n_midi": n_midi,
            "n_wav": n_wav,
            "metrics": avg,
        }

        logger.info(f"  {method_name}: {n_midi} MIDI, {n_wav} WAV files")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export PID steering audio for qualitative evaluation"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=10)
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

    results = export_comparison_set(
        generator,
        dm_vectors,
        encoding,
        args.concept,
        args.alpha,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        output_dir=args.output_dir / args.concept,
        soundfont=args.soundfont,
        device=device,
    )

    # Save summary
    results_path = args.output_dir / args.concept / "export_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Audio Export Summary — {args.concept}")
    print("=" * 70)
    print(
        f"{'Method':<12} {'Samples':<10} {'MIDI':<8} {'WAV':<8} "
        f"{'Avg Pitch':<12} {'PC Entropy':<12}"
    )
    print("-" * 65)
    for method, data in results.items():
        m = data["metrics"]
        print(
            f"{method:<12} {data['n_samples']:<10} {data['n_midi']:<8} "
            f"{data['n_wav']:<8} "
            f"{m.get('average_pitch_mean', 0):<12.2f} "
            f"{m.get('pitch_class_entropy_mean', 0):<12.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
