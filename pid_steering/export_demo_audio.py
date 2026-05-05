"""Export best PID steering demo audio for presentation & demo page.

Generates conditioned samples for all scenarios, ranks them by quality,
and converts top samples to WAV via FluidSynth.

Scenarios produced:
  Single-concept (4): pitch-up, pitch-down, duration-up, duration-down
  Dual-concept (4): LS→HL, HL→LS, LL→HS, HS→LL

Methods compared per scenario:
  temporal_pid, static_smooth (static SAS), baseline

Pipeline:
  1. Run generation → saves MIDIs + JSON metrics (uses existing scripts)
  2. Read JSON → rank samples by (low δ, high |Δattribute|)
  3. Convert top-N MIDIs to WAV via FluidSynth

Usage (EC2):
    # Install FluidSynth + soundfont first:
    #   sudo apt-get install -y fluidsynth fluid-soundfont-gm
    #
    # Single-concept (all 4 directions):
    python pid_steering/export_demo_audio.py single --gpu 0

    # Dual-concept (all 4 conditioned scenarios):
    python pid_steering/export_demo_audio.py dual --gpu 0

    # Both:
    python pid_steering/export_demo_audio.py all --gpu 0

    # Just convert existing MIDIs (skip generation):
    python pid_steering/export_demo_audio.py convert \
        --results_dir exp/sod/pid_steering/experiments/demo_audio
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config_pid

logger = logging.getLogger(__name__)

# Ground truth for degradation
GT = config_pid.GROUND_TRUTH

# Default soundfont search paths
SOUNDFONT_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    str(pathlib.Path.home() / "soundfonts" / "FluidR3_GM.sf2"),
]


def find_soundfont(user_path: Optional[str] = None) -> Optional[str]:
    """Find a GM soundfont for FluidSynth."""
    if user_path and pathlib.Path(user_path).exists():
        return user_path
    for sf in SOUNDFONT_PATHS:
        if pathlib.Path(sf).exists():
            return sf
    return None


def midi_to_wav(
    midi_path: pathlib.Path,
    wav_path: pathlib.Path,
    soundfont: str,
) -> bool:
    """Convert MIDI to WAV using FluidSynth."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
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
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"FluidSynth failed for {midi_path}: {e}")
        return False


# ─── Step 1: Run Generation ──────────────────────────────────────────────────


def run_single_concept_generation(
    output_dir: pathlib.Path, gpu: int, n_songs: int, n_per_song: int
):
    """Run conditioned temporal PID for all single-concept directions."""
    cmd = [
        sys.executable,
        "pid_steering/run_conditioned_temporal_pid.py",
        "--concept",
        "all",
        "--n_songs",
        str(n_songs),
        "--n_per_song",
        str(n_per_song),
        "--output_dir",
        str(output_dir / "single"),
        "--gpu",
        str(gpu),
    ]
    logger.info(f"Running single-concept generation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(config_pid.PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("Single-concept generation failed")
        return False
    return True


def run_dual_concept_generation(
    output_dir: pathlib.Path, gpu: int, n_songs: int, n_per_song: int
):
    """Run dual temporal PID for conditioned scenarios."""
    cmd = [
        sys.executable,
        "pid_steering/run_dual_temporal_pid.py",
        "--mode",
        "conditioned",
        "--n_songs",
        str(n_songs),
        "--n_per_song",
        str(n_per_song),
        "--output_dir",
        str(output_dir / "dual"),
        "--gpu",
        str(gpu),
    ]
    logger.info(f"Running dual-concept generation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(config_pid.PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("Dual-concept generation failed")
        return False
    return True


# ─── Step 2: Rank Samples ────────────────────────────────────────────────────


def compute_degradation(metrics: dict) -> float:
    """Compute δ from individual metrics."""
    H = metrics.get("pitch_class_entropy", metrics.get("pitch_class_entropy_mean", 0))
    S = metrics.get("scale_consistency", metrics.get("scale_consistency_mean", 0))
    G = metrics.get("groove_consistency", metrics.get("groove_consistency_mean", 0))
    return abs(H - GT["H0"]) + max(0, GT["S0"] - S) + max(0, GT["G0"] - G)


def rank_single_concept_samples(results_dir: pathlib.Path) -> List[Dict]:
    """Rank single-concept samples by quality + steering strength."""
    ranked = []

    # Look for the results JSON
    for json_path in results_dir.glob("**/conditioned_temporal_sas_results.json"):
        with open(json_path) as f:
            data = json.load(f)

        for concept, concept_data in data.items():
            for direction, dir_data in concept_data.items():
                pid_samples = dir_data.get("temporal_pid", {}).get("samples", [])
                static_samples = dir_data.get("static_smooth", {}).get("samples", [])
                baseline_samples = dir_data.get("baseline", {}).get("samples", [])

                if not pid_samples:
                    # Try flat structure
                    pid_samples = dir_data.get("temporal_pid", [])
                    static_samples = dir_data.get("static_smooth", [])
                    baseline_samples = dir_data.get("baseline", [])

                if not isinstance(pid_samples, list):
                    continue

                for i, pid_m in enumerate(pid_samples):
                    if not isinstance(pid_m, dict):
                        continue

                    static_m = static_samples[i] if i < len(static_samples) else {}
                    base_m = baseline_samples[i] if i < len(baseline_samples) else {}

                    attr_key = (
                        "average_pitch" if "pitch" in concept else "average_duration"
                    )

                    pid_delta = compute_degradation(pid_m)
                    static_delta = compute_degradation(static_m) if static_m else 999

                    pid_attr = pid_m.get(attr_key, 0)
                    static_attr = static_m.get(attr_key, 0) if static_m else 0
                    base_attr = base_m.get(attr_key, 0) if base_m else 0

                    pid_change = abs(pid_attr - pid_m.get("init_value", base_attr))
                    static_change = abs(
                        static_attr - (static_m or {}).get("init_value", base_attr)
                    )

                    sample_id = pid_m.get("sample_id", f"sample_{i:03d}")

                    # Score: low δ is good, high |change| is good
                    # PID advantage = how much better PID is than static
                    score = (static_delta - pid_delta) + 0.3 * (
                        pid_change - static_change
                    )

                    ranked.append(
                        {
                            "concept": concept,
                            "direction": direction,
                            "sample_id": sample_id,
                            "pid_delta": pid_delta,
                            "static_delta": static_delta,
                            "pid_attr": pid_attr,
                            "static_attr": static_attr,
                            "base_attr": base_attr,
                            "pid_change": pid_change,
                            "score": score,
                            "results_dir": str(json_path.parent),
                        }
                    )

    # Sort by: lowest PID δ first, then highest score
    ranked.sort(key=lambda x: (x["pid_delta"], -x["score"]))
    return ranked


def rank_dual_concept_samples(results_dir: pathlib.Path) -> List[Dict]:
    """Rank dual-concept samples by quality."""
    ranked = []

    for json_path in results_dir.glob("**/dual_conditioned_results.json"):
        with open(json_path) as f:
            data = json.load(f)

        for scenario_name, scenario_data in data.items():
            pid_data = scenario_data.get("dual_pid", {})
            static_data = scenario_data.get("static_sas", {})

            pid_delta = pid_data.get("degradation_mean", 999)
            static_delta = static_data.get("degradation_mean", 999)

            # For dual, we need per-sample data if available
            # The script saves aggregate stats; MIDIs are in conditioned/scenario/method/
            midi_dir = json_path.parent / "conditioned" / scenario_name
            if not midi_dir.exists():
                continue

            for method_dir in midi_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                for midi_file in sorted(method_dir.glob("*.mid")):
                    ranked.append(
                        {
                            "scenario": scenario_name,
                            "method": method,
                            "sample_id": midi_file.stem,
                            "midi_path": str(midi_file),
                            "pid_delta": (
                                pid_delta if method == "dual_pid" else static_delta
                            ),
                            "results_dir": str(json_path.parent),
                        }
                    )

    ranked.sort(key=lambda x: x.get("pid_delta", 999))
    return ranked


# ─── Step 3: Convert Best to WAV ─────────────────────────────────────────────


def convert_single_concept(
    results_dir: pathlib.Path, wav_dir: pathlib.Path, soundfont: str, top_n: int = 5
):
    """Convert top-N single-concept samples per scenario to WAV."""
    ranked = rank_single_concept_samples(results_dir)
    if not ranked:
        logger.warning("No single-concept results found to rank")
        return

    # Group by concept × direction
    scenarios = {}
    for r in ranked:
        key = (r["concept"], r["direction"])
        scenarios.setdefault(key, []).append(r)

    converted = 0
    print(f"\n{'='*90}")
    print("SINGLE-CONCEPT SAMPLE RANKING")
    print(f"{'='*90}")

    for (concept, direction), samples in sorted(scenarios.items()):
        # Sort this scenario: lowest PID δ, then highest score
        samples.sort(key=lambda x: (x["pid_delta"], -x["score"]))
        top = samples[:top_n]

        print(f"\n  {concept} / {direction} — top {min(top_n, len(samples))}")
        print(
            f"  {'Rank':<5} {'Sample':<20} {'PID δ':<8} {'Static δ':<10} "
            f"{'PID attr':<10} {'Change':<8} {'Score':<8}"
        )
        print(f"  {'-'*75}")

        for rank, s in enumerate(top, 1):
            print(
                f"  {rank:<5} {s['sample_id']:<20} {s['pid_delta']:<8.2f} "
                f"{s['static_delta']:<10.2f} {s['pid_attr']:<10.2f} "
                f"{s['pid_change']:<8.2f} {s['score']:<8.3f}"
            )

            # Convert all 3 methods for this sample
            base_dir = pathlib.Path(s["results_dir"])
            for method in ["temporal_pid", "static_smooth", "baseline"]:
                midi_path = (
                    base_dir / concept / direction / method / f"{s['sample_id']}.mid"
                )
                if not midi_path.exists():
                    logger.debug(f"  MIDI not found: {midi_path}")
                    continue
                out_wav = (
                    wav_dir
                    / "single"
                    / concept
                    / direction
                    / f"{s['sample_id']}_{method}.wav"
                )
                if midi_to_wav(midi_path, out_wav, soundfont):
                    converted += 1

    print(f"\n  Converted {converted} WAVs → {wav_dir / 'single'}")


def convert_dual_concept(
    results_dir: pathlib.Path, wav_dir: pathlib.Path, soundfont: str, top_n: int = 5
):
    """Convert top-N dual-concept samples per scenario to WAV."""
    # For dual, just convert ALL samples (typically small n_per_song × n_songs)
    converted = 0

    for json_path in results_dir.glob("**/dual_conditioned_results.json"):
        cond_dir = json_path.parent / "conditioned"
        if not cond_dir.exists():
            continue

        print(f"\n{'='*90}")
        print("DUAL-CONCEPT SAMPLES")
        print(f"{'='*90}")

        for scenario_dir in sorted(cond_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario = scenario_dir.name
            count = 0

            for method_dir in sorted(scenario_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                midis = sorted(method_dir.glob("*.mid"))[:top_n]

                for midi_path in midis:
                    out_wav = (
                        wav_dir / "dual" / scenario / f"{midi_path.stem}_{method}.wav"
                    )
                    if midi_to_wav(midi_path, out_wav, soundfont):
                        converted += 1
                        count += 1

            print(f"  {scenario}: {count} WAVs")

    print(f"\n  Converted {converted} WAVs → {wav_dir / 'dual'}")


def convert_all_midis(
    results_dir: pathlib.Path, wav_dir: pathlib.Path, soundfont: str, top_n: int = 5
):
    """Convert best samples from all available results."""
    # Single concept
    single_dir = results_dir / "single"
    if single_dir.exists():
        convert_single_concept(single_dir, wav_dir, soundfont, top_n)
    else:
        # Try results_dir directly (may already be the single dir)
        if list(results_dir.glob("**/conditioned_temporal_sas_results.json")):
            convert_single_concept(results_dir, wav_dir, soundfont, top_n)

    # Dual concept
    dual_dir = results_dir / "dual"
    if dual_dir.exists():
        convert_dual_concept(dual_dir, wav_dir, soundfont, top_n)
    else:
        if list(results_dir.glob("**/dual_conditioned_results.json")):
            convert_dual_concept(results_dir, wav_dir, soundfont, top_n)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export best PID steering demo audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate + convert everything:
  python pid_steering/export_demo_audio.py all --gpu 0

  # Single-concept only (pitch + duration, up + down):
  python pid_steering/export_demo_audio.py single --gpu 0

  # Dual-concept only (4 conditioned scenarios):
  python pid_steering/export_demo_audio.py dual --gpu 0

  # Just convert existing MIDIs to WAV (no generation):
  python pid_steering/export_demo_audio.py convert \\
      --results_dir exp/sod/pid_steering/experiments/demo_audio

  # More samples for better selection:
  python pid_steering/export_demo_audio.py all --n_songs 15 --n_per_song 3 --gpu 0
""",
    )

    parser.add_argument(
        "mode",
        choices=["single", "dual", "all", "convert"],
        help="What to generate: single-concept, dual-concept, both, or just convert existing MIDIs",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--n_songs",
        type=int,
        default=10,
        help="Number of extreme songs per direction (default: 10)",
    )
    parser.add_argument(
        "--n_per_song",
        type=int,
        default=2,
        help="Continuations per song (default: 2)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Top-N best samples per scenario to convert to WAV (default: 5)",
    )
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "demo_audio",
        help="Directory for results/MIDIs",
    )
    parser.add_argument(
        "--wav_dir",
        type=pathlib.Path,
        default=config_pid.PROJECT_ROOT / "AUDIO EVAL" / "PID AUDIOS",
        help="Output directory for WAVs",
    )
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument(
        "--s3-upload",
        type=str,
        default=None,
        metavar="S3_URI",
        help="Upload WAVs to S3 after conversion (e.g. s3://bucket/path/)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    soundfont = find_soundfont(args.soundfont)
    if soundfont is None and args.mode != "convert":
        logger.warning(
            "No soundfont found — MIDIs will be generated but WAV conversion will be skipped.\n"
            "Install: sudo apt-get install -y fluidsynth fluid-soundfont-gm"
        )

    # ─── Generation ───────────────────────────────────────────────────────
    if args.mode in ("single", "all"):
        logger.info("=== Generating single-concept conditioned samples ===")
        run_single_concept_generation(
            args.results_dir, args.gpu, args.n_songs, args.n_per_song
        )

    if args.mode in ("dual", "all"):
        logger.info("=== Generating dual-concept conditioned samples ===")
        run_dual_concept_generation(
            args.results_dir, args.gpu, args.n_songs, args.n_per_song
        )

    # ─── Conversion ───────────────────────────────────────────────────────
    if soundfont:
        logger.info(f"=== Converting best MIDIs to WAV (soundfont: {soundfont}) ===")
        convert_all_midis(args.results_dir, args.wav_dir, soundfont, args.top_n)
    else:
        logger.info("Skipping WAV conversion (no soundfont). MIDIs are in:")
        logger.info(f"  {args.results_dir}")

    # ─── S3 Upload ────────────────────────────────────────────────────
    if args.s3_upload and args.wav_dir.exists():
        s3_dest = args.s3_upload.rstrip("/") + "/"
        logger.info(f"=== Uploading WAVs to {s3_dest} ===")
        s3_cmd = [
            "aws",
            "s3",
            "sync",
            str(args.wav_dir),
            s3_dest,
            "--exclude",
            "*",
            "--include",
            "*.wav",
        ]
        logger.info(f"  {' '.join(s3_cmd)}")
        s3_result = subprocess.run(s3_cmd, capture_output=True, text=True)
        if s3_result.returncode == 0:
            logger.info(f"  Uploaded to {s3_dest}")
            if s3_result.stdout.strip():
                print(s3_result.stdout)
        else:
            logger.error(f"  S3 upload failed: {s3_result.stderr}")

    # ─── Summary ──────────────────────────────────────────────────────
    n_midis = len(list(args.results_dir.rglob("*.mid")))
    n_wavs = len(list(args.wav_dir.rglob("*.wav"))) if args.wav_dir.exists() else 0

    print(f"\n{'='*60}")
    print("DEMO AUDIO EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  MIDIs:  {n_midis} files in {args.results_dir}")
    print(f"  WAVs:   {n_wavs} files in {args.wav_dir}")
    if args.s3_upload:
        print(f"  S3:     {args.s3_upload}")
    print("\n  To copy WAVs locally:")
    print(
        "    scp -r ec2-user@<ip>:~/mmt/AUDIO\\ EVAL/PID\\ AUDIOS/ './AUDIO EVAL/PID AUDIOS/'"
    )
    print("\n  Or sync MIDIs too:")
    print(
        f"    scp -r ec2-user@<ip>:~/mmt/{args.results_dir.relative_to(config_pid.PROJECT_ROOT)}/ ./{args.results_dir.relative_to(config_pid.PROJECT_ROOT)}/"
    )


if __name__ == "__main__":
    main()
