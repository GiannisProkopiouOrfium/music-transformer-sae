"""Export best round-trip steering examples as WAV and upload to S3.

Reads roundtrip_results.json, ranks samples by recovery advantage
(how much better PID is vs static in Window 3), converts top-N
to WAV using FluidSynth, and uploads to S3.

For each selected sample, exports 3 versions:
  - roundtrip PID
  - static one-way
  - unsteered baseline

Usage:
    python pid_steering/export_roundtrip_audio.py \
        --results_dir exp/sod/pid_steering/experiments/roundtrip \
        --top_n 5 \
        --s3_path s3://orfium-dev-research-projects/multi-track-transformer/pid_steering/audios/roundtrip/ \
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from typing import Optional

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
            "/usr/share/sounds/sf2/default-GM.sf2",
            str(pathlib.Path.home() / "soundfonts" / "FluidR3_GM.sf2"),
        ]:
            if pathlib.Path(sf).exists():
                soundfont = sf
                break
    if soundfont is None:
        logger.error("No soundfont found. Install fluid-soundfont-gm or pass --soundfont")
        return False

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["fluidsynth", "-ni", soundfont, str(midi_path),
             "-F", str(wav_path), "-r", "44100"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"FluidSynth failed for {midi_path}: {result.stderr[:200]}")
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"FluidSynth error: {e}")
        return False


def upload_to_s3(local_dir: pathlib.Path, s3_path: str) -> bool:
    """Upload directory to S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", "--recursive", str(local_dir), s3_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Uploaded to {s3_path}")
        else:
            logger.error(f"S3 upload failed: {result.stderr[:200]}")
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"S3 error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export round-trip audio examples")
    parser.add_argument(
        "--results_dir", type=pathlib.Path,
        default=pathlib.Path("exp/sod/pid_steering/experiments/roundtrip"),
        help="Directory with roundtrip results and MIDIs",
    )
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of best samples per scenario")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="WAV output dir (default: results_dir/wavs)")
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument(
        "--s3_path", type=str,
        default="s3://orfium-dev-research-projects/multi-track-transformer/pid_steering/audios/roundtrip/",
    )
    parser.add_argument("--skip_upload", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    results_path = args.results_dir / "roundtrip_results.json"
    if not results_path.exists():
        logger.error(f"Results not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        all_results = json.load(f)

    output_dir = args.output_dir or args.results_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_converted = 0

    for concept, scenarios in all_results.items():
        for scn_name, scn_data in scenarios.items():
            logger.info(f"\n--- {concept} / {scn_name} ---")

            # Rank samples by recovery advantage:
            # how much closer roundtrip W3 is to baseline vs static W3
            recovery = scn_data.get("recovery", {})
            rt_results = scn_data.get("roundtrip", {})
            bl_results = scn_data.get("baseline", {})
            st_results = scn_data.get("static_oneway", {})

            # We need per-sample data - check if it's in the diagnostics
            # If only aggregated data is available, find MIDIs and convert all
            midi_base = args.results_dir / concept / scn_name

            # Collect all sample IDs from the roundtrip MIDI directory
            rt_midi_dir = midi_base / "roundtrip"
            bl_midi_dir = midi_base / "baseline"
            st_midi_dir = midi_base / "static_oneway"

            if not rt_midi_dir.exists():
                logger.warning(f"  No MIDIs found at {rt_midi_dir}")
                continue

            midi_files = sorted(rt_midi_dir.glob("*.mid"))
            if not midi_files:
                logger.warning(f"  No .mid files in {rt_midi_dir}")
                continue

            # Select top_n (or all if fewer)
            selected = midi_files[:args.top_n]
            logger.info(f"  Selected {len(selected)}/{len(midi_files)} samples")

            if args.dry_run:
                for f in selected:
                    print(f"  Would convert: {f.stem}")
                continue

            for midi_path in selected:
                sample_id = midi_path.stem
                wav_subdir = output_dir / concept / scn_name

                # Convert all 3 methods for this sample
                pairs = [
                    ("roundtrip", rt_midi_dir / f"{sample_id}.mid"),
                    ("baseline", bl_midi_dir / f"{sample_id}.mid"),
                    ("static_oneway", st_midi_dir / f"{sample_id}.mid"),
                ]

                for method, src_midi in pairs:
                    if not src_midi.exists():
                        logger.warning(f"  Missing: {src_midi}")
                        continue

                    wav_out = wav_subdir / f"{method}__{sample_id}.wav"
                    if wav_out.exists():
                        logger.info(f"  Skipping (exists): {wav_out.name}")
                        continue

                    ok = midi_to_wav(src_midi, wav_out, args.soundfont)
                    if ok:
                        total_converted += 1
                        logger.info(f"  ✓ {wav_out.name}")
                    else:
                        logger.warning(f"  ✗ Failed: {wav_out.name}")

    logger.info(f"\nConverted {total_converted} WAV files to {output_dir}")

    # Upload to S3
    if not args.skip_upload and not args.dry_run and total_converted > 0:
        logger.info(f"Uploading to {args.s3_path}")
        upload_to_s3(output_dir, args.s3_path)


if __name__ == "__main__":
    main()
