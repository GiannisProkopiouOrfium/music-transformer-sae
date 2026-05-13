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
        logger.error(
            "No soundfont found. Install fluid-soundfont-gm or pass --soundfont"
        )
        return False

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
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Uploaded to {s3_path}")
        else:
            logger.error(f"S3 upload failed: {result.stderr[:200]}")
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"S3 error: {e}")
        return False


def rank_samples(per_sample_path: pathlib.Path, concept: str):
    """Rank samples by recovery advantage (PID closer to baseline than static in W3).

    Returns list of (sample_id, recovery_advantage, rt_w3, bl_w3, st_w3) sorted best-first.
    """
    if not per_sample_path.exists():
        return []

    with open(per_sample_path) as f:
        samples = json.load(f)

    ranked = []
    import math

    for s in samples:
        sample_id = s.get("sample_id", "")
        rt = s.get("roundtrip", {})
        bl = s.get("baseline", {})
        st = s.get("static_oneway", {})

        rt_windows = rt.get("windows", [])
        bl_windows = bl.get("windows", [])
        st_windows = st.get("windows", [])

        if len(rt_windows) < 3 or len(bl_windows) < 3 or len(st_windows) < 3:
            continue

        rt_w3 = rt_windows[2].get(concept, float("nan"))
        bl_w3 = bl_windows[2].get(concept, float("nan"))
        st_w3 = st_windows[2].get(concept, float("nan"))

        if any(math.isnan(v) for v in [rt_w3, bl_w3, st_w3]):
            continue

        # Recovery advantage: how much closer roundtrip is to baseline vs static
        rt_dist = abs(rt_w3 - bl_w3)
        st_dist = abs(st_w3 - bl_w3)
        advantage = st_dist - rt_dist  # positive = PID recovered better

        ranked.append(
            {
                "sample_id": sample_id,
                "advantage": advantage,
                "rt_w3": rt_w3,
                "bl_w3": bl_w3,
                "st_w3": st_w3,
                "rt_dist": rt_dist,
                "st_dist": st_dist,
            }
        )

    ranked.sort(key=lambda x: x["advantage"], reverse=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Export round-trip audio examples")
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/pid_steering/experiments/roundtrip"),
        help="Directory with roundtrip results and MIDIs",
    )
    parser.add_argument(
        "--top_n", type=int, default=5, help="Number of best samples per scenario"
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="WAV output dir (default: results_dir/wavs)",
    )
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument(
        "--s3_path",
        type=str,
        default="s3://orfium-dev-research-projects/multi-track-transformer/pid_steering/audios/roundtrip/",
    )
    parser.add_argument("--skip_upload", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    output_dir = args.output_dir or args.results_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_converted = 0
    concepts = ["average_pitch", "average_duration"]

    for concept in concepts:
        concept_dir = args.results_dir / concept
        if not concept_dir.exists():
            continue

        for scn_dir in sorted(concept_dir.iterdir()):
            if not scn_dir.is_dir():
                continue
            scn_name = scn_dir.name

            # Skip non-scenario dirs (diagnostics files, etc.)
            rt_midi_dir = scn_dir / "roundtrip"
            if not rt_midi_dir.exists():
                continue

            logger.info(f"\n--- {concept} / {scn_name} ---")

            # Try to rank using per-sample data
            per_sample_path = concept_dir / f"per_sample_{scn_name}.json"
            ranked = rank_samples(per_sample_path, concept)

            bl_midi_dir = scn_dir / "baseline"
            st_midi_dir = scn_dir / "static_oneway"

            if ranked:
                # Use ranked ordering
                selected_ids = [r["sample_id"] for r in ranked[: args.top_n]]
                logger.info(
                    f"  Ranked {len(ranked)} samples, selecting top {len(selected_ids)}:"
                )
                for i, r in enumerate(ranked[: args.top_n]):
                    logger.info(
                        f"    #{i+1} {r['sample_id']}: "
                        f"advantage={r['advantage']:.1f}, "
                        f"RT_W3={r['rt_w3']:.1f}, BL_W3={r['bl_w3']:.1f}, ST_W3={r['st_w3']:.1f}"
                    )
            else:
                # Fallback: just take first N MIDI files
                midi_files = sorted(rt_midi_dir.glob("*.mid"))
                selected_ids = [f.stem for f in midi_files[: args.top_n]]
                logger.info(
                    f"  No per-sample data, selecting first {len(selected_ids)}"
                )

            if args.dry_run:
                continue

            for sample_id in selected_ids:
                wav_subdir = output_dir / concept / scn_name

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
                        logger.info(f"  OK {wav_out.name}")
                    else:
                        logger.warning(f"  FAIL {wav_out.name}")

    logger.info(f"\nConverted {total_converted} WAV files to {output_dir}")

    if not args.skip_upload and not args.dry_run and total_converted > 0:
        logger.info(f"Uploading to {args.s3_path}")
        upload_to_s3(output_dir, args.s3_path)


if __name__ == "__main__":
    main()
