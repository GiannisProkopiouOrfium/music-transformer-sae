"""Select best temporal PID SAS MIDIs and convert to WAV.

Reads MIDIs produced by run_temporal_fmd.py, computes per-sample
quality metrics, selects top-N examples per concept/method, converts
to WAV via FluidSynth, and optionally uploads to S3.

Usage:
    python pid_steering/select_temporal_audios.py \
        --midis_dir exp/sod/pid_steering/experiments/temporal_fmd \
        --top_n 5 \
        --s3_path s3://orfium-dev-research-projects/multi-track-transformer/pid_steering/audios/temporal_sas
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.run_single_attribute import compute_generation_metrics
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
            str(pathlib.Path.home() / "soundfonts" / "FluidR3_GM.sf2"),
        ]:
            if pathlib.Path(sf).exists():
                soundfont = sf
                break
    if soundfont is None:
        logger.error("No soundfont found")
        return False
    try:
        result = subprocess.run(
            ["fluidsynth", "-ni", soundfont, str(midi_path),
             "-F", str(wav_path), "-r", "44100"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def score_midis(midi_dir: pathlib.Path, encoding: Dict, concept: str) -> List[Dict]:
    """Score MIDIs by quality (low degradation from ground truth)."""
    scored = []
    gt = config_pid.GROUND_TRUTH

    for midi_path in sorted(midi_dir.glob("*.mid")):
        try:
            import muspy
            music = muspy.read(str(midi_path))
            if music is None or not music.tracks:
                continue
            # Convert to token representation for metrics
            codes = representation.encode(music, encoding)
            if codes is None or len(codes) == 0:
                continue
            metrics = compute_generation_metrics(codes, encoding)
        except Exception:
            # Fallback: read MIDI directly with muspy for basic metrics
            try:
                import muspy
                music = muspy.read(str(midi_path))
                if music is None:
                    continue
                notes = [n for t in music.tracks for n in t.notes]
                if not notes:
                    continue
                pitches = [n.pitch for n in notes]
                durations = [n.duration for n in notes]
                metrics = {
                    "average_pitch": float(np.mean(pitches)),
                    "pitch_class_entropy": float(muspy.pitch_class_entropy(music)),
                    "scale_consistency": float(muspy.scale_consistency(music) * 100),
                    "groove_consistency": float(muspy.groove_consistency(music, 24) * 100),
                    "average_duration": float(np.mean(durations)),
                }
            except Exception as e:
                logger.warning(f"Skipping {midi_path.name}: {e}")
                continue

        # Compute degradation from ground truth
        entropy_dev = abs(metrics.get("pitch_class_entropy", 0) - gt["pitch_class_entropy"])
        scale_loss = max(0, gt["scale_consistency"] - metrics.get("scale_consistency", 0))
        groove_loss = max(0, gt["groove_consistency"] - metrics.get("groove_consistency", 0))
        degradation = entropy_dev + scale_loss + groove_loss

        # Steering effect: deviation from baseline average
        attr_key = concept.replace("average_", "average_")
        attr_value = metrics.get(attr_key, 0)

        scored.append({
            "midi_path": midi_path,
            "metrics": metrics,
            "degradation": degradation,
            "attribute_value": attr_value,
        })

    # Sort by low degradation (best quality first)
    scored.sort(key=lambda x: x["degradation"])
    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Select best temporal PID SAS MIDIs and convert to WAV"
    )
    parser.add_argument(
        "--midis_dir", type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "temporal_fmd",
    )
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=None,
        help="WAV output dir (default: AUDIO EVAL/SAS PID AUDIOS)",
    )
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument("--s3_path", type=str, default=None)
    parser.add_argument("--keep_local", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    project_root = pathlib.Path(__file__).parent.parent
    output_dir = args.output_dir or (project_root / "AUDIO EVAL" / "SAS PID AUDIOS")

    encoding_path = config_pid.ENCODING_PATH
    with open(encoding_path) as f:
        encoding = json.load(f)

    concepts = ["average_pitch", "average_duration"]
    methods = ["temporal_pid", "static_smooth", "baseline"]

    converted = 0
    failed = 0

    for concept in concepts:
        for method in methods:
            midi_dir = args.midis_dir / concept / method
            if not midi_dir.exists():
                logger.warning(f"Missing: {midi_dir}")
                continue

            n_midis = len(list(midi_dir.glob("*.mid")))
            logger.info(f"\n{concept}/{method}: {n_midis} MIDIs")

            scored = score_midis(midi_dir, encoding, concept)
            if not scored:
                logger.warning(f"  No valid MIDIs")
                continue

            top = scored[:args.top_n]

            print(f"\n{'='*70}")
            print(f"Top {args.top_n} — {concept} / {method}")
            print(f"{'='*70}")
            print(f"{'Rank':<5} {'File':<20} {'Attr':<10} {'Degrad':<10} {'Entropy':<10} {'Scale%':<8}")
            print("-" * 65)
            for i, s in enumerate(top):
                m = s["metrics"]
                print(
                    f"{i+1:<5} {s['midi_path'].name:<20} "
                    f"{s['attribute_value']:<10.2f} "
                    f"{s['degradation']:<10.2f} "
                    f"{m.get('pitch_class_entropy', 0):<10.3f} "
                    f"{m.get('scale_consistency', 0):<8.1f}"
                )

            if args.dry_run:
                continue

            # Convert to WAV
            for i, s in enumerate(top):
                wav_dir = output_dir / concept / method
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / f"top{i+1}_{s['midi_path'].stem}.wav"

                if midi_to_wav(s["midi_path"], wav_path, args.soundfont):
                    logger.info(f"  -> {wav_path.name}")
                    converted += 1
                else:
                    logger.warning(f"  FAILED: {s['midi_path'].name}")
                    failed += 1

    print(f"\nConverted {converted} WAVs, {failed} failed")
    if not args.dry_run:
        print(f"Output: {output_dir}")

    # Upload to S3
    if args.s3_path and converted > 0:
        s3_dest = args.s3_path.rstrip("/")
        print(f"\nUploading to {s3_dest}/ ...")
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", str(output_dir), f"{s3_dest}/",
                 "--recursive", "--exclude", "*", "--include", "*.wav"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                uploaded = result.stdout.count("upload:")
                print(f"Uploaded {uploaded} files to S3")
                if not args.keep_local:
                    import shutil
                    shutil.rmtree(output_dir)
                    print(f"Cleaned up local dir: {output_dir}")
            else:
                logger.error(f"S3 upload failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"S3 upload error: {e}")


if __name__ == "__main__":
    main()
