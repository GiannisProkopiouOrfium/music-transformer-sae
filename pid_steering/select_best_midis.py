"""Select best MIDI examples and convert to WAV.

After running conditioned_pid_evaluator.py (which saves MIDIs),
this script reads the results JSON, ranks songs by PID advantage,
and converts only the top-N to WAV using FluidSynth.

Usage:
    python pid_steering/select_best_midis.py \
        --results_json exp/sod/pid_steering/experiments/conditioned/conditioned_pid_results.json \
        --top_n 5 \
        --output_dir AUDIO\ EVAL/PID\ AUDIOS
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

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
        logger.error("No soundfont found. Install FluidR3_GM or pass --soundfont")
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


def score_songs(results_data: Dict) -> List[Dict]:
    """Score all songs across all concepts/alphas by PID advantage.

    Scoring: PID advantage = (P_degradation - PID_degradation) +
                              (|PID_change| - |P_change|)
    Higher score = PID is clearly better than P-only.
    """
    scored = []

    for concept, concept_results in results_data.items():
        for song_result in concept_results:
            methods = song_result.get("methods", {})
            if "p_only" not in methods or "pid" not in methods:
                continue

            p = methods["p_only"]
            pid = methods["pid"]

            p_deg = p.get(
                "degradation",
                p.get("degradation", {}).get("total_degradation", float("inf")),
            )
            pid_deg = pid.get(
                "degradation",
                pid.get("degradation", {}).get("total_degradation", float("inf")),
            )

            # Handle nested degradation dict
            if isinstance(p_deg, dict):
                p_deg = p_deg.get("total_degradation", float("inf"))
            if isinstance(pid_deg, dict):
                pid_deg = pid_deg.get("total_degradation", float("inf"))

            p_change = abs(p.get("attribute_change", 0))
            pid_change = abs(pid.get("attribute_change", 0))

            degrad_advantage = p_deg - pid_deg  # positive = PID better
            change_advantage = pid_change - p_change  # positive = PID stronger

            # Combined score: prioritize lower degradation, then stronger effect
            score = degrad_advantage + 0.5 * change_advantage

            scored.append(
                {
                    "concept": concept,
                    "song_name": song_result.get("song_name", "unknown"),
                    "category": song_result.get("category", "unknown"),
                    "alpha": song_result.get("alpha", 0),
                    "score": score,
                    "degrad_advantage": degrad_advantage,
                    "change_advantage": change_advantage,
                    "p_degradation": p_deg,
                    "pid_degradation": pid_deg,
                    "p_change": p.get("attribute_change", 0),
                    "pid_change": pid.get("attribute_change", 0),
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def find_midi_files(
    midis_dir: pathlib.Path,
    song_name: str,
    concept: str,
    category: str,
) -> Dict[str, pathlib.Path]:
    """Find MIDI files for all methods of a given song."""
    midis = {}
    for method in ["baseline", "p_only", "pi", "pid"]:
        method_dir = midis_dir / concept / category / method
        if not method_dir.exists():
            continue
        for f in method_dir.glob(f"{song_name}*.mid"):
            midis[method] = f
            break
    return midis


def main():
    parser = argparse.ArgumentParser(
        description="Select best PID MIDI examples and convert to WAV"
    )
    parser.add_argument(
        "--results_json",
        type=pathlib.Path,
        required=True,
        help="Path to conditioned_pid_results.json from conditioned_pid_evaluator.py",
    )
    parser.add_argument(
        "--midis_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing saved MIDIs (default: same as results_json parent)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Where to save WAVs (default: AUDIO EVAL/PID AUDIOS)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of best songs to convert per scenario (concept × category)",
    )
    parser.add_argument(
        "--per_scenario",
        action="store_true",
        default=True,
        help="Select top_n per scenario (concept × category) to cover all directions",
    )
    parser.add_argument(
        "--all_methods",
        action="store_true",
        default=True,
        help="Convert all methods for selected songs (baseline/P/PI/PID)",
    )
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just print rankings without converting",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load results
    with open(args.results_json) as f:
        results_data = json.load(f)

    midis_dir = args.midis_dir or args.results_json.parent

    project_root = pathlib.Path(__file__).parent.parent
    output_dir = args.output_dir or (project_root / "AUDIO EVAL" / "PID AUDIOS")

    # Score all songs
    all_scored = score_songs(results_data)

    if not all_scored:
        logger.error("No songs with P-only and PID results found")
        return

    # Select top-N per scenario (concept × category) for full coverage
    scenarios = sorted(set((s["concept"], s["category"]) for s in all_scored))
    selected = []

    scenario_labels = {
        ("average_pitch", "low"): "Pitch LOW→HIGH (steer up)",
        ("average_pitch", "high"): "Pitch HIGH→LOW (steer down)",
        ("average_duration", "low"): "Duration SHORT→LONG (steer up)",
        ("average_duration", "high"): "Duration LONG→SHORT (steer down)",
    }

    for concept, category in scenarios:
        scenario_scored = [
            s
            for s in all_scored
            if s["concept"] == concept and s["category"] == category
        ]
        # For best audio: prioritize low PID degradation + clear effect
        # Re-sort by: low PID degradation first, then high |PID change|
        scenario_scored.sort(
            key=lambda s: (s["pid_degradation"], -abs(s["pid_change"]))
        )
        top = scenario_scored[: args.top_n]
        selected.extend(top)

        label = scenario_labels.get((concept, category), f"{concept} / {category}")
        print(f"\n{'='*90}")
        print(f"Top {args.top_n} best audio — {label}")
        print(f"{'='*90}")
        print(
            f"{'Rank':<5} {'Song':<25} {'α':<6} "
            f"{'PID Deg':<9} {'PID Δ':<9} {'P Deg':<9} {'P Δ':<9} "
            f"{'ΔDeg':<8} {'Score':<8}"
        )
        print("-" * 88)
        for i, s in enumerate(top):
            print(
                f"{i+1:<5} {s['song_name']:<25} "
                f"{s['alpha']:<6.1f} "
                f"{s['pid_degradation']:<9.2f} "
                f"{s['pid_change']:+<9.1f} "
                f"{s['p_degradation']:<9.2f} "
                f"{s['p_change']:+<9.1f} "
                f"{s['degrad_advantage']:<8.2f} "
                f"{s['score']:<8.3f}"
            )

    if args.dry_run:
        n_scenarios = len(scenarios)
        print(
            f"\nDry run — would convert {len(selected)} songs "
            f"({args.top_n} × {n_scenarios} scenarios) × 4 methods "
            f"= {len(selected) * 4} WAVs"
        )
        return

    # Convert selected songs
    converted = 0
    failed = 0

    for s in selected:
        midis = find_midi_files(midis_dir, s["song_name"], s["concept"], s["category"])
        if not midis:
            logger.warning(
                f"No MIDIs found for {s['song_name']} "
                f"({s['concept']}/{s['category']})"
            )
            failed += 1
            continue

        methods_to_convert = (
            midis
            if args.all_methods
            else {k: v for k, v in midis.items() if k in ("pid", "p_only", "baseline")}
        )

        for method, midi_path in methods_to_convert.items():
            wav_dir = output_dir / s["concept"] / s["category"]
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / f"{midi_path.stem}_{method}.wav"

            if midi_to_wav(midi_path, wav_path, args.soundfont):
                logger.info(f"  ✓ {wav_path.name}")
                converted += 1
            else:
                logger.warning(f"  ✗ Failed: {midi_path}")
                failed += 1

    print(f"\nConverted {converted} WAVs, {failed} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
