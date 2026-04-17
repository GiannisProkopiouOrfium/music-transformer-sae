#!/usr/bin/env python3
"""
Process audio files for human evaluation.
- Converts all WAV files to MP3 (192 kbps)
- Trims to 35 seconds (with 0.5s fade-out at the end)
- Renames files with clear, self-explanatory names for SoundCloud upload
- Mirrors organized folder structure into AUDIO EVAL TRIM/

Requires: ffmpeg (brew install ffmpeg)
"""

import os
import re
import subprocess
import shutil
from pathlib import Path

SRC = Path("AUDIO EVAL")
DST = Path("AUDIO EVAL TRIM")
TRIM_SEC = 35
FADE_OUT = 0.5  # gentle fade at cut point
BITRATE = "192k"


def convert_and_trim(src_wav: Path, dst_mp3: Path):
    """Convert WAV to MP3, trim to TRIM_SEC with fade-out."""
    dst_mp3.parent.mkdir(parents=True, exist_ok=True)
    fade_start = TRIM_SEC - FADE_OUT
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_wav),
        "-t",
        str(TRIM_SEC),
        "-af",
        f"afade=t=out:st={fade_start}:d={FADE_OUT}",
        "-b:a",
        BITRATE,
        "-ar",
        "44100",
        str(dst_mp3),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ──────────────────────────────────────────────
# Filename cleaning helpers
# ──────────────────────────────────────────────


def clean_song_id(name: str) -> str:
    """Extract a clean song identifier like 'Musicalion-3672' or 'song389'."""
    # Try full form first: Musicalion-1234 or Kunstderfuge-1234
    m = re.search(r"(Musicalion|Kunstderfuge)-(\d+)", name, re.I)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Try bare number prefixed by h/l or just digits
    m = re.search(r"[hl]?(\d{3,4})", name)
    if m:
        return f"song{m.group(1)}"
    return name


def is_baseline(name: str) -> bool:
    lower = name.lower()
    stem = lower.rsplit(".", 1)[0]  # remove extension
    if "baseline" in lower or "_base" in lower:
        return True
    # Only match exact zero value at end of stem (e.g. h389-0, l1109-0)
    # NOT partial matches like _0.5 or ap-0.5
    if stem.endswith("-0") or stem.endswith("_0"):
        return True
    return False


# ──────────────────────────────────────────────
# Process each section
# ──────────────────────────────────────────────


def process_dm_single_pitch():
    """DM AUDIOS / Single Steering / Pitch Steering"""
    src_dir = SRC / "DM AUDIOS" / "Single Steering" / "Pitch Steering"
    dst_dir = DST / "DM" / "single_pitch"

    for f in sorted(src_dir.glob("*.wav")):
        name = f.stem.lower()
        # Parse: h389-0, h389-m01, h708-0, h708-m05, l1109-0, l1109-01
        m = re.match(r"([hl])(\d+)[-_](.*)", name)
        if not m:
            print(f"  [SKIP] Cannot parse: {f.name}")
            continue
        direction_char = m.group(1)  # h = originally high pitch, l = originally low
        song_num = m.group(2)
        param_part = m.group(3)

        direction = "high_to_low" if direction_char == "h" else "low_to_high"

        if param_part in ("0", "0.0"):
            role = "baseline"
            out_name = f"DM_pitch_{direction}_song{song_num}_baseline.mp3"
        else:
            # m01 → alpha=-0.1, m05 → alpha=-0.5, 01 → alpha=+0.1
            if param_part.startswith("m"):
                alpha = f"-{param_part[1:]}"
            else:
                alpha = f"+{param_part}"
            # Insert dot if missing: m01 → -0.1, m05 → -0.5
            if "." not in alpha:
                alpha_val = alpha[0] + alpha[1] + "." + alpha[2:]  # e.g. -01 → -0.1
                alpha = alpha_val
            out_name = f"DM_pitch_{direction}_song{song_num}_steered_a{alpha}.mp3"

        print(f"  {f.name} → {out_name}")
        convert_and_trim(f, dst_dir / out_name)


def process_dm_single_duration():
    """DM AUDIOS / Single Steering / Duration Steering / {high_to_low, low_to_high}"""
    base = SRC / "DM AUDIOS" / "Single Steering" / "Duration Steering"
    dst_dir = DST / "DM" / "single_duration"

    for direction in ["high_to_low", "low_to_high"]:
        src_dir = base / direction
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*.wav")):
            song_id = clean_song_id(f.stem)
            if is_baseline(f.name):
                out_name = f"DM_duration_{direction}_{song_id}_baseline.mp3"
            else:
                # Extract alpha: e.g. _m0.5 or _0.5
                m = re.search(
                    r"[_-](m?\d+\.?\d*)",
                    f.stem.split(song_id)[-1] if song_id in f.stem else f.stem,
                )
                alpha_str = ""
                if m:
                    raw = m.group(1)
                    if raw.startswith("m"):
                        alpha_str = f"a-{raw[1:]}"
                    else:
                        alpha_str = f"a+{raw}"
                out_name = f"DM_duration_{direction}_{song_id}_steered_{alpha_str}.mp3"
            print(f"  {f.name} → {out_name}")
            convert_and_trim(f, dst_dir / out_name)


def _parse_dm_dual_scenario(folder_name: str) -> str:
    """Derive a clean scenario label from the DM dual folder name.

    Source folders look like:
      High_pitch_long_dur_to_low_long_3672
      High_pitch_short_dur_to_low_short_2940
      Low_pitch_short_dur_to_high_long_1367
      low_pitch_long_dur_to_low_short
    """
    low = folder_name.lower()
    # Strip trailing song number if present
    low = re.sub(r"_\d{3,4}$", "", low)
    # Normalise separators
    low = re.sub(r"[^a-z0-9]+", "_", low).strip("_")
    # Remove filler words
    for w in ("dur", "duration", "pitch"):
        low = low.replace(w, "")
    low = re.sub(r"_+", "_", low).strip("_")
    return low  # e.g. high_long_to_low_long


def process_dm_dual():
    """DM AUDIOS / Dual Steering / multiple subfolders → one subfolder per scenario."""
    base = SRC / "DM AUDIOS" / "Dual Steering"

    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        scenario = _parse_dm_dual_scenario(sub.name)
        dst_dir = DST / "DM" / "dual" / scenario

        for f in sorted(sub.glob("*.wav")):
            song_id = clean_song_id(f.stem)
            if is_baseline(f.name):
                out_name = f"DM_dual_{scenario}_{song_id}_baseline.mp3"
            else:
                # Extract ap and ad values
                ap_m = re.search(r"ap([+-]?\d+\.?\d*)", f.stem)
                ad_m = re.search(r"ad([+-]?\d+\.?\d*)", f.stem)
                params = ""
                if ap_m:
                    params += f"_ap{ap_m.group(1)}"
                if ad_m:
                    params += f"_ad{ad_m.group(1)}"
                out_name = f"DM_dual_{scenario}_{song_id}_steered{params}.mp3"
            print(f"  {f.name} → {out_name}")
            convert_and_trim(f, dst_dir / out_name)


def process_sas_single():
    """SAS AUDIOS / Single Steering / {pitch_high_low, pitch_low_high, dur_long_short, dur_short_long}"""
    base = SRC / "SAS AUDIOS" / "Single Steering"
    concept_map = {
        "pitch_high_low": ("pitch", "high_to_low"),
        "pitch_low_high": ("pitch", "low_to_high"),
        "dur_long_short": ("duration", "long_to_short"),
        "dur_short_long": ("duration", "short_to_long"),
    }

    for folder_name, (concept, direction) in concept_map.items():
        src_dir = base / folder_name
        if not src_dir.exists():
            continue
        dst_dir = DST / "SAS" / f"single_{concept}"

        for f in sorted(src_dir.glob("*.wav")):
            song_id = clean_song_id(f.stem)
            if is_baseline(f.name):
                out_name = f"SAS_{concept}_{direction}_{song_id}_baseline.mp3"
            else:
                # Extract lambda: e.g. _0.5, _0.75, _-1.5, _0.25
                m = re.search(r"[-_]([-+]?\d+\.?\d*)\.wav$", f.name)
                lam = ""
                if m:
                    val = m.group(1)
                    if not val.startswith(("-", "+")):
                        val = f"+{val}"
                    lam = f"_l{val}"
                out_name = f"SAS_{concept}_{direction}_{song_id}_steered{lam}.mp3"
            print(f"  {f.name} → {out_name}")
            convert_and_trim(f, dst_dir / out_name)


def process_sas_dual():
    """SAS AUDIOS / Dual Steering / {h2l_long, h2l_short, l2h_long, l2h_short} → one subfolder per scenario."""
    base = SRC / "SAS AUDIOS" / "Dual Steering"
    folder_map = {
        "h2l_long": "high_to_low_long",
        "h2l_short": "high_to_low_short",
        "l2h_long": "low_to_high_long",
        "l2h_short": "low_to_high_short",
    }

    for folder_name, scenario in folder_map.items():
        src_dir = base / folder_name
        if not src_dir.exists():
            continue
        dst_dir = DST / "SAS" / "dual" / scenario

        for f in sorted(src_dir.glob("*.wav")):
            if f.name.startswith("."):
                continue
            song_id = clean_song_id(f.stem)

            if is_baseline(f.name):
                out_name = f"SAS_dual_{scenario}_{song_id}_baseline.mp3"
                print(f"  {f.name} → {out_name}")
                convert_and_trim(f, dst_dir / out_name)
                continue

            # Parse: 22_steered_Musicalion-2940_lp-1.00_ld+0.75_deg0.44.wav
            lp_m = re.search(r"lp([+-]?\d+\.?\d*)", f.stem)
            ld_m = re.search(r"ld([+-]?\d+\.?\d*)", f.stem)
            deg_m = re.search(r"deg(\d+\.?\d*)", f.stem)
            params = ""
            if lp_m:
                params += f"_lp{lp_m.group(1)}"
            if ld_m:
                params += f"_ld{ld_m.group(1)}"
            if deg_m:
                params += f"_deg{deg_m.group(1)}"
            out_name = f"SAS_dual_{scenario}_{song_id}_steered{params}.mp3"
            print(f"  {f.name} → {out_name}")
            convert_and_trim(f, dst_dir / out_name)


def process_smooth():
    """SMOOTH / {pitch_high_low, pitch_low_high, duration_long_short, duration_short_long}"""
    base = SRC / "SMOOTH"
    concept_map = {
        "pitch_high_low": ("pitch", "high_to_low"),
        "pitch_low_high": ("pitch", "low_to_high"),
        "duration_long_short": ("duration", "long_to_short"),
        "duration_short_long": ("duration", "short_to_long"),
    }

    for folder_name, (concept, direction) in concept_map.items():
        src_dir = base / folder_name
        if not src_dir.exists():
            continue
        dst_dir = DST / "SMOOTH" / concept

        for f in sorted(src_dir.glob("*.wav")):
            name_lower = f.stem.lower()
            song_id_m = re.search(r"(\d{3,4})", f.stem)
            song_num = song_id_m.group(1) if song_id_m else "unknown"

            if "abrupt" in name_lower:
                mode = "abrupt"
                out_name = f"SMOOTH_{concept}_{direction}_song{song_num}_{mode}.mp3"
            elif "gradual" in name_lower or "warmup" in name_lower:
                # Extract schedule and ramp length
                sched_m = re.search(r"(gradual|warmup_hold)", name_lower)
                ramp_m = re.search(r"(\d+)(?!.*\d)", f.stem)  # last number = ramp beats
                schedule = sched_m.group(1) if sched_m else "smooth"
                ramp = ramp_m.group(1) if ramp_m else ""
                mode = f"{schedule}_{ramp}beats" if ramp else schedule
                out_name = f"SMOOTH_{concept}_{direction}_song{song_num}_{mode}.mp3"
            else:
                # Fallback: might be a baseline-like abrupt with different naming
                # e.g. h389-m1.wav → abrupt steered version
                m = re.match(r"[hl](\d+)[-_](.*)", f.stem.lower())
                if m:
                    song_num = m.group(1)
                    param = m.group(2)
                    if param.startswith("m"):
                        alpha = f"-{param[1:]}"
                    else:
                        alpha = f"+{param}"
                    out_name = f"SMOOTH_{concept}_{direction}_song{song_num}_abrupt_a{alpha}.mp3"
                else:
                    # Last resort: extract what we can
                    out_name = (
                        f"SMOOTH_{concept}_{direction}_song{song_num}_{f.stem}.mp3"
                    )

            print(f"  {f.name} → {out_name}")
            convert_and_trim(f, dst_dir / out_name)


def _normalize_song_id(sid: str) -> str:
    """Normalize song IDs so 'song708' and 'Musicalion-708' match by number."""
    m = re.search(r"(\d{3,4})", sid)
    return m.group(1) if m else sid


def cross_reference_sas_dual_baselines():
    """For each SAS dual scenario folder, check if every unique song has a baseline.
    If not, find it from any other processed folder and copy it in."""

    sas_dual_base = DST / "SAS" / "dual"
    if not sas_dual_base.exists():
        return

    # Build index: number → existing baseline mp3 path (from ALL sections)
    baseline_index = {}  # e.g. "2940" → Path("...baseline.mp3")
    for mp3 in DST.rglob("*_baseline.mp3"):
        num = _normalize_song_id(mp3.stem)
        if num not in baseline_index:
            baseline_index[num] = mp3

    for scenario_dir in sorted(sas_dual_base.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name

        # Collect song IDs that appear in steered files
        steered_songs = set()
        existing_baselines = set()
        for mp3 in scenario_dir.glob("*.mp3"):
            sid = clean_song_id(mp3.stem)
            num = _normalize_song_id(sid)
            if "_baseline" in mp3.stem:
                existing_baselines.add(num)
            else:
                steered_songs.add((num, sid))

        # Copy missing baselines
        for num, sid in sorted(steered_songs):
            if num in existing_baselines:
                continue
            if num in baseline_index:
                src_mp3 = baseline_index[num]
                dst_name = f"SAS_dual_{scenario}_{sid}_baseline.mp3"
                dst_path = scenario_dir / dst_name
                shutil.copy2(src_mp3, dst_path)
                print(f"  [CROSS-REF] {src_mp3.name} → {dst_name}")
            else:
                print(f"  [MISSING] No baseline found for {sid} (song #{num})")


def main():
    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        return

    # Clean destination
    if DST.exists():
        print(f"Removing existing {DST}/...")
        shutil.rmtree(DST)

    print("=" * 60)
    print("Processing audio files for human evaluation")
    print(f"  Source:      {SRC}")
    print(f"  Destination: {DST}")
    print(f"  Trim:        {TRIM_SEC}s with {FADE_OUT}s fade-out")
    print(f"  Format:      MP3 @ {BITRATE}")
    print("=" * 60)

    print("\n── DM Single Pitch ──")
    process_dm_single_pitch()

    print("\n── DM Single Duration ──")
    process_dm_single_duration()

    print("\n── DM Dual ──")
    process_dm_dual()

    print("\n── SAS Single ──")
    process_sas_single()

    print("\n── SAS Dual ──")
    process_sas_dual()

    print("\n── Smooth Steering ──")
    process_smooth()

    # ── Cross-reference baselines for SAS dual ──
    print("\n── Cross-referencing baselines for SAS dual ──")
    cross_reference_sas_dual_baselines()

    # Print summary
    total = sum(1 for _ in DST.rglob("*.mp3"))
    print(f"\n{'=' * 60}")
    print(f"Done! {total} MP3 files written to {DST}/")
    print(f"\nOutput structure:")
    for d in sorted(DST.rglob("*")):
        if d.is_dir():
            depth = len(d.relative_to(DST).parts)
            print(f"  {'  ' * depth}📁 {d.name}/")
        elif d.suffix == ".mp3":
            depth = len(d.relative_to(DST).parts) - 1
            print(f"  {'  ' * depth}🎵 {d.name}")


if __name__ == "__main__":
    main()
