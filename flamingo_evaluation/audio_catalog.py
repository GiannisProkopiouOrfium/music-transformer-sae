"""
Audio catalog: parse processed filenames into structured metadata
and build comparison pairs for each evaluation block.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AudioEntry:
    path: str
    filename: str
    method: str  # DM, SAS, SMOOTH
    concept: str  # pitch, duration, dual
    direction: str  # high_to_low, low_to_high, etc.
    song_id: str  # song389, Musicalion-1951, etc.
    song_num: str  # bare number: 389, 1951 (for cross-matching)
    role: str  # baseline, steered
    params: str  # steering strength: a-0.5, lp-1.00_ld+0.75, etc.
    mode: str = ""  # for SMOOTH: abrupt, gradual_128beats, warmup_hold_32beats


@dataclass
class ComparisonPair:
    block: str  # effectiveness, quality, smooth_vs_abrupt, cross_method
    concept: str  # pitch, duration, dual
    method: str  # DM, SAS, or "DM_vs_SAS"
    song_id: str
    audio_a: AudioEntry
    audio_b: AudioEntry
    label_a: str  # what A represents: "baseline", "abrupt", "DM_steered"
    label_b: str  # what B represents: "steered", "smooth", "SAS_steered"
    direction: str = ""  # steering direction: high_to_low, low_to_high, etc.


def _extract_song_num(song_id: str) -> str:
    m = re.search(r"(\d{3,4})", song_id)
    return m.group(1) if m else song_id


def parse_filename(filepath: Path) -> AudioEntry:
    """Parse a processed MP3 filename into structured AudioEntry."""
    name = filepath.stem

    # Determine method
    if name.startswith("SMOOTH_"):
        method = "SMOOTH"
        rest = name[7:]  # strip SMOOTH_
    elif name.startswith("DM_"):
        method = "DM"
        rest = name[3:]
    elif name.startswith("SAS_"):
        method = "SAS"
        rest = name[4:]
    else:
        raise ValueError(f"Cannot parse method from: {name}")

    # Determine concept
    if rest.startswith("dual_"):
        concept = "dual"
        rest = rest[5:]
    elif rest.startswith("pitch_"):
        concept = "pitch"
        rest = rest[6:]
    elif rest.startswith("duration_"):
        concept = "duration"
        rest = rest[9:]
    else:
        raise ValueError(f"Cannot parse concept from: {name}")

    # Determine role, direction, song_id, params, mode
    role = "baseline" if "_baseline" in name else "steered"

    # Extract song_id
    song_m = re.search(r"((?:Musicalion|Kunstderfuge)-\d+|song\d+)", rest)
    song_id = song_m.group(1) if song_m else "unknown"
    song_num = _extract_song_num(song_id)

    # Direction: everything before the song_id
    if song_m:
        direction = rest[: song_m.start()].rstrip("_")
    else:
        direction = ""

    # Params: everything after song_id and role marker
    params = ""
    if role == "steered":
        param_m = re.search(r"_steered_?(.*)", name)
        if param_m:
            params = param_m.group(1)

    # Mode (for SMOOTH)
    mode = ""
    if method == "SMOOTH":
        if "abrupt" in name:
            mode = "abrupt"
        elif "gradual" in name or "warmup" in name:
            mode_m = re.search(r"(gradual_\d+beats|warmup_hold_\d+beats)", name)
            mode = mode_m.group(1) if mode_m else "smooth"

    return AudioEntry(
        path=str(filepath),
        filename=filepath.name,
        method=method,
        concept=concept,
        direction=direction,
        song_id=song_id,
        song_num=song_num,
        role=role,
        params=params,
        mode=mode,
    )


def build_catalog(audio_dir: Path) -> List[AudioEntry]:
    """Scan directory and parse all MP3 files into AudioEntry objects."""
    catalog = []
    for mp3 in sorted(audio_dir.glob("*.mp3")):
        try:
            entry = parse_filename(mp3)
            catalog.append(entry)
        except ValueError as e:
            print(f"  [SKIP] {mp3.name}: {e}")
    return catalog


def _group_by(entries: List[AudioEntry], *keys) -> Dict[tuple, List[AudioEntry]]:
    """Group entries by arbitrary attribute keys."""
    groups: Dict[tuple, List[AudioEntry]] = {}
    for e in entries:
        key = tuple(getattr(e, k) for k in keys)
        groups.setdefault(key, []).append(e)
    return groups


def build_effectiveness_pairs(catalog: List[AudioEntry]) -> List[ComparisonPair]:
    """
    Block 2: Steering Effectiveness — baseline vs steered.
    For each (method, concept, song), pair baseline with each steered version.
    Only DM and SAS (not SMOOTH).
    """
    pairs = []
    dm_sas = [e for e in catalog if e.method in ("DM", "SAS")]
    groups = _group_by(dm_sas, "method", "concept", "song_num")

    for (method, concept, song_num), entries in sorted(groups.items()):
        baselines = [e for e in entries if e.role == "baseline"]
        steered = [e for e in entries if e.role == "steered"]
        if not baselines or not steered:
            continue
        bl = baselines[0]
        for st in steered:
            pairs.append(
                ComparisonPair(
                    block="effectiveness",
                    concept=concept,
                    method=method,
                    song_id=st.song_id,
                    audio_a=bl,
                    audio_b=st,
                    label_a="baseline",
                    label_b=f"steered ({st.params})",
                    direction=st.direction,
                )
            )
    return pairs


def build_quality_pairs(catalog: List[AudioEntry]) -> List[ComparisonPair]:
    """
    Block 3: Quality Degradation — same pairs as effectiveness but
    scored on quality dimensions. Reuses same pairs.
    """
    pairs = build_effectiveness_pairs(catalog)
    for p in pairs:
        p.block = "quality"
    return pairs


def build_smooth_vs_abrupt_pairs(catalog: List[AudioEntry]) -> List[ComparisonPair]:
    """
    Block 4: Smooth vs Abrupt — pair abrupt with smooth for same song/concept.
    Uses SMOOTH entries.
    """
    pairs = []
    smooth = [e for e in catalog if e.method == "SMOOTH"]
    groups = _group_by(smooth, "concept", "direction", "song_num")

    for (concept, direction, song_num), entries in sorted(groups.items()):
        abrupt = [e for e in entries if e.mode == "abrupt"]
        smooth_modes = [e for e in entries if e.mode and e.mode != "abrupt"]
        if not abrupt or not smooth_modes:
            continue
        ab = abrupt[0]
        for sm in smooth_modes:
            pairs.append(
                ComparisonPair(
                    block="smooth_vs_abrupt",
                    concept=concept,
                    method="SMOOTH",
                    song_id=ab.song_id,
                    audio_a=ab,
                    audio_b=sm,
                    label_a="abrupt",
                    label_b=f"smooth ({sm.mode})",
                    direction=direction,
                )
            )
    return pairs


def build_cross_method_pairs(catalog: List[AudioEntry]) -> List[ComparisonPair]:
    """
    Block 5: Cross-Method — DM steered vs SAS steered on the same song/concept.
    Match by song_num across methods.
    """
    pairs = []
    dm = [e for e in catalog if e.method == "DM" and e.role == "steered"]
    sas = [e for e in catalog if e.method == "SAS" and e.role == "steered"]

    # Group by (concept, song_num)
    dm_groups = _group_by(dm, "concept", "song_num")
    sas_groups = _group_by(sas, "concept", "song_num")

    shared_keys = set(dm_groups.keys()) & set(sas_groups.keys())
    for key in sorted(shared_keys):
        concept, song_num = key
        # Pick one representative from each method
        dm_entry = dm_groups[key][0]
        sas_entry = sas_groups[key][0]
        pairs.append(
            ComparisonPair(
                block="cross_method",
                concept=concept,
                method="DM_vs_SAS",
                song_id=f"song{song_num}",
                audio_a=dm_entry,
                audio_b=sas_entry,
                label_a=f"DM ({dm_entry.params})",
                label_b=f"SAS ({sas_entry.params})",
            )
        )
    return pairs


def build_all_pairs(catalog: List[AudioEntry]) -> Dict[str, List[ComparisonPair]]:
    """Build all comparison pairs organized by evaluation block."""
    return {
        "effectiveness": build_effectiveness_pairs(catalog),
        "quality": build_quality_pairs(catalog),
        "smooth_vs_abrupt": build_smooth_vs_abrupt_pairs(catalog),
        "cross_method": build_cross_method_pairs(catalog),
    }
