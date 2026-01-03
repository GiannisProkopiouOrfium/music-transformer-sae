#!/usr/bin/env python3
"""Comprehensive pre-flight verification for Music Flamingo evaluation."""

import json
import pathlib
import sys
import wave
import numpy as np
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))


def check_wav_file(wav_path: pathlib.Path) -> dict:
    """Check if WAV file is valid and get basic info."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            info = {
                "valid": True,
                "channels": wf.getnchannels(),
                "sample_width": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "frames": wf.getnframes(),
                "duration_sec": wf.getnframes() / wf.getframerate(),
                "size_kb": wav_path.stat().st_size / 1024,
            }
            return info
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "size_kb": wav_path.stat().st_size / 1024 if wav_path.exists() else 0,
        }


def main():
    print("=" * 70)
    print("MUSIC FLAMINGO EVALUATION - PRE-FLIGHT VERIFICATION")
    print("=" * 70)

    # Load manifest
    manifest_path = pathlib.Path("flamingo_eval/manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"\n✅ Manifest loaded: {len(manifest)} entries")

    # Check 1: Verify all WAV files exist
    print(f"\n{'='*70}")
    print("CHECK 1: File Existence")
    print("=" * 70)

    missing = []
    for entry in manifest:
        wav_path = pathlib.Path(entry["wav_path"])
        if not wav_path.exists():
            missing.append(entry["id"])

    if missing:
        print(f"❌ FAILED: {len(missing)} files missing")
        for file_id in missing[:5]:
            print(f"  - {file_id}")
        return 1
    else:
        print(f"✅ PASSED: All {len(manifest)} WAV files exist")

    # Check 2: Verify WAV integrity
    print(f"\n{'='*70}")
    print("CHECK 2: WAV File Integrity")
    print("=" * 70)

    corrupt = []
    too_short = []
    durations = []
    sizes = []

    for entry in tqdm(manifest, desc="Checking WAVs"):
        wav_path = pathlib.Path(entry["wav_path"])
        info = check_wav_file(wav_path)

        if not info["valid"]:
            corrupt.append((entry["id"], info.get("error", "Unknown")))
        else:
            durations.append(info["duration_sec"])
            sizes.append(info["size_kb"])

            if info["duration_sec"] < 5.0:  # Less than 5 seconds
                too_short.append((entry["id"], info["duration_sec"]))

    if corrupt:
        print(f"❌ FAILED: {len(corrupt)} corrupt files")
        for file_id, error in corrupt[:5]:
            print(f"  - {file_id}: {error}")
        return 1
    else:
        print(f"✅ PASSED: All {len(manifest)} WAV files are valid")
        print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        print(f"  Average duration: {np.mean(durations):.1f}s")
        print(f"  Average size: {np.mean(sizes):.0f} KB")

    if too_short:
        print(f"\n⚠️  WARNING: {len(too_short)} files shorter than 5 seconds")
        for file_id, dur in too_short[:3]:
            print(f"  - {file_id}: {dur:.1f}s")

    # Check 3: Category distribution
    print(f"\n{'='*70}")
    print("CHECK 3: Category Distribution")
    print("=" * 70)

    by_category = Counter(e["category"] for e in manifest)
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    expected = {
        "unconditional_single": 14,
        "unconditional_dual": 36,
        "conditional_single": 20,
        "conditional_dual": 27,
    }

    all_match = True
    for cat, exp_count in expected.items():
        actual = by_category.get(cat, 0)
        if actual != exp_count:
            print(f"  ⚠️  {cat}: expected {exp_count}, got {actual}")
            all_match = False

    if all_match:
        print(f"\n✅ PASSED: All categories have expected counts")
    else:
        print(f"\n⚠️  WARNING: Some categories have unexpected counts")

    # Check 4: Manifest structure
    print(f"\n{'='*70}")
    print("CHECK 4: Manifest Structure")
    print("=" * 70)

    required_fields = ["id", "category", "wav_path"]
    missing_fields = []

    for entry in manifest:
        for field in required_fields:
            if field not in entry:
                missing_fields.append((entry.get("id", "unknown"), field))

    if missing_fields:
        print(f"❌ FAILED: {len(missing_fields)} entries missing required fields")
        for file_id, field in missing_fields[:5]:
            print(f"  - {file_id}: missing '{field}'")
        return 1
    else:
        print(f"✅ PASSED: All entries have required fields")

    # Check 5: Config file
    print(f"\n{'='*70}")
    print("CHECK 5: Configuration")
    print("=" * 70)

    config_path = pathlib.Path("flamingo_eval/config.yaml")
    if not config_path.exists():
        print(f"❌ FAILED: config.yaml not found")
        return 1

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_config = ["paths", "prompts", "music_flamingo", "resume"]
    missing_config = [k for k in required_config if k not in config]

    if missing_config:
        print(f"❌ FAILED: Missing config sections: {missing_config}")
        return 1
    else:
        print(f"✅ PASSED: Config file valid")
        print(f"  Resume enabled: {config['resume']['enabled']}")
        print(f"  Checkpoint interval: {config['resume']['checkpoint_interval']}")
        print(f"  Num runs per sample: {config['music_flamingo']['num_runs']}")

    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✅ All checks passed!")
    print(f"\nReady for evaluation:")
    print(f"  Total samples: {len(manifest)}")
    print(f"  Total API calls: {len(manifest) * config['music_flamingo']['num_runs']}")
    print(
        f"  Estimated time: ~{len(manifest) * config['music_flamingo']['num_runs'] * 3 / 60:.0f} minutes"
    )
    print(f"\nNext step: Run test evaluation with sample from each category")

    return 0


if __name__ == "__main__":
    sys.exit(main())
