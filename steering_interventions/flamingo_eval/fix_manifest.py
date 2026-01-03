#!/usr/bin/env python3
"""Fix manifest.json - remove entries without actual WAV files."""

import json
import pathlib
import sys


def main():
    # Run from steering_interventions/ directory
    base_dir = pathlib.Path(".")
    manifest_path = base_dir / "flamingo_eval" / "manifest.json"

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return 1

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Original manifest: {len(manifest)} entries")

    # Filter entries with actual WAV files
    valid_entries = []
    missing_wavs = []

    for entry in manifest:
        # WAV path is relative to base_dir
        wav_path = base_dir / entry["wav_path"]

        if wav_path.exists():
            valid_entries.append(entry)
        else:
            missing_wavs.append(entry)
            print(f"❌ Missing WAV: {entry['wav_path']}")

    print(f"\n✅ Valid entries: {len(valid_entries)}")
    print(f"❌ Missing WAVs: {len(missing_wavs)}")

    if missing_wavs:
        print("\nCategories with missing WAVs:")
        from collections import Counter

        missing_by_cat = Counter(e["category"] for e in missing_wavs)
        for cat, count in missing_by_cat.items():
            print(f"  {cat}: {count}")

    # Save cleaned manifest
    if len(valid_entries) < len(manifest):
        backup_path = pathlib.Path("flamingo_eval/manifest_original.json")
        print(f"\n📝 Saving backup to: {backup_path}")
        with open(backup_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"📝 Saving cleaned manifest to: {manifest_path}")
        with open(manifest_path, "w") as f:
            json.dump(valid_entries, f, indent=2)

        print(f"\n✅ Manifest cleaned: {len(manifest)} → {len(valid_entries)} entries")
    else:
        print("\n✅ All manifest entries have valid WAV files!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
