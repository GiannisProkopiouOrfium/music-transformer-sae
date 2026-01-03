#!/usr/bin/env python3
"""Analyze which WAV files failed and why."""

import json
import pathlib
from collections import Counter

manifest_path = pathlib.Path("flamingo_eval/manifest.json")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

print(f"Total entries: {len(manifest)}\n")

# Categorize entries
valid_wavs = []
missing_wavs = []
invalid_npy = []

for entry in manifest:
    wav_path = pathlib.Path(entry["wav_path"])
    npy_path = pathlib.Path(entry.get("npy_path", "."))

    if wav_path.exists():
        valid_wavs.append(entry)
    elif not npy_path.exists() or str(npy_path) == "." or npy_path.is_dir():
        invalid_npy.append(entry)
        missing_wavs.append(entry)
    else:
        missing_wavs.append(entry)

print("=" * 70)
print(f"✅ Valid WAVs: {len(valid_wavs)}")
print(f"❌ Missing WAVs: {len(missing_wavs)}")
print(f"  └─ Invalid NPY paths: {len(invalid_npy)}")
print(f"  └─ Valid NPY, failed conversion: {len(missing_wavs) - len(invalid_npy)}")

# Show breakdown by category
print("\n" + "=" * 70)
print("Missing WAVs by category:")
missing_by_cat = Counter(e["category"] for e in missing_wavs)
for cat, count in sorted(missing_by_cat.items()):
    print(f"  {cat}: {count}")

# Show invalid NPY paths
if invalid_npy:
    print("\n" + "=" * 70)
    print(f"Entries with invalid NPY paths ({len(invalid_npy)}):")
    for entry in invalid_npy[:10]:
        print(f"  ID: {entry['id']}")
        print(f"    Category: {entry['category']}")
        print(f"    NPY path: {entry.get('npy_path', 'MISSING')}")
        print(f"    WAV path: {entry['wav_path']}")
        print()

    if len(invalid_npy) > 10:
        print(f"  ... and {len(invalid_npy) - 10} more")

# Show a few failed conversions (valid NPY but no WAV)
valid_npy_failed = [e for e in missing_wavs if e not in invalid_npy]
if valid_npy_failed:
    print("\n" + "=" * 70)
    print(f"Failed conversions (valid NPY, no WAV) ({len(valid_npy_failed)}):")
    for entry in valid_npy_failed[:5]:
        npy_path = pathlib.Path(entry["npy_path"])
        print(f"  ID: {entry['id']}")
        print(f"    Category: {entry['category']}")
        print(f"    NPY exists: {npy_path.exists()}")
        print(f"    WAV path: {entry['wav_path']}")
        print()

print("\n" + "=" * 70)
print("Summary by category (valid/total):")
for cat in sorted(set(e["category"] for e in manifest)):
    total = len([e for e in manifest if e["category"] == cat])
    valid = len([e for e in valid_wavs if e["category"] == cat])
    print(f"  {cat}: {valid}/{total} ({100*valid/total:.1f}%)")
