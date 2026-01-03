#!/usr/bin/env python3
"""Copy Category IV WAV files from listening_samples to prepared_audio."""

import json
import pathlib
import shutil
from tqdm import tqdm

manifest_path = pathlib.Path("flamingo_eval/manifest.json")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

# Filter Category IV entries
cat4_entries = [e for e in manifest if e["category"] == "conditional_dual"]

print(f"Found {len(cat4_entries)} Category IV entries\n")

# Source directory
source_dir = pathlib.Path(
    "flamingo_exp/dual/conditional/phase4_conditioned/listening_samples"
)

if not source_dir.exists():
    print(f"❌ Source directory not found: {source_dir}")
    exit(1)

print(f"✅ Found listening_samples at: {source_dir}")

# Create lookup map of source files by filename
source_wavs = {}
for wav_file in source_dir.glob("*.wav"):
    source_wavs[wav_file.name] = wav_file

print(f"Found {len(source_wavs)} WAV files in listening_samples\n")

# Copy files
success = 0
failed = 0
skipped = 0
not_found = []

print("Copying files...")
for entry in tqdm(cat4_entries):
    dest_path = pathlib.Path(entry["wav_path"])

    # Check if already exists
    if dest_path.exists():
        skipped += 1
        continue

    # Get source filename from destination path
    source_filename = dest_path.name

    # Find source file
    if source_filename in source_wavs:
        source_path = source_wavs[source_filename]

        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            success += 1
        except Exception as e:
            print(f"\n❌ Failed to copy {source_filename}: {e}")
            failed += 1
    else:
        not_found.append(source_filename)
        failed += 1

print(f"\n{'='*70}")
print(f"Copy complete:")
print(f"  ✅ Copied: {success}")
print(f"  ⏭️  Skipped (already exist): {skipped}")
print(f"  ❌ Failed: {failed}")

if not_found:
    print(f"\n⚠️  Files not found in source ({len(not_found)}):")
    for filename in not_found[:10]:
        print(f"  - {filename}")
    if len(not_found) > 10:
        print(f"  ... and {len(not_found) - 10} more")

# Verify final count
valid_now = sum(1 for e in manifest if pathlib.Path(e["wav_path"]).exists())
print(f"\nTotal valid WAVs: {valid_now}/{len(manifest)}")

# Show category breakdown
print(f"\nBy category:")
for cat in [
    "unconditional_single",
    "unconditional_dual",
    "conditional_single",
    "conditional_dual",
]:
    total = len([e for e in manifest if e["category"] == cat])
    valid = len(
        [
            e
            for e in manifest
            if e["category"] == cat and pathlib.Path(e["wav_path"]).exists()
        ]
    )
    print(f"  {cat}: {valid}/{total} ({100*valid/total:.0f}%)")
