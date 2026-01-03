#!/usr/bin/env python3
"""Show which 15 WAV files actually exist."""

import json
import pathlib
from collections import Counter

manifest_path = pathlib.Path("flamingo_eval/manifest.json")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

valid = [e for e in manifest if pathlib.Path(e["wav_path"]).exists()]

print(f"Valid WAV files: {len(valid)}\n")
print("=" * 70)

by_category = {}
for entry in valid:
    cat = entry["category"]
    if cat not in by_category:
        by_category[cat] = []
    by_category[cat].append(entry)

for cat, entries in sorted(by_category.items()):
    print(f"\n{cat}: {len(entries)} files")
    for e in entries:
        wav_name = pathlib.Path(e["wav_path"]).name
        if cat == "unconditional_single":
            print(f"  - {e['concept']} α={e['alpha']:+.1f}")
        elif cat == "unconditional_dual":
            print(
                f"  - scenario={e.get('scenario', 'unknown')} αp={e.get('alpha_pitch', '?')} αd={e.get('alpha_duration', '?')}"
            )
        else:
            print(f"  - {wav_name}")

print("\n" + "=" * 70)
print(f"\nSummary:")
for cat, entries in sorted(by_category.items()):
    print(f"  {cat}: {len(entries)}")
