#!/usr/bin/env python3
"""Diagnose why NPY to WAV conversion failed for missing files."""

import json
import pathlib
import sys
import numpy as np

# Add paths
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))

try:
    import representation

    HAS_REPRESENTATION = True
except ImportError:
    HAS_REPRESENTATION = False
    print("⚠️  Warning: Could not import representation module")


def main():
    manifest_path = pathlib.Path("flamingo_eval/manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    missing_files = [e for e in manifest if not pathlib.Path(e["wav_path"]).exists()]

    print(f"Analyzing {len(missing_files)} missing WAV files...\n")

    # Check if NPY files exist
    npy_exists = 0
    npy_missing = 0
    categories = {}

    for entry in missing_files[:10]:  # Check first 10
        npy_path = pathlib.Path(entry.get("npy_path", ""))
        category = entry["category"]

        categories[category] = categories.get(category, 0) + 1

        if npy_path.exists():
            npy_exists += 1
            print(f"✅ NPY exists: {npy_path.name}")

            # Try to load it
            try:
                tokens = np.load(npy_path)
                print(f"   Shape: {tokens.shape}, dtype: {tokens.dtype}")

                if HAS_REPRESENTATION:
                    # Try to decode
                    try:
                        # Load encoding
                        encoding_path = pathlib.Path("../baseline/encoding_mmm.json")
                        if encoding_path.exists():
                            with open(encoding_path) as f:
                                encoding = json.load(f)

                            music = representation.decode(tokens, encoding)
                            print(f"   ✅ Decoding successful")
                        else:
                            print(f"   ⚠️  Encoding file not found: {encoding_path}")
                    except Exception as e:
                        print(f"   ❌ Decoding failed: {e}")
            except Exception as e:
                print(f"   ❌ Failed to load NPY: {e}")
        else:
            npy_missing += 1
            print(f"❌ NPY missing: {npy_path}")

    print(f"\n" + "=" * 70)
    print(f"Summary of first 10 missing files:")
    print(f"  NPY files exist: {npy_exists}/10")
    print(f"  NPY files missing: {npy_missing}/10")

    print(f"\nMissing by category:")
    for cat, count in sorted(categories.items()):
        total = len([e for e in missing_files if e["category"] == cat])
        print(f"  {cat}: {total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
