#!/usr/bin/env python3
"""
Debug script to inspect token file format and validate data.

Usage:
    python inspect_tokens.py [--sample-files 5]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import representation


def inspect_token_file(npy_path: pathlib.Path, encoding: dict):
    """Inspect a single .npy token file."""
    print(f"\n📄 File: {npy_path.name}")
    print(f"   Path: {npy_path}")

    try:
        tokens = np.load(npy_path)
        print(f"   ✓ Loaded successfully")
        print(f"   Shape: {tokens.shape}")
        print(f"   Dtype: {tokens.dtype}")

        if len(tokens.shape) == 2:
            seq_len, n_dims = tokens.shape
            print(f"   Sequence length: {seq_len}")
            print(f"   Dimensions: {n_dims}")

            # Show value ranges for each dimension
            print(f"\n   Value ranges by dimension:")
            dim_names = ["type", "beat", "position", "pitch", "duration", "instrument"]
            for dim_idx in range(n_dims):
                min_val = tokens[:, dim_idx].min()
                max_val = tokens[:, dim_idx].max()
                dim_name = (
                    dim_names[dim_idx] if dim_idx < len(dim_names) else f"dim_{dim_idx}"
                )

                # Get vocab size from encoding
                vocab_size = None
                if dim_name in encoding:
                    vocab_size = len(encoding[dim_name])

                status = (
                    "✓"
                    if vocab_size is None or max_val < vocab_size
                    else "✗ OUT OF BOUNDS!"
                )
                vocab_info = f"(vocab: {vocab_size})" if vocab_size else ""

                print(
                    f"     [{dim_idx}] {dim_name:12s}: [{min_val:6d}, {max_val:6d}] {vocab_info} {status}"
                )

                if vocab_size and max_val >= vocab_size:
                    print(
                        f"         ⚠️  ERROR: max value {max_val} >= vocab size {vocab_size}"
                    )
                    # Show some examples of out-of-bounds values
                    bad_indices = np.where(tokens[:, dim_idx] >= vocab_size)[0]
                    print(f"         Found {len(bad_indices)} out-of-bounds values")
                    if len(bad_indices) > 0:
                        print(
                            f"         First few: {tokens[bad_indices[:5], dim_idx].tolist()}"
                        )

            # Show first few tokens
            print(f"\n   First 5 tokens:")
            for i in range(min(5, seq_len)):
                token = tokens[i]
                token_str = " ".join(f"{int(v):4d}" for v in token)
                print(f"     [{i}] {token_str}")

            return True

        else:
            print(f"   ✗ Unexpected shape: {tokens.shape}")
            return False

    except Exception as e:
        print(f"   ✗ Error loading: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect token file format")
    parser.add_argument(
        "--sample-files",
        type=int,
        default=5,
        help="Number of sample files to inspect (default: 5)",
    )
    parser.add_argument(
        "--segment-file",
        type=pathlib.Path,
        default=None,
        help="Specific segment JSON file to inspect (e.g., high_velocity_segments.json)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TOKEN FILE INSPECTOR")
    print("=" * 80)

    # Load encoding
    encoding_file = config.NOTES_DIR / "encoding.json"
    encoding = representation.load_encoding(encoding_file)
    print(f"\n✓ Loaded encoding from: {encoding_file}")

    # Show vocabulary sizes
    print(f"\nVocabulary sizes:")
    for key in ["type", "beat", "position", "pitch", "duration", "instrument"]:
        if key in encoding:
            print(f"  {key:12s}: {len(encoding[key]):6d}")

    # Determine which files to inspect
    if args.segment_file and args.segment_file.exists():
        # Load segments from specified file
        with open(args.segment_file, "r") as f:
            segments = json.load(f)

        print(f"\n✓ Loaded {len(segments)} segments from: {args.segment_file}")

        # Inspect files from segments
        print(
            f"\nInspecting {min(args.sample_files, len(segments))} sample files from segments..."
        )

        success_count = 0
        for i, segment in enumerate(segments[: args.sample_files]):
            file_name = segment["file_name"]
            sub_folder_name = file_name.split("-")[0]
            npy_path = config.NOTES_DIR / sub_folder_name / f"{file_name}.npy"

            if npy_path.exists():
                if inspect_token_file(npy_path, encoding):
                    success_count += 1
            else:
                print(f"\n📄 File: {file_name}")
                print(f"   ✗ Not found: {npy_path}")

        print(f"\n{'=' * 80}")
        print(
            f"Summary: Successfully inspected {success_count}/{args.sample_files} files"
        )

    else:
        # Just scan the directory for any .npy files
        print(f"\nScanning {config.NOTES_DIR} for .npy files...")

        npy_files = list(config.NOTES_DIR.rglob("*.npy"))[: args.sample_files]

        if not npy_files:
            print("✗ No .npy files found!")
            return

        print(f"Found {len(npy_files)} files, inspecting first {args.sample_files}...")

        success_count = 0
        for npy_path in npy_files:
            if inspect_token_file(npy_path, encoding):
                success_count += 1

        print(f"\n{'=' * 80}")
        print(f"Summary: Successfully inspected {success_count}/{len(npy_files)} files")

    print()


if __name__ == "__main__":
    main()
