#!/usr/bin/env python3
"""
Convert MusPy JSON files to .pt token files

This script converts MusPy JSON files to encoded token tensors (.pt files)
that can be used for song-conditioned interventions.

Usage:
    python convert_json_to_pt.py \
        --json-dir ../data/sod/processed/json/Kunstderfuge \
        --output-dir ../data/sod/processed/notes/Kunstderfuge \
        --song-names contrasting_songs_results_3_182/contrasting_songs_layer3_top10.txt
"""

import argparse
import sys
from pathlib import Path
import torch
import muspy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmt import representation


def convert_json_to_pt(json_path: Path, output_path: Path, encoding: dict):
    """
    Convert a single MusPy JSON file to encoded tokens (.pt).

    Args:
        json_path: Path to input .json file
        output_path: Path to output .pt file
        encoding: Encoding dictionary
    """
    try:
        # Load MusPy JSON
        music = muspy.load(str(json_path))

        # Encode to tokens using the representation
        tokens = representation.encode(music, encoding)

        # Convert to tensor and save
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Save as .pt file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens_tensor, output_path)

        print(
            f"✅ Converted: {json_path.name} -> {output_path.name} (shape: {tokens_tensor.shape})"
        )
        return True

    except Exception as e:
        print(f"❌ Failed to convert {json_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert MusPy JSON files to encoded token .pt files"
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory containing MusPy JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for .pt files"
    )
    parser.add_argument(
        "--song-names",
        type=Path,
        default=None,
        help="Text file with song names to convert (one per line). If not provided, converts all JSON files.",
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default=Path("data/sod/processed/notes/encoding.json"),
        help="Path to encoding.json file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MusPy JSON to .pt Converter")
    print("=" * 80)
    print(f"JSON directory: {args.json_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Encoding: {args.encoding_path}")
    print("")

    # Load encoding
    if not args.encoding_path.exists():
        print(f"❌ Error: Encoding file not found: {args.encoding_path}")
        sys.exit(1)

    print(f"Loading encoding from: {args.encoding_path}")
    encoding = representation.load_encoding(args.encoding_path)
    print(f"✅ Encoding loaded")
    print("")

    # Get list of songs to convert
    if args.song_names and args.song_names.exists():
        print(f"Loading song names from: {args.song_names}")
        with open(args.song_names, "r") as f:
            song_names = [line.strip() for line in f if line.strip()]
        print(f"Found {len(song_names)} songs to convert")
        json_files = [args.json_dir / f"{name}.json" for name in song_names]
    else:
        print(f"Converting all JSON files in: {args.json_dir}")
        json_files = list(args.json_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")

    print("")
    print("Starting conversion...")
    print("-" * 80)

    # Convert each file
    success_count = 0
    failed_count = 0

    for json_file in json_files:
        if not json_file.exists():
            print(f"⚠️  File not found: {json_file}")
            failed_count += 1
            continue

        # Determine output path (preserve subdirectory structure if any)
        output_file = args.output_dir / f"{json_file.stem}.pt"

        # Convert
        if convert_json_to_pt(json_file, output_file, encoding):
            success_count += 1
        else:
            failed_count += 1

    print("-" * 80)
    print("")
    print("=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"✅ Successfully converted: {success_count} files")
    print(f"❌ Failed: {failed_count} files")
    print(f"📁 Output directory: {args.output_dir}")
    print("")


if __name__ == "__main__":
    main()
