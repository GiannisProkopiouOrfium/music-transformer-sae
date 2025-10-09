#!/usr/bin/env python3
"""
Convert Intervention PyTorch Tensors to Analysis-Ready Formats

This script converts your existing .pt intervention results to formats
compatible with the deterministic analysis pipeline.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Add MMT modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your representation module
import representation


def convert_pt_to_analysis_formats(pt_file_path: str, output_dir: str = None):
    """
    Convert a single .pt file to multiple analysis-ready formats.

    Args:
        pt_file_path: Path to the .pt file
        output_dir: Output directory (default: same as input file)
    """
    pt_path = Path(pt_file_path)

    if output_dir is None:
        output_dir = pt_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {pt_path.name}")

    try:
        # Load the tensor data
        tensor_data = torch.load(pt_path, map_location="cpu")

        # Extract the generated sequence
        if "generated" in tensor_data:
            sequence = tensor_data["generated"]
        else:
            print(f"Warning: No 'generated' key in {pt_path.name}")
            return False

        # Convert to numpy
        if isinstance(sequence, torch.Tensor):
            seq_np = sequence.numpy()
        else:
            seq_np = sequence

        # Remove batch dimension if present
        if len(seq_np.shape) == 3 and seq_np.shape[0] == 1:
            seq_np = seq_np[0]

        # Get base filename without extension
        base_name = pt_path.stem

        # Load encoding (you might need to adjust this path)
        encoding_path = "data/sod/processed/notes/encoding.json"
        if not os.path.exists(encoding_path):
            # Try alternative paths
            possible_paths = [
                "mmt/data/sod/processed/notes/encoding.json",
                "../data/sod/processed/notes/encoding.json",
                "baseline/encoding.json",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    encoding_path = path
                    break
            else:
                print(f"Error: Could not find encoding.json file")
                return False

        encoding = representation.load_encoding(encoding_path)

        # Convert to MusPy Music object
        music = representation.decode(seq_np, encoding)

        # Save in multiple formats for deterministic analysis

        # 1. Save as MIDI file (most important for analysis)
        midi_file = output_dir / f"{base_name}.mid"
        music.write(str(midi_file))
        print(f"  → Saved MIDI: {midi_file}")

        # 2. Save as MusPy JSON (backup format)
        json_file = output_dir / f"{base_name}.json"
        music.save(str(json_file))
        print(f"  → Saved JSON: {json_file}")

        # 3. Save as NumPy array (original sequence)
        npy_file = output_dir / f"{base_name}.npy"
        np.save(str(npy_file), seq_np)
        print(f"  → Saved NPY: {npy_file}")

        # 4. Save metadata as separate JSON
        metadata_file = output_dir / f"{base_name}_metadata.json"
        metadata = {
            "condition_name": tensor_data.get("condition_name", "unknown"),
            "intervention_type": tensor_data.get("intervention_type", "unknown"),
            "strength": tensor_data.get("strength", 0.0),
            "feature_id": tensor_data.get("feature_id", "unknown"),
            "layer": tensor_data.get("layer", 0),
            "generation_params": tensor_data.get("generation_params", {}),
            "original_file": str(pt_path),
            "sequence_shape": list(seq_np.shape),
            "music_duration": music.get_end_time(),
            "total_notes": len(
                [note for track in music.tracks for note in track.notes]
            ),
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  → Saved metadata: {metadata_file}")

        return True

    except Exception as e:
        print(f"Error converting {pt_path.name}: {e}")
        return False


def convert_intervention_directory(input_dir: str, output_dir: str = None):
    """
    Convert all .pt files in a directory structure to analysis-ready formats.

    Args:
        input_dir: Directory containing intervention results
        output_dir: Output directory for converted files (default: create 'analysis_ready' subdir)
    """
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = input_path / "analysis_ready"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting intervention results from: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all .pt files
    pt_files = list(input_path.rglob("*.pt"))

    if not pt_files:
        print("No .pt files found!")
        return

    print(f"Found {len(pt_files)} .pt files to convert")
    print()

    success_count = 0

    for pt_file in tqdm(pt_files, desc="Converting files"):
        # Preserve directory structure in output
        relative_path = pt_file.relative_to(input_path)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        if convert_pt_to_analysis_formats(str(pt_file), str(output_subdir)):
            success_count += 1

    print(f"\n✅ Conversion complete!")
    print(f"Successfully converted: {success_count}/{len(pt_files)} files")
    print(f"Results saved to: {output_dir}")

    return output_dir


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert intervention .pt files to analysis formats"
    )
    parser.add_argument("input_dir", help="Directory containing intervention .pt files")
    parser.add_argument("--output-dir", help="Output directory for converted files")
    parser.add_argument(
        "--single-file", help="Convert single file instead of directory"
    )

    args = parser.parse_args()

    if args.single_file:
        # Convert single file
        success = convert_pt_to_analysis_formats(args.single_file, args.output_dir)
        if success:
            print("✅ File converted successfully!")
        else:
            print("❌ File conversion failed!")
            return 1
    else:
        # Convert directory
        output_dir = convert_intervention_directory(args.input_dir, args.output_dir)
        print(f"\n🎵 Ready for deterministic analysis!")
        print(
            f"Run: python -m deterministic_analysis.batch_deterministic_analyzer {output_dir}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
