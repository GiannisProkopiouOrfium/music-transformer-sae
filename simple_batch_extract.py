#!/usr/bin/env python3
"""
Simple Batch Musical Dataset Processor

This script processes many JSON files while maintaining input-activation
associations using the existing MMT pipeline infrastructure.
"""

import os
import sys
import subprocess
import pathlib
import time
from typing import List


def find_json_files(data_dir: str, max_files: int = None) -> List[str]:
    """Find all JSON files in the data directory."""
    data_path = pathlib.Path(data_dir)

    json_files = []
    for json_file in data_path.rglob("*.json"):
        if json_file.is_file() and "json-names.txt" not in json_file.name:
            # Get relative name for the pipeline
            relative_name = json_file.stem  # filename without .json
            json_files.append(relative_name)

    json_files.sort()

    if max_files:
        json_files = json_files[:max_files]

    print(f"Found {len(json_files)} JSON files to process")
    return json_files


def create_temporary_filelist(json_files: List[str], dataset: str) -> str:
    """Create a temporary file list for the pipeline."""
    temp_file = f"temp_filelist_{dataset}_{len(json_files)}.txt"

    with open(temp_file, "w") as f:
        for filename in json_files:
            f.write(f"{filename}\n")

    print(f"Created temporary file list: {temp_file}")
    return temp_file


def run_extraction(
    dataset: str,
    representation: str,
    layers: List[int],
    temp_filelist: str,
    max_files: int,
) -> None:
    """Run the extraction using the pipeline."""

    # First, backup the original filelist
    original_filelist = f"data/{dataset}/processed/json-names.txt"
    backup_filelist = f"data/{dataset}/processed/json-names.txt.backup"

    try:
        # Backup original if it exists
        if os.path.exists(original_filelist):
            print(f"Backing up original file list to {backup_filelist}")
            subprocess.run(["cp", original_filelist, backup_filelist], check=True)

        # Replace with our temporary list
        print("Using temporary file list for processing")
        subprocess.run(["cp", temp_filelist, original_filelist], check=True)

        # Build the extraction command
        cmd = [
            "python",
            "-m",
            "mmt.pipeline.main",
            "--extract-only",
            "--dataset",
            dataset,
            "--representation",
            representation,
            "--layers",
        ] + [str(layer) for layer in layers]

        print(f"Running: {' '.join(cmd)}")
        print(f"Processing {max_files} files from dataset '{dataset}'")
        print("=" * 60)

        # Run the extraction
        subprocess.run(cmd, check=True)

        print("=" * 60)
        print("✅ EXTRACTION COMPLETED SUCCESSFULLY")

    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)

    finally:
        # Restore original filelist
        if os.path.exists(backup_filelist):
            print("Restoring original file list")
            subprocess.run(["mv", backup_filelist, original_filelist])

        # Clean up temporary file
        if os.path.exists(temp_filelist):
            os.remove(temp_filelist)
            print(f"Cleaned up temporary file: {temp_filelist}")


def main():
    """Main processing function."""
    if len(sys.argv) < 5:
        print(
            "Usage: python simple_batch_extract.py <dataset> <representation> <layers> <max_files>"
        )
        print("Example: python simple_batch_extract.py sod ape 3 50")
        print("         python simple_batch_extract.py sod ape '3,4,5' 100")
        sys.exit(1)

    dataset = sys.argv[1]
    representation = sys.argv[2]
    layers_arg = sys.argv[3]
    max_files = int(sys.argv[4])

    # Parse layers
    if "," in layers_arg:
        layers = [int(x.strip()) for x in layers_arg.split(",")]
    else:
        layers = [int(layers_arg)]

    print("🎵 MUSICAL DATASET BATCH PROCESSOR")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Representation: {representation}")
    print(f"Layers: {layers}")
    print(f"Max files: {max_files}")
    print("=" * 60)

    # Find JSON files
    data_dir = f"data/{dataset}/processed"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    json_files = find_json_files(data_dir, max_files)

    if not json_files:
        print(f"❌ No JSON files found in {data_dir}")
        sys.exit(1)

    # Create temporary file list
    temp_filelist = create_temporary_filelist(json_files, dataset)

    # Run extraction
    start_time = time.time()
    run_extraction(dataset, representation, layers, temp_filelist, max_files)
    end_time = time.time()

    # Show results
    layers_str = "_".join(map(str, layers))
    output_file = f"exp/{dataset}/{representation}/activations_layers_{layers_str}.h5"

    print()
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Processing time: {end_time - start_time:.1f} seconds")
    print(f"📁 Output file: {output_file}")

    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"📊 File size: {file_size:.1f} MB")

    print()
    print("Next steps:")
    print("1. Analyze the dataset:")
    print(f"   python analyze_dataset.py {output_file}")
    print()
    print("2. Train SAE:")
    print(
        f"   python mmt/pipeline/main.py --train-only --dataset {dataset} --representation {representation}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
