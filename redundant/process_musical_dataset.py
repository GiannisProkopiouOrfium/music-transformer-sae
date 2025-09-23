#!/usr/bin/env python3
"""
Comprehensive Batch Processing for Musical SAE Dataset Creation

This script uses the existing MMT pipeline to process many JSON files
while maintaining input-activation associations for interpretability.
"""

import argparse
import pathlib
import json
import logging
import subprocess
import time
import os
from typing import List
import sys


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
    )


def discover_input_files(
    data_dir: pathlib.Path, max_files: int = None
) -> List[pathlib.Path]:
    """
    Discover JSON files to process.

    Args:
        data_dir: Data directory containing JSON files
        max_files: Maximum number of files to process

    Returns:
        List of JSON file paths
    """
    json_files = []

    # Find all JSON files
    for json_file in data_dir.glob("**/*.json"):
        if json_file.is_file() and json_file.name != "json-names.txt":
            json_files.append(json_file)

    json_files.sort()

    if max_files:
        json_files = json_files[:max_files]

    logging.info(f"Found {len(json_files)} JSON files to process")
    return json_files


def create_file_list(
    json_files: List[pathlib.Path], output_path: pathlib.Path
) -> pathlib.Path:
    """
    Create a file list for the MMT pipeline.

    Args:
        json_files: List of JSON file paths
        output_path: Output file list path

    Returns:
        Path to created file list
    """
    # Create relative paths from the JSON files
    file_names = []
    for json_file in json_files:
        # Extract the filename without extension and directory structure
        # The pipeline expects names like "Kunstderfuge-0"
        relative_name = json_file.stem  # Gets filename without .json
        file_names.append(relative_name)

    # Write to file list
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for name in file_names:
            f.write(f"{name}\n")

    logging.info(f"Created file list with {len(file_names)} files: {output_path}")
    return output_path


def run_extraction_pipeline(
    dataset: str,
    representation: str,
    layers: List[int],
    file_list: pathlib.Path,
    output_dir: pathlib.Path,
    gpu: int = -1,
    batch_size: int = 4,
) -> pathlib.Path:
    """
    Run the MMT extraction pipeline on the file list.

    Args:
        dataset: Dataset name (e.g., 'sod')
        representation: Representation type (e.g., 'ape')
        layers: List of layers to extract
        file_list: Path to file list
        output_dir: Output directory
        gpu: GPU index (-1 for CPU)
        batch_size: Batch size for processing

    Returns:
        Path to extracted activations
    """
    layer_filename = "_".join(map(str, layers))

    # Build the extraction command
    cmd = (
        [
            "python",
            "run_pipeline.py",
            "--extract-only",
            "--dataset",
            dataset,
            "--representation",
            representation,
            "--layers",
        ]
        + [str(layer) for layer in layers]
        + ["--output", str(output_dir), "--batch-size", str(batch_size)]
    )

    if gpu >= 0:
        cmd.extend(["--gpu", str(gpu)])

    # Set the file list environment variable or modify config
    env = {"MMT_FILE_LIST": str(file_list)}

    logging.info(f"Running extraction command: {' '.join(cmd)}")
    print(f"Running extraction command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=pathlib.Path.cwd(),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            # Look for the generated activations file
            activations_path = (
                output_dir / "activations" / f"activations_layers_{layer_filename}.h5"
            )

            if activations_path.exists():
                logging.info(f"✅ Extraction successful: {activations_path}")
                return activations_path
            else:
                raise FileNotFoundError(
                    f"Expected activations file not found: {activations_path}"
                )
        else:
            logging.error(f"Extraction failed with return code {result.returncode}")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            raise RuntimeError("Extraction pipeline failed")

    except subprocess.TimeoutExpired:
        logging.error("Extraction pipeline timed out")
        raise RuntimeError("Extraction pipeline timed out")


def process_batch(
    dataset: str,
    representation: str,
    layers: List[int],
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    max_files: int = None,
    batch_size: int = 4,
    gpu: int = -1,
) -> pathlib.Path:
    """
    Process a batch of JSON files through the extraction pipeline.

    Args:
        dataset: Dataset name
        representation: Representation type
        layers: Layers to extract
        data_dir: Directory containing JSON files
        output_dir: Output directory
        max_files: Maximum files to process
        batch_size: Batch size
        gpu: GPU index

    Returns:
        Path to extracted activations
    """
    logging.info("=== STARTING BATCH MUSICAL DATASET PROCESSING ===")
    print("=== STARTING BATCH MUSICAL DATASET PROCESSING ===")

    # Discover files
    json_files = discover_input_files(data_dir, max_files)

    if not json_files:
        raise RuntimeError(f"No JSON files found in {data_dir}")

    # Create temporary file list
    layers_str = "_".join(map(str, layers))
    file_list_path = (
        output_dir / f"temp_file_list_{len(json_files)}files_layers_{layers_str}.txt"
    )
    create_file_list(json_files, file_list_path)

    # Run extraction
    try:
        activations_path = run_extraction_pipeline(
            dataset=dataset,
            representation=representation,
            layers=layers,
            file_list=file_list_path,
            output_dir=output_dir,
            gpu=gpu,
            batch_size=batch_size,
        )

        # Create metadata file
        metadata = {
            "dataset": dataset,
            "representation": representation,
            "layers": layers,
            "total_files": len(json_files),
            "source_files": [
                str(f) for f in json_files[:100]
            ],  # First 100 for reference
            "data_directory": str(data_dir),
            "processing_time": time.time(),
            "file_list": str(file_list_path),
        }

        metadata_path = output_dir / f"batch_metadata_{len(json_files)}files.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"✅ Metadata saved: {metadata_path}")

        return activations_path

    finally:
        # Cleanup temporary file list
        if file_list_path.exists():
            file_list_path.unlink()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch Musical Dataset Processing")
    parser.add_argument("-d", "--dataset", default="sod", help="Dataset name")
    parser.add_argument(
        "-r", "--representation", default="ape", help="Representation type"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[3], help="Layers to extract"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing JSON files"
    )
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--max-files", type=int, help="Maximum files to process")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index (-1 for CPU)")

    args = parser.parse_args()

    setup_logging()

    # Setup paths
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = pathlib.Path(f"exp/{args.dataset}/{args.representation}")

    try:
        # Process the batch
        activations_path = process_batch(
            dataset=args.dataset,
            representation=args.representation,
            layers=args.layers,
            data_dir=data_dir,
            output_dir=output_dir,
            max_files=args.max_files,
            batch_size=args.batch_size,
            gpu=args.gpu,
        )

        print()
        print("=" * 80)
        print("BATCH PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Dataset: {activations_path}")
        print(f"Files processed: {args.max_files or 'all available'}")
        print(f"Layers: {args.layers}")
        print(f"File size: {activations_path.stat().st_size / (1024**2):.1f} MB")
        print()
        print("Next steps:")
        print("  1. Analyze dataset: python analyze_dataset.py", str(activations_path))
        print("  2. Train SAE: python run_pipeline.py --train-only --skip-extraction")
        print("=" * 80)

    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
