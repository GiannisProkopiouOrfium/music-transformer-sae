#!/usr/bin/env python3
"""
Batch Activation Extraction Pipeline

This script processes multiple JSON files to extract activations while
maintaining associations with source inputs for interpretability.
"""

import argparse
import pathlib
import json
import logging
import h5py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import sys
from tqdm import tqdm

# Add mmt module to path
sys.path.append(str(pathlib.Path(__file__).parent))

from mmt.pipeline.config import PipelineConfig
from mmt.activation_extractor import ActivationExtractor
from mmt.music_x_transformers import MusicXTransformer
from mmt.representation import get_fast_repr_class


def setup_logging():
    """Setup logging for batch processing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def discover_json_files(
    data_dir: pathlib.Path, max_files: Optional[int] = None
) -> List[pathlib.Path]:
    """
    Discover all JSON files in the dataset.

    Args:
        data_dir: Root data directory
        max_files: Maximum number of files to process (for testing)

    Returns:
        List of JSON file paths
    """
    json_files = []

    # Search in all subdirectories
    for json_file in data_dir.glob("**/*.json"):
        if json_file.is_file() and json_file.name != "json-names.txt":
            json_files.append(json_file)

    # Sort for reproducible ordering
    json_files.sort()

    if max_files:
        json_files = json_files[:max_files]

    logging.info(f"Discovered {len(json_files)} JSON files")
    return json_files


def extract_activations_from_file(
    json_file: pathlib.Path,
    model: MusicXTransformer,
    extractor: ActivationExtractor,
    config: PipelineConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract activations from a single JSON file.

    Args:
        json_file: Path to JSON file
        model: Loaded transformer model
        extractor: Activation extractor
        config: Pipeline configuration

    Returns:
        Tuple of (activations, metadata)
    """
    try:
        # Load the JSON data
        with open(json_file, "r") as f:
            json_data = json.load(f)

        # For now, we'll use a simple approach - convert JSON to a token sequence
        # This is a simplified version - in practice you'd use the proper representation
        if isinstance(json_data, list) and len(json_data) > 0:
            # Assume it's already a token sequence
            tokens = json_data[: config.extraction.max_seq_len]
        else:
            # Skip files that don't have the expected format
            logging.warning(f"Skipping {json_file}: unexpected format")
            return None, None

        if len(tokens) == 0:
            logging.warning(f"Skipping {json_file}: empty sequence")
            return None, None

        # Convert to numpy array
        tokens = np.array(tokens, dtype=np.int64)

        # Extract activations
        activations = extractor.extract_activations(
            tokens, layers=config.extraction.layers
        )

        # Create metadata
        metadata = {
            "file_path": str(json_file),
            "file_name": json_file.name,
            "source": json_file.parent.name,  # e.g., "Kunstderfuge"
            "dataset": json_file.parent.parent.parent.name,  # e.g., "sod"
            "num_tokens": len(tokens),
            "num_activations": (
                activations.shape[1] if len(activations.shape) > 1 else 1
            ),
            "layers": config.extraction.layers,
            "sequence_tokens": tokens[:100].tolist(),  # First 100 tokens for reference
        }

        return activations, metadata

    except Exception as e:
        logging.warning(f"Failed to process {json_file}: {e}")
        return None, None


def create_batch_activation_dataset(
    config: PipelineConfig, json_files: List[pathlib.Path], output_path: pathlib.Path
) -> pathlib.Path:
    """
    Create a comprehensive activation dataset from multiple JSON files.

    Args:
        config: Pipeline configuration
        json_files: List of JSON file paths
        output_path: Output HDF5 file path

    Returns:
        Path to created dataset
    """
    logging.info("=== STARTING BATCH ACTIVATION EXTRACTION ===")
    print("=== STARTING BATCH ACTIVATION EXTRACTION ===")

    # Setup model and extractor
    device = "mps" if config.model.gpu == -1 else f"cuda:{config.model.gpu}"

    # Load model
    logging.info(f"Loading model for dataset: {config.model.dataset}")
    model = MusicXTransformer.load_model(
        dataset=config.model.dataset,
        representation=config.model.representation,
        model_steps=config.model.model_steps,
        device=device,
    )

    # Setup representation (removed - using direct JSON loading)
    # repr_class = get_fast_repr_class(config.model.representation)()

    # Setup extractor
    extractor = ActivationExtractor(model, device=device)

    # Prepare output file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process files and collect activations
    all_activations = []
    all_metadata = []
    processed_files = 0
    failed_files = 0

    logging.info(f"Processing {len(json_files)} files...")
    print(f"Processing {len(json_files)} files...")

    for i, json_file in enumerate(tqdm(json_files, desc="Extracting")):
        activations, metadata = extract_activations_from_file(
            json_file, model, extractor, config
        )

        if activations is not None:
            # Store activations for each layer
            for layer_idx, layer_activations in enumerate(activations):
                layer_num = config.extraction.layers[layer_idx]

                # Add layer information to metadata
                layer_metadata = metadata.copy()
                layer_metadata["layer"] = layer_num
                layer_metadata["activation_shape"] = layer_activations.shape

                all_activations.append(layer_activations)
                all_metadata.append(layer_metadata)

            processed_files += 1
        else:
            failed_files += 1

        # Progress update every 100 files
        if (i + 1) % 100 == 0:
            progress_msg = f"Processed {i + 1}/{len(json_files)} files ({processed_files} successful, {failed_files} failed)"
            logging.info(progress_msg)
            print(progress_msg)

    logging.info(
        f"Extraction complete: {processed_files} successful, {failed_files} failed"
    )
    print(f"Extraction complete: {processed_files} successful, {failed_files} failed")

    # Convert to numpy arrays
    if all_activations:
        # Stack all activations
        stacked_activations = np.vstack(all_activations)

        logging.info(f"Final dataset shape: {stacked_activations.shape}")
        print(f"Final dataset shape: {stacked_activations.shape}")

        # Save to HDF5 with metadata
        with h5py.File(output_path, "w") as f:
            # Save activations
            for layer_num in config.extraction.layers:
                layer_acts = []
                layer_meta = []

                for act, meta in zip(all_activations, all_metadata):
                    if meta["layer"] == layer_num:
                        layer_acts.append(act)
                        layer_meta.append(meta)

                if layer_acts:
                    layer_data = np.vstack(layer_acts)
                    f.create_dataset(f"layer_{layer_num}", data=layer_data)

                    # Save metadata as JSON strings
                    meta_strings = [json.dumps(meta) for meta in layer_meta]
                    f.create_dataset(
                        f"layer_{layer_num}_metadata",
                        data=meta_strings,
                        dtype=h5py.string_dtype(),
                    )

            # Save global metadata
            global_meta = {
                "config": config.__dict__,
                "total_files_processed": processed_files,
                "total_files_failed": failed_files,
                "total_activations": len(all_activations),
                "layers": config.extraction.layers,
                "sources": list(set(meta["source"] for meta in all_metadata)),
            }

            f.attrs["metadata"] = json.dumps(global_meta)

        logging.info(f"Dataset saved to: {output_path}")
        print(f"Dataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")

        return output_path

    else:
        raise RuntimeError("No activations were successfully extracted!")


def main():
    """Main function for batch activation extraction."""
    parser = argparse.ArgumentParser(description="Batch Activation Extraction")
    parser.add_argument("-d", "--dataset", default="sod", help="Dataset name")
    parser.add_argument(
        "-r", "--representation", default="ape", help="Representation type"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[3], help="Layers to extract"
    )
    parser.add_argument(
        "--max-files", type=int, help="Maximum files to process (for testing)"
    )
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--data-dir", help="Data directory path")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Create configuration
    config = PipelineConfig()
    config.model.dataset = args.dataset
    config.model.representation = args.representation
    config.extraction.layers = args.layers
    config.extraction.max_seq_len = 1024  # Reasonable limit

    # Determine data directory
    if args.data_dir:
        data_dir = pathlib.Path(args.data_dir)
    else:
        data_dir = pathlib.Path(f"data/{args.dataset}/processed/json")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Discover JSON files
    json_files = discover_json_files(data_dir, args.max_files)

    if not json_files:
        raise RuntimeError(f"No JSON files found in {data_dir}")

    # Determine output path
    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        layer_str = "_".join(map(str, args.layers))
        output_path = pathlib.Path(
            f"exp/{args.dataset}/{args.representation}/activations/"
            f"batch_activations_layers_{layer_str}_{len(json_files)}files.h5"
        )

    # Extract activations
    final_path = create_batch_activation_dataset(config, json_files, output_path)

    print()
    print("=" * 60)
    print("BATCH EXTRACTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Dataset: {final_path}")
    print(f"Files processed: {len(json_files)}")
    print(f"Layers: {args.layers}")
    print()
    print("Next steps:")
    print(f"  1. Train SAE: python run_pipeline.py --train-only --skip-extraction")
    print(f"  2. Use custom dataset: Modify config to point to {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
