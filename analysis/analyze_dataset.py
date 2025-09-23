#!/usr/bin/env python3
"""
Dataset Analysis and Inspection Tool

This script analyzes extracted activation datasets and provides insights
about the musical content and activation patterns.
"""

import argparse
import pathlib
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
    )


def analyze_activation_dataset(dataset_path: pathlib.Path) -> Dict[str, Any]:
    """
    Analyze an activation dataset and return comprehensive statistics.

    Args:
        dataset_path: Path to HDF5 activation dataset

    Returns:
        Dictionary containing analysis results
    """
    results = {}

    with h5py.File(dataset_path, "r") as f:
        # Load global metadata
        if "metadata" in f.attrs:
            global_meta = json.loads(f.attrs["metadata"])
            results["global"] = global_meta

        # Analyze each layer
        layers_info = {}

        for layer_key in f.keys():
            if layer_key.startswith("layer_") and not layer_key.endswith("_metadata"):
                layer_num = layer_key.split("_")[1]

                # Load activation data
                activations = f[layer_key][:]

                # Load metadata if available
                metadata = []
                meta_key = f"{layer_key}_metadata"
                if meta_key in f:
                    meta_strings = f[meta_key][:]
                    metadata = [json.loads(meta.decode()) for meta in meta_strings]

                # Compute statistics
                layer_info = {
                    "layer_number": int(layer_num),
                    "shape": activations.shape,
                    "num_sequences": activations.shape[0],
                    "activation_dim": activations.shape[1],
                    "mean_activation": float(np.mean(activations)),
                    "std_activation": float(np.std(activations)),
                    "sparsity_ratio": float(np.mean(activations == 0)),
                    "max_activation": float(np.max(activations)),
                    "min_activation": float(np.min(activations)),
                    "metadata_count": len(metadata),
                }

                # Analyze metadata if available
                if metadata:
                    sources = [meta.get("source", "unknown") for meta in metadata]
                    file_names = [meta.get("file_name", "unknown") for meta in metadata]
                    num_tokens = [meta.get("num_tokens", 0) for meta in metadata]

                    layer_info["sources"] = list(set(sources))
                    layer_info["num_sources"] = len(set(sources))
                    layer_info["source_distribution"] = {
                        src: sources.count(src) for src in set(sources)
                    }
                    layer_info["avg_tokens_per_sequence"] = (
                        float(np.mean(num_tokens)) if num_tokens else 0
                    )
                    layer_info["total_unique_files"] = len(set(file_names))

                layers_info[layer_num] = layer_info

        results["layers"] = layers_info

    return results


def print_analysis_report(analysis: Dict[str, Any], dataset_path: pathlib.Path):
    """Print a comprehensive analysis report."""

    print("=" * 80)
    print(f"ACTIVATION DATASET ANALYSIS: {dataset_path.name}")
    print("=" * 80)

    # File info
    file_size_mb = dataset_path.stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"File path: {dataset_path}")
    print()

    # Global metadata
    if "global" in analysis:
        global_meta = analysis["global"]
        print("GLOBAL METADATA:")
        print(
            f"  Total files processed: {global_meta.get('total_files_processed', 'N/A')}"
        )
        print(f"  Total files failed: {global_meta.get('total_files_failed', 'N/A')}")
        print(f"  Total activations: {global_meta.get('total_activations', 'N/A')}")
        print(f"  Sources: {', '.join(global_meta.get('sources', []))}")
        print()

    # Layer analysis
    print("LAYER ANALYSIS:")
    for layer_num, layer_info in analysis["layers"].items():
        print(f"  Layer {layer_num}:")
        print(f"    Shape: {layer_info['shape']}")
        print(f"    Sequences: {layer_info['num_sequences']:,}")
        print(f"    Activation dimension: {layer_info['activation_dim']}")
        print(f"    Mean activation: {layer_info['mean_activation']:.4f}")
        print(f"    Std activation: {layer_info['std_activation']:.4f}")
        print(
            f"    Sparsity ratio: {layer_info['sparsity_ratio']:.3f} ({layer_info['sparsity_ratio']*100:.1f}% zeros)"
        )
        print(
            f"    Activation range: [{layer_info['min_activation']:.4f}, {layer_info['max_activation']:.4f}]"
        )

        if "sources" in layer_info:
            print(f"    Unique sources: {layer_info['num_sources']}")
            print(f"    Sources: {', '.join(layer_info['sources'])}")
            print(f"    Source distribution: {layer_info['source_distribution']}")
            print(
                f"    Avg tokens per sequence: {layer_info['avg_tokens_per_sequence']:.1f}"
            )
            print(f"    Unique files: {layer_info['total_unique_files']:,}")
        print()

    # Data quality assessment
    print("DATA QUALITY ASSESSMENT:")
    for layer_num, layer_info in analysis["layers"].items():
        print(f"  Layer {layer_num}:")

        # Check for reasonable activation values
        if abs(layer_info["mean_activation"]) > 10:
            print(f"    ⚠️  High mean activation: {layer_info['mean_activation']:.4f}")
        else:
            print(f"    ✅ Normal mean activation: {layer_info['mean_activation']:.4f}")

        # Check sparsity
        if layer_info["sparsity_ratio"] > 0.9:
            print(f"    ⚠️  Very sparse activations: {layer_info['sparsity_ratio']:.3f}")
        elif layer_info["sparsity_ratio"] > 0.5:
            print(f"    ✅ Moderately sparse: {layer_info['sparsity_ratio']:.3f}")
        else:
            print(f"    ✅ Dense activations: {layer_info['sparsity_ratio']:.3f}")

        # Check data diversity
        if "num_sources" in layer_info and layer_info["num_sources"] > 1:
            print(f"    ✅ Multiple sources: {layer_info['num_sources']}")
        elif "num_sources" in layer_info:
            print(f"    ⚠️  Single source only: {layer_info['sources']}")

        print()

    print("=" * 80)
    print("RECOMMENDATIONS:")

    total_sequences = sum(info["num_sequences"] for info in analysis["layers"].values())
    if total_sequences < 10000:
        print("  📈 Consider processing more files for larger dataset")
    elif total_sequences > 100000:
        print("  ✅ Large dataset - good for training robust SAEs")
    else:
        print("  ✅ Medium dataset - suitable for SAE training")

    # Check if ready for SAE training
    activation_dims = [info["activation_dim"] for info in analysis["layers"].values()]
    if len(set(activation_dims)) == 1:
        print(f"  ✅ Consistent activation dimension: {activation_dims[0]}")
        print("  ✅ Dataset ready for SAE training")
    else:
        print(f"  ⚠️  Inconsistent activation dimensions: {set(activation_dims)}")

    print()
    print("USAGE EXAMPLES:")
    print("  # Train SAE on this dataset:")
    print(f"  python run_pipeline.py --train-only --skip-extraction")
    print(f"  # (Modify config to point to: {dataset_path})")
    print()
    print(f"  # Analyze specific layer activations:")
    print(
        f"  python -c \"import h5py; f=h5py.File('{dataset_path}'); print(list(f.keys()))\""
    )
    print("=" * 80)


def compare_datasets(dataset_paths: List[pathlib.Path]):
    """Compare multiple activation datasets."""

    print("=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)

    analyses = []
    for path in dataset_paths:
        if path.exists():
            analysis = analyze_activation_dataset(path)
            analysis["path"] = path
            analyses.append(analysis)
        else:
            print(f"⚠️  Dataset not found: {path}")

    if len(analyses) < 2:
        print("Need at least 2 datasets to compare")
        return

    # Compare key metrics
    comparison_data = []
    for analysis in analyses:
        for layer_num, layer_info in analysis["layers"].items():
            comparison_data.append(
                {
                    "dataset": analysis["path"].name,
                    "layer": layer_num,
                    "sequences": layer_info["num_sequences"],
                    "activation_dim": layer_info["activation_dim"],
                    "mean_activation": layer_info["mean_activation"],
                    "sparsity": layer_info["sparsity_ratio"],
                    "sources": layer_info.get("num_sources", 1),
                    "unique_files": layer_info.get("total_unique_files", 0),
                }
            )

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()

    print("RECOMMENDATIONS:")
    # Find dataset with most sequences
    max_seq_dataset = df.loc[df["sequences"].idxmax()]
    print(
        f"  📊 Largest dataset: {max_seq_dataset['dataset']} ({max_seq_dataset['sequences']:,} sequences)"
    )

    # Find most diverse dataset
    max_sources_dataset = df.loc[df["sources"].idxmax()]
    print(
        f"  🎵 Most diverse: {max_sources_dataset['dataset']} ({max_sources_dataset['sources']} sources)"
    )

    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze activation datasets")
    parser.add_argument("datasets", nargs="+", help="Dataset file paths")
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple datasets"
    )

    args = parser.parse_args()

    setup_logging()

    dataset_paths = [pathlib.Path(path) for path in args.datasets]

    if args.compare and len(dataset_paths) > 1:
        compare_datasets(dataset_paths)
    else:
        for dataset_path in dataset_paths:
            if dataset_path.exists():
                analysis = analyze_activation_dataset(dataset_path)
                print_analysis_report(analysis, dataset_path)
            else:
                print(f"Dataset not found: {dataset_path}")


if __name__ == "__main__":
    main()
