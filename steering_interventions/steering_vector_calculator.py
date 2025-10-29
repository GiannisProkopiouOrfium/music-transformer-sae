"""Step C: Steering Vector Calculation.

This module:
1. Loads activations for high and low concept datasets
2. Calculates mean vectors for each layer
3. Computes steering vectors as the difference
4. Saves steering vectors for later use
"""

import argparse
import json
import logging
import pathlib
from typing import Dict

import h5py
import numpy as np
import torch

import config


def load_activations(filepath: pathlib.Path) -> Dict[int, np.ndarray]:
    """Load activations from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary mapping layer_idx -> activations array
    """
    activations = {}
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        if 'metadata_json' in f.attrs:
            metadata = json.loads(f.attrs['metadata_json'])
        
        # Load activations for each layer
        for key in f.keys():
            if key.startswith('layer_'):
                layer_idx = int(key.split('_')[1])
                activations[layer_idx] = f[key][:]
                logging.debug(f"Loaded {key}: shape {activations[layer_idx].shape}")
    
    logging.info(f"Loaded activations from: {filepath}")
    logging.info(f"Number of layers: {len(activations)}")
    
    return activations, metadata


def calculate_steering_vectors(
    high_activations: Dict[int, np.ndarray],
    low_activations: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Calculate steering vectors from high and low activations.
    
    Args:
        high_activations: Activations for high concept segments
        low_activations: Activations for low concept segments
        
    Returns:
        Dictionary mapping layer_idx -> steering_vector
    """
    steering_vectors = {}
    
    # Get all layer indices (should be the same for both)
    layer_indices = sorted(set(high_activations.keys()) & set(low_activations.keys()))
    
    logging.info(f"Calculating steering vectors for {len(layer_indices)} layers")
    
    for layer_idx in layer_indices:
        high_acts = high_activations[layer_idx]  # shape: (n_high, dim)
        low_acts = low_activations[layer_idx]    # shape: (n_low, dim)
        
        # Calculate mean vectors
        mean_high = np.mean(high_acts, axis=0)  # shape: (dim,)
        mean_low = np.mean(low_acts, axis=0)    # shape: (dim,)
        
        # Steering vector = difference
        steering_vector = mean_high - mean_low  # shape: (dim,)
        
        steering_vectors[layer_idx] = steering_vector
        
        # Log statistics
        norm = np.linalg.norm(steering_vector)
        logging.info(f"Layer {layer_idx}: "
                    f"norm={norm:.4f}, "
                    f"mean={np.mean(steering_vector):.4f}, "
                    f"std={np.std(steering_vector):.4f}")
    
    return steering_vectors


def save_steering_vectors(
    steering_vectors: Dict[int, np.ndarray],
    output_path: pathlib.Path,
    metadata: Dict,
):
    """Save steering vectors to file.
    
    Args:
        steering_vectors: Dictionary mapping layer_idx -> steering_vector
        output_path: Output file path (.pt for PyTorch)
        metadata: Metadata to save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyTorch tensors for easier use during generation
    steering_tensors = {
        layer_idx: torch.from_numpy(vec).float()
        for layer_idx, vec in steering_vectors.items()
    }
    
    # Save as PyTorch checkpoint
    torch.save({
        'steering_vectors': steering_tensors,
        'metadata': metadata,
    }, output_path)
    
    logging.info(f"Saved steering vectors to: {output_path}")
    
    # Also save as JSON for inspection
    json_path = output_path.with_suffix('.json')
    
    # Calculate statistics for each layer
    stats = {}
    for layer_idx, vec in steering_vectors.items():
        stats[f'layer_{layer_idx}'] = {
            'norm': float(np.linalg.norm(vec)),
            'mean': float(np.mean(vec)),
            'std': float(np.std(vec)),
            'min': float(np.min(vec)),
            'max': float(np.max(vec)),
            'shape': list(vec.shape),
        }
    
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': metadata,
            'statistics': stats,
        }, f, indent=2)
    
    logging.info(f"Saved statistics to: {json_path}")


def visualize_steering_vectors(steering_vectors: Dict[int, np.ndarray]):
    """Print visualization of steering vector norms across layers.
    
    Args:
        steering_vectors: Dictionary mapping layer_idx -> steering_vector
    """
    print("\n" + "=" * 60)
    print("Steering Vector Norms by Layer")
    print("=" * 60)
    
    layer_indices = sorted(steering_vectors.keys())
    norms = [np.linalg.norm(steering_vectors[i]) for i in layer_indices]
    max_norm = max(norms)
    
    for layer_idx, norm in zip(layer_indices, norms):
        bar_length = int(40 * norm / max_norm)
        bar = '█' * bar_length
        print(f"Layer {layer_idx:2d}: {bar} {norm:.4f}")
    
    print("=" * 60)
    print(f"Mean norm: {np.mean(norms):.4f}")
    print(f"Std norm:  {np.std(norms):.4f}")
    print(f"Min norm:  {np.min(norms):.4f} (layer {layer_indices[np.argmin(norms)]})")
    print(f"Max norm:  {np.max(norms):.4f} (layer {layer_indices[np.argmax(norms)]})")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Calculate steering vectors")
    parser.add_argument(
        "--concept",
        type=str,
        default="velocity",
        help="Which concept to use"
    )
    parser.add_argument(
        "--activations_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "activations",
        help="Directory with extracted activations"
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "steering_vectors",
        help="Output directory for steering vectors"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize steering vector norms"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load activations
    high_acts_file = args.activations_dir / f"high_{args.concept}_activations.h5"
    low_acts_file = args.activations_dir / f"low_{args.concept}_activations.h5"
    
    if not high_acts_file.exists() or not low_acts_file.exists():
        logging.error(f"Activation files not found in {args.activations_dir}")
        logging.error("Please run activation_extractor.py first!")
        return
    
    logging.info("Loading high concept activations...")
    high_activations, high_metadata = load_activations(high_acts_file)
    
    logging.info("Loading low concept activations...")
    low_activations, low_metadata = load_activations(low_acts_file)
    
    # Calculate steering vectors
    logging.info("Calculating steering vectors...")
    steering_vectors = calculate_steering_vectors(high_activations, low_activations)
    
    # Visualize if requested
    if args.visualize:
        visualize_steering_vectors(steering_vectors)
    
    # Prepare metadata
    metadata = {
        'concept': args.concept,
        'num_layers': len(steering_vectors),
        'high_metadata': high_metadata,
        'low_metadata': low_metadata,
        'n_high_samples': high_activations[0].shape[0] if 0 in high_activations else 0,
        'n_low_samples': low_activations[0].shape[0] if 0 in low_activations else 0,
    }
    
    # Save steering vectors
    output_path = args.output_dir / f"{args.concept}_steering_vectors.pt"
    save_steering_vectors(steering_vectors, output_path, metadata)
    
    logging.info("Steering vector calculation complete!")


if __name__ == "__main__":
    main()
