"""
Optimized SAE Data Loading for GPU Training

This module provides optimized data loading specifically designed to maximize GPU utilization
by minimizing CPU bottlenecks and improving data throughput.
"""

import torch
import torch.utils.data
import h5py
import numpy as np
from typing import Optional


class OptimizedActivationDataset(torch.utils.data.Dataset):
    """
    Optimized dataset for SAE training with focus on GPU utilization.
    
    Key optimizations:
    - Pre-loads data into memory when possible
    - Uses vectorized operations
    - Minimizes HDF5 file access
    - Optimized for multi-worker data loading
    """

    def __init__(
        self,
        h5_file_path: str,
        layer_key: str = None,
        normalize: bool = True,
        subsample: Optional[int] = None,
        preload_data: bool = True,
    ):
        """
        Initialize optimized activation dataset.

        Args:
            h5_file_path: Path to HDF5 file containing activations
            layer_key: Key for the layer to load
            normalize: Whether to normalize activations
            subsample: If provided, randomly subsample this many examples
            preload_data: If True, load all data into memory for faster access
        """
        self.h5_file_path = h5_file_path
        self.normalize = normalize
        self.preload_data = preload_data

        # Get dataset info
        with h5py.File(h5_file_path, "r") as f:
            available_keys = list(f.keys())

            if layer_key is None:
                data_keys = [k for k in available_keys if k != "metadata"]
                layer_key = data_keys[0] if data_keys else available_keys[0]

            if layer_key not in available_keys:
                raise KeyError(
                    f"Key '{layer_key}' not found. Available keys: {available_keys}"
                )

            self.layer_key = layer_key
            dataset = f[layer_key]
            self.original_shape = dataset.shape
            self.dtype = dataset.dtype

        # Create indices for subsampling
        total_samples = self.original_shape[0]
        if subsample is not None and subsample < total_samples:
            rng = np.random.default_rng(42)
            self.indices = rng.choice(total_samples, subsample, replace=False)
            self.indices.sort()
        else:
            self.indices = np.arange(total_samples)

        self.num_samples = len(self.indices)

        # Pre-load data if requested and feasible
        if preload_data:
            self._preload_all_data()
        else:
            self.activations = None
            if normalize:
                self._compute_normalization_stats()

    def _preload_all_data(self):
        """Pre-load all data into memory for fastest access."""
        print(f"Pre-loading {self.num_samples} samples into memory...")
        
        with h5py.File(self.h5_file_path, "r") as f:
            dataset = f[self.layer_key]
            
            # Load data in chunks to avoid memory spikes
            chunk_size = 10000
            data_chunks = []
            
            for i in range(0, len(self.indices), chunk_size):
                chunk_indices = self.indices[i : i + chunk_size]
                chunk_data = dataset[chunk_indices]
                data_chunks.append(chunk_data)
            
            # Concatenate all chunks
            self.activations = np.concatenate(data_chunks, axis=0)
            
        # Convert to float32 for GPU efficiency
        self.activations = self.activations.astype(np.float32)
        
        # Apply normalization if requested
        if self.normalize:
            self.mean = np.mean(self.activations, axis=0, keepdims=True)
            self.std = np.std(self.activations, axis=0, keepdims=True)
            self.std = np.maximum(self.std, 1e-8)  # Avoid division by zero
            self.activations = (self.activations - self.mean) / self.std
        
        # Convert to PyTorch tensor
        self.activations = torch.from_numpy(self.activations)
        
        print(f"✓ Data pre-loaded: {self.activations.shape}")

    def _compute_normalization_stats(self):
        """Compute normalization statistics efficiently."""
        chunk_size = 10000
        running_sum = 0
        running_sum_sq = 0
        count = 0

        with h5py.File(self.h5_file_path, "r") as f:
            dataset = f[self.layer_key]

            for i in range(0, len(self.indices), chunk_size):
                chunk_indices = self.indices[i : i + chunk_size]
                chunk_data = dataset[chunk_indices].astype(np.float32)
                
                running_sum += np.sum(chunk_data, axis=0)
                running_sum_sq += np.sum(chunk_data ** 2, axis=0)
                count += len(chunk_data)

        self.mean = running_sum / count
        self.variance = (running_sum_sq / count) - (self.mean ** 2)
        self.std = np.sqrt(np.maximum(self.variance, 1e-8))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.activations is not None:
            # Fast path: data is pre-loaded
            return self.activations[idx]
        else:
            # Slower path: load from file
            actual_idx = self.indices[idx]
            with h5py.File(self.h5_file_path, "r") as f:
                data = f[self.layer_key][actual_idx].astype(np.float32)
            
            if self.normalize:
                data = (data - self.mean) / self.std
            
            return torch.from_numpy(data)


def create_optimized_sae_dataloader(
    h5_file_path: str,
    layer_key: str = None,
    batch_size: int = 512,
    shuffle: bool = True,
    normalize: bool = True,
    subsample: Optional[int] = None,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    preload_data: bool = True,
):
    """
    Create an optimized DataLoader for SAE training.
    
    Args:
        h5_file_path: Path to HDF5 activations file
        layer_key: Layer key to load
        batch_size: Batch size (recommended: 512-2048 for GPUs)
        shuffle: Whether to shuffle data
        normalize: Whether to normalize activations
        subsample: Number of samples to use (None for all)
        num_workers: Number of data loading workers (recommended: 4-8)
        pin_memory: Use pinned memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        preload_data: Pre-load all data into memory if possible
    
    Returns:
        DataLoader and dataset info
    """
    
    dataset = OptimizedActivationDataset(
        h5_file_path=h5_file_path,
        layer_key=layer_key,
        normalize=normalize,
        subsample=subsample,
        preload_data=preload_data,
    )

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }

    # Add multiprocessing optimizations if using multiple workers
    if num_workers > 0:
        dataloader_kwargs.update({
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        })

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # Get dataset info
    info = {
        "layer_key": dataset.layer_key,
        "shape": (dataset.num_samples, dataset.original_shape[1]),
        "samples": dataset.num_samples,
        "feature_dim": dataset.original_shape[1],
        "preloaded": dataset.activations is not None,
    }

    return dataloader, info


if __name__ == "__main__":
    # Test the optimized data loader
    print("Testing optimized SAE DataLoader...")
    
    import pathlib
    activations_file = "exp/sod/ape/activations/activations_train_layers_3.h5"
    
    if not pathlib.Path(activations_file).exists():
        print(f"File not found: {activations_file}")
        exit(1)
    
    # Create optimized data loader
    dataloader, info = create_optimized_sae_dataloader(
        activations_file,
        layer_key="layer_3",
        batch_size=512,
        num_workers=4,
    )
    
    print(f"Dataset info: {info}")
    
    # Test data loading speed
    import time
    start_time = time.time()
    batch_count = 0
    
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 10:  # Test first 10 batches
            break
    
    elapsed = time.time() - start_time
    print(f"Loaded {batch_count} batches in {elapsed:.2f}s ({batch_count/elapsed:.1f} batches/s)")
    print(f"Last batch shape: {batch.shape}")
