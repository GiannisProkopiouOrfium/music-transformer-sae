import torch
import torch.utils.data
import h5py
import numpy as np
from typing import Optional


class ActivationDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset for loading extracted activations for SAE training."""

    def __init__(
        self,
        h5_file_path: str,
        layer_key: str = None,
        normalize: bool = True,
        subsample: Optional[int] = None,
        memory_efficient: bool = True,
        cache_size: int = 1000,  # Number of samples to cache in memory
    ):
        """
        Initialize activation dataset.

        Args:
            h5_file_path: Path to HDF5 file containing activations
            layer_key: Key for the layer to load (e.g., 'layer_6'). If None, uses first available key
            normalize: Whether to normalize activations
            subsample: If provided, randomly subsample this many examples
            memory_efficient: If True, use memory-efficient loading (recommended for large datasets)
            cache_size: Number of recently accessed samples to keep in memory cache
        """
        self.h5_file_path = h5_file_path
        self.normalize = normalize
        self.memory_efficient = memory_efficient
        self.cache_size = cache_size
        self.cache = {}  # Simple LRU-like cache

        # Get dataset info without loading all data
        with h5py.File(h5_file_path, "r") as f:
            available_keys = list(f.keys())

            if layer_key is None:
                if len(available_keys) == 0:
                    raise ValueError(f"No datasets found in {h5_file_path}")
                # Filter out metadata keys
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
            rng = np.random.default_rng(42)  # Use fixed seed for reproducibility
            self.indices = rng.choice(total_samples, subsample, replace=False)
            self.indices.sort()  # Sort for efficient HDF5 access
        else:
            self.indices = np.arange(total_samples)

        self.num_samples = len(self.indices)

        # Compute normalization statistics if needed (memory-efficient way)
        if normalize:
            self._compute_normalization_stats()
        else:
            self.mean = None
            self.std = None

        # If not memory efficient or dataset is small, load everything
        if not memory_efficient or self.num_samples < 10000:
            self._load_all_data()
            self.activations = torch.from_numpy(self.activations).float()
        else:
            self.activations = None

    def _compute_normalization_stats(self):
        """Compute mean and std without loading all data into memory."""
        chunk_size = 1000
        means = []
        variances = []
        counts = []

        with h5py.File(self.h5_file_path, "r") as f:
            dataset = f[self.layer_key]

            for i in range(0, len(self.indices), chunk_size):
                chunk_indices = self.indices[i : i + chunk_size]
                chunk_data = dataset[chunk_indices]

                means.append(np.mean(chunk_data, axis=0))
                variances.append(np.var(chunk_data, axis=0))
                counts.append(len(chunk_data))

        # Combine statistics across chunks
        means = np.array(means)
        variances = np.array(variances)
        counts = np.array(counts)

        # Weighted mean
        total_count = np.sum(counts)
        self.mean = np.sum(means * counts[:, None], axis=0) / total_count

        # Weighted variance
        self.std = np.sqrt(np.sum(variances * counts[:, None], axis=0) / total_count)
        self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero

        # Reshape for broadcasting
        self.mean = self.mean.reshape(1, -1)
        self.std = self.std.reshape(1, -1)

    def _load_all_data(self):
        """Load all data into memory (for smaller datasets)."""
        with h5py.File(self.h5_file_path, "r") as f:
            self.activations = f[self.layer_key][self.indices]

        # Apply normalization
        if self.normalize and self.mean is not None:
            self.activations = (self.activations - self.mean) / self.std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.activations is not None:
            # Data is already loaded in memory
            return self.activations[idx]
        else:
            # Check cache first
            if idx in self.cache:
                return self.cache[idx]

            # Load data on-demand
            actual_idx = self.indices[idx]
            with h5py.File(self.h5_file_path, "r") as f:
                data = f[self.layer_key][actual_idx : actual_idx + 1][
                    0
                ]  # Load single sample

            # Apply normalization if needed
            if self.normalize and self.mean is not None:
                data = (data - self.mean.flatten()) / self.std.flatten()

            tensor_data = torch.from_numpy(data).float()

            # Add to cache (simple FIFO cache management)
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[idx] = tensor_data
            return tensor_data

    def get_data_info(self):
        """Get information about the dataset."""
        # For memory-efficient mode, compute stats from a sample
        if self.activations is not None:
            sample_data = self.activations
        else:
            # Load a small sample to get stats
            sample_size = min(1000, self.num_samples)
            sample_indices = self.indices[:sample_size]
            with h5py.File(self.h5_file_path, "r") as f:
                sample_data = f[self.layer_key][sample_indices]

            if self.normalize and self.mean is not None:
                sample_data = (sample_data - self.mean) / self.std

            sample_data = torch.from_numpy(sample_data).float()

        return {
            "layer_key": self.layer_key,
            "shape": (self.num_samples, self.original_shape[1]),
            "mean": torch.mean(sample_data, dim=0),
            "std": torch.std(sample_data, dim=0),
            "sparsity": torch.mean((sample_data == 0).float()).item(),
        }


def create_sae_dataloader(
    h5_file_path: str,
    layer_key: str = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    normalize: bool = True,
    subsample: Optional[int] = None,
    num_workers: int = 0,  # Default to 0 to avoid multiprocessing issues
    memory_efficient: bool = True,  # Enable memory-efficient loading by default
    cache_size: int = 1000,  # Cache size for memory-efficient mode
):
    """Create a DataLoader for SAE training."""
    dataset = ActivationDataset(
        h5_file_path,
        layer_key,
        normalize,
        subsample,
        memory_efficient=memory_efficient,
        cache_size=cache_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        # Use persistent workers to avoid recreating processes
        persistent_workers=num_workers > 0,
    )

    return dataloader, dataset.get_data_info()


# Example usage for future SAE training
if __name__ == "__main__":
    print("Creating SAE DataLoader...")

    # Use the correct file that we just created
    activations_file = "exp/sod/ape/activations/activations_layers_3.h5"

    # Check if file exists
    import pathlib

    if not pathlib.Path(activations_file).exists():
        print(f"File not found: {activations_file}")
        print("Available files in exp/sod/ape/activations/:")
        activations_dir = pathlib.Path("exp/sod/ape/activations")
        if activations_dir.exists():
            for file in activations_dir.glob("*.h5"):
                print(f"  {file}")
        else:
            print("  Directory does not exist")
        exit(1)

    # Test the data loader - let it auto-detect the layer key
    dataloader, info = create_sae_dataloader(
        activations_file,
        layer_key="layer_3",  # Specify the layer
        batch_size=512,
    )

    print(f"Dataset info: {info}")

    # Check a batch
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch mean: {torch.mean(batch):.4f}")
        print(f"Batch std: {torch.std(batch, dim=None):.4f}")
        break
