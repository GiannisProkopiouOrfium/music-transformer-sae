import torch
import torch.utils.data
import h5py
import numpy as np
from typing import Optional


class ActivationDataset(torch.utils.data.Dataset):
    """Dataset for loading extracted activations for SAE training."""

    def __init__(
        self,
        h5_file_path: str,
        layer_key: str = None,
        normalize: bool = True,
        subsample: Optional[int] = None,
    ):
        """
        Initialize activation dataset.

        Args:
            h5_file_path: Path to HDF5 file containing activations
            layer_key: Key for the layer to load (e.g., 'layer_6'). If None, uses first available key
            normalize: Whether to normalize activations
            subsample: If provided, randomly subsample this many examples
        """
        self.h5_file_path = h5_file_path
        self.normalize = normalize

        # Load activations
        with h5py.File(h5_file_path, "r") as f:
            available_keys = list(f.keys())
            # print(f"Available keys in {h5_file_path}: {available_keys}")

            if layer_key is None:
                if len(available_keys) == 0:
                    raise ValueError(f"No datasets found in {h5_file_path}")
                layer_key = available_keys[0]
                # print(f"Using first available key: {layer_key}")

            if layer_key not in available_keys:
                raise KeyError(
                    f"Key '{layer_key}' not found. Available keys: {available_keys}"
                )

            self.layer_key = layer_key
            self.activations = f[layer_key][:]
            # print(f"Loaded activations with shape: {self.activations.shape}")

        # Subsample if requested
        if subsample is not None and subsample < len(self.activations):
            indices = np.random.choice(len(self.activations), subsample, replace=False)
            self.activations = self.activations[indices]
            # print(f"Subsampled to {len(self.activations)} examples")

        # Normalize if requested
        if normalize:
            self.mean = np.mean(self.activations, axis=0, keepdims=True)
            self.std = np.std(self.activations, axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero
            self.activations = (self.activations - self.mean) / self.std
            # print("Applied normalization")

        self.activations = torch.from_numpy(self.activations).float()

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]

    def get_data_info(self):
        """Get information about the dataset."""
        return {
            "layer_key": self.layer_key,
            "shape": self.activations.shape,
            "mean": torch.mean(self.activations, dim=0),
            "std": torch.std(self.activations, dim=0),
            "sparsity": torch.mean((self.activations == 0).float()).item(),
        }


def create_sae_dataloader(
    h5_file_path: str,
    layer_key: str = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    normalize: bool = True,
    subsample: Optional[int] = None,
    num_workers: int = 0,  # Default to 0 to avoid multiprocessing issues
):
    """Create a DataLoader for SAE training."""
    dataset = ActivationDataset(h5_file_path, layer_key, normalize, subsample)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
        layer_key=None,  # Auto-detect
        batch_size=512,
    )

    print(f"Dataset info: {info}")

    # Check a batch
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch mean: {torch.mean(batch):.4f}")
        print(f"Batch std: {torch.std(batch):.4f}")
        break
