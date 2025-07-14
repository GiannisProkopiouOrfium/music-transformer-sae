"""Storage optimization utilities for HDF5 and large datasets."""

import h5py
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time


class HDF5Manager:
    """Manages optimized HDF5 storage for SAE activations and results."""

    def __init__(
        self,
        compression: str = "gzip",
        compression_opts: int = 9,
        shuffle: bool = True,
        fletcher32: bool = True,
    ):
        """
        Initialize HDF5 manager with optimization settings.

        Args:
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            shuffle: Enable shuffle filter for better compression
            fletcher32: Enable Fletcher32 checksum for data integrity
        """
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.logger = logging.getLogger(__name__)

    def create_optimized_dataset(
        self,
        h5_file: h5py.File,
        name: str,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        chunk_size: Optional[Tuple[int, ...]] = None,
    ) -> h5py.Dataset:
        """
        Create an optimized HDF5 dataset.

        Args:
            h5_file: Open HDF5 file
            name: Dataset name
            shape: Dataset shape
            dtype: Data type
            chunk_size: Chunk size for chunked storage

        Returns:
            Created dataset
        """
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunks(shape, dtype)

        dataset = h5_file.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunk_size,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            maxshape=tuple(
                None if i == 0 else s for i, s in enumerate(shape)
            ),  # Allow growth in first dimension
        )

        # Add metadata
        dataset.attrs["created"] = time.time()
        dataset.attrs["compression"] = self.compression
        dataset.attrs["chunk_size"] = chunk_size

        self.logger.info(
            f"Created dataset '{name}' with shape {shape}, chunks {chunk_size}"
        )
        return dataset

    def _calculate_optimal_chunks(
        self, shape: Tuple[int, ...], dtype: str
    ) -> Tuple[int, ...]:
        """Calculate optimal chunk size for given shape and dtype."""
        # Target chunk size of ~1MB
        target_size = 1024 * 1024  # 1MB

        # Get dtype size
        dtype_sizes = {"float32": 4, "float64": 8, "int32": 4, "int64": 8}
        element_size = dtype_sizes.get(dtype, 4)

        # Calculate chunk dimensions
        if len(shape) == 2:  # 2D array (samples, features)
            samples, features = shape
            elements_per_chunk = target_size // element_size

            if features > 0:
                samples_per_chunk = max(1, elements_per_chunk // features)
                samples_per_chunk = min(
                    samples_per_chunk, samples, 1000
                )  # Cap at 1000 samples
                return (samples_per_chunk, features)
            else:
                return (min(1000, samples), 1)

        elif len(shape) == 3:  # 3D array (samples, seq_len, features)
            samples, seq_len, features = shape
            elements_per_chunk = target_size // element_size

            # Try to fit at least one complete sequence
            if seq_len * features > 0:
                samples_per_chunk = max(1, elements_per_chunk // (seq_len * features))
                samples_per_chunk = min(
                    samples_per_chunk, samples, 100
                )  # Cap at 100 samples for 3D
                return (samples_per_chunk, seq_len, features)
            else:
                return (min(100, samples), seq_len, features)

        else:
            # Default chunking for other dimensions
            chunk = list(shape)
            if chunk[0] > 1000:
                chunk[0] = 1000
            return tuple(chunk)

    def batch_write(
        self, h5_file: h5py.File, dataset_name: str, data: Any, start_idx: int = 0
    ) -> None:
        """
        Write data to HDF5 dataset in optimized batches.

        Args:
            h5_file: Open HDF5 file
            dataset_name: Name of dataset
            data: Data to write
            start_idx: Starting index for writing
        """
        if dataset_name not in h5_file:
            raise ValueError(f"Dataset '{dataset_name}' not found in file")

        dataset = h5_file[dataset_name]
        end_idx = start_idx + len(data)

        # Resize dataset if needed
        if end_idx > dataset.shape[0]:
            new_shape = list(dataset.shape)
            new_shape[0] = end_idx
            dataset.resize(new_shape)

        # Write data
        dataset[start_idx:end_idx] = data

        self.logger.debug(
            f"Wrote {len(data)} items to '{dataset_name}' at indices {start_idx}:{end_idx}"
        )

    def get_storage_stats(self, file_path: str) -> Dict[str, Any]:
        """Get storage statistics for HDF5 file."""
        if not os.path.exists(file_path):
            return {"exists": False}

        file_size = os.path.getsize(file_path)

        stats = {
            "exists": True,
            "file_size_mb": file_size / (1024**2),
            "file_size_gb": file_size / (1024**3),
            "datasets": {},
        }

        try:
            with h5py.File(file_path, "r") as f:
                for name, dataset in f.items():
                    if isinstance(dataset, h5py.Dataset):
                        stats["datasets"][name] = {
                            "shape": dataset.shape,
                            "dtype": str(dataset.dtype),
                            "size_mb": dataset.nbytes / (1024**2),
                            "compression": dataset.compression,
                            "chunks": dataset.chunks,
                        }
        except Exception as e:
            stats["error"] = str(e)

        return stats

    def optimize_existing_file(
        self, input_path: str, output_path: str
    ) -> Dict[str, Any]:
        """
        Optimize an existing HDF5 file by recompressing with better settings.

        Args:
            input_path: Path to input file
            output_path: Path to optimized output file

        Returns:
            Optimization statistics
        """
        start_time = time.time()
        original_stats = self.get_storage_stats(input_path)

        with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
            for name, dataset in src.items():
                if isinstance(dataset, h5py.Dataset):
                    # Calculate optimal chunks
                    chunks = self._calculate_optimal_chunks(
                        dataset.shape, str(dataset.dtype)
                    )

                    # Create optimized dataset
                    new_dataset = self.create_optimized_dataset(
                        dst, name, dataset.shape, str(dataset.dtype), chunks
                    )

                    # Copy data in chunks
                    chunk_size = chunks[0] if chunks else 1000
                    for i in range(0, dataset.shape[0], chunk_size):
                        end_i = min(i + chunk_size, dataset.shape[0])
                        new_dataset[i:end_i] = dataset[i:end_i]

                    # Copy attributes
                    for attr_name, attr_value in dataset.attrs.items():
                        new_dataset.attrs[attr_name] = attr_value

        new_stats = self.get_storage_stats(output_path)
        optimization_time = time.time() - start_time

        return {
            "optimization_time": optimization_time,
            "original_size_mb": original_stats["file_size_mb"],
            "optimized_size_mb": new_stats["file_size_mb"],
            "compression_ratio": original_stats["file_size_mb"]
            / new_stats["file_size_mb"],
            "space_saved_mb": original_stats["file_size_mb"]
            - new_stats["file_size_mb"],
        }


def optimize_chunk_size(
    data_shape: Tuple[int, ...], dtype_size: int = 4, target_mb: float = 1.0
) -> Tuple[int, ...]:
    """
    Calculate optimal chunk size for HDF5 storage.

    Args:
        data_shape: Shape of the data
        dtype_size: Size of data type in bytes
        target_mb: Target chunk size in MB

    Returns:
        Optimal chunk shape
    """
    manager = HDF5Manager()
    return manager._calculate_optimal_chunks(
        data_shape, "float32" if dtype_size == 4 else "float64"
    )


def create_storage_directory(base_path: str, experiment_name: str) -> Dict[str, str]:
    """
    Create organized directory structure for SAE experiment storage.

    Args:
        base_path: Base path for storage
        experiment_name: Name of the experiment

    Returns:
        Dictionary of created directory paths
    """
    base_dir = Path(base_path) / experiment_name

    directories = {
        "base": str(base_dir),
        "activations": str(base_dir / "activations"),
        "models": str(base_dir / "models"),
        "results": str(base_dir / "results"),
        "logs": str(base_dir / "logs"),
        "configs": str(base_dir / "configs"),
    }

    for path in directories.values():
        os.makedirs(path, exist_ok=True)

    return directories


def estimate_storage_requirements(
    num_samples: int,
    sequence_length: int,
    hidden_size: int,
    compression_ratio: float = 0.3,
) -> Dict[str, float]:
    """
    Estimate storage requirements for SAE data.

    Args:
        num_samples: Number of activation samples
        sequence_length: Length of sequences
        hidden_size: Hidden dimension size
        compression_ratio: Expected compression ratio

    Returns:
        Storage estimates
    """
    # Calculate raw data size
    raw_bytes = num_samples * sequence_length * hidden_size * 4  # float32
    compressed_bytes = raw_bytes * compression_ratio

    # Add overhead for metadata and multiple datasets
    overhead_factor = 1.2
    total_bytes = compressed_bytes * overhead_factor

    return {
        "raw_gb": raw_bytes / (1024**3),
        "compressed_gb": compressed_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
        "compression_ratio": compression_ratio,
    }
