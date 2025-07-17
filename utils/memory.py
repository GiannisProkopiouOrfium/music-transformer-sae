"""Memory optimization utilities for large-scale SAE analysis."""

import psutil
import logging
from typing import Dict, Any, Optional


class MemoryOptimizer:
    """Optimizes memory usage for large-scale SAE training and analysis."""

    def __init__(self, reserve_ratio: float = 0.2):
        """
        Initialize memory optimizer.

        Args:
            reserve_ratio: Fraction of memory to keep in reserve
        """
        self.reserve_ratio = reserve_ratio
        self.logger = logging.getLogger(__name__)
        self.system_memory = psutil.virtual_memory().total
        self.available_memory = self.system_memory * (1 - reserve_ratio)

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "usable_gb": self.available_memory / (1024**3),
        }

    def calculate_batch_size(
        self,
        item_size_bytes: int,
        model_memory_gb: float = 2.0,
        safety_factor: float = 0.8,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            item_size_bytes: Size of each data item in bytes
            model_memory_gb: Memory used by model and overhead
            safety_factor: Safety margin (0.8 = use 80% of available)

        Returns:
            Optimal batch size
        """
        available_gb = self.get_memory_info()["available_gb"]
        usable_gb = (available_gb - model_memory_gb) * safety_factor
        usable_bytes = usable_gb * (1024**3)

        batch_size = max(1, int(usable_bytes / item_size_bytes))

        self.logger.info(f"Memory calculation:")
        self.logger.info(f"  Available: {available_gb:.1f} GB")
        self.logger.info(f"  Model overhead: {model_memory_gb:.1f} GB")
        self.logger.info(f"  Usable: {usable_gb:.1f} GB")
        self.logger.info(f"  Item size: {item_size_bytes / (1024**2):.1f} MB")
        self.logger.info(f"  Batch size: {batch_size}")

        return batch_size

    def estimate_activation_memory(
        self,
        sequence_length: int,
        hidden_size: int,
        num_samples: int,
        dtype_bytes: int = 4,
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for activation storage.

        Args:
            sequence_length: Length of sequences
            hidden_size: Hidden dimension size
            num_samples: Number of samples
            dtype_bytes: Bytes per element (4 for float32)

        Returns:
            Memory estimates in GB
        """
        single_activation_bytes = sequence_length * hidden_size * dtype_bytes
        total_bytes = single_activation_bytes * num_samples

        # Add overhead for HDF5 compression and metadata
        overhead_factor = 1.2
        total_with_overhead = total_bytes * overhead_factor

        return {
            "single_activation_mb": single_activation_bytes / (1024**2),
            "total_gb": total_bytes / (1024**3),
            "with_overhead_gb": total_with_overhead / (1024**3),
            "recommended_chunk_size": self._calculate_chunk_size(
                single_activation_bytes
            ),
        }

    def _calculate_chunk_size(
        self, single_item_bytes: int, target_chunk_mb: int = 500
    ) -> int:
        """Calculate optimal chunk size for HDF5 storage."""
        target_bytes = target_chunk_mb * (1024**2)
        chunk_size = max(1, target_bytes // single_item_bytes)
        return min(chunk_size, 10000)  # Cap at 10k items per chunk

    def optimize_for_sae_training(
        self,
        hidden_size: int,
        sae_expansion: int = 4,
        gpu_memory_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Optimize settings specifically for SAE training.

        Args:
            hidden_size: SAE input dimension
            sae_expansion: SAE expansion factor
            gpu_memory_gb: Available GPU memory (if applicable)

        Returns:
            Optimized training settings
        """
        # Estimate SAE model memory
        sae_hidden_size = hidden_size * sae_expansion
        model_params = 2 * hidden_size * sae_hidden_size  # encoder + decoder
        model_memory_bytes = model_params * 4 * 2  # params + gradients, float32
        model_memory_gb = model_memory_bytes / (1024**3)

        # Calculate batch memory
        activation_size_bytes = hidden_size * 4  # float32

        if gpu_memory_gb:
            # GPU training
            usable_memory_gb = gpu_memory_gb * 0.7  # Conservative estimate
            batch_memory_gb = usable_memory_gb - model_memory_gb
            batch_size = max(
                1, int((batch_memory_gb * (1024**3)) / activation_size_bytes)
            )
        else:
            # CPU training
            batch_size = self.calculate_batch_size(
                activation_size_bytes, model_memory_gb, safety_factor=0.6
            )

        # Gradient accumulation to maintain effective batch size
        target_effective_batch = 1024
        grad_accumulation = max(1, target_effective_batch // batch_size)

        settings = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accumulation,
            "effective_batch_size": batch_size * grad_accumulation,
            "model_memory_gb": model_memory_gb,
            "activation_memory_mb": activation_size_bytes / (1024**2),
            "recommended_dataloader_workers": min(4, psutil.cpu_count()),
        }

        self.logger.info(f"SAE training optimization:")
        self.logger.info(f"  Model memory: {model_memory_gb:.2f} GB")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Gradient accumulation: {grad_accumulation}")
        self.logger.info(f"  Effective batch size: {settings['effective_batch_size']}")

        return settings


def calculate_optimal_batch_size(
    sequence_length: int, hidden_size: int, target_memory_gb: float = 4.0
) -> int:
    """
    Calculate optimal batch size for given constraints.

    Args:
        sequence_length: Length of input sequences
        hidden_size: Hidden dimension size
        target_memory_gb: Target memory usage in GB

    Returns:
        Optimal batch size
    """
    optimizer = MemoryOptimizer()

    # Single item memory (float32)
    item_bytes = sequence_length * hidden_size * 4

    # Account for model overhead
    return optimizer.calculate_batch_size(
        item_size_bytes=item_bytes,
        model_memory_gb=1.0,  # Conservative model overhead
        safety_factor=0.8,
    )


def get_system_limits() -> Dict[str, float]:
    """Get system memory limits and recommendations."""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()

    return {
        "total_memory_gb": memory.total / (1024**3),
        "available_memory_gb": memory.available / (1024**3),
        "cpu_cores": cpu_count,
        "recommended_workers": min(8, cpu_count),
        "max_safe_memory_gb": memory.total * 0.8 / (1024**3),
    }
