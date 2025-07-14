"""GPU utilities for AWS EC2 optimization."""

import torch
import psutil
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple


class GPUManager:
    """Manages GPU resources and optimization for AWS EC2 instances."""

    def __init__(self):
        self.device = self._detect_device()
        self.gpu_info = self._get_gpu_info()
        self.logger = logging.getLogger(__name__)

    def _detect_device(self) -> torch.device:
        """Detect and configure optimal device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("CUDA not available, using CPU")
        return device

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        if not torch.cuda.is_available():
            return {"available": False}

        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_available": torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(),
        }

        # Add driver and CUDA version info
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_info["driver_version"] = result.stdout.strip()
        except:
            gpu_info["driver_version"] = "Unknown"

        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["pytorch_version"] = torch.__version__

        return gpu_info

    def optimize_memory(self) -> None:
        """Optimize GPU memory settings."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Set memory fraction if needed
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                torch.cuda.set_per_process_memory_fraction(0.95)

            # Enable cudnn optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.logger.info("GPU memory optimizations applied")

    def get_optimal_settings(self, task_type: str = "training") -> Dict[str, Any]:
        """Get optimal settings for different tasks."""
        settings = {
            "device": self.device,
            "mixed_precision": torch.cuda.is_available(),
            "pin_memory": torch.cuda.is_available(),
            "num_workers": min(8, psutil.cpu_count()),
        }

        if torch.cuda.is_available():
            memory_gb = self.gpu_info["memory_total"] / (1024**3)

            if task_type == "training":
                if memory_gb >= 24:  # A100/V100
                    settings.update(
                        {
                            "batch_size": 32,
                            "gradient_accumulation": 1,
                            "max_tokens": 100000,
                        }
                    )
                elif memory_gb >= 16:  # T4
                    settings.update(
                        {
                            "batch_size": 16,
                            "gradient_accumulation": 2,
                            "max_tokens": 50000,
                        }
                    )
                else:  # Smaller GPUs
                    settings.update(
                        {
                            "batch_size": 8,
                            "gradient_accumulation": 4,
                            "max_tokens": 25000,
                        }
                    )
            elif task_type == "extraction":
                settings.update(
                    {
                        "batch_size": min(64, int(memory_gb * 2)),
                        "chunk_size": min(10000, int(memory_gb * 1000)),
                    }
                )
        else:
            # CPU settings
            settings.update(
                {
                    "batch_size": 4,
                    "gradient_accumulation": 8,
                    "max_tokens": 10000,
                    "mixed_precision": False,
                }
            )

        return settings

    def monitor_usage(self) -> Dict[str, float]:
        """Monitor current GPU usage."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "memory_allocated": torch.cuda.memory_allocated() / (1024**3),
            "memory_cached": torch.cuda.memory_reserved() / (1024**3),
            "memory_total": self.gpu_info["memory_total"] / (1024**3),
            "utilization_percent": self._get_gpu_utilization(),
        }

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except:
            return 0.0


def setup_device(prefer_gpu: bool = True) -> Tuple[torch.device, Dict[str, Any]]:
    """Setup and return optimal device with configuration."""
    manager = GPUManager()
    manager.optimize_memory()

    device = manager.device if prefer_gpu else torch.device("cpu")
    config = manager.get_optimal_settings()

    return device, config


def log_system_info():
    """Log comprehensive system information."""
    logger = logging.getLogger(__name__)

    # System info
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # GPU info
    manager = GPUManager()
    if manager.gpu_info["available"]:
        logger.info(f"GPU: {manager.gpu_info['device_name']}")
        logger.info(
            f"GPU Memory: {manager.gpu_info['memory_total'] / (1024**3):.1f} GB"
        )
        logger.info(f"CUDA Version: {manager.gpu_info['cuda_version']}")
        logger.info(f"Driver Version: {manager.gpu_info['driver_version']}")
    else:
        logger.info("No GPU available")
