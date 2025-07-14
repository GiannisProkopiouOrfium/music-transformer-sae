#!/usr/bin/env python3
"""
GPU Memory Profiler for SAE Analysis
Profiles memory usage during different stages of the pipeline.
"""

import torch
import psutil
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List


class MemoryProfiler:
    """Profiles memory usage during SAE analysis."""

    def __init__(self, output_dir: str = "./profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.profiles = []
        self.start_time = time.time()

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()

    def profile_memory(self, stage: str, description: str = "") -> Dict[str, float]:
        """Profile current memory usage."""
        timestamp = time.time() - self.start_time

        # System memory
        memory = psutil.virtual_memory()
        sys_memory = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
        }

        # GPU memory
        gpu_memory = {"available": False}
        if self.gpu_available:
            gpu_memory = {
                "available": True,
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
            }

        profile = {
            "timestamp": timestamp,
            "stage": stage,
            "description": description,
            "system_memory": sys_memory,
            "gpu_memory": gpu_memory,
        }

        self.profiles.append(profile)

        print(f"[{timestamp:.1f}s] {stage}: {description}")
        print(
            f"  System: {sys_memory['used_gb']:.1f}/{sys_memory['total_gb']:.1f} GB ({sys_memory['percent']:.1f}%)"
        )
        if gpu_memory["available"]:
            print(
                f"  GPU: {gpu_memory['allocated_gb']:.1f} GB allocated, {gpu_memory['reserved_gb']:.1f} GB reserved"
            )
        print()

        return profile

    def save_profiles(self, filename: str = "memory_profile.json"):
        """Save memory profiles to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(self.profiles, f, indent=2)
        print(f"Memory profiles saved to {output_file}")

    def create_plots(self, filename: str = "memory_usage.png"):
        """Create memory usage plots."""
        if not self.profiles:
            print("No profiles to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        timestamps = [p["timestamp"] for p in self.profiles]
        stages = [p["stage"] for p in self.profiles]

        # System memory plot
        sys_used = [p["system_memory"]["used_gb"] for p in self.profiles]
        sys_total = [p["system_memory"]["total_gb"] for p in self.profiles]

        axes[0].plot(timestamps, sys_used, "b-", label="Used Memory")
        axes[0].axhline(y=sys_total[0], color="r", linestyle="--", label="Total Memory")
        axes[0].set_ylabel("Memory (GB)")
        axes[0].set_title("System Memory Usage")
        axes[0].legend()
        axes[0].grid(True)

        # Add stage markers
        for i, stage in enumerate(stages):
            if i == 0 or stage != stages[i - 1]:
                axes[0].axvline(x=timestamps[i], color="gray", linestyle=":", alpha=0.7)
                axes[0].text(
                    timestamps[i], max(sys_used) * 0.9, stage, rotation=90, fontsize=8
                )

        # GPU memory plot
        if self.gpu_available and any(
            p["gpu_memory"]["available"] for p in self.profiles
        ):
            gpu_allocated = [
                p["gpu_memory"]["allocated_gb"]
                for p in self.profiles
                if p["gpu_memory"]["available"]
            ]
            gpu_reserved = [
                p["gpu_memory"]["reserved_gb"]
                for p in self.profiles
                if p["gpu_memory"]["available"]
            ]
            gpu_timestamps = [
                p["timestamp"] for p in self.profiles if p["gpu_memory"]["available"]
            ]

            axes[1].plot(gpu_timestamps, gpu_allocated, "g-", label="Allocated")
            axes[1].plot(gpu_timestamps, gpu_reserved, "orange", label="Reserved")
            axes[1].set_ylabel("GPU Memory (GB)")
            axes[1].set_title("GPU Memory Usage")
            axes[1].legend()
            axes[1].grid(True)

            # Add stage markers
            for i, stage in enumerate(stages):
                if i == 0 or stage != stages[i - 1]:
                    axes[1].axvline(
                        x=timestamps[i], color="gray", linestyle=":", alpha=0.7
                    )
        else:
            axes[1].text(
                0.5,
                0.5,
                "No GPU Available",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("GPU Memory Usage (Not Available)")

        axes[1].set_xlabel("Time (seconds)")

        plt.tight_layout()
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Memory usage plot saved to {output_file}")


def profile_pipeline_stage(profiler: MemoryProfiler, stage: str, func, *args, **kwargs):
    """Profile a pipeline stage."""
    profiler.profile_memory(stage, f"Starting {stage}")

    # Run the function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    profiler.profile_memory(stage, f"Completed {stage} in {end_time - start_time:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Profiler for SAE Analysis")
    parser.add_argument(
        "--output-dir", default="./profiles", help="Output directory for profiles"
    )
    parser.add_argument(
        "--test-allocation", action="store_true", help="Test GPU memory allocation"
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=10.0,
        help="Maximum memory to allocate for testing",
    )

    args = parser.parse_args()

    profiler = MemoryProfiler(args.output_dir)

    # Initial profile
    profiler.profile_memory("initialization", "Starting memory profiler")

    if args.test_allocation and profiler.gpu_available:
        print("Testing GPU memory allocation...")

        # Test different allocation sizes
        test_sizes = [1, 2, 4, 8, args.max_memory_gb]

        for size_gb in test_sizes:
            if size_gb <= args.max_memory_gb:
                try:
                    # Allocate tensor
                    elements = int(size_gb * (1024**3) / 4)  # float32 = 4 bytes
                    tensor = torch.randn(elements, device="cuda")

                    profiler.profile_memory(
                        "allocation", f"Allocated {size_gb} GB tensor"
                    )

                    # Clean up
                    del tensor
                    torch.cuda.empty_cache()

                    profiler.profile_memory(
                        "cleanup", f"Cleaned up {size_gb} GB tensor"
                    )

                except Exception as e:
                    profiler.profile_memory(
                        "error", f"Failed to allocate {size_gb} GB: {str(e)}"
                    )
                    break

    # Final profile
    profiler.profile_memory("completion", "Memory profiling completed")

    # Save results
    profiler.save_profiles()
    profiler.create_plots()

    print(f"\nProfiler completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
