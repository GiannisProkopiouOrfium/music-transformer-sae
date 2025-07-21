#!/usr/bin/env python3
"""
Comprehensive SAE Analysis Script

This script provides detailed analysis and visualization for trained SAE models.
It can be run standalone after training to generate comprehensive reports.

Usage:
    python sae/analyze_trained_sae.py --model-path exp/sod/ape/sae_models/sae_layer_2048d.pt

    # Or analyze multiple models
    python sae/analyze_trained_sae.py --model-dir exp/sod/ape/sae_models/
"""

import argparse
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from sae.sae_data import create_sae_dataloader
from sae.train_sae import SparseAutoencoder


class SAEAnalyzer:
    """Comprehensive analyzer for trained SAE models."""

    def __init__(self, model_path: str, activations_path: Optional[str] = None):
        """
        Initialize analyzer with trained model.

        Args:
            model_path: Path to trained SAE model
            activations_path: Optional path to activations data
        """
        self.model_path = pathlib.Path(model_path)
        self.activations_path = activations_path
        self.checkpoint = None
        self.model = None
        self.analysis_results = {}

        self._load_model()

    def _load_model(self):
        """Load the trained SAE model."""
        print(f"📦 Loading SAE model from: {self.model_path}")

        self.checkpoint = torch.load(self.model_path, map_location="cpu")

        # Recreate model
        self.model = SparseAutoencoder(
            input_dim=self.checkpoint["input_dim"],
            hidden_dim=self.checkpoint["config"]["hidden_dim"],
            sparsity_coeff=self.checkpoint["config"]["sparsity_coeff"],
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        print(
            f"✅ Loaded SAE: {self.checkpoint['input_dim']} -> {self.checkpoint['config']['hidden_dim']}"
        )
        print(f"   Sparsity coeff: {self.checkpoint['config']['sparsity_coeff']}")
        print(f"   Best val loss: {self.checkpoint.get('best_val_loss', 'unknown')}")
        print(f"   Final sparsity: {self.checkpoint.get('final_sparsity', 'unknown')}")

    def analyze_model_weights(self) -> Dict:
        """Analyze the learned weights and architecture."""
        print("🔍 Analyzing model weights...")

        encoder_weights = self.model.encoder.weight.data
        decoder_weights = self.model.decoder.weight.data

        # Weight statistics
        analysis = {
            "encoder_stats": {
                "mean": torch.mean(encoder_weights).item(),
                "std": torch.std(encoder_weights).item(),
                "min": torch.min(encoder_weights).item(),
                "max": torch.max(encoder_weights).item(),
                "norm": torch.norm(encoder_weights).item(),
            },
            "decoder_stats": {
                "mean": torch.mean(decoder_weights).item(),
                "std": torch.std(decoder_weights).item(),
                "min": torch.min(decoder_weights).item(),
                "max": torch.max(decoder_weights).item(),
                "column_norms": torch.norm(decoder_weights, dim=0).numpy(),
            },
            "architecture": {
                "input_dim": self.checkpoint["input_dim"],
                "hidden_dim": self.checkpoint["config"]["hidden_dim"],
                "expansion_factor": self.checkpoint["config"]["hidden_dim"]
                / self.checkpoint["input_dim"],
            },
        }

        # Check decoder normalization
        decoder_norms = torch.norm(decoder_weights, dim=0)
        analysis["decoder_stats"]["norm_violation"] = torch.mean(
            torch.abs(decoder_norms - 1.0)
        ).item()

        return analysis

    def analyze_activations(self, subsample: int = 5000) -> Dict:
        """Analyze feature activations on validation data."""
        if not self.activations_path:
            print("⚠️ No activations path provided, skipping activation analysis")
            return {}

        print(f"🎯 Analyzing activations (subsample: {subsample})...")

        # Load activations data
        layer_key = self.checkpoint.get("layer_key", "layer_3")
        dataloader, _ = create_sae_dataloader(
            self.activations_path,
            layer_key=layer_key,
            batch_size=512,
            shuffle=False,
            subsample=subsample,
        )

        # Collect activations
        all_inputs = []
        all_hidden = []
        all_reconstructed = []

        with torch.no_grad():
            for batch in dataloader:
                reconstructed, hidden = self.model(batch)

                all_inputs.append(batch)
                all_hidden.append(hidden)
                all_reconstructed.append(reconstructed)

        inputs = torch.cat(all_inputs, dim=0)
        hidden_acts = torch.cat(all_hidden, dim=0)
        reconstructed = torch.cat(all_reconstructed, dim=0)

        print(f"   Processed {inputs.shape[0]} samples")

        # Feature analysis
        feature_frequencies = torch.mean((hidden_acts > 0).float(), dim=0)
        feature_magnitudes = torch.mean(hidden_acts, dim=0)
        feature_max_acts = torch.max(hidden_acts, dim=0)[0]

        # Reconstruction analysis
        recon_errors = torch.mean((inputs - reconstructed) ** 2, dim=1)

        # Sparsity analysis
        sparsity_ratio = torch.mean((hidden_acts == 0).float()).item()

        # Feature correlation analysis
        n_features_for_corr = min(100, hidden_acts.shape[1])
        sample_features = torch.randperm(hidden_acts.shape[1])[:n_features_for_corr]
        feature_correlations = torch.corrcoef(hidden_acts[:, sample_features].T)

        analysis = {
            "feature_statistics": {
                "activation_frequencies": feature_frequencies.numpy(),
                "mean_magnitudes": feature_magnitudes.numpy(),
                "max_activations": feature_max_acts.numpy(),
                "sparsity_ratio": sparsity_ratio,
                "active_features": torch.sum(feature_frequencies > 0).item(),
                "highly_active_features": torch.sum(feature_frequencies > 0.1).item(),
                "rare_features": torch.sum(feature_frequencies < 0.01).item(),
                "dead_features": torch.sum(feature_frequencies == 0).item(),
            },
            "reconstruction_statistics": {
                "mean_error": torch.mean(recon_errors).item(),
                "median_error": torch.median(recon_errors).item(),
                "std_error": torch.std(recon_errors).item(),
                "max_error": torch.max(recon_errors).item(),
                "min_error": torch.min(recon_errors).item(),
            },
            "correlation_analysis": {
                "feature_correlations": feature_correlations.numpy(),
                "high_correlation_pairs": self._find_high_correlation_pairs(
                    feature_correlations, sample_features
                ),
            },
            "data_info": {
                "samples_analyzed": inputs.shape[0],
                "layer_key": layer_key,
            },
        }

        return analysis

    def _find_high_correlation_pairs(
        self,
        correlations: torch.Tensor,
        feature_indices: torch.Tensor,
        threshold: float = 0.7,
    ) -> List[Tuple]:
        """Find pairs of features with high correlation."""
        pairs = []
        n_features = correlations.shape[0]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = correlations[i, j].item()
                if abs(corr) > threshold:
                    pairs.append(
                        (feature_indices[i].item(), feature_indices[j].item(), corr)
                    )

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def analyze_training_history(self) -> Dict:
        """Analyze training history if available."""
        training_history = self.checkpoint.get("training_history", [])

        if not training_history:
            print("⚠️ No training history available")
            return {}

        print(f"📈 Analyzing training history ({len(training_history)} epochs)")

        # Extract data with error handling
        epochs = []
        train_losses = []
        val_losses = []
        sparsity_ratios = []

        for entry in training_history:
            try:
                epochs.append(entry["epoch"])
                train_losses.append(entry["train"]["total_loss"])
                val_losses.append(entry["val"]["total_loss"])
                sparsity_ratios.append(entry["val"]["sparsity_ratio"])
            except (KeyError, TypeError) as e:
                print(f"⚠️ Skipping malformed training entry: {e}")
                continue

        if not epochs:
            print("⚠️ No valid training history entries found")
            return {}

        # Find best epoch safely
        best_epoch = 1
        if val_losses:
            try:
                best_epoch = (
                    min(range(len(val_losses)), key=lambda i: val_losses[i]) + 1
                )
            except (ValueError, IndexError):
                best_epoch = 1

        analysis = {
            "training_progress": {
                "epochs": epochs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "sparsity_ratios": sparsity_ratios,
                "best_epoch": best_epoch,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "final_val_loss": val_losses[-1] if val_losses else 0.0,
                "best_val_loss": min(val_losses) if val_losses else 0.0,
                "convergence_info": self._analyze_convergence(val_losses),
            }
        }

        return analysis

    def _analyze_convergence(self, losses: List[float], window: int = 10) -> Dict:
        """Analyze training convergence."""
        if len(losses) < 2:
            return {
                "converged": False,
                "reason": "insufficient_data",
                "recent_variance": 0.0,
                "recent_mean": losses[0] if losses else 0.0,
                "is_improving": False,
                "plateau_epochs": 0,
            }

        # Adjust window size if we don't have enough data
        effective_window = min(window, len(losses))

        # Check if loss has plateaued in the last window epochs
        recent_losses = losses[-effective_window:]
        loss_variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)

        # Check for monotonic improvement (only if we have enough data)
        is_improving = False
        if len(losses) >= 2:
            # Check last few pairs for improvement
            comparison_window = min(effective_window - 1, len(losses) - 1)
            if comparison_window > 0:
                is_improving = all(
                    losses[-(comparison_window + 1) + i]
                    >= losses[-(comparison_window + 1) + i + 1]
                    for i in range(comparison_window)
                )

        # Check convergence criteria
        converged = (
            loss_variance < 0.0001 and not is_improving and len(losses) >= window
        )

        return {
            "converged": converged,
            "recent_variance": loss_variance,
            "recent_mean": mean_loss,
            "is_improving": is_improving,
            "plateau_epochs": effective_window if converged else 0,
        }

    def create_comprehensive_report(self, output_dir: str):
        """Create comprehensive analysis report with visualizations."""
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"📊 Creating comprehensive analysis report...")

        # Run all analyses
        weight_analysis = self.analyze_model_weights()
        activation_analysis = self.analyze_activations()
        training_analysis = self.analyze_training_history()

        # Store results
        self.analysis_results = {
            "model_info": {
                "model_path": str(self.model_path),
                "activations_path": self.activations_path,
                "checkpoint_keys": list(self.checkpoint.keys()),
            },
            "weight_analysis": weight_analysis,
            "activation_analysis": activation_analysis,
            "training_analysis": training_analysis,
        }

        # Create visualizations
        self._create_weight_visualizations(output_path)
        if activation_analysis:
            self._create_activation_visualizations(output_path, activation_analysis)
        if training_analysis:
            self._create_training_visualizations(output_path, training_analysis)

        # Save comprehensive report
        report_file = output_path / "sae_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(
                self._prepare_json_serializable(self.analysis_results), f, indent=2
            )

        # Create summary report
        self._create_summary_report(output_path)

        print(f"✅ Analysis complete! Results saved to: {output_path}")
        return output_path

    def _prepare_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._prepare_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _create_weight_visualizations(self, output_path: pathlib.Path):
        """Create weight analysis visualizations."""
        print("   📈 Creating weight visualizations...")

        encoder_weights = self.model.encoder.weight.data.numpy()
        decoder_weights = self.model.decoder.weight.data.numpy()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Encoder weight distribution
        axes[0, 0].hist(encoder_weights.flatten(), bins=50, alpha=0.7, color="blue")
        axes[0, 0].set_title("Encoder Weight Distribution")
        axes[0, 0].set_xlabel("Weight Value")
        axes[0, 0].set_ylabel("Frequency")

        # Decoder weight distribution
        axes[0, 1].hist(decoder_weights.flatten(), bins=50, alpha=0.7, color="red")
        axes[0, 1].set_title("Decoder Weight Distribution")
        axes[0, 1].set_xlabel("Weight Value")
        axes[0, 1].set_ylabel("Frequency")

        # Decoder column norms
        decoder_norms = np.linalg.norm(decoder_weights, axis=0)
        axes[0, 2].hist(decoder_norms, bins=50, alpha=0.7, color="green")
        axes[0, 2].axvline(1.0, color="red", linestyle="--", label="Target norm")
        axes[0, 2].set_title("Decoder Column Norms")
        axes[0, 2].set_xlabel("L2 Norm")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].legend()

        # Weight matrices as heatmaps (sample)
        sample_size = min(100, encoder_weights.shape[0])
        encoder_sample = encoder_weights[:sample_size, :sample_size]
        im1 = axes[1, 0].imshow(encoder_sample, cmap="RdBu_r", aspect="auto")
        axes[1, 0].set_title(f"Encoder Weights Sample ({sample_size}x{sample_size})")
        plt.colorbar(im1, ax=axes[1, 0])

        decoder_sample = decoder_weights[:sample_size, :sample_size]
        im2 = axes[1, 1].imshow(decoder_sample, cmap="RdBu_r", aspect="auto")
        axes[1, 1].set_title(f"Decoder Weights Sample ({sample_size}x{sample_size})")
        plt.colorbar(im2, ax=axes[1, 1])

        # Weight norm vs feature index
        axes[1, 2].plot(decoder_norms, alpha=0.7)
        axes[1, 2].set_title("Decoder Norms by Feature")
        axes[1, 2].set_xlabel("Feature Index")
        axes[1, 2].set_ylabel("L2 Norm")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "weight_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _create_activation_visualizations(
        self, output_path: pathlib.Path, analysis: Dict
    ):
        """Create activation analysis visualizations."""
        print("   🎯 Creating activation visualizations...")

        feature_stats = analysis["feature_statistics"]
        recon_stats = analysis["reconstruction_statistics"]

        frequencies = feature_stats["activation_frequencies"]
        magnitudes = feature_stats["mean_magnitudes"]

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Feature activation frequency
        axes[0, 0].hist(frequencies, bins=50, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("Feature Activation Frequency Distribution")
        axes[0, 0].set_xlabel("Activation Frequency")
        axes[0, 0].set_ylabel("Number of Features")
        axes[0, 0].axvline(0.01, color="red", linestyle="--", label="1%")
        axes[0, 0].axvline(0.1, color="orange", linestyle="--", label="10%")
        axes[0, 0].legend()

        # Feature magnitude distribution
        axes[0, 1].hist(magnitudes, bins=50, alpha=0.7, color="lightgreen")
        axes[0, 1].set_title("Feature Magnitude Distribution")
        axes[0, 1].set_xlabel("Mean Activation Magnitude")
        axes[0, 1].set_ylabel("Number of Features")

        # Feature ranking
        sorted_freq = np.sort(frequencies)[::-1]
        axes[0, 2].plot(sorted_freq)
        axes[0, 2].set_title("Features Ranked by Frequency")
        axes[0, 2].set_xlabel("Feature Rank")
        axes[0, 2].set_ylabel("Activation Frequency")
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(True, alpha=0.3)

        # Sparsity statistics
        categories = [
            "Dead\n(0%)",
            "Rare\n(<1%)",
            "Moderate\n(1-10%)",
            "Active\n(>10%)",
        ]
        counts = [
            feature_stats["dead_features"],
            feature_stats["rare_features"] - feature_stats["dead_features"],
            feature_stats["active_features"] - feature_stats["highly_active_features"],
            feature_stats["highly_active_features"],
        ]
        colors = ["red", "orange", "yellow", "green"]

        axes[1, 0].bar(categories, counts, color=colors, alpha=0.7)
        axes[1, 0].set_title("Feature Activity Categories")
        axes[1, 0].set_ylabel("Number of Features")

        # Frequency vs magnitude scatter
        axes[1, 1].scatter(frequencies, magnitudes, alpha=0.6, s=10)
        axes[1, 1].set_xlabel("Activation Frequency")
        axes[1, 1].set_ylabel("Mean Activation Magnitude")
        axes[1, 1].set_title("Frequency vs Magnitude")
        axes[1, 1].grid(True, alpha=0.3)

        # Reconstruction error statistics
        recon_data = [
            recon_stats["min_error"],
            recon_stats["median_error"],
            recon_stats["mean_error"],
            recon_stats["max_error"],
        ]
        labels = ["Min", "Median", "Mean", "Max"]

        axes[1, 2].bar(labels, recon_data, alpha=0.7, color="purple")
        axes[1, 2].set_title("Reconstruction Error Statistics")
        axes[1, 2].set_ylabel("Error Value")

        # Feature correlation heatmap (if available)
        if "correlation_analysis" in analysis:
            corr_matrix = analysis["correlation_analysis"]["feature_correlations"]
            im = axes[2, 0].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[2, 0].set_title("Feature Correlation Matrix (Sample)")
            plt.colorbar(im, ax=axes[2, 0])
        else:
            axes[2, 0].text(
                0.5,
                0.5,
                "Correlation\ndata not\navailable",
                ha="center",
                va="center",
                transform=axes[2, 0].transAxes,
            )

        # Sparsity overview pie chart
        total_features = len(frequencies)
        sparsity_data = [
            feature_stats["dead_features"] / total_features * 100,
            (feature_stats["rare_features"] - feature_stats["dead_features"])
            / total_features
            * 100,
            (feature_stats["active_features"] - feature_stats["highly_active_features"])
            / total_features
            * 100,
            feature_stats["highly_active_features"] / total_features * 100,
        ]

        axes[2, 1].pie(
            sparsity_data, labels=categories, colors=colors, autopct="%1.1f%%"
        )
        axes[2, 1].set_title("Feature Activity Distribution")

        # Summary statistics text
        axes[2, 2].axis("off")
        summary_text = f"""
ACTIVATION ANALYSIS SUMMARY

Total Features: {total_features:,}
Active Features: {feature_stats['active_features']:,} ({feature_stats['active_features']/total_features*100:.1f}%)
Highly Active: {feature_stats['highly_active_features']:,} ({feature_stats['highly_active_features']/total_features*100:.1f}%)
Dead Features: {feature_stats['dead_features']:,} ({feature_stats['dead_features']/total_features*100:.1f}%)

Sparsity Ratio: {feature_stats['sparsity_ratio']:.3f}

RECONSTRUCTION QUALITY
Mean Error: {recon_stats['mean_error']:.4f}
Median Error: {recon_stats['median_error']:.4f}
Std Error: {recon_stats['std_error']:.4f}

Samples Analyzed: {analysis['data_info']['samples_analyzed']:,}
        """

        axes[2, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[2, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(
            output_path / "activation_analysis.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _create_training_visualizations(
        self, output_path: pathlib.Path, analysis: Dict
    ):
        """Create training progress visualizations."""
        print("   📈 Creating training visualizations...")

        progress = analysis["training_progress"]

        # Safety check for empty data
        if not progress.get("epochs") or len(progress["epochs"]) == 0:
            print("   ⚠️ No training data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = progress["epochs"]
        train_losses = progress["train_losses"]
        val_losses = progress["val_losses"]
        sparsity_ratios = progress["sparsity_ratios"]

        # Training and validation loss
        if train_losses and val_losses:
            axes[0, 0].plot(epochs, train_losses, "b-", label="Training", linewidth=2)
            axes[0, 0].plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Training Progress")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "Loss data\nnot available",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Training Progress")

        # Sparsity evolution
        if sparsity_ratios:
            axes[0, 1].plot(epochs, sparsity_ratios, "g-", linewidth=2)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Sparsity Ratio")
            axes[0, 1].set_title("Sparsity Evolution")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Sparsity data\nnot available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Sparsity Evolution")

        # Loss improvement
        if val_losses and len(val_losses) > 1:
            loss_improvement = [val_losses[0] - loss for loss in val_losses]
            axes[1, 0].plot(epochs, loss_improvement, "purple", linewidth=2)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss Improvement from Start")
            axes[1, 0].set_title("Cumulative Loss Improvement")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Insufficient data\nfor improvement",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Cumulative Loss Improvement")

        # Training summary text
        axes[1, 1].axis("off")

        convergence = progress.get("convergence_info", {})

        # Calculate improvement metrics safely
        total_reduction = 0.0
        relative_improvement = 0.0
        if val_losses and len(val_losses) > 1 and val_losses[0] > 0:
            total_reduction = val_losses[0] - val_losses[-1]
            relative_improvement = (total_reduction / val_losses[0]) * 100

        summary_text = f"""
TRAINING SUMMARY

Total Epochs: {len(epochs)}
Best Epoch: {progress.get('best_epoch', 'N/A')}

FINAL METRICS
Train Loss: {progress.get('final_train_loss', 0.0):.4f}
Val Loss: {progress.get('final_val_loss', 0.0):.4f}
Best Val Loss: {progress.get('best_val_loss', 0.0):.4f}

CONVERGENCE ANALYSIS
Converged: {convergence.get('converged', False)}
Recent Variance: {convergence.get('recent_variance', 0.0):.6f}
Is Improving: {convergence.get('is_improving', False)}

IMPROVEMENT
Total Loss Reduction: {total_reduction:.4f}
Relative Improvement: {relative_improvement:.1f}%
        """

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output_path / "training_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _create_summary_report(self, output_path: pathlib.Path):
        """Create a human-readable summary report."""
        print("   📋 Creating summary report...")

        summary_file = output_path / "analysis_summary.md"

        with open(summary_file, "w") as f:
            f.write("# SAE Analysis Report\n\n")
            f.write(f"**Model Path:** `{self.model_path}`\n")
            f.write(f"**Analysis Date:** {pathlib.Path().resolve()}\n\n")

            # Model architecture
            f.write("## Model Architecture\n\n")
            if "weight_analysis" in self.analysis_results:
                arch = self.analysis_results["weight_analysis"]["architecture"]
                f.write(f"- **Input Dimension:** {arch['input_dim']}\n")
                f.write(f"- **Hidden Dimension:** {arch['hidden_dim']}\n")
                f.write(f"- **Expansion Factor:** {arch['expansion_factor']:.1f}x\n\n")

            # Training summary
            if (
                "training_analysis" in self.analysis_results
                and self.analysis_results["training_analysis"]
            ):
                f.write("## Training Summary\n\n")
                progress = self.analysis_results["training_analysis"][
                    "training_progress"
                ]
                f.write(f"- **Total Epochs:** {len(progress['epochs'])}\n")
                f.write(f"- **Best Epoch:** {progress['best_epoch']}\n")
                f.write(
                    f"- **Final Validation Loss:** {progress['final_val_loss']:.4f}\n"
                )
                f.write(
                    f"- **Best Validation Loss:** {progress['best_val_loss']:.4f}\n\n"
                )

            # Feature analysis
            if (
                "activation_analysis" in self.analysis_results
                and self.analysis_results["activation_analysis"]
            ):
                f.write("## Feature Analysis\n\n")
                stats = self.analysis_results["activation_analysis"][
                    "feature_statistics"
                ]
                total = len(
                    self.analysis_results["activation_analysis"]["feature_statistics"][
                        "activation_frequencies"
                    ]
                )

                f.write(f"- **Total Features:** {total:,}\n")
                f.write(
                    f"- **Active Features:** {stats['active_features']:,} ({stats['active_features']/total*100:.1f}%)\n"
                )
                f.write(
                    f"- **Highly Active Features (>10%):** {stats['highly_active_features']:,} ({stats['highly_active_features']/total*100:.1f}%)\n"
                )
                f.write(
                    f"- **Dead Features:** {stats['dead_features']:,} ({stats['dead_features']/total*100:.1f}%)\n"
                )
                f.write(f"- **Overall Sparsity:** {stats['sparsity_ratio']:.3f}\n\n")

                f.write("## Reconstruction Quality\n\n")
                recon = self.analysis_results["activation_analysis"][
                    "reconstruction_statistics"
                ]
                f.write(f"- **Mean Error:** {recon['mean_error']:.4f}\n")
                f.write(f"- **Median Error:** {recon['median_error']:.4f}\n")
                f.write(f"- **Standard Deviation:** {recon['std_error']:.4f}\n\n")

            # Generated files
            f.write("## Generated Analysis Files\n\n")
            f.write(
                "- `weight_analysis.png` - Weight distribution and structure analysis\n"
            )
            f.write(
                "- `activation_analysis.png` - Feature activation patterns and statistics\n"
            )
            f.write(
                "- `training_analysis.png` - Training progress and convergence analysis\n"
            )
            f.write(
                "- `sae_analysis_report.json` - Complete analysis data in JSON format\n"
            )
            f.write("- `analysis_summary.md` - This human-readable summary\n")

        print(f"   ✅ Summary report saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SAE Analysis Tool")
    parser.add_argument("--model-path", required=True, help="Path to trained SAE model")
    parser.add_argument(
        "--activations-path", help="Path to activations data (optional)"
    )
    parser.add_argument("--output-dir", help="Output directory for analysis results")
    parser.add_argument(
        "--subsample",
        type=int,
        default=5000,
        help="Number of samples to use for activation analysis",
    )

    args = parser.parse_args()

    # Set default output directory
    if not args.output_dir:
        model_dir = pathlib.Path(args.model_path).parent
        args.output_dir = model_dir / "analysis_results"

    # Infer activations path if not provided
    if not args.activations_path:
        model_dir = pathlib.Path(args.model_path).parent.parent
        activations_dir = model_dir / "activations"
        if activations_dir.exists():
            activation_files = list(activations_dir.glob("*.h5"))
            if activation_files:
                args.activations_path = str(activation_files[0])
                print(f"📁 Auto-detected activations: {args.activations_path}")

    print("🎼 COMPREHENSIVE SAE ANALYSIS")
    print("=" * 50)

    # Create analyzer and run analysis
    analyzer = SAEAnalyzer(args.model_path, args.activations_path)
    output_path = analyzer.create_comprehensive_report(args.output_dir)

    print("\n" + "=" * 50)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📋 Summary report: {output_path}/analysis_summary.md")
    print(f"📊 Visualizations: {output_path}/*.png")
    print(f"📄 Data: {output_path}/sae_analysis_report.json")


if __name__ == "__main__":
    main()
