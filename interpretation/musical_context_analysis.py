import torch
import numpy as np
import pathlib
import h5py
from typing import List, Dict
import matplotlib.pyplot as plt


def analyze_feature_musical_context(
    sae_model_path: str,
    activations_path: str,
    interesting_feature_ids: List[int],
    top_k: int = 5,
):
    """
    Analyze what musical contexts trigger specific SAE features.
    """

    print("🎼 MUSICAL CONTEXT ANALYSIS 🎼")
    print("=" * 50)

    # Load SAE model
    from train_sae import SparseAutoencoder

    checkpoint = torch.load(sae_model_path, weights_only=False)
    model = SparseAutoencoder(
        checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["sparsity_coeff"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load activations with musical context
    from sae_data import create_sae_dataloader

    dataloader, _ = create_sae_dataloader(
        activations_path, batch_size=32, shuffle=False, subsample=None  # Use all data
    )

    print(f"Analyzing {len(interesting_feature_ids)} features...")

    # Collect all activations and find top activating examples
    all_inputs = []
    all_hidden = []
    sample_indices = []

    with torch.no_grad():
        sample_idx = 0
        for batch in dataloader:
            _, hidden = model(batch)
            all_inputs.append(batch)
            all_hidden.append(hidden)

            # Track sample indices for this batch
            batch_indices = list(range(sample_idx, sample_idx + batch.shape[0]))
            sample_indices.extend(batch_indices)
            sample_idx += batch.shape[0]

    all_inputs = torch.cat(all_inputs, dim=0)
    all_hidden = torch.cat(all_hidden, dim=0)

    print(f"Collected {all_inputs.shape[0]} samples")

    # Analyze each interesting feature
    feature_analysis = {}

    for feat_id in interesting_feature_ids[:10]:  # Analyze top 10 features
        print(f"\n🎵 FEATURE {feat_id} ANALYSIS:")
        print("-" * 30)

        feature_activations = all_hidden[:, feat_id]

        # Find top activating samples
        top_indices = torch.topk(feature_activations, top_k).indices
        top_values = feature_activations[top_indices]

        print(f"Top {top_k} activating samples:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            sample_input = all_inputs[idx]
            print(f"  Sample {idx.item():4d}: activation = {val.item():.3f}")

            # Analyze the input pattern that caused this activation
            input_stats = {
                "mean": torch.mean(sample_input).item(),
                "std": torch.std(sample_input).item(),
                "max": torch.max(sample_input).item(),
                "min": torch.min(sample_input).item(),
                "sparsity": torch.mean((sample_input == 0).float()).item(),
            }

            print(
                f"    Input: mean={input_stats['mean']:.3f}, "
                f"std={input_stats['std']:.3f}, "
                f"max={input_stats['max']:.3f}, "
                f"sparsity={input_stats['sparsity']:.3f}"
            )

        # Analyze what this feature typically responds to
        high_activation_mask = feature_activations > torch.quantile(
            feature_activations, 0.9
        )
        high_activation_inputs = all_inputs[high_activation_mask]

        if len(high_activation_inputs) > 0:
            avg_high_input = torch.mean(high_activation_inputs, dim=0)

            # Find the most important input dimensions for this feature
            top_input_dims = torch.topk(torch.abs(avg_high_input), 5).indices

            print(f"  Most important input dimensions:")
            for dim in top_input_dims:
                weight = avg_high_input[dim].item()
                print(f"    Dim {dim.item():3d}: {weight:+.4f}")

        feature_analysis[feat_id] = {
            "top_samples": top_indices.tolist(),
            "top_activations": top_values.tolist(),
            "activation_stats": {
                "mean": torch.mean(feature_activations).item(),
                "std": torch.std(feature_activations).item(),
                "max": torch.max(feature_activations).item(),
                "activation_rate": torch.mean((feature_activations > 0).float()).item(),
            },
        }

    # Cross-feature analysis
    print(f"\n🔗 CROSS-FEATURE ANALYSIS:")
    print("-" * 30)

    # Find samples that activate multiple interesting features
    interesting_activations = all_hidden[:, interesting_feature_ids[:20]]
    multi_feature_samples = torch.sum(interesting_activations > 0, dim=1)

    print(f"Multi-feature activation distribution:")
    for n_features in range(1, min(11, len(interesting_feature_ids) + 1)):
        count = torch.sum(multi_feature_samples == n_features).item()
        if count > 0:
            print(f"  {n_features:2d} features active: {count:4d} samples")

    # Find the most "musical" samples (activate many features)
    most_musical_samples = torch.topk(multi_feature_samples, 10).indices
    print(f"\nMost 'musical' samples (activate many features):")
    for i, sample_idx in enumerate(most_musical_samples):
        n_active = multi_feature_samples[sample_idx].item()
        print(f"  Sample {sample_idx.item():4d}: {n_active} features active")

    return feature_analysis


def visualize_feature_patterns(
    feature_analysis: Dict, save_path: str = "feature_patterns.png"
):
    """Create visualizations of feature activation patterns."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Feature activation rates
    feature_ids = list(feature_analysis.keys())
    activation_rates = [
        feature_analysis[fid]["activation_stats"]["activation_rate"]
        for fid in feature_ids
    ]

    axes[0, 0].bar(
        range(len(feature_ids)), activation_rates, color="skyblue", alpha=0.7
    )
    axes[0, 0].set_xlabel("Feature Index")
    axes[0, 0].set_ylabel("Activation Rate")
    axes[0, 0].set_title("Feature Activation Rates")
    axes[0, 0].set_xticks(range(len(feature_ids)))
    axes[0, 0].set_xticklabels([str(fid) for fid in feature_ids], rotation=45)

    # Plot 2: Max activations
    max_activations = [
        feature_analysis[fid]["activation_stats"]["max"] for fid in feature_ids
    ]

    axes[0, 1].bar(range(len(feature_ids)), max_activations, color="coral", alpha=0.7)
    axes[0, 1].set_xlabel("Feature Index")
    axes[0, 1].set_ylabel("Max Activation")
    axes[0, 1].set_title("Maximum Feature Activations")
    axes[0, 1].set_xticks(range(len(feature_ids)))
    axes[0, 1].set_xticklabels([str(fid) for fid in feature_ids], rotation=45)

    # Plot 3: Activation distribution for first feature
    if feature_ids:
        first_feature = feature_ids[0]
        top_activations = feature_analysis[first_feature]["top_activations"]

        axes[1, 0].hist(top_activations, bins=10, alpha=0.7, color="lightgreen")
        axes[1, 0].set_xlabel("Activation Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(f"Top Activations for Feature {first_feature}")

    # Plot 4: Feature statistics comparison
    mean_acts = [
        feature_analysis[fid]["activation_stats"]["mean"] for fid in feature_ids
    ]
    std_acts = [feature_analysis[fid]["activation_stats"]["std"] for fid in feature_ids]

    axes[1, 1].scatter(mean_acts, std_acts, alpha=0.7, s=50, color="purple")
    axes[1, 1].set_xlabel("Mean Activation")
    axes[1, 1].set_ylabel("Std Activation")
    axes[1, 1].set_title("Feature Activation Statistics")

    for i, fid in enumerate(feature_ids):
        axes[1, 1].annotate(
            str(fid),
            (mean_acts[i], std_acts[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved feature pattern analysis to '{save_path}'")


if __name__ == "__main__":
    # Use the most selective features from previous analysis
    interesting_features = [1184, 1395, 804, 997, 534, 1988, 306, 549, 1780, 482]

    # Analyze musical context
    analysis = analyze_feature_musical_context(
        sae_model_path="sae_models/sae_layer_2048d.pt",
        activations_path="exp/sod/ape/activations/activations_layers_3.h5",
        interesting_feature_ids=interesting_features,
        top_k=5,
    )

    # Create visualizations
    visualize_feature_patterns(analysis, "musical_feature_analysis.png")

    print("\n🎉 Musical context analysis complete!")
