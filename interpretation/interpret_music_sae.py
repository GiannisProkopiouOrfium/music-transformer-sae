import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pathlib
from sae.analyze_sae import analyze_sae_features


def interpret_music_features(results, activations_path, top_k=10):
    """Interpret what the SAE features might represent in musical terms."""

    model = results["model"]
    feature_frequencies = results["feature_frequencies"]
    hidden_acts = results["hidden_activations"]
    interesting_features = results["interesting_features"]

    print("🎵 INTERPRETING MUSIC SAE FEATURES 🎵")
    print("=" * 60)

    # Load original data to understand what triggers each feature
    from sae_data import create_sae_dataloader

    dataloader, data_info = create_sae_dataloader(
        activations_path, batch_size=64, shuffle=False, subsample=1000
    )

    # Get a representative sample
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        _, sample_hidden = model(sample_batch)

    print(f"Analyzing {len(interesting_features)} interesting features...")

    # Analysis 1: Find features with extreme activations
    print("\n🔥 TOP FEATURES BY ACTIVATION PATTERNS:")
    print("-" * 40)

    # Most selective features (activate rarely but strongly)
    feature_selectivity = torch.std(hidden_acts, dim=0) / (
        torch.mean(hidden_acts, dim=0) + 1e-8
    )
    most_selective = torch.topk(feature_selectivity, top_k).indices

    print("Most Selective Features (high variance/mean ratio):")
    for i, feat_idx in enumerate(most_selective):
        freq = feature_frequencies[feat_idx].item()
        selectivity = feature_selectivity[feat_idx].item()
        mean_act = torch.mean(hidden_acts[:, feat_idx]).item()
        max_act = torch.max(hidden_acts[:, feat_idx]).item()
        print(
            f"  Feature {feat_idx:4d}: {freq:.1%} active, selectivity={selectivity:.2f}, max_act={max_act:.2f}"
        )

    # Analysis 2: Feature clustering by activation patterns
    print(f"\n🎼 FEATURE ACTIVATION CLUSTERS:")
    print("-" * 40)

    # Sample a subset of features for clustering analysis
    n_sample_features = min(100, len(interesting_features))
    sample_features = interesting_features[
        torch.randperm(len(interesting_features))[:n_sample_features]
    ]

    # Compute pairwise correlations
    feature_correlations = torch.corrcoef(hidden_acts[:, sample_features].T)

    # Find highly correlated feature pairs (potential feature families)
    correlation_threshold = 0.3
    high_corr_pairs = []

    for i in range(feature_correlations.shape[0]):
        for j in range(i + 1, feature_correlations.shape[1]):
            corr = feature_correlations[i, j].item()
            if abs(corr) > correlation_threshold:
                feat_i = sample_features[i].item()
                feat_j = sample_features[j].item()
                high_corr_pairs.append((feat_i, feat_j, corr))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(
        f"Found {len(high_corr_pairs)} highly correlated feature pairs (|corr| > {correlation_threshold}):"
    )
    for feat_i, feat_j, corr in high_corr_pairs[:10]:
        print(f"  Features {feat_i:4d} ↔ {feat_j:4d}: correlation = {corr:+.3f}")

    # Analysis 3: Feature activation statistics
    print(f"\n📊 FEATURE STATISTICS:")
    print("-" * 40)

    # Distribution of activation frequencies
    freq_bins = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("Feature activation frequency distribution:")

    prev_threshold = 0.0
    for threshold in freq_bins:
        count = torch.sum(
            (feature_frequencies > prev_threshold) & (feature_frequencies <= threshold)
        ).item()
        print(f"  {prev_threshold:.1%} - {threshold:.1%}: {count:4d} features")
        prev_threshold = threshold

    count = torch.sum(feature_frequencies > prev_threshold).item()
    print(f"  >{prev_threshold:.1%}: {count:4d} features")

    # Analysis 4: Decoder weight analysis (what each feature reconstructs)
    print(f"\n🎹 DECODER PATTERN ANALYSIS:")
    print("-" * 40)

    decoder_weights = model.decoder.weight.data  # Shape: [512, 2048]

    # Find features with strongest reconstruction patterns
    decoder_norms = torch.norm(decoder_weights, dim=0)  # Norm for each feature
    strongest_features = torch.topk(decoder_norms, top_k).indices

    print("Features with strongest decoder patterns:")
    for i, feat_idx in enumerate(strongest_features):
        norm = decoder_norms[feat_idx].item()
        freq = feature_frequencies[feat_idx].item()

        # Analyze what this feature reconstructs
        decoder_pattern = decoder_weights[:, feat_idx]
        max_weight_idx = torch.argmax(torch.abs(decoder_pattern)).item()
        max_weight = decoder_pattern[max_weight_idx].item()

        print(
            f"  Feature {feat_idx:4d}: norm={norm:.3f}, freq={freq:.1%}, "
            f"peak_weight@{max_weight_idx}={max_weight:+.3f}"
        )

    # Analysis 5: Save feature activation heatmap
    print(f"\n💾 SAVING VISUALIZATIONS:")
    print("-" * 40)

    # Create a detailed heatmap of feature activations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Feature frequency distribution
    axes[0, 0].hist(feature_frequencies.numpy(), bins=50, alpha=0.7, color="skyblue")
    axes[0, 0].set_xlabel("Activation Frequency")
    axes[0, 0].set_ylabel("Number of Features")
    axes[0, 0].set_title("Feature Activation Frequency Distribution")
    axes[0, 0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="50% line")
    axes[0, 0].legend()

    # Plot 2: Selectivity vs Frequency scatter
    axes[0, 1].scatter(
        feature_frequencies.numpy(),
        feature_selectivity.numpy(),
        alpha=0.6,
        s=20,
        color="coral",
    )
    axes[0, 1].set_xlabel("Activation Frequency")
    axes[0, 1].set_ylabel("Selectivity (std/mean)")
    axes[0, 1].set_title("Feature Selectivity vs Activation Frequency")

    # Plot 3: Decoder weight norms
    axes[1, 0].hist(decoder_norms.numpy(), bins=50, alpha=0.7, color="lightgreen")
    axes[1, 0].set_xlabel("Decoder Weight Norm")
    axes[1, 0].set_ylabel("Number of Features")
    axes[1, 0].set_title("Decoder Pattern Strength Distribution")

    # Plot 4: Feature activation heatmap (sample)
    sample_features_for_heatmap = interesting_features[
        :50
    ]  # Top 50 interesting features
    sample_activations = hidden_acts[
        :100, sample_features_for_heatmap
    ]  # First 100 samples

    im = axes[1, 1].imshow(sample_activations.T.numpy(), aspect="auto", cmap="viridis")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Feature Index")
    axes[1, 1].set_title("Feature Activation Heatmap\n(50 features × 100 samples)")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("detailed_sae_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅ Saved detailed analysis to 'detailed_sae_analysis.png'")

    return {
        "most_selective_features": most_selective,
        "feature_correlations": high_corr_pairs,
        "strongest_decoder_features": strongest_features,
        "feature_selectivity": feature_selectivity,
        "decoder_norms": decoder_norms,
    }


if __name__ == "__main__":
    # Load previous results
    results = analyze_sae_features(
        "sae_models/sae_layer_2048d.pt",
        "exp/sod/ape/activations/activations_layers_3.h5",
    )

    # Detailed interpretation
    interpretation = interpret_music_features(
        results, "exp/sod/ape/activations/activations_layers_3.h5", top_k=15
    )
