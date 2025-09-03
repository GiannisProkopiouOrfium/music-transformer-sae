import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .sae_data import create_sae_dataloader


def analyze_sae_features(model_path: str, activations_path: str):
    """Analyze the learned SAE features."""

    # Load the trained model
    checkpoint = torch.load(model_path, map_location="cpu")

    # Create a new model instance
    from train_sae import SparseAutoencoder

    model = SparseAutoencoder(
        checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["sparsity_coeff"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    dataloader, _ = create_sae_dataloader(
        activations_path,
        batch_size=512,
        shuffle=False,
        subsample=2000,  # Use subset for analysis
    )

    print("Analyzing SAE features...")

    # Collect activations and features
    all_inputs = []
    all_hidden = []
    all_reconstructed = []

    with torch.no_grad():
        for batch in dataloader:
            reconstructed, hidden = model(batch)

            all_inputs.append(batch)
            all_hidden.append(hidden)
            all_reconstructed.append(reconstructed)

    inputs = torch.cat(all_inputs, dim=0)
    hidden_acts = torch.cat(all_hidden, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)

    print(f"Collected {inputs.shape[0]} samples")

    # Analysis 1: Feature activation frequency
    feature_activation_freq = torch.mean((hidden_acts > 0).float(), dim=0)

    plt.figure(figsize=(12, 8))

    # Plot 1: Feature activation frequency histogram
    plt.subplot(2, 3, 1)
    plt.hist(feature_activation_freq.numpy(), bins=50, alpha=0.7)
    plt.xlabel("Activation Frequency")
    plt.ylabel("Number of Features")
    plt.title("Feature Activation Frequency Distribution")

    # Plot 2: Most and least active features
    plt.subplot(2, 3, 2)
    sorted_freq, indices = torch.sort(feature_activation_freq, descending=True)
    plt.plot(sorted_freq.numpy())
    plt.xlabel("Feature Rank")
    plt.ylabel("Activation Frequency")
    plt.title("Features Ranked by Activation Frequency")
    plt.yscale("log")

    # Plot 3: Feature activation magnitudes
    plt.subplot(2, 3, 3)
    mean_magnitudes = torch.mean(hidden_acts, dim=0)
    plt.hist(mean_magnitudes.numpy(), bins=50, alpha=0.7)
    plt.xlabel("Mean Activation Magnitude")
    plt.ylabel("Number of Features")
    plt.title("Feature Activation Magnitude Distribution")

    # Analysis 2: Reconstruction quality
    recon_error = torch.mean((inputs - reconstructed) ** 2, dim=1)

    plt.subplot(2, 3, 4)
    plt.hist(recon_error.numpy(), bins=50, alpha=0.7)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of Samples")
    plt.title("Per-Sample Reconstruction Error")

    # Plot 5: Input vs Reconstructed correlation
    plt.subplot(2, 3, 5)
    correlations = []
    for i in range(min(100, inputs.shape[0])):  # Sample 100 examples
        corr = torch.corrcoef(torch.stack([inputs[i], reconstructed[i]]))[0, 1]
        if not torch.isnan(corr):
            correlations.append(corr.item())

    plt.hist(correlations, bins=30, alpha=0.7)
    plt.xlabel("Input-Reconstruction Correlation")
    plt.ylabel("Number of Samples")
    plt.title("Reconstruction Quality (Correlation)")

    # Plot 6: Feature co-activation
    plt.subplot(2, 3, 6)
    # Sample random pairs of features to analyze co-activation
    n_features_to_sample = 20
    random_features = torch.randperm(hidden_acts.shape[1])[:n_features_to_sample]
    coactivation_matrix = torch.corrcoef(hidden_acts[:, random_features].T)

    sns.heatmap(
        coactivation_matrix.numpy(),
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title(f"Feature Co-activation Matrix\n(Random {n_features_to_sample} features)")

    plt.tight_layout()
    plt.savefig("sae_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print statistics
    print("\n" + "=" * 50)
    print("SAE FEATURE ANALYSIS")
    print("=" * 50)

    print(f"Total features: {hidden_acts.shape[1]}")
    print(f"Average sparsity: {torch.mean((hidden_acts == 0).float()):.3f}")
    print(
        f"Most active feature: {torch.max(feature_activation_freq):.3f} activation rate"
    )
    print(
        f"Least active feature: {torch.min(feature_activation_freq):.3f} activation rate"
    )
    print(
        f"Features with >10% activation: {torch.sum(feature_activation_freq > 0.1).item()}"
    )
    print(
        f"Features with <1% activation: {torch.sum(feature_activation_freq < 0.01).item()}"
    )

    print(f"\nReconstruction Quality:")
    print(f"Mean reconstruction error: {torch.mean(recon_error):.4f}")
    print(f"Median reconstruction error: {torch.median(recon_error):.4f}")
    if correlations:
        print(f"Mean input-reconstruction correlation: {np.mean(correlations):.4f}")

    # Find most interesting features (moderate activation frequency)
    interesting_features = torch.where(
        (feature_activation_freq > 0.05) & (feature_activation_freq < 0.5)
    )[0]
    print(f"\nInteresting features (5-50% activation): {len(interesting_features)}")

    return {
        "model": model,
        "feature_frequencies": feature_activation_freq,
        "hidden_activations": hidden_acts,
        "reconstruction_errors": recon_error,
        "interesting_features": interesting_features,
    }


if __name__ == "__main__":
    results = analyze_sae_features(
        "sae_models/sae_layer_2048d.pt",
        "exp/sod/ape/activations/activations_layers_3.h5",
    )
