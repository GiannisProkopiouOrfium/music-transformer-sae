"""Verify trained SAE models are ready for Stage 2."""

import argparse
import pathlib
import torch
import numpy as np
import h5py
from sae_model import SparseAutoencoder
from config_sas import (
    SAE_CHECKPOINT_DIR,
    SAE_TRAINING_DATA_DIR,
    HIDDEN_DIM,
    SPARSE_DIM,
)


def load_sae(checkpoint_path: pathlib.Path, k: int) -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    sae = SparseAutoencoder(
        input_dim=HIDDEN_DIM,
        sparse_dim=SPARSE_DIM,
        k=k,
        tied_weights=True,
        normalize_input=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Backward compatibility: add _normalization_fitted if missing
    if "_normalization_fitted" not in state_dict:
        # Check if normalization was fitted by checking if input_mean is non-zero
        if "input_mean" in state_dict:
            has_normalization = state_dict["input_mean"].abs().sum() > 0
            state_dict["_normalization_fitted"] = torch.tensor(1 if has_normalization else 0)
        else:
            state_dict["_normalization_fitted"] = torch.tensor(0)
    
    sae.load_state_dict(state_dict)
    sae.eval()
    
    return sae, checkpoint


def get_adaptive_k(layer_idx: int) -> int:
    """Get adaptive K for layer."""
    if layer_idx == 0:
        return 32
    elif layer_idx < 4:
        return 64
    elif layer_idx < 8:
        return 96
    else:
        return 128


def verify_sae(layer_idx: int, checkpoint_dir: pathlib.Path, data_file: pathlib.Path):
    """Verify a single SAE model."""
    print(f"\nLayer {layer_idx}:")
    print("-" * 40)

    # Get adaptive K
    k = get_adaptive_k(layer_idx)

    # Load checkpoint
    checkpoint_path = checkpoint_dir / f"sae_layer_{layer_idx}_best.pt"
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    sae, checkpoint = load_sae(checkpoint_path, k)

    # Check normalization parameters
    if not sae.normalization_fitted:
        print("✗ Normalization not fitted!")
        return False

    print(f"✓ Loaded checkpoint (epoch {checkpoint['epoch']})")
    print(f"✓ K={k}, sparse_dim={SPARSE_DIM}")
    print(
        f"✓ Normalization: mean={sae.input_mean.mean():.4f}, std={sae.input_std.mean():.4f}"
    )

    # Load test data
    with h5py.File(data_file, "r") as f:
        layer_key = f"layer_{layer_idx}"
        if layer_key not in f:
            print(f"✗ Layer data not found: {layer_key}")
            return False

        # Take a small sample
        data = f[layer_key][:100]
        data_tensor = torch.from_numpy(data).float()

    # Test forward pass
    with torch.no_grad():
        reconstruction, sparse_features = sae(data_tensor)

    # Check sparsity
    l0 = (sparse_features != 0).float().sum(dim=-1).mean().item()
    mse = ((reconstruction - data_tensor) ** 2).mean().item()

    if abs(l0 - k) > 1:
        print(f"✗ Sparsity mismatch: L0={l0:.1f} (expected {k})")
        return False

    print(f"✓ Forward pass: MSE={mse:.6f}, L0={l0:.1f}")

    # Check feature statistics
    active_features = (sparse_features != 0).float()
    feature_freq = active_features.mean(dim=0)
    n_dead_features = (feature_freq == 0).sum().item()

    print(
        f"✓ Dead features: {n_dead_features}/{SPARSE_DIM} ({n_dead_features/SPARSE_DIM*100:.1f}%)"
    )

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify trained SAE models")
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=SAE_CHECKPOINT_DIR,
        help="Directory with SAE checkpoints",
    )
    parser.add_argument(
        "--data_file",
        type=pathlib.Path,
        default=SAE_TRAINING_DATA_DIR / "random_activations_full.h5",
        help="Training data file for testing",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to verify (default: all)",
    )

    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(f"✗ Checkpoint directory not found: {args.checkpoint_dir}")
        return

    if not args.data_file.exists():
        print(f"✗ Data file not found: {args.data_file}")
        return

    print("=" * 60)
    print("Verifying Trained SAE Models")
    print("=" * 60)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Data file: {args.data_file}")

    # Determine layers to verify
    if args.layers is not None:
        layers = args.layers
    else:
        layers = list(range(12))

    # Verify each layer
    all_passed = True
    for layer_idx in layers:
        passed = verify_sae(layer_idx, args.checkpoint_dir, args.data_file)
        if not passed:
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ ALL SAE MODELS VERIFIED - Ready for Stage 2!")
        print()
        print("Next steps:")
        print("1. Stage 2: Encode existing concept activations")
        print("   - Load pitch/duration high/low activations from DiffMean")
        print("   - Pass through trained SAE encoders")
        print("   - Compute SAS vectors with τ filtering")
        print()
        print("2. Stage 3: Implement inference-time steering")
        print("   - Single steering (pitch, duration)")
        print("   - Dual steering (pitch + duration)")
        print("   - Conditioned steering (fight scenarios)")
    else:
        print("✗ SOME VERIFICATIONS FAILED - Check errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
