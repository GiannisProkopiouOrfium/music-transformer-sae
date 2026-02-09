"""Debug script to check if dataset labels are correct."""

import torch
import pathlib
import numpy as np

print("=" * 70)
print("CHECKING SPARSE ACTIVATION FILES")
print("=" * 70)

sas_dir = pathlib.Path("exp/sod/sparse_steering/sas_vectors")

# Load sparse activations
high_path = sas_dir / "average_pitch_high_sparse.pt"
low_path = sas_dir / "average_pitch_low_sparse.pt"

if high_path.exists():
    print(f"\n✓ Found: {high_path}")
    high_data = torch.load(high_path, map_location="cpu")

    # Check layer 0 as example
    if 0 in high_data:
        high_layer0 = high_data[0]
        print(f"  Layer 0 shape: {high_layer0.shape}")
        print(f"  Layer 0 mean activation: {high_layer0[high_layer0 != 0].mean():.4f}")
        print(
            f"  Layer 0 L0 (avg non-zeros): {(high_layer0 != 0).sum(axis=1).mean():.1f}"
        )
else:
    print(f"\n✗ NOT FOUND: {high_path}")

if low_path.exists():
    print(f"\n✓ Found: {low_path}")
    low_data = torch.load(low_path, map_location="cpu")

    # Check layer 0 as example
    if 0 in low_data:
        low_layer0 = low_data[0]
        print(f"  Layer 0 shape: {low_layer0.shape}")
        print(f"  Layer 0 mean activation: {low_layer0[low_layer0 != 0].mean():.4f}")
        print(
            f"  Layer 0 L0 (avg non-zeros): {(low_layer0 != 0).sum(axis=1).mean():.1f}"
        )
else:
    print(f"\n✗ NOT FOUND: {low_path}")

print("\n" + "=" * 70)
print("CHECKING SAS VECTOR COMPUTATION")
print("=" * 70)

if high_path.exists() and low_path.exists():
    # Recompute v_SAS for layer 0 manually to verify
    tau = 0.08

    sparse_high = high_data[0]
    sparse_low = low_data[0]

    print(f"\nLayer 0 manual verification (τ={tau}):")
    print(f"  High samples: {len(sparse_high)}")
    print(f"  Low samples: {len(sparse_low)}")

    # Compute frequency
    freq_high = (sparse_high != 0).mean(axis=0)
    freq_low = (sparse_low != 0).mean(axis=0)

    # Compute v+ (from high)
    v_pos = np.zeros(sparse_high.shape[1])
    valid_features_high = freq_high >= tau
    for c in range(len(v_pos)):
        if valid_features_high[c]:
            active_rows = sparse_high[:, c] != 0
            if active_rows.any():
                v_pos[c] = sparse_high[active_rows, c].mean()

    # Compute v- (from low)
    v_neg = np.zeros(sparse_low.shape[1])
    valid_features_low = freq_low >= tau
    for c in range(len(v_neg)):
        if valid_features_low[c]:
            active_rows = sparse_low[:, c] != 0
            if active_rows.any():
                v_neg[c] = sparse_low[active_rows, c].mean()

    # Remove shared
    common = (v_pos != 0) & (v_neg != 0)
    v_pos[common] = 0
    v_neg[common] = 0

    # Final vector
    v_sas = v_pos - v_neg

    print(
        f"  v_pos: L0={(v_pos != 0).sum()}, mean={v_pos[v_pos != 0].mean() if (v_pos != 0).any() else 0:.4f}"
    )
    print(
        f"  v_neg: L0={(v_neg != 0).sum()}, mean={v_neg[v_neg != 0].mean() if (v_neg != 0).any() else 0:.4f}"
    )
    print(f"  v_SAS: L0={(v_sas != 0).sum()}, mean={v_sas.mean():+.6f}")
    print(f"    positive features: {(v_sas > 0).sum()}")
    print(f"    negative features: {(v_sas < 0).sum()}")

    # Load saved SAS vector for comparison
    sas_path = sas_dir / "average_pitch_sas_vectors.pt"
    if sas_path.exists():
        saved_sas = torch.load(sas_path, map_location="cpu")
        if 0 in saved_sas:
            saved_vec = saved_sas[0]
            if isinstance(saved_vec, torch.Tensor):
                saved_vec = saved_vec.numpy()

            print(
                f"\n  Saved v_SAS: L0={(saved_vec != 0).sum()}, mean={saved_vec.mean():+.6f}"
            )
            print(f"    Match: {np.allclose(v_sas, saved_vec, atol=1e-6)}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

vec_mean = v_sas.mean() if "v_sas" in locals() else None
if vec_mean is not None:
    if vec_mean > 0:
        print("\nThe SAS vector has POSITIVE mean.")
        print("This means: λ > 0 should INCREASE pitch (add positive features)")
        print("            λ < 0 should DECREASE pitch (subtract positive features)")
    else:
        print("\nThe SAS vector has NEGATIVE mean.")
        print("This means: λ > 0 should DECREASE pitch (add negative features)")
        print("            λ < 0 should INCREASE pitch (subtract negative features)")

    print("\nBut you observed:")
    print("  λ=-2.0 → pitch=68.32 (HIGHEST)")
    print("  λ=+2.0 → pitch=64.65 (LOWEST)")

    if vec_mean > 0:
        print("\n⚠️  PROBLEM: Vector mean is positive but λ<0 gives HIGHER pitch!")
        print("   → The labels are SWAPPED in your dataset files!")
        print("   → 'high_average_pitch' songs actually have LOW pitch")
        print("   → 'low_average_pitch' songs actually have HIGH pitch")
        print("\n✓ SOLUTION: Swap the dataset labels or flip v_SAS sign")
