"""Test that compute_sas_vectors.py follows Algorithm 1 exactly.

This test verifies the SAS vector computation against a manual implementation
of Algorithm 1 from the paper.
"""

import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from compute_sas_vectors import compute_sas_vector


def compute_sas_vector_manual(sparse_high, sparse_low, tau):
    """Manual implementation of Algorithm 1 for verification."""
    n_high, d = sparse_high.shape
    n_low, _ = sparse_low.shape

    # Step 1: Define R+[c] and R-[c] (rows where each column is non-zero)
    # We'll compute this implicitly through frequency

    # Step 2: Define C+_τ and C-_τ (columns with freq >= τ)
    is_active_high = sparse_high != 0
    is_active_low = sparse_low != 0

    freq_high = is_active_high.mean(axis=0)  # |R+[c]|/|D|
    freq_low = is_active_low.mean(axis=0)  # |R-[c]|/|D|

    C_plus_tau = freq_high >= tau
    C_minus_tau = freq_low >= tau

    # Step 3: Compute v+ and v-
    v_plus = np.zeros(d)
    v_minus = np.zeros(d)

    # For v+[c]: average over non-zero values if c in C+_τ
    for c in range(d):
        if C_plus_tau[c]:
            active_rows = sparse_high[:, c] != 0
            if active_rows.any():
                v_plus[c] = sparse_high[active_rows, c].mean()

        if C_minus_tau[c]:
            active_rows = sparse_low[:, c] != 0
            if active_rows.any():
                v_minus[c] = sparse_low[active_rows, c].mean()

    # Step 4: Zero out common columns C = {c | v+[c] ≠ 0 ∧ v-[c] ≠ 0}
    common = (v_plus != 0) & (v_minus != 0)
    v_plus[common] = 0
    v_minus[common] = 0

    # Step 5: Final vector
    v_sas = v_plus - v_minus

    return v_sas, v_plus, v_minus, freq_high, freq_low


def test_sas_algorithm():
    """Test that our implementation matches Algorithm 1."""
    np.random.seed(42)

    # Create synthetic sparse data
    n_high, n_low = 100, 80
    d = 1000
    k = 50  # Active features per sample

    # Generate sparse activations (TopK-like)
    sparse_high = np.zeros((n_high, d))
    sparse_low = np.zeros((n_low, d))

    for i in range(n_high):
        active_idx = np.random.choice(d, k, replace=False)
        sparse_high[i, active_idx] = np.random.randn(k) * 2 + 5  # Positive values

    for i in range(n_low):
        active_idx = np.random.choice(d, k, replace=False)
        sparse_low[i, active_idx] = np.random.randn(k) * 2 + 3  # Different distribution

    # Test with different tau values
    for tau in [0.1, 0.3, 0.5, 0.7]:
        print(f"\nTesting τ = {tau}")

        # Our implementation
        v_sas_ours, stats_ours = compute_sas_vector(sparse_high, sparse_low, tau)

        # Manual implementation
        v_sas_manual, v_plus_manual, v_minus_manual, freq_high, freq_low = (
            compute_sas_vector_manual(sparse_high, sparse_low, tau)
        )

        # Compare
        diff = np.abs(v_sas_ours - v_sas_manual).max()
        print(f"  Max difference: {diff:.10f}")

        if diff > 1e-10:
            print(f"  ✗ FAIL: Vectors don't match!")
            print(f"  Our L0: {(v_sas_ours != 0).sum()}")
            print(f"  Manual L0: {(v_sas_manual != 0).sum()}")

            # Find where they differ
            differ_idx = np.where(np.abs(v_sas_ours - v_sas_manual) > 1e-10)[0]
            if len(differ_idx) > 0:
                idx = differ_idx[0]
                print(f"  First difference at feature {idx}:")
                print(f"    Ours: {v_sas_ours[idx]:.6f}")
                print(f"    Manual: {v_sas_manual[idx]:.6f}")
                print(
                    f"    freq_high: {freq_high[idx]:.3f}, freq_low: {freq_low[idx]:.3f}"
                )
            return False
        else:
            print(f"  ✓ PASS: Vectors match!")
            print(f"  L0: {(v_sas_ours != 0).sum()}")
            print(f"  ||v_SAS||: {np.linalg.norm(v_sas_ours):.4f}")

    return True


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    # Case 1: No overlap in active features
    print("\nCase 1: Disjoint features")
    sparse_high = np.zeros((10, 100))
    sparse_low = np.zeros((10, 100))

    sparse_high[:, :20] = np.random.randn(10, 20) + 5
    sparse_low[:, 20:40] = np.random.randn(10, 20) + 5

    v_sas, stats = compute_sas_vector(sparse_high, sparse_low, tau=0.5)
    print(f"  Shared features removed: {stats['n_features_shared_removed']}")
    print(
        f"  Should be 0: {'✓ PASS' if stats['n_features_shared_removed'] == 0 else '✗ FAIL'}"
    )

    # Case 2: Complete overlap
    print("\nCase 2: Complete overlap")
    sparse_high = np.random.randn(10, 100) + 5
    sparse_low = np.random.randn(10, 100) + 3

    v_sas, stats = compute_sas_vector(sparse_high, sparse_low, tau=0.5)
    print(f"  Shared features removed: {stats['n_features_shared_removed']}")
    print(f"  Final L0: {stats['l0_vector']}")
    print(
        f"  Should be 0 (all shared): {'✓ PASS' if stats['l0_vector'] == 0 else '✗ FAIL'}"
    )

    # Case 3: Partial overlap
    print("\nCase 3: Partial overlap")
    sparse_high = np.zeros((10, 100))
    sparse_low = np.zeros((10, 100))

    sparse_high[:, :30] = np.random.randn(10, 30) + 5  # Features 0-29
    sparse_low[:, 20:50] = np.random.randn(10, 30) + 3  # Features 20-49

    v_sas, stats = compute_sas_vector(sparse_high, sparse_low, tau=0.5)
    print(f"  High-only: 0-19 (20 features)")
    print(f"  Shared: 20-29 (10 features, should be removed)")
    print(f"  Low-only: 30-49 (20 features)")
    print(f"  Shared removed: {stats['n_features_shared_removed']}")
    print(f"  Final L0: {stats['l0_vector']}")
    print(f"  Expected L0: 40, Got: {stats['l0_vector']}")
    print(f"  {'✓ PASS' if stats['l0_vector'] == 40 else '✗ FAIL'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SAS Algorithm 1 Implementation")
    print("=" * 60)

    success = test_sas_algorithm()
    test_edge_cases()

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)
