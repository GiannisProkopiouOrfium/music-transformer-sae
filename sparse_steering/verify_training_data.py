"""Verify SAE training data quality before expensive training run."""

import argparse
import pathlib
import h5py
import numpy as np


def verify_training_data(data_path: pathlib.Path):
    """Verify extracted training data is valid.
    
    Checks:
    - File exists and is readable
    - All 12 layers present with correct shape (N, 512)
    - No NaN or Inf values
    - Reasonable value ranges
    - Consistent sample count across layers
    """
    print("=" * 60)
    print("Verifying SAE Training Data")
    print("=" * 60)
    print(f"File: {data_path}")
    print()
    
    # Check file exists
    if not data_path.exists():
        print(f"✗ ERROR: File not found: {data_path}")
        return False
    
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print()
    
    all_checks_passed = True
    
    try:
        with h5py.File(data_path, "r") as f:
            # Check metadata
            print("Metadata:")
            if "metadata" in f.attrs:
                for key, value in f.attrs.items():
                    if key == "metadata":
                        continue
                    print(f"  {key}: {value}")
            print()
            
            # Get number of layers
            layer_keys = [k for k in f.keys() if k.startswith("layer_")]
            num_layers = len(layer_keys)
            
            if num_layers != 12:
                print(f"✗ WARNING: Expected 12 layers, found {num_layers}")
                all_checks_passed = False
            else:
                print(f"✓ Found 12 layers")
            
            # Check each layer
            print()
            print("Layer-wise checks:")
            print("-" * 60)
            
            sample_counts = []
            
            for layer_idx in range(num_layers):
                layer_key = f"layer_{layer_idx}"
                
                if layer_key not in f:
                    print(f"✗ Layer {layer_idx}: MISSING")
                    all_checks_passed = False
                    continue
                
                data = f[layer_key][:]
                shape = data.shape
                sample_counts.append(shape[0])
                
                # Check shape
                if len(shape) != 2:
                    print(f"✗ Layer {layer_idx}: Invalid shape {shape} (expected 2D)")
                    all_checks_passed = False
                    continue
                
                if shape[1] != 512:
                    print(f"✗ Layer {layer_idx}: Invalid dim {shape[1]} (expected 512)")
                    all_checks_passed = False
                    continue
                
                # Check for NaN/Inf
                has_nan = np.isnan(data).any()
                has_inf = np.isinf(data).any()
                
                if has_nan or has_inf:
                    print(f"✗ Layer {layer_idx}: Contains NaN={has_nan}, Inf={has_inf}")
                    all_checks_passed = False
                    continue
                
                # Check value ranges
                mean = np.mean(data)
                std = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                
                # Check for suspicious values (transformer activations should be reasonable)
                if abs(mean) > 100 or std > 100:
                    print(f"✗ Layer {layer_idx}: Suspicious stats (mean={mean:.2f}, std={std:.2f})")
                    all_checks_passed = False
                    continue
                
                print(
                    f"✓ Layer {layer_idx}: shape={shape}, "
                    f"mean={mean:.4f}, std={std:.4f}, "
                    f"range=[{min_val:.4f}, {max_val:.4f}]"
                )
            
            # Check sample count consistency
            print()
            print("-" * 60)
            if len(set(sample_counts)) != 1:
                print(f"✗ Inconsistent sample counts: {sample_counts}")
                all_checks_passed = False
            else:
                print(f"✓ Consistent sample count: {sample_counts[0]} across all layers")
            
            # Additional checks
            print()
            print("Additional checks:")
            print("-" * 60)
            
            # Check if we have enough samples
            n_samples = sample_counts[0] if sample_counts else 0
            if n_samples < 1000:
                print(f"✗ Too few samples: {n_samples} < 1000")
                all_checks_passed = False
            else:
                print(f"✓ Sufficient samples: {n_samples}")
            
            # Check train/val split will work (90/10 split)
            train_size = int(n_samples * 0.9)
            val_size = n_samples - train_size
            
            if val_size < 10:
                print(f"✗ Validation set too small: {val_size} < 10")
                all_checks_passed = False
            else:
                print(f"✓ Train/val split: {train_size}/{val_size}")
            
            # Estimate training time
            batch_size = 64
            epochs = 50
            n_batches_per_epoch = (train_size + batch_size - 1) // batch_size
            total_batches = n_batches_per_epoch * epochs * 12  # 12 layers
            
            # Rough estimate: ~100 batches/sec during training
            estimated_seconds = total_batches / 100
            estimated_hours = estimated_seconds / 3600
            
            print(f"✓ Estimated training time: ~{estimated_hours:.1f} hours (very rough)")
            
    except Exception as e:
        print(f"✗ ERROR reading file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 60)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - Ready for training!")
        print()
        print("Start training with:")
        print("  python sparse_steering/train_sae.py --gpu 0")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before training")
    print("=" * 60)
    
    return all_checks_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify SAE training data quality"
    )
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=None,
        help="Path to training data h5 file (default: auto-detect)",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Verify quick test data instead of full data",
    )
    
    args = parser.parse_args()
    
    # Auto-detect path if not provided
    if args.data_path is None:
        base_dir = pathlib.Path(__file__).parent.parent / "exp/sod/sparse_steering/sae_training_data"
        if args.quick_test:
            args.data_path = base_dir / "random_activations_quick_test.h5"
        else:
            args.data_path = base_dir / "random_activations_full.h5"
    
    verify_training_data(args.data_path)


if __name__ == "__main__":
    main()
