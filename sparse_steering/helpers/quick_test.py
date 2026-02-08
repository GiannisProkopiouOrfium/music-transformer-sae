"""Quick test script to validate SAE configuration before full training.

This script runs a complete pipeline test with minimal data to ensure:
1. SAE model works correctly
2. Data extraction works
3. Training loop executes
4. Validation metrics are computed

Run this BEFORE starting full 10K segment training to catch any issues early.
"""

import argparse
import logging
import pathlib
import sys

# Add sparse_steering to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from config_sas import print_config, get_quick_test_config


def test_sae_model():
    """Test SAE model functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: SAE Model Architecture")
    print("=" * 60)

    from sae_model import test_sae_model

    test_sae_model()


def test_data_extraction(gpu: int = None):
    """Test data extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: Random Activation Extraction")
    print("=" * 60)

    import subprocess

    # Build command
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).parent / "extract_sae_training_data.py"),
        "--quick_test",
    ]

    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("✗ Data extraction test FAILED")
        return False

    print("✓ Data extraction test PASSED")
    return True


def test_training(gpu: int = None, layer: int = 0):
    """Test SAE training on a single layer."""
    print("\n" + "=" * 60)
    print(f"TEST 3: SAE Training (Layer {layer} only)")
    print("=" * 60)

    import subprocess

    # Build command
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).parent / "train_sae.py"),
        "--quick_test",
        "--layers",
        str(layer),
    ]

    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("✗ Training test FAILED")
        return False

    print("✓ Training test PASSED")
    return True


def main():
    """Run quick test suite."""
    parser = argparse.ArgumentParser(
        description="Quick test SAE configuration before full training"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use for testing",
    )
    parser.add_argument(
        "--skip_model",
        action="store_true",
        help="Skip model architecture test",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip data extraction test",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training test",
    )
    parser.add_argument(
        "--test_layer",
        type=int,
        default=0,
        help="Which layer to test training on (default: 0)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("SPARSE AUTOENCODER QUICK TEST SUITE")
    print("=" * 60)
    print("This will test the complete SAE pipeline with minimal data")
    print("to catch configuration issues before full training.")
    print("=" * 60)

    # Print config
    config = get_quick_test_config()
    print_config(quick_test=True)

    # Run tests
    results = {}

    if not args.skip_model:
        try:
            test_sae_model()
            results["model"] = True
        except Exception as e:
            print(f"✗ Model test FAILED: {e}")
            results["model"] = False

    if not args.skip_extraction:
        try:
            results["extraction"] = test_data_extraction(gpu=args.gpu)
        except Exception as e:
            print(f"✗ Extraction test FAILED: {e}")
            results["extraction"] = False

    if not args.skip_training:
        try:
            results["training"] = test_training(gpu=args.gpu, layer=args.test_layer)
        except Exception as e:
            print(f"✗ Training test FAILED: {e}")
            results["training"] = False

    # Print summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.capitalize()}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✓ All tests PASSED! Ready for full training.")
        print("\nNext steps:")
        print("1. Extract full training data:")
        print(
            f"   python extract_sae_training_data.py --gpu {args.gpu if args.gpu is not None else 0}"
        )
        print("\n2. Train all SAEs:")
        print(f"   python train_sae.py --gpu {args.gpu if args.gpu is not None else 0}")
    else:
        print("✗ Some tests FAILED. Please fix issues before full training.")
        return 1

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
