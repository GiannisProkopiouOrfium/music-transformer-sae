#!/usr/bin/env python3
"""Setup verification script for steering interventions.

This script checks that all dependencies and data are in place
before running the pipeline.
"""

import pathlib
import sys

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_mark(passed):
    """Return colored check mark or X."""
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def main():
    """Run all checks."""
    print("=" * 60)
    print("Steering Interventions - Setup Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: Python version
    print("Checking Python version...")
    if sys.version_info >= (3, 8):
        print(
            f"  {check_mark(True)} Python {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        print(
            f"  {check_mark(False)} Python {sys.version_info.major}.{sys.version_info.minor} (need >= 3.8)"
        )
        all_passed = False

    # Check 2: Required packages
    print("\nChecking required packages...")
    required_packages = [
        "torch",
        "numpy",
        "h5py",
        "matplotlib",
        "scipy",
        "tqdm",
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"  {check_mark(True)} {package}")
        except ImportError:
            print(
                f"  {check_mark(False)} {package} (install with: pip install {package})"
            )
            all_passed = False

    # Check 3: MMT modules
    print("\nChecking MMT modules...")
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

    mmt_modules = [
        "music_x_transformers",
        "representation",
        "dataset",
        "utils",
    ]

    for module in mmt_modules:
        try:
            __import__(module)
            print(f"  {check_mark(True)} {module}")
        except ImportError as e:
            print(f"  {check_mark(False)} {module} ({e})")
            all_passed = False

    # Check 4: Data directories
    print("\nChecking data directories...")
    project_root = pathlib.Path(__file__).parent.parent

    data_paths = {
        "JSON data": project_root / "data" / "sod" / "processed" / "json",
        "Notes data": project_root / "data" / "sod" / "processed" / "notes",
        "Encoding": project_root
        / "data"
        / "sod"
        / "processed"
        / "notes"
        / "encoding.json",
        "Train names": project_root / "data" / "sod" / "processed" / "train-names.txt",
    }

    for name, path in data_paths.items():
        if path.exists():
            if path.is_file():
                print(f"  {check_mark(True)} {name}: {path}")
            else:
                n_files = len(list(path.rglob("*")))
                print(f"  {check_mark(True)} {name}: {path} ({n_files} files)")
        else:
            print(f"  {check_mark(False)} {name}: {path} (not found)")
            all_passed = False

    # Check 5: Model checkpoint exp/sod/ape/checkpoints/best_model.pt
    print("\nChecking model checkpoint...")
    model_dir = project_root / "exp" / "sod" / "ape"

    if model_dir.exists():
        print(f"  {check_mark(True)} Model directory: {model_dir}")

        checkpoint = model_dir / "checkpoints" / "best_model.pt"
        train_args = model_dir / "train-args.json"

        if checkpoint.exists():
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            print(f"  {check_mark(True)} Checkpoint: {checkpoint} ({size_mb:.1f} MB)")
        else:
            print(f"  {check_mark(False)} Checkpoint: {checkpoint} (not found)")
            print(
                f"      {YELLOW}Note: You need a trained model to run steering interventions{RESET}"
            )
            all_passed = False

        if train_args.exists():
            print(f"  {check_mark(True)} Training args: {train_args}")
        else:
            print(f"  {check_mark(False)} Training args: {train_args} (not found)")
            all_passed = False
    else:
        print(f"  {check_mark(False)} Model directory: {model_dir} (not found)")
        print(f"      {YELLOW}Note: You need to train a model first{RESET}")
        all_passed = False

    # Check 6: Output directories
    print("\nChecking/creating output directories...")
    output_dir = pathlib.Path(__file__).parent / "outputs"

    subdirs = [
        "datasets",
        "activations",
        "steering_vectors",
        "generated_samples",
        "evaluation",
    ]

    for subdir in subdirs:
        path = output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {check_mark(True)} {path}")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(
            f"{GREEN}✓ All checks passed! Ready to run steering interventions.{RESET}"
        )
        print("\nNext steps:")
        print("  1. Run the pipeline:")
        print("     python run_pipeline.py --concept velocity --n_beats 16")
        print()
        print("  2. Or try the simple example:")
        print("     python example.py")
    else:
        print(f"{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
        print("\nCommon fixes:")
        print(
            "  - Install missing packages: pip install torch numpy h5py matplotlib scipy tqdm"
        )
        print("  - Make sure you have trained a model (see main README.md)")
        print("  - Check that data is processed and available")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
