#!/usr/bin/env python3
"""Pre-flight verification for Music Flamingo evaluation pipeline."""

import pathlib
import sys
import json


def verify_directory_structure():
    """Check if all expected directories exist."""
    print("=" * 70)
    print("STEP 1: Verifying Directory Structure")
    print("=" * 70)

    base_path = pathlib.Path("steering_interventions/flamingo_exp")

    expected_dirs = [
        "dual/unconditional/low_long",
        "dual/unconditional/low_short",
        "dual/unconditional/high_long",
        "dual/unconditional/high_short",
        "dual/unconditional/neutral",
        "dual/conditional/phase4_conditioned/listening_samples",
        "single/unconditional/average_pitch",
        "single/unconditional/average_duration",
        "single/conditional/flamingo_exp/high_pitch",
        "single/conditional/flamingo_exp/low_pitch",
        "single/conditional/flamingo_exp/high_duration",
        "single/conditional/flamingo_exp/low_duration",
    ]

    all_exist = True
    for dir_path in expected_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}")
        if not exists:
            all_exist = False

    return all_exist


def verify_sample_files():
    """Check if sample files exist in key directories."""
    print("\n" + "=" * 70)
    print("STEP 2: Verifying Sample Files")
    print("=" * 70)

    base_path = pathlib.Path("steering_interventions/flamingo_exp")

    checks = [
        ("dual/unconditional/neutral", "*.npy", "NPY files in neutral folder"),
        (
            "dual/unconditional/neutral",
            "sample_metrics_p*.json",
            "sample_metrics JSONs in neutral",
        ),
        (
            "single/unconditional/average_pitch",
            "sample_metrics.json",
            "sample_metrics.json in single/unconditional",
        ),
        (
            "dual/conditional/phase4_conditioned/listening_samples",
            "*.wav",
            "WAV files in listening_samples",
        ),
    ]

    all_pass = True
    for dir_path, pattern, description in checks:
        full_path = base_path / dir_path
        if not full_path.exists():
            print(f"❌ {description}: Directory not found")
            all_pass = False
            continue

        files = list(full_path.glob(pattern))
        if files:
            print(f"✅ {description}: {len(files)} files found")
        else:
            print(f"❌ {description}: No files found")
            all_pass = False

    return all_pass


def verify_baseline_files():
    """Check if baseline (p0.0_d0.0) files exist in neutral folder."""
    print("\n" + "=" * 70)
    print("STEP 3: Verifying Baseline Files")
    print("=" * 70)

    neutral_path = pathlib.Path(
        "steering_interventions/flamingo_exp/dual/unconditional/neutral"
    )

    if not neutral_path.exists():
        print("❌ Neutral folder not found")
        return False

    # Look for p0.0_d0.0 baseline files
    baseline_files = list(neutral_path.glob("*p0.0_d0.0*.npy"))

    if baseline_files:
        print(f"✅ Found {len(baseline_files)} baseline files (p0.0_d0.0)")
        for f in baseline_files[:3]:
            print(f"   - {f.name}")
        if len(baseline_files) > 3:
            print(f"   ... and {len(baseline_files) - 3} more")
        return True
    else:
        print("❌ No baseline files (p0.0_d0.0) found in neutral folder")
        return False


def verify_config_file():
    """Check if config.yaml exists."""
    print("\n" + "=" * 70)
    print("STEP 4: Verifying Configuration")
    print("=" * 70)

    config_path = pathlib.Path("steering_interventions/flamingo_eval/config.yaml")

    if not config_path.exists():
        print("❌ config.yaml not found")
        return False

    print(f"✅ config.yaml exists")

    # Try to load and validate basic structure
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["paths", "ground_truth", "prompts", "music_flamingo"]
        missing = [k for k in required_keys if k not in config]

        if missing:
            print(f"❌ Missing config keys: {missing}")
            return False

        print(f"✅ Config structure valid")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False


def verify_scripts():
    """Check if all pipeline scripts exist."""
    print("\n" + "=" * 70)
    print("STEP 5: Verifying Pipeline Scripts")
    print("=" * 70)

    scripts = [
        "steering_interventions/flamingo_eval/prepare_audios.py",
        "steering_interventions/flamingo_eval/evaluate_music_flamingo.py",
        "steering_interventions/flamingo_eval/analyze_results.py",
    ]

    all_exist = True
    for script in scripts:
        script_path = pathlib.Path(script)
        exists = script_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {script_path.name}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MUSIC FLAMINGO PIPELINE VERIFICATION" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    results.append(("Directory Structure", verify_directory_structure()))
    results.append(("Sample Files", verify_sample_files()))
    results.append(("Baseline Files", verify_baseline_files()))
    results.append(("Configuration", verify_config_file()))
    results.append(("Pipeline Scripts", verify_scripts()))

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 All checks passed! Ready to run the pipeline.\n")
        print("Next steps:")
        print(
            "  1. Install dependencies: pip install -r steering_interventions/flamingo_eval/requirements.txt"
        )
        print(
            "  2. Run Step 1: python steering_interventions/flamingo_eval/prepare_audios.py"
        )
        print(
            "  3. Run Step 2: python steering_interventions/flamingo_eval/evaluate_music_flamingo.py"
        )
        print(
            "  4. Run Step 3: python steering_interventions/flamingo_eval/analyze_results.py"
        )
        print()
        return 0
    else:
        print(
            "\n❌ Some checks failed. Please fix the issues above before proceeding.\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
