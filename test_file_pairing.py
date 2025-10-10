#!/usr/bin/env python3
"""
Quick test to verify baseline-intervention file pairing logic works correctly.
Tests the updated find_intervention_files() method.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deterministic_analysis.batch_deterministic_analyzer import (
    BatchDeterministicAnalyzer,
)


def test_file_pairing(interventions_dir: str):
    """Test file pairing on actual directory structure."""

    print("=" * 80)
    print("TESTING FILE PAIRING LOGIC")
    print("=" * 80)
    
    # Initialize analyzer (uses default features_config internally)
    analyzer = BatchDeterministicAnalyzer(
        output_dir="test_results"
    )    # Find intervention files
    print(f"\nScanning directory: {interventions_dir}")
    files_by_feature = analyzer.find_intervention_files(interventions_dir)

    print(f"\nFound {len(files_by_feature)} features with interventions")
    print("=" * 80)

    # Display file pairing for each feature
    for feature_id, file_groups in sorted(files_by_feature.items()):
        print(f"\n📁 FEATURE: {feature_id}")
        print("-" * 80)

        # Show baseline files
        baseline_files = file_groups.get("baseline", [])
        print(f"\n  🔵 BASELINE ({len(baseline_files)} files):")
        for bf in baseline_files:
            print(f"    - {Path(bf).name}")

        # Show intervention files by type
        interventions = file_groups.get("interventions", {})

        if interventions:
            print(f"\n  🔴 INTERVENTIONS:")
            for intervention_type, intervention_files in sorted(interventions.items()):
                print(
                    f"\n    Type: {intervention_type} ({len(intervention_files)} files)"
                )
                for if_ in intervention_files:
                    print(f"      - {Path(if_).name}")
        else:
            print("\n  ⚠️  No interventions found")

        # Validation checks
        print(f"\n  ✓ VALIDATION:")
        if not baseline_files:
            print(f"    ❌ ERROR: No baseline files found for feature {feature_id}")
        else:
            print(f"    ✅ Has {len(baseline_files)} baseline file(s)")

        if not interventions:
            print(f"    ❌ ERROR: No intervention files found for feature {feature_id}")
        else:
            intervention_count = sum(len(files) for files in interventions.values())
            print(
                f"    ✅ Has {intervention_count} intervention file(s) across {len(interventions)} types"
            )

        # Check if baseline is feature-specific (not shared)
        if baseline_files:
            baseline_name = Path(baseline_files[0]).name
            if feature_id in baseline_name:
                print(f"    ✅ Baseline is feature-specific (contains '{feature_id}')")
            else:
                print(
                    f"    ⚠️  WARNING: Baseline doesn't contain feature ID '{feature_id}'"
                )

        print("-" * 80)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return files_by_feature


def main():
    """Run the test."""

    # Default path - adjust if needed
    default_dir = "batch_extractions_interventions/interventions"

    if len(sys.argv) > 1:
        interventions_dir = sys.argv[1]
    else:
        interventions_dir = default_dir

    if not os.path.exists(interventions_dir):
        print(f"❌ ERROR: Directory not found: {interventions_dir}")
        print(f"\nUsage: python test_file_pairing.py [interventions_directory]")
        print(f"Example: python test_file_pairing.py {default_dir}")
        sys.exit(1)

    files_by_feature = test_file_pairing(interventions_dir)

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  Total features detected: {len(files_by_feature)}")

    features_with_baseline = sum(
        1 for fg in files_by_feature.values() if fg.get("baseline")
    )
    features_with_interventions = sum(
        1 for fg in files_by_feature.values() if fg.get("interventions")
    )

    print(f"  Features with baseline: {features_with_baseline}")
    print(f"  Features with interventions: {features_with_interventions}")
    print(
        f"  Features ready for analysis: {min(features_with_baseline, features_with_interventions)}"
    )


if __name__ == "__main__":
    main()
