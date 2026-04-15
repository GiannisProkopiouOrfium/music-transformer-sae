#!/usr/bin/env python3
"""Run FMD with Ledoit-Wolf and OAS covariance estimators.

Re-uses your existing evaluate_fmd.py but with --estimator flag changed.
The CLaMP2 embeddings are cached so re-running with a different estimator
only recomputes the Gaussian fit + Fréchet distance (fast).

Usage (on EC2):
    python feedback_exp/run_fmd_estimators.py

    # Or with explicit workspace:
    python feedback_exp/run_fmd_estimators.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace \
        --gpu 0
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time


PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def run_fmd(workspace_dir: pathlib.Path, estimator: str, gpu: int) -> dict:
    """Run evaluate_fmd.py with a specific estimator and return parsed results."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "metrics_evaluation" / "evaluate_fmd.py"),
        "--workspace_dir",
        str(workspace_dir),
        "--estimator",
        estimator,
        "--gpu",
        str(gpu),
    ]

    print(f"\n{'─' * 70}")
    print(f"  Running FMD with estimator: {estimator}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'─' * 70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  ❌ FAILED (exit code {result.returncode})")
        print(result.stderr[-1000:])
        return {}

    print(f"  ✅ Completed in {elapsed:.1f}s")

    # The script overwrites fmd_results.json — read it back
    results_path = workspace_dir / "fmd_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def compare_estimators(mle_results: list, alt_results: list, alt_name: str):
    """Compare FMD scores between MLE and an alternative estimator."""
    # Build lookup by comparison name
    mle_lookup = {}
    for r in mle_results:
        if r.get("fmd") is not None:
            mle_lookup[r["comparison"]] = r["fmd"]

    alt_lookup = {}
    for r in alt_results:
        if r.get("fmd") is not None:
            alt_lookup[r["comparison"]] = r["fmd"]

    print(f"\n{'=' * 90}")
    print(f"  FMD COMPARISON: MLE vs {alt_name}")
    print(f"{'=' * 90}")
    print(f"  {'Comparison':<55s}  {'MLE':>8s}  {alt_name:>8s}  {'Δ%':>8s}")
    print(f"  {'-' * 55}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    common_keys = sorted(set(mle_lookup.keys()) & set(alt_lookup.keys()))
    for key in common_keys:
        mle_val = mle_lookup[key]
        alt_val = alt_lookup[key]
        pct_change = ((alt_val - mle_val) / mle_val * 100) if mle_val != 0 else 0
        print(f"  {key[:55]:<55s}  {mle_val:8.3f}  {alt_val:8.3f}  {pct_change:+7.1f}%")

    # Summarize: do relative rankings (SAS vs DiffMean) hold?
    print(f"\n  Rank preservation check (do SAS < DiffMean orderings hold?):")
    sas_keys = [k for k in common_keys if "SAS" in k or "sas" in k.lower()]
    dm_keys = [
        k
        for k in common_keys
        if "DiffMean" in k or "diffmean" in k.lower() or "gram_schmidt" in k.lower()
    ]

    if sas_keys and dm_keys:
        sas_fmd_mle = [mle_lookup[k] for k in sas_keys if k in mle_lookup]
        dm_fmd_mle = [mle_lookup[k] for k in dm_keys if k in mle_lookup]
        sas_fmd_alt = [alt_lookup[k] for k in sas_keys if k in alt_lookup]
        dm_fmd_alt = [alt_lookup[k] for k in dm_keys if k in alt_lookup]

        if sas_fmd_mle and dm_fmd_mle and sas_fmd_alt and dm_fmd_alt:
            avg_sas_mle = sum(sas_fmd_mle) / len(sas_fmd_mle)
            avg_dm_mle = sum(dm_fmd_mle) / len(dm_fmd_mle)
            avg_sas_alt = sum(sas_fmd_alt) / len(sas_fmd_alt)
            avg_dm_alt = sum(dm_fmd_alt) / len(dm_fmd_alt)

            gap_mle = (avg_dm_mle - avg_sas_mle) / avg_dm_mle * 100
            gap_alt = (avg_dm_alt - avg_sas_alt) / avg_dm_alt * 100

            print(
                f"    MLE:        SAS avg={avg_sas_mle:.3f}, DM avg={avg_dm_mle:.3f} → SAS {gap_mle:+.1f}% better"
            )
            print(
                f"    {alt_name:10s}: SAS avg={avg_sas_alt:.3f}, DM avg={avg_dm_alt:.3f} → SAS {gap_alt:+.1f}% better"
            )
            print(
                f"    → Rankings {'PRESERVED ✅' if (gap_mle > 0) == (gap_alt > 0) else 'CHANGED ⚠️'}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run FMD with alternative covariance estimators"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "fmd_workspace",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("feedback_exp/fmd_estimator_results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Back up existing MLE results
    mle_results_path = args.workspace_dir / "fmd_results.json"
    mle_results = []
    if mle_results_path.exists():
        try:
            with open(mle_results_path) as f:
                mle_results = json.load(f)
            # Back up
            backup_path = args.output_dir / "fmd_results_mle.json"
            with open(backup_path, "w") as f:
                json.dump(mle_results, f, indent=2)
            print(
                f"✅ Backed up MLE results ({len(mle_results)} comparisons) to {backup_path}"
            )
        except json.JSONDecodeError as e:
            print(f"⚠ Existing fmd_results.json is corrupted ({e})")
            print("  Will re-run MLE estimator from scratch...")
            mle_results = run_fmd(args.workspace_dir, "mle", args.gpu)
            if isinstance(mle_results, list) and mle_results:
                backup_path = args.output_dir / "fmd_results_mle.json"
                with open(backup_path, "w") as f:
                    json.dump(mle_results, f, indent=2)
                print(
                    f"✅ Fresh MLE results ({len(mle_results)} comparisons) saved to {backup_path}"
                )
    else:
        print("⚠ No existing MLE results found — will run MLE first")
        mle_results = run_fmd(args.workspace_dir, "mle", args.gpu)
        if isinstance(mle_results, list):
            with open(args.output_dir / "fmd_results_mle.json", "w") as f:
                json.dump(mle_results, f, indent=2)

    # Step 2: Run Ledoit-Wolf (note: typo in evaluate_fmd.py is 'leodit_wolf')
    lw_results = run_fmd(args.workspace_dir, "leodit_wolf", args.gpu)
    if isinstance(lw_results, list) and lw_results:
        with open(args.output_dir / "fmd_results_ledoit_wolf.json", "w") as f:
            json.dump(lw_results, f, indent=2)

    # Step 3: Run OAS
    oas_results = run_fmd(args.workspace_dir, "oas", args.gpu)
    if isinstance(oas_results, list) and oas_results:
        with open(args.output_dir / "fmd_results_oas.json", "w") as f:
            json.dump(oas_results, f, indent=2)

    # Step 4: Restore MLE results file (so the main workspace is unchanged)
    if mle_results:
        with open(mle_results_path, "w") as f:
            json.dump(mle_results if isinstance(mle_results, list) else [], f, indent=2)
        print(f"\n✅ Restored original MLE results to {mle_results_path}")

    # Step 5: Compare
    if isinstance(mle_results, list) and mle_results:
        if isinstance(lw_results, list) and lw_results:
            compare_estimators(mle_results, lw_results, "Ledoit-Wolf")
        if isinstance(oas_results, list) and oas_results:
            compare_estimators(mle_results, oas_results, "OAS")

    # Step 6: Save combined comparison
    combined = {
        "mle": mle_results if isinstance(mle_results, list) else [],
        "ledoit_wolf": lw_results if isinstance(lw_results, list) else [],
        "oas": oas_results if isinstance(oas_results, list) else [],
    }
    combined_path = args.output_dir / "fmd_estimator_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✅ Combined comparison saved to {combined_path}")


if __name__ == "__main__":
    main()
