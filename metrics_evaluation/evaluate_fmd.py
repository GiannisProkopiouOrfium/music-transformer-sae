#!/usr/bin/env python3
"""Compute Frechet Music Distance (FMD) for all experiment groups.

Reads the manifest.json produced by prepare_fmd_midis.py and computes
FMD scores for every relevant (reference, test) pair.

Comparisons performed:
  1. FMD(SOD, baseline_conditioned)        — model quality (conditioned)
  2. FMD(SOD, baseline_unconditioned)       — model quality (unconditioned)
  3. FMD(SOD, strategy_X_conditioned)       — steered absolute quality
  4. FMD(SOD, strategy_X_unconditioned)     — steered absolute quality
  5. FMD(baseline, strategy_X_conditioned)  — steering perturbation cost
  6. FMD(baseline, strategy_X_unconditioned)— steering perturbation cost
  7. Per-scenario FMD(SOD, scenario_strategy) — scenario-level quality

Usage:
    python metrics_evaluation/evaluate_fmd.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace \
        --gpu 0
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_midis(d: pathlib.Path) -> int:
    """Count .mid files in directory."""
    if not d.exists():
        return 0
    return len(list(d.glob("*.mid")))


def compute_fmd(ref_path: str, test_path: str, metric) -> dict:
    """Compute FMD between two directories.

    Returns:
        {"fmd": float, "n_ref": int, "n_test": int} or {"fmd": None, "error": str}
    """
    ref_dir = pathlib.Path(ref_path)
    test_dir = pathlib.Path(test_path)

    n_ref = count_midis(ref_dir)
    n_test = count_midis(test_dir)

    if n_ref < 2 or n_test < 2:
        return {
            "fmd": None,
            "n_ref": n_ref,
            "n_test": n_test,
            "error": f"Too few MIDIs (ref={n_ref}, test={n_test})",
        }

    try:
        t0 = time.time()
        score = metric.score(
            reference_path=str(ref_dir),
            test_path=str(test_dir),
        )
        elapsed = time.time() - t0
        return {
            "fmd": float(score),
            "n_ref": n_ref,
            "n_test": n_test,
            "time_sec": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "fmd": None,
            "n_ref": n_ref,
            "n_test": n_test,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FMD across all experiment groups"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        required=True,
        help="FMD workspace dir (created by prepare_fmd_midis.py)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for CLaMP2")
    parser.add_argument(
        "--model",
        choices=["clamp2", "clamp"],
        default="clamp2",
        help="Embedding model for FMD",
    )
    parser.add_argument(
        "--estimator",
        choices=["mle", "bootstrap", "oas", "shrinkage", "leodit_wolf"],
        default="mle",
        help="Gaussian estimator",
    )
    parser.add_argument("--clear_cache", action="store_true")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load manifest
    manifest_path = args.workspace_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"manifest.json not found in {args.workspace_dir}")
        logger.error("Run prepare_fmd_midis.py first!")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest with {len(manifest)} directories")
    for k, v in manifest.items():
        logger.info(f"  {k}: {v['n_midis']} MIDIs")

    # Initialize FMD
    from frechet_music_distance import FrechetMusicDistance

    logger.info(f"Initializing FMD (model={args.model}, estimator={args.estimator})")
    metric = FrechetMusicDistance(
        feature_extractor=args.model,
        gaussian_estimator=args.estimator,
        verbose=True,
    )

    if args.clear_cache:
        from frechet_music_distance.utils import clear_cache

        clear_cache()
        logger.info("Cache cleared")

    # ── Define all comparisons ──
    comparisons = []

    ref_sod = manifest.get("reference_sod", {}).get("path")
    ref_cond = manifest.get("baseline_conditioned", {}).get("path")
    ref_uncond = manifest.get("baseline_unconditioned", {}).get("path")

    # Skip references with too few MIDIs
    min_midis = 2
    if (
        ref_cond
        and manifest.get("baseline_conditioned", {}).get("n_midis", 0) < min_midis
    ):
        logger.warning(
            f"Skipping conditioned baseline — only {manifest['baseline_conditioned']['n_midis']} MIDIs"
        )
        ref_cond = None
    if (
        ref_uncond
        and manifest.get("baseline_unconditioned", {}).get("n_midis", 0) < min_midis
    ):
        logger.warning(
            f"Skipping unconditioned baseline — only {manifest['baseline_unconditioned']['n_midis']} MIDIs"
        )
        ref_uncond = None

    strategies = ["expanded_k_2x", "gram_schmidt_ek2"]

    # 1. Baseline quality: FMD(SOD, baselines)
    if ref_sod and ref_cond:
        comparisons.append(("SOD vs Baseline (conditioned)", ref_sod, ref_cond))
    if ref_sod and ref_uncond:
        comparisons.append(("SOD vs Baseline (unconditioned)", ref_sod, ref_uncond))

    # 2. Steered absolute quality: FMD(SOD, steered)
    for strat in strategies:
        key_c = f"conditioned/{strat}"
        key_u = f"unconditioned/{strat}"
        if ref_sod and key_c in manifest:
            comparisons.append(
                (f"SOD vs {strat} (conditioned)", ref_sod, manifest[key_c]["path"])
            )
        if ref_sod and key_u in manifest:
            comparisons.append(
                (f"SOD vs {strat} (unconditioned)", ref_sod, manifest[key_u]["path"])
            )

    # 3. Steering perturbation cost: FMD(baseline, steered)
    for strat in strategies:
        key_c = f"conditioned/{strat}"
        key_u = f"unconditioned/{strat}"
        if ref_cond and key_c in manifest:
            comparisons.append(
                (
                    f"Baseline vs {strat} (conditioned)",
                    ref_cond,
                    manifest[key_c]["path"],
                )
            )
        if ref_uncond and key_u in manifest:
            comparisons.append(
                (
                    f"Baseline vs {strat} (unconditioned)",
                    ref_uncond,
                    manifest[key_u]["path"],
                )
            )

    # 4. Per-scenario: FMD(SOD, scenario_strategy)
    for key, info in manifest.items():
        if key.startswith("conditioned/per_scenario/"):
            label = key.replace("conditioned/per_scenario/", "")
            if ref_sod:
                comparisons.append((f"SOD vs {label}", ref_sod, info["path"]))

    # 5. Per-lambda: FMD(SOD, lambda_pair) — for heatmap
    per_lambda_comparisons = []
    for key, info in sorted(manifest.items()):
        if "/per_lambda/" in key:
            n = info.get("n_midis", 0)
            if n < 2:
                continue
            # key like "conditioned/per_lambda/expanded_k_2x__lp+0.50_ld+0.75"
            #       or "unconditioned/per_lambda/expanded_k_2x__p+0.50_d+1.00"
            short = key.split("/per_lambda/")[1]
            mode = "cond" if key.startswith("conditioned") else "uncond"
            label = f"[{mode}] {short}"
            if ref_sod:
                per_lambda_comparisons.append((label, ref_sod, info["path"]))

    # ── Run main comparisons ──
    logger.info(f"\nRunning {len(comparisons)} main FMD comparisons...\n")
    results = []

    for label, ref, test in comparisons:
        logger.info(f"Computing: {label}")
        result = compute_fmd(ref, test, metric)
        result["comparison"] = label
        result["reference"] = ref
        result["test"] = test
        results.append(result)

        fmd_str = f"{result['fmd']:.4f}" if result["fmd"] is not None else "FAILED"
        logger.info(
            f"  FMD = {fmd_str}  (ref={result['n_ref']}, test={result['n_test']})"
        )
        if "error" in result:
            logger.warning(f"  Error: {result['error']}")

    # ── Run per-lambda comparisons ──
    per_lambda_results = []
    if per_lambda_comparisons:
        logger.info(
            f"\nRunning {len(per_lambda_comparisons)} per-lambda FMD comparisons...\n"
        )
        for label, ref, test in per_lambda_comparisons:
            logger.info(f"Computing: {label}")
            result = compute_fmd(ref, test, metric)
            result["comparison"] = label
            result["reference"] = ref
            result["test"] = test
            per_lambda_results.append(result)

            fmd_str = f"{result['fmd']:.4f}" if result["fmd"] is not None else "FAILED"
            logger.info(
                f"  FMD = {fmd_str}  (ref={result['n_ref']}, test={result['n_test']})"
            )
            if "error" in result:
                logger.warning(f"  Error: {result['error']}")

    # ── Print summary table ──
    print("\n" + "=" * 90)
    print(" FRECHET MUSIC DISTANCE (FMD) RESULTS")
    print("=" * 90)

    # Simpler: just print all in order
    print(f"\n{'Comparison':<55} {'FMD':>8} {'#Ref':>5} {'#Test':>5}")
    print("-" * 80)

    # Baselines
    for r in results:
        if "Baseline" in r["comparison"] and "vs Baseline" in r["comparison"]:
            fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
            print(f"{r['comparison']:<55} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}")

    print()
    # Steered vs SOD
    for r in results:
        if (
            r["comparison"].startswith("SOD vs ")
            and "Baseline" not in r["comparison"]
            and "__" not in r["comparison"]
        ):
            fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
            print(f"{r['comparison']:<55} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}")

    print()
    # Steering cost
    for r in results:
        if r["comparison"].startswith("Baseline vs "):
            fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
            print(f"{r['comparison']:<55} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}")

    print()
    # Per scenario
    per_scenario = [r for r in results if "__" in r.get("comparison", "")]
    if per_scenario:
        print("Per-scenario (FMD vs SOD):")
        for r in sorted(per_scenario, key=lambda x: x["comparison"]):
            fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
            label = r["comparison"].replace("SOD vs ", "  ")
            print(f"{label:<55} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}")

    # Per-lambda heatmap
    if per_lambda_results:
        print()
        print("=" * 90)
        print(" PER-LAMBDA FMD HEATMAP (FMD vs SOD)")
        print("=" * 90)

        # Parse lambda values from comparison labels and group by (strategy, mode)
        import re

        groups = {}  # (strategy, mode) -> {(lp, ld): fmd}
        for r in per_lambda_results:
            label = r["comparison"]  # e.g. "[cond] expanded_k_2x__lp+0.50_ld+0.75"
            mode_match = re.match(r"\[(cond|uncond)\]\s+(.+?)__(.*)", label)
            if not mode_match:
                continue
            mode = mode_match.group(1)
            strat = mode_match.group(2)
            lam_str = mode_match.group(3)

            # Parse lambda values
            if mode == "cond":
                lam_match = re.match(r"lp([+-]?\d+\.\d+)_ld([+-]?\d+\.\d+)", lam_str)
            else:
                lam_match = re.match(r"p([+-]?\d+\.\d+)_d([+-]?\d+\.\d+)", lam_str)

            if not lam_match:
                continue

            lp_val = float(lam_match.group(1))
            ld_val = float(lam_match.group(2))

            key = (strat, mode)
            if key not in groups:
                groups[key] = {}
            groups[key][(lp_val, ld_val)] = r["fmd"]

        for (strat, mode), fmd_map in sorted(groups.items()):
            print(f"\n  {strat} ({mode}):")

            # Build grid
            all_lp = sorted(set(lp for lp, _ in fmd_map.keys()))
            all_ld = sorted(set(ld for _, ld in fmd_map.keys()))

            # Header
            col_label = "λ_p \\ λ_d"
            header = f"  {col_label:>10}"
            for ld in all_ld:
                header += f" {ld:>8.2f}"
            print(header)
            print("  " + "-" * (11 + 9 * len(all_ld)))

            # Rows
            best_fmd = None
            best_pair = None
            for lp in all_lp:
                row = f"  {lp:>10.2f}"
                for ld in all_ld:
                    val = fmd_map.get((lp, ld))
                    if val is not None:
                        row += f" {val:>8.1f}"
                        if best_fmd is None or val < best_fmd:
                            best_fmd = val
                            best_pair = (lp, ld)
                    else:
                        row += f" {'N/A':>8}"
                print(row)

            if best_pair:
                print(
                    f"  → Best: λ_p={best_pair[0]:+.2f}, λ_d={best_pair[1]:+.2f} → FMD={best_fmd:.1f}"
                )

    print("=" * 90)

    # Save all results
    all_results = results + per_lambda_results
    results_path = args.workspace_dir / "fmd_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved {len(all_results)} results: {results_path}")


if __name__ == "__main__":
    main()
