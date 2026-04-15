#!/usr/bin/env python3
"""Compute 95% confidence intervals for success rates and δ values.

Reads the result JSONs from both SAS and DiffMean dual-steering experiments
and produces a summary table with Mean ± CI for each strategy.

Usage (on EC2):
    python feedback_exp/compute_confidence_intervals.py

    # Or specify paths explicitly:
    python feedback_exp/compute_confidence_intervals.py \
        --sas_uncond  exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json \
        --sas_cond    exp/sod/sparse_steering/dual_steering/conditioned/conditioned_results.json \
        --dm_uncond   steering_interventions/dual_steering/outputs/phase3_grid_search/phase3_results.json \
        --dm_cond     steering_interventions/dual_steering/outputs/phase4_conditioned/conditioned_results.json \
        --output_dir  feedback_exp/ci_results
"""

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np


# ── Confidence interval helpers ─────────────────────────────────────────────


def ci95_proportion(successes: int, n: int) -> dict:
    """Wilson score interval for a binomial proportion (better than normal approx for small n)."""
    if n == 0:
        return {"mean": float("nan"), "ci": float("nan"), "n": 0}
    p_hat = successes / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    # Also provide simple normal CI for reporting
    simple_ci = z * math.sqrt(p_hat * (1 - p_hat) / n) if n > 1 else 0
    return {
        "mean": p_hat,
        "ci_normal": simple_ci,
        "ci_wilson_lo": max(0, centre - margin),
        "ci_wilson_hi": min(1, centre + margin),
        "n": n,
    }


def ci95_continuous(values: List[float]) -> dict:
    """95% CI for continuous metric: mean ± 1.96 * s / sqrt(n)."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci": float("nan"), "n": 0}
    arr = np.array(clean)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci": ci, "n": n}


def fmt_pct(mean: float, ci: float) -> str:
    """Format as percentage with CI."""
    if math.isnan(mean):
        return "N/A"
    return f"{mean * 100:.1f}% ± {ci * 100:.1f}%"


def fmt_val(mean: float, ci: float, decimals: int = 2) -> str:
    """Format continuous value with CI."""
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.{decimals}f} ± {ci:.{decimals}f}"


# ── Result parsers ──────────────────────────────────────────────────────────


def parse_unconditioned_aggregated(results: list, method_label: str) -> dict:
    """Parse aggregated unconditioned results (SAS or DiffMean).

    These JSONs only have success_count / n_samples per config — no per-sample data.
    We aggregate across all (λ_p, λ_d) pairs for each strategy.
    """
    strategy_data = defaultdict(
        lambda: {
            "pitch_successes": 0,
            "dur_successes": 0,
            "both_successes": 0,
            "total_n": 0,
            "degradations": [],
            "config_count": 0,
        }
    )

    for r in results:
        if r.get("error") or r.get("pitch_mean") is None:
            continue

        strat = r.get("strategy", "unknown")
        lp = r.get("lambda_pitch", r.get("alpha_pitch", 0))
        ld = r.get("lambda_duration", r.get("alpha_duration", 0))

        # Skip baseline (0,0) — success is undefined when λ=0
        if abs(lp) < 1e-6 and abs(ld) < 1e-6:
            continue

        n = r.get("valid_samples", r.get("n_samples", r.get("success_rate_n", 5)))

        # Success counts — field names vary between SAS and DiffMean
        p_count = r.get("pitch_success_count", 0)
        d_count = r.get("duration_success_count", 0)
        b_count = r.get("both_success_count", 0)

        # If only rates are available (no counts), reconstruct counts
        if p_count == 0 and "pitch_success_rate" in r:
            p_count = round(r["pitch_success_rate"] * n)
            d_count = round(r["duration_success_rate"] * n)
            b_count = round(r.get("both_success_rate", 0) * n)

        sd = strategy_data[strat]

        # Only count pitch success when pitch is being steered
        if abs(lp) > 1e-6:
            sd["pitch_successes"] += p_count
        if abs(ld) > 1e-6:
            sd["dur_successes"] += d_count
        if abs(lp) > 1e-6 and abs(ld) > 1e-6:
            sd["both_successes"] += b_count

        sd["total_n"] += n
        sd["config_count"] += 1

        # Degradation
        deg = r.get("degradation", r.get("total_degradation"))
        if isinstance(deg, dict):
            val = deg.get("total_degradation")
        else:
            val = deg
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            sd["degradations"].append(val)

    # Build summary
    summary = {}
    for strat, sd in sorted(strategy_data.items()):
        n = sd["total_n"]
        pitch_ci = ci95_proportion(sd["pitch_successes"], n)
        dur_ci = ci95_proportion(sd["dur_successes"], n)
        both_ci = ci95_proportion(sd["both_successes"], n)
        deg_ci = ci95_continuous(sd["degradations"])

        summary[strat] = {
            "method": method_label,
            "pitch_success": pitch_ci,
            "duration_success": dur_ci,
            "both_success": both_ci,
            "degradation": deg_ci,
            "n_configs": sd["config_count"],
        }

    return summary


def parse_conditioned_per_sample(results: list, method_label: str) -> dict:
    """Parse per-sample conditioned results (SAS or DiffMean).

    Each entry is one generation with boolean success fields.
    """
    strategy_data = defaultdict(
        lambda: {
            "pitch_successes": [],
            "dur_successes": [],
            "both_successes": [],
            "degradations": [],
        }
    )

    for r in results:
        strat = r.get("strategy", "unknown")
        sd = strategy_data[strat]

        # Field names differ between SAS and DiffMean
        ps = r.get("pitch_success", r.get("pitch_steering_success"))
        ds = r.get("duration_success", r.get("duration_steering_success"))
        bs = r.get("both_success", r.get("overall_success"))

        if ps is not None:
            sd["pitch_successes"].append(bool(ps))
        if ds is not None:
            sd["dur_successes"].append(bool(ds))
        if bs is not None:
            sd["both_successes"].append(bool(bs))

        # Degradation
        deg = r.get("degradation", {})
        if isinstance(deg, dict):
            val = deg.get("total_degradation")
        else:
            val = deg
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            sd["degradations"].append(val)

    summary = {}
    for strat, sd in sorted(strategy_data.items()):
        p_succ = sum(sd["pitch_successes"])
        p_n = len(sd["pitch_successes"])
        d_succ = sum(sd["dur_successes"])
        d_n = len(sd["dur_successes"])
        b_succ = sum(sd["both_successes"])
        b_n = len(sd["both_successes"])

        summary[strat] = {
            "method": method_label,
            "pitch_success": ci95_proportion(p_succ, p_n),
            "duration_success": ci95_proportion(d_succ, d_n),
            "both_success": ci95_proportion(b_succ, b_n),
            "degradation": ci95_continuous(sd["degradations"]),
            "n_samples": b_n,
        }

    return summary


# ── Main ────────────────────────────────────────────────────────────────────


def load_json(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        print(f"  ⚠ Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def print_table(title: str, summary: dict):
    """Print a formatted table of results."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(
        f"  {'Strategy':<30s}  {'Pitch SR':<18s}  {'Duration SR':<18s}  "
        f"{'Both SR':<18s}  {'δ (degradation)':<18s}  {'n'}"
    )
    print(f"  {'-' * 30}  {'-' * 18}  {'-' * 18}  {'-' * 18}  {'-' * 18}  {'-' * 5}")

    for strat, s in summary.items():
        ps = s["pitch_success"]
        ds = s["duration_success"]
        bs = s["both_success"]
        dg = s["degradation"]
        n = s.get("n_samples", s.get("n_configs", "?"))

        print(
            f"  {strat:<30s}  "
            f"{fmt_pct(ps['mean'], ps['ci_normal']):<18s}  "
            f"{fmt_pct(ds['mean'], ds['ci_normal']):<18s}  "
            f"{fmt_pct(bs['mean'], bs['ci_normal']):<18s}  "
            f"{fmt_val(dg['mean'], dg['ci']):<18s}  "
            f"{n}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compute 95% CIs for paper tables")
    parser.add_argument(
        "--sas_uncond",
        type=pathlib.Path,
        default=pathlib.Path(
            "exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json"
        ),
    )
    parser.add_argument(
        "--sas_cond",
        type=pathlib.Path,
        default=pathlib.Path(
            "exp/sod/sparse_steering/dual_steering/conditioned/conditioned_results.json"
        ),
    )
    parser.add_argument(
        "--dm_uncond",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase3_grid_search/phase3_results.json"
        ),
    )
    parser.add_argument(
        "--dm_cond",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase4_conditioned/conditioned_results.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("feedback_exp/ci_results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    # ── SAS Unconditioned ───────────────────────────────────────────
    print("\n📊 Loading SAS unconditioned results...")
    data = load_json(args.sas_uncond)
    if data:
        results = data.get("results", data if isinstance(data, list) else [])
        summary = parse_unconditioned_aggregated(results, "SAS")
        print_table("SAS Dual — Unconditioned (Table 3)", summary)
        all_summaries["sas_unconditioned"] = summary

    # ── SAS Conditioned ─────────────────────────────────────────────
    print("\n📊 Loading SAS conditioned results...")
    data = load_json(args.sas_cond)
    if data:
        results = data.get("results", data if isinstance(data, list) else [])
        summary = parse_conditioned_per_sample(results, "SAS")
        print_table("SAS Dual — Conditioned (Table 4)", summary)
        all_summaries["sas_conditioned"] = summary

    # ── DiffMean Unconditioned ──────────────────────────────────────
    print("\n📊 Loading DiffMean unconditioned results...")
    data = load_json(args.dm_uncond)
    if data:
        results = data.get("results", data if isinstance(data, list) else [])
        summary = parse_unconditioned_aggregated(results, "DiffMean")
        print_table("DiffMean Dual — Unconditioned (Table 3)", summary)
        all_summaries["diffmean_unconditioned"] = summary

    # ── DiffMean Conditioned ────────────────────────────────────────
    print("\n📊 Loading DiffMean conditioned results...")
    data = load_json(args.dm_cond)
    if data:
        results = data.get("results", data if isinstance(data, list) else [])
        summary = parse_conditioned_per_sample(results, "DiffMean")
        print_table("DiffMean Dual — Conditioned (Table 4)", summary)
        all_summaries["diffmean_conditioned"] = summary

    # ── Save combined JSON ──────────────────────────────────────────
    # Convert numpy/special types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    out_path = args.output_dir / "confidence_intervals.json"
    with open(out_path, "w") as f:
        json.dump(sanitize(all_summaries), f, indent=2)
    print(f"\n✅ Full results saved to {out_path}")

    # ── LaTeX-ready snippet ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  LATEX-READY SNIPPETS (copy into paper)")
    print("=" * 90)
    for table_key, table_label in [
        ("sas_unconditioned", "SAS Unconditioned"),
        ("sas_conditioned", "SAS Conditioned"),
        ("diffmean_unconditioned", "DiffMean Unconditioned"),
        ("diffmean_conditioned", "DiffMean Conditioned"),
    ]:
        if table_key not in all_summaries:
            continue
        print(f"\n% --- {table_label} ---")
        for strat, s in all_summaries[table_key].items():
            bs = s["both_success"]
            dg = s["degradation"]
            b_str = fmt_pct(bs["mean"], bs["ci_normal"])
            d_str = fmt_val(dg["mean"], dg["ci"])
            print(f"% {strat}: Both SR = {b_str}, δ = {d_str}")


if __name__ == "__main__":
    main()
