#!/usr/bin/env python3
"""Compute 95% confidence intervals for success rates and δ values.

Reads the result JSONs from both SAS and DiffMean dual-steering experiments
and produces a summary table with Mean ± CI for each strategy.

Usage (on EC2):
    python feedback_exp/compute_confidence_intervals.py

    # Or specify paths explicitly:
    python feedback_exp/compute_confidence_intervals.py \
        --sas_uncond  exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json \
        --sas_cond    exp/sod/sparse_steering/dual_steering/conditioned_gs_both/conditioned_results.json \
        --dm_uncond   steering_interventions/dual_steering/outputs/diffmean_unconditioned/phase3_results.json \
        --dm_cond     steering_interventions/dual_steering/outputs/diffmean_conditioned/conditioned_results.json \
        --output_dir  feedback_exp/ci_results

    # Auto-scan all SAS conditioned experiment dirs:
    python feedback_exp/compute_confidence_intervals.py --scan_sas_cond
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


def _extract_pitch_mean(r: dict) -> Optional[float]:
    """Extract pitch mean from result dict, handling both SAS and DiffMean formats."""
    # SAS format: {"pitch_mean": float}
    if "pitch_mean" in r and r["pitch_mean"] is not None:
        return r["pitch_mean"]
    # DiffMean format: {"pitch_control": {"mean": float}}
    pc = r.get("pitch_control", {})
    if isinstance(pc, dict) and pc.get("mean") is not None and pc["mean"] != 0:
        return pc["mean"]
    return None


def _extract_duration_mean(r: dict) -> Optional[float]:
    """Extract duration mean from result dict, handling both formats."""
    # SAS format: {"duration_mean": float}
    if "duration_mean" in r and r["duration_mean"] is not None:
        return r["duration_mean"]
    # DiffMean format: {"duration_control": {"mean": float}}
    dc = r.get("duration_control", {})
    if isinstance(dc, dict) and dc.get("mean") is not None and dc["mean"] != 0:
        return dc["mean"]
    return None


def _extract_degradation(r: dict) -> Optional[float]:
    """Extract total degradation value, handling nested and flat formats."""
    deg = r.get("degradation", r.get("total_degradation"))
    if deg is None:
        return None
    if isinstance(deg, (int, float)):
        return float(deg) if not math.isnan(deg) else None
    if isinstance(deg, dict):
        td = deg.get("total_degradation")
        if td is None:
            return None
        # DiffMean: {"total_degradation": {"mean": float, "std": float}}
        if isinstance(td, dict):
            val = td.get("mean")
            return float(val) if val is not None and not math.isnan(val) else None
        # SAS: {"total_degradation": float}
        return float(td) if not math.isnan(td) else None
    return None


def parse_unconditioned_aggregated(results: list, method_label: str) -> dict:
    """Parse aggregated unconditioned results (SAS or DiffMean).

    The unconditioned JSONs have per-config aggregated metrics (pitch_mean,
    duration_mean) but NO success counts.  Success must be computed post-hoc
    by comparing each config's mean to the baseline (λ_p=0, λ_d=0) config.
    """
    # ── Step 1: Find baselines per strategy ──────────────────────────
    strategy_baselines = {}
    for r in results:
        if r.get("error"):
            continue
        pm = _extract_pitch_mean(r)
        dm = _extract_duration_mean(r)
        if pm is None or dm is None:
            continue
        strat = r.get("strategy", "unknown")
        lp = r.get("lambda_pitch", r.get("alpha_pitch", 0))
        ld = r.get("lambda_duration", r.get("alpha_duration", 0))
        if abs(lp) < 1e-6 and abs(ld) < 1e-6:
            strategy_baselines[strat] = {"pitch": pm, "duration": dm}

    # If no per-strategy baseline, use a global one (average across all baselines)
    global_baseline = None
    if strategy_baselines:
        global_baseline = {
            "pitch": float(np.mean([b["pitch"] for b in strategy_baselines.values()])),
            "duration": float(
                np.mean([b["duration"] for b in strategy_baselines.values()])
            ),
        }

    # ── Step 2: Compute success per config and aggregate ─────────────
    strategy_data = defaultdict(
        lambda: {
            "pitch_successes": [],
            "dur_successes": [],
            "both_successes": [],
            "degradations": [],
            "config_count": 0,
        }
    )

    for r in results:
        if r.get("error"):
            continue
        pitch_mean = _extract_pitch_mean(r)
        dur_mean = _extract_duration_mean(r)
        if pitch_mean is None or dur_mean is None:
            continue

        strat = r.get("strategy", "unknown")
        lp = r.get("lambda_pitch", r.get("alpha_pitch", 0))
        ld = r.get("lambda_duration", r.get("alpha_duration", 0))

        # Skip baseline (0,0) — success is undefined when λ=0
        if abs(lp) < 1e-6 and abs(ld) < 1e-6:
            continue

        baseline = strategy_baselines.get(strat, global_baseline)
        if baseline is None:
            continue

        sd = strategy_data[strat]
        sd["config_count"] += 1

        # Compute success post-hoc: did pitch_mean shift in the λ direction?
        p_success = None
        d_success = None

        if abs(lp) > 1e-6:
            if lp > 0:
                p_success = pitch_mean > baseline["pitch"]
            else:
                p_success = pitch_mean < baseline["pitch"]
            sd["pitch_successes"].append(p_success)

        if abs(ld) > 1e-6:
            if ld > 0:
                d_success = dur_mean > baseline["duration"]
            else:
                d_success = dur_mean < baseline["duration"]
            sd["dur_successes"].append(d_success)

        if abs(lp) > 1e-6 and abs(ld) > 1e-6:
            sd["both_successes"].append(p_success and d_success)

        # Degradation
        val = _extract_degradation(r)
        if val is not None:
            sd["degradations"].append(val)

    # ── Step 3: Build summary with CIs ───────────────────────────────
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
            "exp/sod/sparse_steering/dual_steering/conditioned_gs_both/conditioned_results.json"
        ),
    )
    parser.add_argument(
        "--scan_sas_cond",
        action="store_true",
        help="Auto-scan all conditioned_* dirs under exp/sod/sparse_steering/dual_steering/",
    )
    parser.add_argument(
        "--dm_uncond",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/diffmean_unconditioned/phase3_results.json"
        ),
    )
    parser.add_argument(
        "--dm_cond",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/diffmean_conditioned/conditioned_results.json"
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
    if args.scan_sas_cond:
        print("\n📊 Scanning all SAS conditioned result dirs...")
        sas_dual_base = pathlib.Path("exp/sod/sparse_steering/dual_steering")
        all_sas_cond_results = []
        for d in sorted(sas_dual_base.glob("conditioned_*/conditioned_results.json")):
            print(f"  Found: {d.parent.name}")
            data = load_json(d)
            if data:
                results = data.get("results", data if isinstance(data, list) else [])
                all_sas_cond_results.extend(results)
        if all_sas_cond_results:
            summary = parse_conditioned_per_sample(all_sas_cond_results, "SAS")
            print_table("SAS Dual — Conditioned (all dirs, Table 4)", summary)
            all_summaries["sas_conditioned"] = summary
    else:
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
