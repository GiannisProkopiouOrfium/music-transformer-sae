#!/usr/bin/env python3
"""Extract all data needed for PhD slides from FMD + steering-success CSVs.

Run on EC2:
    python metrics_evaluation/extract_slide_data.py \
        --fmd_csv exp/sod/sparse_steering/fmd_workspace/fmd_per_lambda.csv \
        --fmd_json exp/sod/sparse_steering/fmd_workspace/fmd_results.json \
        --success_csv exp/sod/sparse_steering/fmd_workspace/steering_success.csv
"""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

# Paper reference FMD (Gui et al. 2024)
PAPER_FMD = {"uncond": 363.57, "cond": 328.74}

# Friendly labels
STRAT_LABEL = {
    "gram_schmidt_ek2": "SAS gs-ek2",
    "expanded_k_2x": "SAS ek2x",
    "DM_gram_schmidt_pitch": "DiffMean",
    "gram_schmidt_pitch": "DiffMean",
    "gram_schmidt_duration": "SAS gs-dur",
    # JSON comparison strings use SAS_ prefix
    "SAS_gram_schmidt_ek2": "SAS gs-ek2",
    "SAS_expanded_k_2x": "SAS ek2x",
    "SAS_gram_schmidt_duration": "SAS gs-dur",
}

# Map success-CSV strategy names -> FMD-CSV strategy names
# (DiffMean uses 'gram_schmidt_pitch' in success CSV but 'DM_gram_schmidt_pitch' in FMD CSV)
SUCCESS_TO_FMD_STRAT = {
    "gram_schmidt_pitch": "DM_gram_schmidt_pitch",
}

FMD_TO_SUCCESS_STRAT = {v: k for k, v in SUCCESS_TO_FMD_STRAT.items()}


def fmd_strat(success_strat: str) -> str:
    """Convert success-CSV strategy name to FMD-CSV strategy name."""
    return SUCCESS_TO_FMD_STRAT.get(success_strat, success_strat)


def success_strat(fmd_strat_name: str) -> str:
    """Convert FMD-CSV strategy name to success-CSV strategy name."""
    return FMD_TO_SUCCESS_STRAT.get(fmd_strat_name, fmd_strat_name)


SCENARIO_SHORT = {
    "low_pitch_short_duration_to_high_long": "↑P ↑D",
    "low_pitch_long_duration_to_high_short": "↑P ↓D",
    "high_pitch_short_duration_to_low_long": "↓P ↑D",
    "high_pitch_long_duration_to_low_short": "↓P ↓D",
}


def gap(fmd, mode):
    """Percent gap vs paper reference."""
    ref = PAPER_FMD[mode]
    return (fmd - ref) / ref * 100


def load_fmd_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["fmd"] = float(r["fmd"]) if r["fmd"] else None
            r["lambda_pitch"] = float(r["lambda_pitch"]) if r["lambda_pitch"] else None
            r["lambda_duration"] = (
                float(r["lambda_duration"]) if r["lambda_duration"] else None
            )
            rows.append(r)
    return rows


def load_success_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ["lambda_pitch", "lambda_duration"]:
                r[k] = float(r[k]) if r[k] else None
            for k in ["both_success", "pitch_success", "duration_success"]:
                v = r.get(k, "")
                r[k] = True if v == "True" else (False if v == "False" else None)
            for k in ["total_degradation"]:
                v = r.get(k, "")
                r[k] = float(v) if v and v != "" else None
            rows.append(r)
    return rows


def main():
    default_ws = pathlib.Path("exp/sod/sparse_steering/fmd_workspace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fmd_csv",
        type=pathlib.Path,
        default=default_ws / "fmd_per_lambda.csv",
    )
    parser.add_argument(
        "--fmd_json",
        type=pathlib.Path,
        default=default_ws / "fmd_results.json",
    )
    parser.add_argument(
        "--success_csv",
        type=pathlib.Path,
        default=default_ws / "steering_success.csv",
    )
    args = parser.parse_args()

    fmd_rows = load_fmd_csv(args.fmd_csv)
    success_rows = load_success_csv(args.success_csv)

    with open(args.fmd_json) as f:
        fmd_json = json.load(f)

    # Helper: look up FMD for a (mode, strategy_from_success, lp, ld) config
    def lookup_fmd(mode, strat, lp, ld, concept="dual"):
        fs = fmd_strat(strat)
        for r in fmd_rows:
            if (
                r["concept"] == concept
                and r["strategy"] == fs
                and r["mode"] == mode
                and r["lambda_pitch"] is not None
                and r["lambda_duration"] is not None
                and abs(r["lambda_pitch"] - lp) < 1e-3
                and abs(r["lambda_duration"] - ld) < 1e-3
                and r["fmd"] is not None
            ):
                return r["fmd"]
        return None

    def lookup_marginal_fmd(mode, strat, concept, lam_val):
        fs = fmd_strat(strat)
        fmd_key = "lambda_pitch" if "pitch" in concept else "lambda_duration"
        for r in fmd_rows:
            if (
                r["concept"] == concept
                and r["strategy"] == fs
                and r["mode"] == mode
                and r.get(fmd_key) is not None
                and abs(r[fmd_key] - lam_val) < 1e-3
                and r["fmd"] is not None
            ):
                return r["fmd"]
        return None

    # ================================================================
    # 1. AGGREGATE FMD PER MODE (for slides 4/5: unconditioned-specific)
    # ================================================================
    print("=" * 80)
    print("=== 1. AGGREGATE FMD (from fmd_results.json) ===")
    print(f"  Paper reference: uncond={PAPER_FMD['uncond']}, cond={PAPER_FMD['cond']}")
    print()

    for r in sorted(fmd_json, key=lambda x: x.get("comparison", "")):
        comp = r.get("comparison", "")
        fmd_val = r.get("fmd")
        if fmd_val is None:
            continue
        # Match "SOD vs {X} (conditioned|unconditioned)" but not per-scenario "__"
        m = re.match(r"SOD vs (.+?) \((conditioned|unconditioned)\)$", comp)
        if m and "__" not in comp:
            label = m.group(1)
            label_nice = STRAT_LABEL.get(label, label)
            mode = "cond" if m.group(2) == "conditioned" else "uncond"
            g = gap(fmd_val, mode)
            print(
                f"  {mode:<7} {label_nice:<28} FMD={fmd_val:.1f}  "
                f"gap={g:+.1f}% vs paper"
            )

    # ================================================================
    # 2. BEST DUAL FMD (both λ ≠ 0)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 2. BEST DUAL FMD (both λ ≠ 0) ===")

    dual_fmd = [
        r
        for r in fmd_rows
        if r["concept"] == "dual"
        and r["fmd"] is not None
        and r["lambda_pitch"] is not None
        and r["lambda_duration"] is not None
        and abs(r["lambda_pitch"]) > 1e-6
        and abs(r["lambda_duration"]) > 1e-6
    ]

    best_dual = {}  # (mode, strategy) -> best row
    for r in dual_fmd:
        key = (r["mode"], r["strategy"])
        if key not in best_dual or r["fmd"] < best_dual[key]["fmd"]:
            best_dual[key] = r

    for (mode, strat), r in sorted(best_dual.items()):
        sl = STRAT_LABEL.get(strat, strat)
        g = gap(r["fmd"], mode)
        print(
            f"  {mode:<7} {sl:<28} FMD={r['fmd']:.1f} "
            f"lp={r['lambda_pitch']:+.2f} ld={r['lambda_duration']:+.2f}  "
            f"gap={g:+.1f}%"
        )

    # ================================================================
    # 3. SUCCESS + DEGRADATION AT BEST DUAL FMD CONFIGS
    # ================================================================
    print()
    print("=" * 80)
    print("=== 3. SUCCESS + DEGRADATION AT BEST DUAL FMD CONFIGS ===")

    for (mode, strat), fmd_row in sorted(best_dual.items()):
        lp = fmd_row["lambda_pitch"]
        ld = fmd_row["lambda_duration"]
        ss_strat = success_strat(strat)  # map FMD name -> success name

        # Match success rows at this config
        matching = [
            s
            for s in success_rows
            if s["mode"] == ("uncond" if mode == "uncond" else "cond")
            and s["strategy"] == ss_strat
            and s["lambda_pitch"] is not None
            and s["lambda_duration"] is not None
            and abs(s["lambda_pitch"] - lp) < 1e-3
            and abs(s["lambda_duration"] - ld) < 1e-3
            and s["both_success"] is not None
        ]
        n_total = len(matching)
        n_ok = sum(1 for s in matching if s["both_success"] is True)
        rate = (n_ok / n_total * 100) if n_total else float("nan")

        degs = [
            s["total_degradation"]
            for s in matching
            if s["total_degradation"] is not None
        ]
        avg_deg = np.nanmean(degs) if degs else float("nan")

        sl = STRAT_LABEL.get(strat, strat)
        print(
            f"  {mode:<7} {sl:<28} "
            f"lp={lp:+.2f} ld={ld:+.2f}  "
            f"success={rate:.0f}% ({n_ok}/{n_total})  "
            f"degradation={avg_deg:.2f}"
        )

    # ================================================================
    # 4. BEST MARGINAL FMD (single-concept)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 4. BEST MARGINAL FMD (single-concept, pitch / duration) ===")

    for concept in ["marginal_pitch", "marginal_duration"]:
        marginal = [
            r for r in fmd_rows if r["concept"] == concept and r["fmd"] is not None
        ]
        best_m = {}  # (mode, strategy) -> best row
        for r in marginal:
            key = (r["mode"], r["strategy"])
            if key not in best_m or r["fmd"] < best_m[key]["fmd"]:
                best_m[key] = r

        c_label = "pitch" if "pitch" in concept else "duration"
        for (mode, strat), r in sorted(best_m.items()):
            sl = STRAT_LABEL.get(strat, strat)
            lam = (
                r["lambda_pitch"]
                if r["lambda_pitch"] is not None
                else r["lambda_duration"]
            )
            g = gap(r["fmd"], mode)
            print(
                f"  {mode:<7} {sl:<28} {c_label:<9} "
                f"FMD={r['fmd']:.1f}  λ={lam:+.2f}  gap={g:+.1f}%"
            )

    # ================================================================
    # 5. BEST UNCONDITIONED CONFIGS (success + FMD + degradation)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 5. BEST UNCOND CONFIGS (dual, both λ ≠ 0) ===")
    print("  Ranked by: max both_success%, then min degradation")

    uncond_success = [
        s
        for s in success_rows
        if s["mode"] == "uncond"
        and s["both_success"] is not None
        and s["lambda_pitch"] is not None
        and s["lambda_duration"] is not None
        and abs(s["lambda_pitch"]) > 1e-6
        and abs(s["lambda_duration"]) > 1e-6
    ]

    # Aggregate by (strategy, lp, ld)
    agg = defaultdict(lambda: {"ok": 0, "total": 0, "degs": []})
    for s in uncond_success:
        key = (s["strategy"], s["lambda_pitch"], s["lambda_duration"])
        agg[key]["total"] += 1
        if s["both_success"]:
            agg[key]["ok"] += 1
        if s["total_degradation"] is not None:
            agg[key]["degs"].append(s["total_degradation"])

    configs = []
    for (strat, lp, ld), a in agg.items():
        rate = a["ok"] / a["total"] if a["total"] else 0
        avg_deg = np.nanmean(a["degs"]) if a["degs"] else float("nan")
        fmd_val = lookup_fmd("uncond", strat, lp, ld)
        configs.append(
            {
                "strat": strat,
                "lp": lp,
                "ld": ld,
                "rate": rate,
                "deg": avg_deg,
                "fmd": fmd_val,
                "n": a["total"],
            }
        )

    configs.sort(key=lambda c: (-c["rate"], c["deg"]))
    print(
        f"  {'Strategy':<28} {'λ_p':>6} {'λ_d':>6} "
        f"{'Both%':>7} {'Degrad':>8} {'FMD':>8} {'N':>4}"
    )
    print("  " + "-" * 75)
    for c in configs[:15]:
        fmd_s = f"{c['fmd']:.1f}" if c["fmd"] else "N/A"
        sl = STRAT_LABEL.get(c["strat"], c["strat"])
        print(
            f"  {sl:<28} {c['lp']:>+6.2f} {c['ld']:>+6.2f} "
            f"{c['rate']*100:>6.0f}% {c['deg']:>8.2f} {fmd_s:>8} {c['n']:>4}"
        )

    # ================================================================
    # 6. BEST CONDITIONED CONFIGS (dual, both λ ≠ 0)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 6. BEST COND CONFIGS (dual, both λ ≠ 0) ===")
    print("  Ranked by: max both_success%, then min degradation")

    cond_success = [
        s
        for s in success_rows
        if s["mode"] == "cond"
        and s["both_success"] is not None
        and s["lambda_pitch"] is not None
        and s["lambda_duration"] is not None
        and abs(s["lambda_pitch"]) > 1e-6
        and abs(s["lambda_duration"]) > 1e-6
    ]

    agg_c = defaultdict(lambda: {"ok": 0, "total": 0, "degs": []})
    for s in cond_success:
        key = (s["strategy"], s["lambda_pitch"], s["lambda_duration"])
        agg_c[key]["total"] += 1
        if s["both_success"]:
            agg_c[key]["ok"] += 1
        if s["total_degradation"] is not None:
            agg_c[key]["degs"].append(s["total_degradation"])

    configs_c = []
    for (strat, lp, ld), a in agg_c.items():
        rate = a["ok"] / a["total"] if a["total"] else 0
        avg_deg = np.nanmean(a["degs"]) if a["degs"] else float("nan")
        fmd_val = lookup_fmd("cond", strat, lp, ld)
        configs_c.append(
            {
                "strat": strat,
                "lp": lp,
                "ld": ld,
                "rate": rate,
                "deg": avg_deg,
                "fmd": fmd_val,
                "n": a["total"],
            }
        )

    configs_c.sort(key=lambda c: (-c["rate"], c["deg"]))
    print(
        f"  {'Strategy':<28} {'λ_p':>6} {'λ_d':>6} "
        f"{'Both%':>7} {'Degrad':>8} {'FMD':>8} {'N':>4}"
    )
    print("  " + "-" * 75)
    for c in configs_c[:15]:
        fmd_s = f"{c['fmd']:.1f}" if c["fmd"] else "N/A"
        sl = STRAT_LABEL.get(c["strat"], c["strat"])
        print(
            f"  {sl:<28} {c['lp']:>+6.2f} {c['ld']:>+6.2f} "
            f"{c['rate']*100:>6.0f}% {c['deg']:>8.2f} {fmd_s:>8} {c['n']:>4}"
        )

    # ================================================================
    # 7. PER-SCENARIO FMD (from fmd_results.json)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 7. PER-SCENARIO FMD (conditioned, FMD vs SOD) ===")

    per_scenario = [
        r
        for r in fmd_json
        if r.get("comparison", "").startswith("SOD vs ")
        and "__" in r.get("comparison", "")
        and r.get("fmd") is not None
    ]
    for r in sorted(per_scenario, key=lambda x: x["comparison"]):
        comp = r["comparison"].replace("SOD vs ", "")
        # Parse strategy and scenario
        parts = comp.split("__", 1)
        if len(parts) == 2:
            strat, scenario = parts
            sc_short = SCENARIO_SHORT.get(scenario, scenario[:30])
            strat_short = STRAT_LABEL.get(strat, strat)
            g = gap(r["fmd"], "cond")
            print(
                f"  {strat_short:<20} {sc_short:<10} "
                f"FMD={r['fmd']:.1f}  gap={g:+.1f}%  "
                f"(n_test={r.get('n_test', '?')})"
            )

    # ================================================================
    # 8. PER-SCENARIO SUCCESS + FMD COMBINED (for slide 6)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 8. PER-SCENARIO: SUCCESS + FMD COMBINED ===")

    # Build FMD lookup by (strategy, scenario)
    scenario_fmd = {}
    for r in per_scenario:
        comp = r["comparison"].replace("SOD vs ", "")
        parts = comp.split("__", 1)
        if len(parts) == 2:
            strat, scenario = parts
            scenario_fmd[(strat, scenario)] = r["fmd"]

    # Build success by (strategy, scenario)
    cond_all = [s for s in success_rows if s["mode"] == "cond"]
    scenario_success = defaultdict(lambda: {"ok": 0, "total": 0, "degs": []})
    for s in cond_all:
        if s["both_success"] is None:
            continue
        scenario = s.get("scenario", "")
        strat = s["strategy"]
        key = (strat, scenario)
        scenario_success[key]["total"] += 1
        if s["both_success"]:
            scenario_success[key]["ok"] += 1
        if s["total_degradation"] is not None:
            scenario_success[key]["degs"].append(s["total_degradation"])

    print(
        f"  {'Strategy':<28} {'Scenario':<10} "
        f"{'Both%':>7} {'Degrad':>8} {'FMD':>8} {'N':>4}"
    )
    print("  " + "-" * 75)

    for strat, scenario in sorted(scenario_success.keys()):
        ss = scenario_success[(strat, scenario)]
        rate = ss["ok"] / ss["total"] * 100 if ss["total"] else 0
        avg_deg = np.nanmean(ss["degs"]) if ss["degs"] else float("nan")
        fmd_val = scenario_fmd.get((fmd_strat(strat), scenario))
        sc_short = SCENARIO_SHORT.get(scenario, scenario[:30])
        strat_short = STRAT_LABEL.get(strat, strat)

        fmd_s = f"{fmd_val:.1f}" if fmd_val is not None else "N/A"
        print(
            f"  {strat_short:<28} {sc_short:<10} "
            f"{rate:>6.0f}% {avg_deg:>8.2f} {fmd_s:>8} {ss['total']:>4}"
        )

    # ================================================================
    # 9. UNCOND SINGLE-CONCEPT CONFIGS (pitch-only and duration-only)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 9. UNCOND SINGLE-CONCEPT (pitch-only / duration-only) ===")

    for concept_name, lam_key, zero_key in [
        ("pitch-only", "lambda_pitch", "lambda_duration"),
        ("duration-only", "lambda_duration", "lambda_pitch"),
    ]:
        print(f"\n  --- {concept_name} ---")
        single = [
            s
            for s in success_rows
            if s["mode"] == "uncond"
            and s[lam_key] is not None
            and s[zero_key] is not None
            and abs(s[lam_key]) > 1e-6
            and abs(s[zero_key]) < 1e-6
        ]
        # Aggregate by (strategy, lambda)
        single_agg = defaultdict(lambda: {"ok": 0, "total": 0, "degs": []})
        for s in single:
            key = (s["strategy"], s[lam_key])
            sc_key = "pitch_success" if "pitch" in concept_name else "duration_success"
            single_agg[key]["total"] += 1
            if s.get(sc_key) is True:
                single_agg[key]["ok"] += 1
            if s["total_degradation"] is not None:
                single_agg[key]["degs"].append(s["total_degradation"])

        sc_configs = []
        for (strat, lam_val), a in single_agg.items():
            rate = a["ok"] / a["total"] if a["total"] else 0
            avg_deg = np.nanmean(a["degs"]) if a["degs"] else float("nan")

            fmd_concept = (
                "marginal_pitch" if "pitch" in concept_name else "marginal_duration"
            )
            fmd_val = lookup_marginal_fmd("uncond", strat, fmd_concept, lam_val)

            sc_configs.append(
                {
                    "strat": strat,
                    "lam": lam_val,
                    "rate": rate,
                    "deg": avg_deg,
                    "fmd": fmd_val,
                    "n": a["total"],
                }
            )

        sc_configs.sort(key=lambda c: (-c["rate"], c["deg"]))
        print(
            f"  {'Strategy':<28} {'λ':>6} "
            f"{'Succ%':>7} {'Degrad':>8} {'FMD':>8} {'N':>4}"
        )
        print("  " + "-" * 65)
        for c in sc_configs[:12]:
            fmd_s = f"{c['fmd']:.1f}" if c["fmd"] else "N/A"
            sl = STRAT_LABEL.get(c["strat"], c["strat"])
            print(
                f"  {sl:<28} {c['lam']:>+6.2f} "
                f"{c['rate']*100:>6.0f}% {c['deg']:>8.2f} {fmd_s:>8} {c['n']:>4}"
            )

    # ================================================================
    # 10. COND SINGLE-CONCEPT CONFIGS
    # ================================================================
    print()
    print("=" * 80)
    print("=== 10. COND SINGLE-CONCEPT (pitch-only / duration-only) ===")

    for concept_name, lam_key, zero_key in [
        ("pitch-only", "lambda_pitch", "lambda_duration"),
        ("duration-only", "lambda_duration", "lambda_pitch"),
    ]:
        print(f"\n  --- {concept_name} ---")
        single = [
            s
            for s in success_rows
            if s["mode"] == "cond"
            and s[lam_key] is not None
            and s[zero_key] is not None
            and abs(s[lam_key]) > 1e-6
            and abs(s[zero_key]) < 1e-6
        ]
        single_agg = defaultdict(lambda: {"ok": 0, "total": 0, "degs": []})
        for s in single:
            key = (s["strategy"], s[lam_key])
            sc_key = "pitch_success" if "pitch" in concept_name else "duration_success"
            single_agg[key]["total"] += 1
            if s.get(sc_key) is True:
                single_agg[key]["ok"] += 1
            if s["total_degradation"] is not None:
                single_agg[key]["degs"].append(s["total_degradation"])

        sc_configs = []
        for (strat, lam_val), a in single_agg.items():
            rate = a["ok"] / a["total"] if a["total"] else 0
            avg_deg = np.nanmean(a["degs"]) if a["degs"] else float("nan")

            fmd_concept = (
                "marginal_pitch" if "pitch" in concept_name else "marginal_duration"
            )
            fmd_val = lookup_marginal_fmd("cond", strat, fmd_concept, lam_val)

            sc_configs.append(
                {
                    "strat": strat,
                    "lam": lam_val,
                    "rate": rate,
                    "deg": avg_deg,
                    "fmd": fmd_val,
                    "n": a["total"],
                }
            )

        sc_configs.sort(key=lambda c: (-c["rate"], c["deg"]))
        print(
            f"  {'Strategy':<28} {'λ':>6} "
            f"{'Succ%':>7} {'Degrad':>8} {'FMD':>8} {'N':>4}"
        )
        print("  " + "-" * 65)
        for c in sc_configs[:12]:
            fmd_s = f"{c['fmd']:.1f}" if c["fmd"] else "N/A"
            sl = STRAT_LABEL.get(c["strat"], c["strat"])
            print(
                f"  {sl:<28} {c['lam']:>+6.2f} "
                f"{c['rate']*100:>6.0f}% {c['deg']:>8.2f} {fmd_s:>8} {c['n']:>4}"
            )

    # ================================================================
    # 11. FMD AT BEST SUCCESS CONFIGS (both dual modes)
    # ================================================================
    print()
    print("=" * 80)
    print("=== 11. FMD AT TOP SUCCESS CONFIGS ===")
    print("  (picking configs with success=100%, lowest degradation, both λ ≠ 0)")

    for mode_label, cfg_list in [("uncond", configs[:10]), ("cond", configs_c[:10])]:
        print(f"\n  --- {mode_label} ---")
        for c in cfg_list:
            if c["rate"] < 1.0:
                break
            fmd_s = f"{c['fmd']:.1f}" if c["fmd"] else "N/A"
            g = gap(c["fmd"], mode_label) if c["fmd"] else float("nan")
            sl = STRAT_LABEL.get(c["strat"], c["strat"])
            print(
                f"  {sl:<28} lp={c['lp']:+.2f} ld={c['ld']:+.2f}  "
                f"success=100%  deg={c['deg']:.2f}  FMD={fmd_s}  gap={g:+.1f}%"
            )

    print()
    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
