#!/usr/bin/env python3
"""Tabulate smooth (beat-wise) vs abrupt steering results for the journal paper.

Consumes the per-sample `comparison_results.json` files written by
  steering_interventions/beatwise_vs_abrupt_dm.py      (dense DiffMean)
  sparse_steering/beatwise_vs_abrupt_comparison.py     (static SAS)
and emits the aggregated rows for Table "smooth-abrupt" of the manuscript:
per (method, mode): steering success rate, mean degradation with 95% CI,
and abrupt-vs-beatwise significance (via significance_tests.py policy).

Record schema (both scripts): flat dicts with
  concept, category, mode ("abrupt"|"beatwise"), alpha (or lambda),
  init_pitch, init_duration, gen_mean_pitch, gen_mean_duration,
  pitch_change, duration_change, total_degradation, quality{...}

Usage (after syncing results from EC2, see journal/EXPERIMENTS_RUNBOOK.md):
    python feedback_exp/beatwise_table.py \
        --dm  steering_interventions/outputs/beatwise_vs_abrupt/comparison_results.json \
        --sas sparse_steering/outputs/beatwise_vs_abrupt/comparison_results.json \
        --latex

Self-test on synthetic fixtures:
    python feedback_exp/beatwise_table.py --self-test
"""

import argparse
import json
import math
import pathlib
import sys
import tempfile
from collections import defaultdict
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from significance_tests import (  # noqa: E402
    compare_groups, holm_correction, load_records, stars,
)


def _strength(r: dict) -> Optional[float]:
    for key in ("alpha", "lambda", "lam", "steering_strength"):
        if key in r and r[key] is not None:
            return float(r[key])
    return None


def _change(r: dict) -> Optional[float]:
    concept = str(r.get("concept", ""))
    key = "duration_change" if "duration" in concept else "pitch_change"
    v = r.get(key)
    return None if v is None else float(v)


def _degradation(r: dict) -> Optional[float]:
    v = r.get("total_degradation")
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def aggregate(records: List[dict], method_label: str) -> List[dict]:
    """Per (concept, mode): success rate, mean delta with 95% CI, samples."""
    groups = defaultdict(list)
    for r in records:
        s = _strength(r)
        if s is None or abs(s) < 1e-9:  # skip baselines
            continue
        groups[(str(r.get("concept", "?")), str(r.get("mode", "?")))].append(r)

    rows = []
    for (concept, mode), recs in sorted(groups.items()):
        changes = [( _change(r), _strength(r)) for r in recs]
        changes = [(c, s) for c, s in changes if c is not None]
        n = len(changes)
        successes = sum(1 for c, s in changes if (s > 0) == (c > 0) and c != 0)
        degs = np.array([d for r in recs if (d := _degradation(r)) is not None])
        deg_mean = float(np.mean(degs)) if len(degs) else float("nan")
        deg_ci = (1.96 * float(np.std(degs, ddof=1)) / math.sqrt(len(degs))
                  if len(degs) > 1 else float("nan"))
        rows.append({
            "method": method_label,
            "concept": concept,
            "mode": mode,
            "n": n,
            "success_rate": successes / n if n else float("nan"),
            "delta_mean": deg_mean,
            "delta_ci": deg_ci,
            "_deg_samples": degs.tolist(),
        })
    return rows


def significance(rows: List[dict]) -> List[dict]:
    """Abrupt-vs-beatwise test per (method, concept), Holm-corrected."""
    index = {(r["method"], r["concept"], r["mode"]): r for r in rows}
    comparisons = []
    for (method, concept, mode), r in index.items():
        if mode != "abrupt":
            continue
        other = index.get((method, concept, "beatwise"))
        if other is None:
            continue
        res = compare_groups(np.array(r["_deg_samples"]),
                             np.array(other["_deg_samples"]))
        res["name"] = f"{method} {concept}: abrupt vs beatwise (delta)"
        comparisons.append(res)
    holm_correction(comparisons)
    return comparisons


def render(rows: List[dict], comps: List[dict], latex: bool) -> str:
    lines = []
    header = (f"{'method':<10} {'concept':<18} {'mode':<9} {'n':>4} "
              f"{'success%':>9} {'delta':>14}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['method']:<10} {r['concept']:<18} {r['mode']:<9} {r['n']:>4} "
            f"{r['success_rate']*100:>8.1f}% "
            f"{r['delta_mean']:>7.2f} ± {r['delta_ci']:.2f}"
        )
    lines.append("")
    for c in comps:
        p = c.get("p_holm", float("nan"))
        lines.append(f"{c['name']}: {c.get('test','')} p_holm={p:.4g} "
                     f"{stars(p)}  Cliff's delta={c.get('cliffs_delta', float('nan')):.2f}")
    if latex:
        lines.append("")
        lines.append("% LaTeX rows for Table smooth-abrupt "
                     "(Method & Onset & Success & delta):")
        for r in rows:
            onset = "smooth" if r["mode"] == "beatwise" else "abrupt"
            lines.append(
                f"% {r['method']} ({r['concept']}) & {onset} & "
                f"{r['success_rate']*100:.1f}\\% & "
                f"{r['delta_mean']:.2f} $\\pm$ {r['delta_ci']:.2f} \\\\"
            )
    return "\n".join(lines)


def self_test() -> int:
    rng = np.random.default_rng(1)

    def rec(mode, concept, alpha, change, deg):
        key = "duration_change" if "duration" in concept else "pitch_change"
        return {"concept": concept, "mode": mode, "alpha": alpha,
                key: float(change), "total_degradation": float(deg)}

    records = []
    # abrupt: high degradation; beatwise: low degradation; both succeed mostly
    for _ in range(30):
        records.append(rec("abrupt", "average_pitch", 1.0,
                           rng.normal(10, 2), rng.normal(3.0, 0.4)))
        records.append(rec("beatwise", "average_pitch", 1.0,
                           rng.normal(9, 2), rng.normal(1.0, 0.4)))
        records.append(rec("abrupt", "average_pitch", 0.0, 0.0, 0.5))  # baseline: skipped

    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "comparison_results.json"
        f.write_text(json.dumps(records))
        rows = aggregate(load_records(f), "Dense")
        comps = significance(rows)
        print(render(rows, comps, latex=True))

    assert len(rows) == 2, rows
    assert all(r["n"] == 30 for r in rows)
    assert all(r["success_rate"] > 0.9 for r in rows)
    assert comps and comps[0]["p_holm"] < 0.001
    print("\nself-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dm", type=pathlib.Path,
                    help="DiffMean comparison_results.json")
    ap.add_argument("--sas", type=pathlib.Path,
                    help="SAS comparison_results.json")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.dm and not args.sas:
        ap.error("provide --dm and/or --sas, or --self-test")

    rows = []
    if args.dm:
        rows += aggregate(load_records(args.dm), "Dense")
    if args.sas:
        rows += aggregate(load_records(args.sas), "Sparse")
    comps = significance(rows)
    out = render(rows, comps, latex=args.latex)
    print(out)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": [{k: v for k, v in r.items() if k != "_deg_samples"}
                            for r in rows],
                   "comparisons": comps}
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
