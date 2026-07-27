#!/usr/bin/env python3
"""Significance tests and effect sizes for steering comparisons.

Addresses reviewer requests (ISMIR #262, R2/R3/meta): pairwise statistical
tests beyond confidence intervals, with effect sizes and multiple-comparison
correction, for every dense-vs-sparse and PID-vs-static comparison.

Test policy
-----------
For each comparison of two independent per-sample groups:
  * Shapiro-Wilk normality check on both groups (when 3 <= n <= 5000).
  * Both normal (p > .05)  -> Welch's t-test (unequal variances).
  * Otherwise              -> Mann-Whitney U test.
  * Effect sizes: Cliff's delta always; Cohen's d (pooled SD) alongside.
  * Holm-Bonferroni correction across the family of comparisons in one run.

Input formats (auto-detected)
-----------------------------
A "results file" is a JSON containing per-sample records, either as a
top-level list of dicts, or nested under one of: "results", "samples",
"all_results", "records", "per_sample". Records are flat dicts; nested
dicts are flattened with dotted keys (e.g. "quality.scale_consistency").

Usage
-----
# Single comparison: two files (or the same file twice with filters)
python feedback_exp/significance_tests.py \
    --name "SAS vs DM dual uncond (delta)" \
    --a exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json \
    --b steering_interventions/dual_steering/outputs/diffmean_unconditioned/phase3_results.json \
    --metric total_degradation

# Within-file comparison via filters (e.g. abrupt vs beatwise):
python feedback_exp/significance_tests.py \
    --name "DM smooth vs abrupt (pitch, delta)" \
    --a steering_interventions/outputs/beatwise_vs_abrupt/comparison_results.json \
    --filter-a mode=abrupt --filter-a concept=average_pitch \
    --b steering_interventions/outputs/beatwise_vs_abrupt/comparison_results.json \
    --filter-b mode=beatwise --filter-b concept=average_pitch \
    --metric total_degradation

# Batch mode with Holm correction across all comparisons:
python feedback_exp/significance_tests.py --config feedback_exp/comparisons.json

# Self-test on synthetic fixtures (no data needed):
python feedback_exp/significance_tests.py --self-test
"""

import argparse
import json
import math
import pathlib
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None

RECORD_CONTAINER_KEYS = ("results", "samples", "all_results", "records",
                         "per_sample", "per_sample_metrics")

# SOD ground-truth statistics for the derived degradation metric.
H0, S0, G0 = 2.974, 92.26, 93.05

# Aliases so "delta" / "total_degradation" resolve to the same derived metric,
# and scale/groove/entropy resolve regardless of the evaluator's field name.
_METRIC_ALIASES = {
    "scale": "scale_consistency",
    "groove": "groove_consistency",
    "entropy": "pitch_class_entropy",
    "pitch": "average_pitch",
    "duration": "average_duration",
}


def _derived_delta(r: dict):
    """Compute delta = |H-H0| + max(0,S0-S) + max(0,G0-G) from a record.

    Handles scale/groove given either on a 0-100 or 0-1 scale.
    """
    try:
        H = float(r["pitch_class_entropy"])
        S = float(r["scale_consistency"])
        G = float(r["groove_consistency"])
    except (KeyError, TypeError, ValueError):
        return None
    if S <= 1.0:  # stored as fraction
        S *= 100.0
    if G <= 1.0:
        G *= 100.0
    return abs(H - H0) + max(0.0, S0 - S) + max(0.0, G0 - G)


# ── Record loading ───────────────────────────────────────────────────────────


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def load_records(path: pathlib.Path) -> List[dict]:
    """Load per-sample records from a results JSON (list or nested list)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in RECORD_CONTAINER_KEYS:
            if isinstance(data.get(key), list):
                records = data[key]
                break
        if records is None:
            # dict keyed by method/group, each value either a list of dicts
            # (e.g. {"pid": [...], "static": [...]}) or a dict wrapping a
            # per-sample list (e.g. temporal PID files:
            #   {"temporal_pid": {"per_sample_metrics": [...]}, "static_sas": {...}})
            records = []
            for group, val in data.items():
                sub = None
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    sub = val
                elif isinstance(val, dict):
                    for key in RECORD_CONTAINER_KEYS:
                        if isinstance(val.get(key), list):
                            sub = val[key]
                            break
                if sub:
                    for r in sub:
                        if not isinstance(r, dict):
                            continue
                        r = dict(r)
                        r.setdefault("group", group)
                        records.append(r)
        if not records:
            raise ValueError(f"No per-sample record list found in {path}")
    else:
        raise ValueError(f"Unsupported JSON top-level type in {path}")
    return [_flatten(r) for r in records if isinstance(r, dict)]


def apply_filters(records: List[dict], filters: List[str]) -> List[dict]:
    """Filter records by repeated key=value expressions (numeric-aware)."""
    for expr in filters:
        key, _, raw = expr.partition("=")
        if not _:
            raise ValueError(f"Bad filter (expected key=value): {expr}")

        def match(r, key=key, raw=raw):
            if key not in r:
                return False
            val = r[key]
            try:
                return math.isclose(float(val), float(raw), abs_tol=1e-9)
            except (TypeError, ValueError):
                return str(val) == raw

        records = [r for r in records if match(r)]
    return records


# Flattened keys that may hold a precomputed degradation value.
_DEG_KEYS = ("total_degradation", "degradation", "degradation.total_degradation",
             "degradation.total_degradation.mean", "total_degradation.mean")


def _extract_value(r: dict, metric: str):
    """Resolve one metric from a (flattened) record, robust to schema variants."""
    if metric in ("delta", "total_degradation"):
        for k in _DEG_KEYS:
            if k in r and r[k] is not None:
                try:
                    return float(r[k])
                except (TypeError, ValueError):
                    pass
        return _derived_delta(r)  # compute from H/S/G when not precomputed
    key = _METRIC_ALIASES.get(metric, metric)
    return r.get(key)


def extract_metric(records: List[dict], metric: str) -> np.ndarray:
    vals = []
    for r in records:
        v = _extract_value(r, metric)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def filter_diagnostics(all_records: List[dict], filtered: List[dict],
                       filters: List[str], label: str) -> None:
    """On a zero-match, print available keys and the distinct values of each
    filter key, so the user can correct the filter without guessing."""
    if filtered or not filters:
        return
    keys = sorted({k for r in all_records for k in r})
    print(f"  [!] filter {filters} matched 0 of {len(all_records)} records "
          f"in group '{label}'.", file=sys.stderr)
    print(f"      available keys: {', '.join(keys[:40])}", file=sys.stderr)
    for expr in filters:
        k = expr.partition("=")[0]
        vals = sorted({str(r[k]) for r in all_records if k in r})
        if vals:
            print(f"      values for '{k}': {', '.join(vals[:30])}", file=sys.stderr)
        else:
            print(f"      key '{k}' not present in any record", file=sys.stderr)


# ── Statistics ───────────────────────────────────────────────────────────────


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta via the Mann-Whitney U statistic (O(n log n))."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    if sps is not None:
        u, _ = sps.mannwhitneyu(a, b, alternative="two-sided")
        return float(2.0 * u / (na * nb) - 1.0)
    # Fallback: O(n^2)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (na * nb))


def cliff_magnitude(d: float) -> str:
    ad = abs(d)
    if math.isnan(ad):
        return "n/a"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def compare_groups(a: np.ndarray, b: np.ndarray) -> dict:
    """Normality-gated two-sided test plus effect sizes."""
    if sps is None:
        raise RuntimeError("scipy is required for significance testing")
    res = {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)) if len(a) else float("nan"),
        "mean_b": float(np.mean(b)) if len(b) else float("nan"),
        "std_a": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
        "std_b": float(np.std(b, ddof=1)) if len(b) > 1 else float("nan"),
    }
    if len(a) < 3 or len(b) < 3:
        res.update(test="insufficient-n", p=float("nan"),
                   cohens_d=float("nan"), cliffs_delta=float("nan"))
        return res

    def normal(x):
        if len(x) < 3 or len(x) > 5000:
            return False
        if np.ptp(x) == 0:
            return False
        return sps.shapiro(x).pvalue > 0.05

    if normal(a) and normal(b):
        stat, p = sps.ttest_ind(a, b, equal_var=False)
        test = "welch-t"
    else:
        stat, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        test = "mann-whitney-u"
    cd = cliffs_delta(a, b)
    res.update(
        test=test,
        statistic=float(stat),
        p=float(p),
        cohens_d=cohens_d(a, b),
        cliffs_delta=cd,
        cliff_magnitude=cliff_magnitude(cd),
    )
    return res


def holm_correction(results: List[dict]) -> None:
    """In-place Holm-Bonferroni adjusted p-values across the comparison family."""
    valid = [r for r in results if not math.isnan(r.get("p", float("nan")))]
    m = len(valid)
    order = sorted(range(m), key=lambda i: valid[i]["p"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * valid[idx]["p"])
        running_max = max(running_max, adj)
        valid[idx]["p_holm"] = running_max


def stars(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Reporting ────────────────────────────────────────────────────────────────


def report(results: List[dict], latex: bool) -> str:
    lines = []
    header = (
        f"{'comparison':<44} {'test':<16} {'p':>9} {'p_holm':>9} "
        f"{'sig':>4} {'d':>7} {'cliff':>7} {'mag':<10} {'n':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        p_holm = r.get("p_holm", float("nan"))
        lines.append(
            f"{r['name']:<44} {r.get('test',''):<16} "
            f"{r.get('p', float('nan')):>9.4g} {p_holm:>9.4g} "
            f"{stars(p_holm):>4} {r.get('cohens_d', float('nan')):>7.2f} "
            f"{r.get('cliffs_delta', float('nan')):>7.2f} "
            f"{r.get('cliff_magnitude',''):<10} "
            f"{r.get('n_a',0)}/{r.get('n_b',0):>4}"
        )
    if latex:
        lines.append("")
        lines.append("% LaTeX rows (name & test & p_holm & Cliff's delta):")
        for r in results:
            p_holm = r.get("p_holm", float("nan"))
            lines.append(
                f"% {r['name']} & {r.get('test','')} & "
                f"$p={p_holm:.3g}$\\,{stars(p_holm)} & "
                f"$\\delta_{{\\mathrm{{Cliff}}}}={r.get('cliffs_delta', float('nan')):.2f}$ "
                f"({r.get('cliff_magnitude','')}) \\\\"
            )
    return "\n".join(lines)


# ── Batch config ─────────────────────────────────────────────────────────────


def run_comparison_spec(spec: dict) -> dict:
    all_a = load_records(pathlib.Path(spec["file_a"]))
    all_b = load_records(pathlib.Path(spec["file_b"]))
    recs_a = apply_filters(all_a, spec.get("filters_a", []))
    recs_b = apply_filters(all_b, spec.get("filters_b", []))
    filter_diagnostics(all_a, recs_a, spec.get("filters_a", []), "A")
    filter_diagnostics(all_b, recs_b, spec.get("filters_b", []), "B")
    a = extract_metric(recs_a, spec["metric"])
    b = extract_metric(recs_b, spec["metric"])
    out = compare_groups(a, b)
    out["name"] = spec.get("name", f"{spec['file_a']} vs {spec['file_b']}")
    out["metric"] = spec["metric"]
    return out


# ── Self-test ────────────────────────────────────────────────────────────────


def self_test() -> int:
    """End-to-end check on synthetic fixtures mimicking evaluator output."""
    rng = np.random.default_rng(0)

    def records(mode, concept, deltas):
        return [
            {
                "song_name": f"s{i}",
                "mode": mode,
                "concept": concept,
                "alpha": 1.0,
                "quality": {"scale_consistency": 92.0 + rng.normal(0, 1)},
                "total_degradation": float(d),
            }
            for i, d in enumerate(deltas)
        ]

    shifted = records("abrupt", "average_pitch", rng.normal(3.0, 0.5, 40)) + \
        records("beatwise", "average_pitch", rng.normal(1.0, 0.5, 40))
    null = records("abrupt", "average_duration", rng.normal(2.0, 0.5, 40)) + \
        records("beatwise", "average_duration", rng.normal(2.0, 0.5, 40))

    with tempfile.TemporaryDirectory() as td:
        f1 = pathlib.Path(td) / "shifted.json"
        f2 = pathlib.Path(td) / "null.json"
        f1.write_text(json.dumps({"results": shifted}))
        f2.write_text(json.dumps(null))

        specs = [
            {"name": "shifted abrupt vs beatwise", "file_a": str(f1),
             "filters_a": ["mode=abrupt"], "file_b": str(f1),
             "filters_b": ["mode=beatwise"], "metric": "total_degradation"},
            {"name": "null abrupt vs beatwise", "file_a": str(f2),
             "filters_a": ["mode=abrupt"], "file_b": str(f2),
             "filters_b": ["mode=beatwise"], "metric": "total_degradation"},
            {"name": "flattened nested key", "file_a": str(f1),
             "filters_a": ["mode=abrupt"], "file_b": str(f1),
             "filters_b": ["mode=beatwise"],
             "metric": "quality.scale_consistency"},
        ]
        # Temporal-PID file schema: {method: {per_sample_metrics: [...]}} with
        # H/S/G fields and a *derived* delta (no precomputed total_degradation).
        def hsg(scale_mean, n):
            return {"per_sample_metrics": [
                {"n_notes": 200, "average_pitch": 60 + rng.normal(0, 2),
                 "average_duration": 19.0,
                 "pitch_class_entropy": 3.0 + rng.normal(0, 0.1),
                 "scale_consistency": scale_mean + rng.normal(0, 2),
                 "groove_consistency": 98 + rng.normal(0, 0.5)}
                for _ in range(n)]}
        f3 = pathlib.Path(td) / "temporal_comparison.json"
        f3.write_text(json.dumps(
            {"temporal_pid": hsg(84.7, 40), "static_sas": hsg(92.0, 40),
             "baseline": hsg(94.0, 40)}))

        results = [run_comparison_spec(s) for s in specs]
        holm_correction(results)
        print(report(results, latex=True))

        assert results[0]["p_holm"] < 0.001, "expected significant shift"
        assert results[0]["cliff_magnitude"] == "large"
        assert results[1]["p_holm"] > 0.05, "expected null to be ns"
        assert results[0]["n_a"] == results[0]["n_b"] == 40
        assert not math.isnan(results[2]["p"]), "nested metric extraction failed"

        # nested per_sample_metrics + derived delta (the temporal-PID schema)
        pid = run_comparison_spec(
            {"name": "PID vs static (derived delta)", "file_a": str(f3),
             "filters_a": ["group=temporal_pid"], "file_b": str(f3),
             "filters_b": ["group=static_sas"], "metric": "delta"})
        holm_correction([pid])
        print("\n" + report([pid], latex=False))
        assert pid["n_a"] == pid["n_b"] == 40, "nested schema not parsed"
        assert not math.isnan(pid["p"]), "derived delta not computed"
        assert pid["mean_a"] > pid["mean_b"], "derived delta direction wrong"
    print("\nself-test OK")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path,
                    help="JSON list of comparison specs (batch mode)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--a", type=pathlib.Path, help="results JSON for group A")
    ap.add_argument("--b", type=pathlib.Path, help="results JSON for group B")
    ap.add_argument("--filter-a", action="append", default=[], dest="filters_a")
    ap.add_argument("--filter-b", action="append", default=[], dest="filters_b")
    ap.add_argument("--metric", default="total_degradation")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX rows")
    ap.add_argument("--output", type=pathlib.Path, help="write JSON results here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.config:
        specs = json.loads(args.config.read_text())
    elif args.a and args.b:
        specs = [{
            "name": args.name or f"{args.a.name} vs {args.b.name} ({args.metric})",
            "file_a": str(args.a), "filters_a": args.filters_a,
            "file_b": str(args.b), "filters_b": args.filters_b,
            "metric": args.metric,
        }]
    else:
        ap.error("provide --config, or both --a and --b, or --self-test")

    results = [run_comparison_spec(s) for s in specs]
    holm_correction(results)
    print(report(results, latex=args.latex))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
