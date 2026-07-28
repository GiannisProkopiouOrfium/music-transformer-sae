#!/usr/bin/env python3
"""Sensitivity of the corpus-relative structural deviation (delta) to weighting.

Addresses the JASMP reviewer request: delta = |H-H0| + max(0,S0-S) + max(0,G0-G)
sums terms with different units (bits vs. percentages), so it embeds an implicit
weighting. This script recomputes delta from per-sample records under three
schemes and checks whether the *method ranking* is preserved:

  raw     : the paper's formula (unit sum).
  zscore  : each term divided by its corpus-level standard deviation before
            summing (equalizes scales); SDs are estimated from the pooled
            per-sample records unless provided via --sd H,S,G.
  reweight : user weights w_H,w_S,w_G on the three (raw) terms.

For each scheme it prints the per-method mean delta and the induced ranking, and
reports whether the ranking is identical to the raw ranking (Kendall tau = 1).

Input: one or more results JSONs, each a group of per-sample records with
pitch_class_entropy, scale_consistency, groove_consistency (0-100 or 0-1). Use
--group to name each file's method; nested {method: {per_sample_metrics: [...]}}
files are auto-split by group (reusing significance_tests loaders).

Usage:
    python feedback_exp/delta_sensitivity.py \
        --file diffmean.json:DiffMean --file sas.json:SAS \
        --weights 1,1,1 --weights 1,2,2 --weights 2,1,1

    # nested temporal-PID file (auto group split):
    python feedback_exp/delta_sensitivity.py --file temporal_comparison_*.json

    python feedback_exp/delta_sensitivity.py --self-test
"""

import argparse
import json
import math
import pathlib
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from significance_tests import load_records, apply_filters  # noqa: E402

H0, S0, G0 = 2.974, 92.26, 93.05


def _find(r, name):
    """Resolve a metric that may be top-level or nested (flattened as quality.*)."""
    for k in (name, f"quality.{name}", f"metrics.{name}", f"quality_metrics.{name}"):
        if k in r and r[k] is not None:
            return r[k]
    return None


def _hsg(r):
    try:
        H = float(_find(r, "pitch_class_entropy"))
        S = float(_find(r, "scale_consistency"))
        G = float(_find(r, "groove_consistency"))
    except (KeyError, TypeError, ValueError):
        return None
    if S <= 1.0:
        S *= 100.0
    if G <= 1.0:
        G *= 100.0
    return H, S, G


def _terms(r):
    hsg = _hsg(r)
    if hsg is None:
        return None
    H, S, G = hsg
    return abs(H - H0), max(0.0, S0 - S), max(0.0, G0 - G)


def delta_raw(t):
    return t[0] + t[1] + t[2]


def delta_zscore(t, sd):
    return sum(ti / si if si > 0 else 0.0 for ti, si in zip(t, sd))


def delta_reweight(t, w):
    return sum(wi * ti for wi, ti in zip(t, w))


def ranking(method_means):
    """Return method names ordered by ascending mean delta (best first)."""
    return [m for m, _ in sorted(method_means.items(), key=lambda kv: kv[1])]


def evaluate(groups, weight_schemes):
    """groups: {method: [term_tuples]}. Returns per-scheme means + rankings."""
    # SDs for z-scoring: std of each term across all pooled records.
    pooled = np.array([t for terms in groups.values() for t in terms])
    sd = pooled.std(axis=0, ddof=1) if len(pooled) > 1 else np.ones(3)

    schemes = {"raw": lambda t: delta_raw(t),
               "zscore": lambda t: delta_zscore(t, sd)}
    for w in weight_schemes:
        schemes[f"reweight{w}"] = (lambda t, w=w: delta_reweight(t, w))

    out = {}
    for name, fn in schemes.items():
        means = {m: float(np.mean([fn(t) for t in terms]))
                 for m, terms in groups.items()}
        out[name] = {"means": means, "ranking": ranking(means)}
    return out, sd.tolist()


def report(out, sd):
    raw_rank = out["raw"]["ranking"]
    lines = [f"z-score term SDs (H,S,G): "
             f"{sd[0]:.3f}, {sd[1]:.3f}, {sd[2]:.3f}", ""]
    header = f"{'scheme':<14} {'ranking (best->worst)':<40} {'== raw?':>8}"
    lines += [header, "-" * len(header)]
    all_preserved = True
    for name, d in out.items():
        preserved = d["ranking"] == raw_rank
        all_preserved &= preserved
        means = "  ".join(f"{m}={v:.2f}" for m, v in
                          sorted(d["means"].items(), key=lambda kv: kv[1]))
        lines.append(f"{name:<14} {means:<40} {'yes' if preserved else 'NO':>8}")
    lines.append("")
    lines.append(f"All schemes preserve the raw ranking: "
                 f"{'YES' if all_preserved else 'NO'}")
    return "\n".join(lines), all_preserved


def build_groups(file_specs):
    groups = defaultdict(list)
    for spec in file_specs:
        path, _, label = spec.partition(":")
        recs = load_records(pathlib.Path(path))
        if label:  # single named method
            for r in recs:
                t = _terms(r)
                if t:
                    groups[label].append(t)
        else:  # split by 'group' field (nested files)
            by = defaultdict(list)
            for r in recs:
                by[str(r.get("group", pathlib.Path(path).stem))].append(r)
            for g, rs in by.items():
                for r in rs:
                    t = _terms(r)
                    if t:
                        groups[g].append(t)
    return groups


def self_test():
    rng = np.random.default_rng(0)

    def recs(scale_mean, n):
        return [{"pitch_class_entropy": 2.9 + rng.normal(0, 0.1),
                 "scale_consistency": scale_mean + rng.normal(0, 2),
                 "groove_consistency": 95 + rng.normal(0, 1)} for _ in range(n)]

    with tempfile.TemporaryDirectory() as td:
        # SAS clearly better (scale near corpus), DiffMean worse (lower scale)
        f1 = pathlib.Path(td) / "sas.json"
        f2 = pathlib.Path(td) / "dm.json"
        f1.write_text(json.dumps(recs(92.0, 40)))
        f2.write_text(json.dumps(recs(84.0, 40)))
        groups = build_groups([f"{f1}:SAS", f"{f2}:DiffMean"])
        out, sd = evaluate(groups, [(1, 1, 1), (1, 2, 2), (2, 1, 1)])
        text, preserved = report(out, sd)
        print(text)
        assert out["raw"]["ranking"][0] == "SAS"
        assert preserved, "expected a clear separation to survive reweighting"
    print("\nself-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="path[:MethodName] (repeatable)")
    ap.add_argument("--weights", action="append", default=[],
                    help="reweight scheme wH,wS,wG (repeatable)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.file:
        ap.error("provide --file path[:Name] (or --self-test)")
    weights = [tuple(float(x) for x in w.split(",")) for w in args.weights] \
        or [(1, 2, 2), (2, 1, 1)]
    groups = build_groups(args.file)
    out, sd = evaluate(groups, weights)
    text, _ = report(out, sd)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
