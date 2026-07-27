#!/usr/bin/env python3
"""Bootstrap confidence interval for the between-method FMD difference.

Addresses the reviewer request (JASMP review 4, Major Concern 9): FMD is a
distributional statistic and the shrinkage-estimator control (Sect. 7.7) shows
the SAS<DiffMean ordering is not a covariance artifact, but it does not quantify
*sampling* uncertainty in the difference. This script bootstraps

    Delta_FMD = FMD(ref, DiffMean) - FMD(ref, SAS)

by resampling the candidate generation sets and recomputing FMD each time,
reporting a 95% percentile CI. If Delta_FMD's CI excludes 0, SAS is closer to
the reference with bootstrap support.

Inputs are CLaMP 2 embedding matrices saved as .npy (shape [n_generations, d]),
which the existing FMD pipeline already produces as an intermediate
(metrics_evaluation/evaluate_fmd.py). Provide the reference-corpus embeddings and
the two candidate sets.

Usage (EC2, after extracting/saving CLaMP2 embeddings):
    python feedback_exp/fmd_bootstrap.py \
        --ref  path/to/ref_embeddings.npy \
        --a    path/to/diffmean_embeddings.npy \
        --b    path/to/sas_embeddings.npy \
        --estimator ledoit_wolf --n_boot 1000

Self-test on synthetic Gaussians (no data needed):
    python feedback_exp/fmd_bootstrap.py --self-test
"""

import argparse
import sys

import numpy as np

try:
    from scipy import linalg as sla
except ImportError:  # pragma: no cover
    sla = None


def _cov(x, estimator):
    if estimator == "mle":
        return np.cov(x, rowvar=False)
    from sklearn.covariance import LedoitWolf, OAS
    est = LedoitWolf() if estimator == "ledoit_wolf" else OAS()
    return est.fit(x).covariance_


def frechet_distance(mu_r, cov_r, mu_g, cov_g):
    """Standard Frechet distance between two Gaussians (as in FMD/FID)."""
    diff = mu_r - mu_g
    covmean, _ = sla.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


def fmd(ref_emb, gen_emb, estimator):
    mu_r, cov_r = ref_emb.mean(0), _cov(ref_emb, estimator)
    mu_g, cov_g = gen_emb.mean(0), _cov(gen_emb, estimator)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def bootstrap_delta(ref, a, b, estimator="ledoit_wolf", n_boot=1000, seed=0):
    """CI for FMD(ref,a) - FMD(ref,b), resampling a and b with replacement."""
    if sla is None:
        raise RuntimeError("scipy is required")
    rng = np.random.default_rng(seed)
    point = fmd(ref, a, estimator) - fmd(ref, b, estimator)
    deltas = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        ai = a[rng.integers(0, na, na)]
        bi = b[rng.integers(0, nb, nb)]
        deltas[i] = fmd(ref, ai, estimator) - fmd(ref, bi, estimator)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_point": point,
        "delta_mean": float(deltas.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "favors_b_if_positive": point > 0,
        "significant": (lo > 0) or (hi < 0),
        "estimator": estimator, "n_boot": n_boot,
        "n_a": na, "n_b": nb,
    }


def _report(r):
    verdict = ("SAS closer (CI excludes 0)" if r["significant"] and r["delta_point"] > 0
               else "DiffMean closer (CI excludes 0)" if r["significant"]
               else "difference not significant (CI includes 0)")
    print(f"Delta_FMD = FMD(ref,DiffMean) - FMD(ref,SAS)")
    print(f"  point   : {r['delta_point']:.2f}")
    print(f"  95% CI  : [{r['ci_low']:.2f}, {r['ci_high']:.2f}]  ({r['estimator']}, "
          f"n_boot={r['n_boot']}, n={r['n_a']}/{r['n_b']})")
    print(f"  verdict : {verdict}")
    print(f"% LaTeX: $\\Delta\\mathrm{{FMD}}={r['delta_point']:.1f}$ "
          f"(95\\% CI [{r['ci_low']:.1f}, {r['ci_high']:.1f}]) \\\\")


def self_test():
    rng = np.random.default_rng(0)
    d = 16
    ref = rng.normal(0, 1, (300, d))
    b = rng.normal(0.2, 1, (120, d))     # SAS: close to ref
    a = rng.normal(0.8, 1, (120, d))     # DiffMean: farther from ref
    r = bootstrap_delta(ref, a, b, estimator="mle", n_boot=300)
    _report(r)
    assert r["delta_point"] > 0, "expected DiffMean farther -> positive delta"
    assert r["significant"], "expected a clear separation to be significant"
    print("\nself-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="reference-corpus embeddings .npy")
    ap.add_argument("--a", help="method A (DiffMean) embeddings .npy")
    ap.add_argument("--b", help="method B (SAS) embeddings .npy")
    ap.add_argument("--estimator", default="ledoit_wolf",
                    choices=["mle", "ledoit_wolf", "oas"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.ref and args.a and args.b):
        ap.error("provide --ref, --a, --b (or --self-test)")
    r = bootstrap_delta(np.load(args.ref), np.load(args.a), np.load(args.b),
                        estimator=args.estimator, n_boot=args.n_boot)
    _report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
