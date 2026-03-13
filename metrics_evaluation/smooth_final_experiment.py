#!/usr/bin/env python3
"""Smooth vs Abrupt Steering — Final Statistical Significance Experiment.

Runs warmup_hold (cosine S-curve ramp) with n_ramp=32 vs abrupt for
both DM and SAS on pitch and duration concepts.

Round 1 (n_ramp=128) showed:
  - Ramp shape doesn't matter (gradual ≈ warmup_hold)
  - 128 was too long: ~24% magnitude loss for DM, catastrophic for SAS
  - No significant degradation reduction for any combo

Round 2 tests n_ramp=32 (224/256 = 88% of tokens at full strength).
Short enough to preserve steering power, long enough to smooth onset.
If 32 doesn't help, the conclusion is clean: smooth ramping does not
improve the quality-effectiveness tradeoff.

Reuses existing abrupt baselines from the same output directory.
4 new smooth runs only (2 methods × 2 concepts × 1 ramp length).

Statistical tests (paired by song × strength):
  - Paired t-test & Wilcoxon signed-rank on |change| and degradation
  - McNemar's test on binary steering success
  - Cohen's d effect sizes
  - Bootstrap 95% CIs for mean differences

Usage
-----
    # Full run (generate + analyze, reuses existing abrupt):
    python metrics_evaluation/smooth_final_experiment.py --n_songs 15 --gpu 0

    # Analyze all existing data in the output dir:
    python metrics_evaluation/smooth_final_experiment.py --analyze_only

    # Generate only (analyze later):
    python metrics_evaluation/smooth_final_experiment.py --n_songs 15 --gpu 0 \
        --generate_only
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Known concepts ───────────────────────────────────────────────────────────
KNOWN_CONCEPTS = ["average_pitch", "average_duration"]

# ── Smooth configs to compare against abrupt ─────────────────────────────────
# warmup_hold only (round 1: gradual ≈ warmup_hold), n_ramp=32 only.
# Abrupt baselines already exist → only 4 new smooth evaluator runs.
SMOOTH_CONFIGS = [
    {"method": "dm",  "concept": "average_pitch",    "mode": "warmup_hold", "n_ramp": 32},
    {"method": "dm",  "concept": "average_duration",  "mode": "warmup_hold", "n_ramp": 32},
    {"method": "sas", "concept": "average_pitch",    "mode": "warmup_hold", "n_ramp": 32},
    {"method": "sas", "concept": "average_duration",  "mode": "warmup_hold", "n_ramp": 32},
]

DM_ALPHAS = "0.0,0.5,1.0,1.5,2.0,-0.5,-1.0,-1.5,-2.0"
SAS_LAMBDAS = "0.0,0.5,1.0,1.5,2.0,-0.5,-1.0,-1.5,-2.0"


# ── Experiment runner ────────────────────────────────────────────────────────


def run_evaluator(
    method: str,
    concept: str,
    n_songs: int,
    output_dir: pathlib.Path,
    experiment_name: str,
    gpu: Optional[int] = None,
    smooth: bool = False,
    mode: Optional[str] = None,
    n_ramp: Optional[int] = None,
    skip_wav: bool = True,
    alphas: str = DM_ALPHAS,
    lambdas: str = SAS_LAMBDAS,
) -> pathlib.Path:
    """Run a DM or SAS evaluator subprocess. Returns path to results JSON."""
    if method == "dm":
        evaluator = REPO_ROOT / "steering_interventions" / "conditioned_evaluator.py"
        results_file = output_dir / experiment_name / "conditioned_results.json"
        strength_arg = f"--alphas={alphas}"
    else:
        evaluator = REPO_ROOT / "sparse_steering" / "conditioned_evaluator_sas.py"
        results_file = (
            output_dir / experiment_name / f"conditioned_results_{concept}.json"
        )
        strength_arg = f"--lambdas={lambdas}"

    if results_file.exists():
        logger.info(f"  Skipping (exists): {results_file}")
        return results_file

    cmd = [
        sys.executable,
        str(evaluator),
        "--concept",
        concept,
        "--n_songs",
        str(n_songs),
        strength_arg,
        "--conditioning_beats",
        "16",
        "--continuation_len",
        "256",
        "--output_dir",
        str(output_dir),
        "--experiment_name",
        experiment_name,
    ]
    if method == "sas":
        cmd += ["--layers", "10"]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if skip_wav:
        cmd += ["--skip_wav"]
    if smooth:
        cmd += [
            "--smooth",
            "--mode",
            mode,
            "--schedule",
            "cosine",
            "--n_ramp",
            str(n_ramp),
        ]

    logger.info(f"  Running: {experiment_name}")
    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"  FAILED: {experiment_name} (exit code {result.returncode})")
    return results_file


def generate_experiments(configs, n_songs, output_dir, gpu, skip_wav, alphas, lambdas):
    """Run abrupt + smooth modes for each method×concept combo.

    Abrupt baselines are shared: one abrupt run per unique (method, concept).
    """
    # Run abrupt baselines first (deduplicated by method×concept)
    done_abrupt = set()
    for cfg in configs:
        method, concept = cfg["method"], cfg["concept"]
        key = (method, concept)
        if key in done_abrupt:
            continue
        done_abrupt.add(key)

        logger.info(f"\n{'='*60}")
        logger.info(f"Abrupt baseline: {method.upper()} × {concept}")
        logger.info(f"{'='*60}")

        abrupt_name = f"{method}_{concept}_abrupt"
        run_evaluator(
            method,
            concept,
            n_songs,
            output_dir,
            abrupt_name,
            gpu=gpu,
            smooth=False,
            skip_wav=skip_wav,
            alphas=alphas,
            lambdas=lambdas,
        )

    # Run each smooth config
    for cfg in configs:
        method, concept = cfg["method"], cfg["concept"]
        mode, n_ramp = cfg["mode"], cfg["n_ramp"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Smooth: {method.upper()} × {concept} × {mode}_{n_ramp}")
        logger.info(f"{'='*60}")

        smooth_name = f"{method}_{concept}_{mode}_{n_ramp}"
        run_evaluator(
            method,
            concept,
            n_songs,
            output_dir,
            smooth_name,
            gpu=gpu,
            smooth=True,
            mode=mode,
            n_ramp=n_ramp,
            skip_wav=skip_wav,
            alphas=alphas,
            lambdas=lambdas,
        )


# ── Auto-detection ───────────────────────────────────────────────────────────


def parse_experiment_name(name: str):
    """Parse directory name into (method, concept, mode_label, is_abrupt).

    E.g. 'dm_average_pitch_gradual_192' → ('dm', 'average_pitch', 'gradual_192', False)
    """
    if name.startswith("dm_"):
        method = "dm"
        rest = name[3:]
    elif name.startswith("sas_"):
        method = "sas"
        rest = name[4:]
    else:
        return None

    concept = None
    for c in KNOWN_CONCEPTS:
        if rest.startswith(c + "_"):
            concept = c
            mode_info = rest[len(c) + 1 :]
            break
        elif rest == c:
            concept = c
            mode_info = "abrupt"
            break

    if concept is None:
        return None

    is_abrupt = mode_info == "abrupt"
    return method, concept, mode_info, is_abrupt


def results_path_for(exp_dir: pathlib.Path, method: str, concept: str) -> pathlib.Path:
    """Return the expected results JSON path for an experiment."""
    if method == "dm":
        return exp_dir / "conditioned_results.json"
    return exp_dir / f"conditioned_results_{concept}.json"


def auto_detect_pairs(output_dir: pathlib.Path) -> List[dict]:
    """Scan output_dir, find (abrupt, smooth) pairs grouped by method+concept.

    Returns a list of dicts, one per pair:
      {method, concept, smooth_label, abrupt_dir, smooth_dir,
       abrupt_file, smooth_file}
    """
    # Index all experiment dirs
    experiments = (
        {}
    )  # (method, concept) → {"abrupt": dir, "smooth": [(label, dir), ...]}

    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_experiment_name(subdir.name)
        if parsed is None:
            continue

        method, concept, mode_label, is_abrupt = parsed
        key = (method, concept)

        if key not in experiments:
            experiments[key] = {"abrupt": None, "smooth": []}

        if is_abrupt:
            experiments[key]["abrupt"] = subdir
        else:
            experiments[key]["smooth"].append((mode_label, subdir))

    # Build pairs
    pairs = []
    for (method, concept), info in sorted(experiments.items()):
        if info["abrupt"] is None:
            logger.warning(f"  No abrupt experiment for {method}×{concept} — skipping")
            continue

        abrupt_dir = info["abrupt"]
        abrupt_file = results_path_for(abrupt_dir, method, concept)
        if not abrupt_file.exists():
            logger.warning(f"  Missing abrupt results: {abrupt_file}")
            continue

        for smooth_label, smooth_dir in info["smooth"]:
            smooth_file = results_path_for(smooth_dir, method, concept)
            if not smooth_file.exists():
                logger.warning(f"  Missing smooth results: {smooth_file}")
                continue

            pairs.append(
                {
                    "method": method,
                    "concept": concept,
                    "smooth_label": smooth_label,
                    "abrupt_file": abrupt_file,
                    "smooth_file": smooth_file,
                }
            )

    return pairs


# ── Statistical analysis ─────────────────────────────────────────────────────


def load_json(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def match_samples(
    abrupt_data: dict,
    smooth_data: dict,
    method: str,
) -> List[Tuple[dict, dict, str, float]]:
    """Match samples by (song_name, strength). Returns (abrupt, smooth, song, strength)."""
    strength_key = "alpha" if method == "dm" else "lambda"

    abrupt_idx = {}
    for r in abrupt_data.get("results", []):
        k = (r.get("song_name", ""), r.get(strength_key, 0))
        abrupt_idx[k] = r

    matched = []
    for r in smooth_data.get("results", []):
        k = (r.get("song_name", ""), r.get(strength_key, 0))
        if k in abrupt_idx:
            matched.append((abrupt_idx[k], r, k[0], k[1]))

    return matched


def run_significance_tests(
    matched: List[Tuple[dict, dict, str, float]],
    concept: str,
) -> dict:
    """Run paired significance tests on matched (abrupt, smooth) samples."""
    from scipy import stats

    change_key = "duration_change" if "duration" in concept else "pitch_change"

    # Collect paired arrays (exclude baseline strength=0)
    a_mag, s_mag = [], []
    a_deg, s_deg = [], []
    a_succ, s_succ = [], []
    a_ent, s_ent = [], []
    a_scale, s_scale = [], []
    a_groove, s_groove = [], []
    pair_strengths = []

    for a, s, song, strength in matched:
        if abs(strength) < 0.01:
            continue

        a_mag.append(abs(a.get(change_key, 0)))
        s_mag.append(abs(s.get(change_key, 0)))

        ad = a.get("degradation", {}).get("total_degradation", np.nan)
        sd = s.get("degradation", {}).get("total_degradation", np.nan)
        if not (np.isnan(ad) or np.isnan(sd)):
            a_deg.append(ad)
            s_deg.append(sd)

        # Individual degradation components
        for arr_a, arr_s, key in [
            (a_ent, s_ent, "entropy_diff"),
            (a_scale, s_scale, "scale_diff"),
            (a_groove, s_groove, "groove_diff"),
        ]:
            av = a.get("degradation", {}).get(key, np.nan)
            sv = s.get("degradation", {}).get(key, np.nan)
            if not (np.isnan(av) or np.isnan(sv)):
                arr_a.append(av)
                arr_s.append(sv)

        # Success: change in the intended direction
        a_dir = a.get(change_key, 0)
        s_dir = s.get(change_key, 0)
        a_succ.append(
            1 if (strength > 0 and a_dir > 0) or (strength < 0 and a_dir < 0) else 0
        )
        s_succ.append(
            1 if (strength > 0 and s_dir > 0) or (strength < 0 and s_dir < 0) else 0
        )
        pair_strengths.append(strength)

    result = {"n_pairs": len(a_mag), "n_pairs_with_degradation": len(a_deg)}
    if len(a_mag) < 3:
        return result

    rng = np.random.default_rng(42)
    a_mag = np.array(a_mag)
    s_mag = np.array(s_mag)
    a_succ = np.array(a_succ)
    s_succ = np.array(s_succ)

    def cohens_d(x, y):
        pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
        return float(np.mean(x - y) / pooled) if pooled > 0 else 0.0

    def bootstrap_ci(diff, n_boot=10000, alpha=0.05):
        boots = np.array(
            [
                np.mean(rng.choice(diff, size=len(diff), replace=True))
                for _ in range(n_boot)
            ]
        )
        return [
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))),
        ]

    def safe_wilcoxon(diff, alternative="two-sided"):
        try:
            w, p = stats.wilcoxon(diff, alternative=alternative)
            return float(w), float(p)
        except ValueError:
            return np.nan, 1.0

    # ── Magnitude ────────────────────────────────────────────────────────
    diff_mag = s_mag - a_mag
    t_mag, p_mag = stats.ttest_rel(s_mag, a_mag)
    w_mag, wp_mag = safe_wilcoxon(diff_mag)

    result["magnitude"] = {
        "abrupt_mean": float(np.mean(a_mag)),
        "abrupt_std": float(np.std(a_mag, ddof=1)),
        "smooth_mean": float(np.mean(s_mag)),
        "smooth_std": float(np.std(s_mag, ddof=1)),
        "mean_diff": float(np.mean(diff_mag)),
        "retention_pct": (
            float(np.mean(s_mag) / np.mean(a_mag) * 100)
            if np.mean(a_mag) > 0
            else np.nan
        ),
        "paired_ttest": {"t": float(t_mag), "p": float(p_mag)},
        "wilcoxon": {"W": w_mag, "p": wp_mag},
        "cohens_d": cohens_d(s_mag, a_mag),
        "bootstrap_ci_95": bootstrap_ci(diff_mag),
    }

    # ── Degradation ──────────────────────────────────────────────────────
    if len(a_deg) >= 3:
        a_d = np.array(a_deg)
        s_d = np.array(s_deg)
        diff_deg = s_d - a_d
        t_deg, p_deg_2 = stats.ttest_rel(s_d, a_d)
        p_deg_1 = p_deg_2 / 2 if t_deg < 0 else 1 - p_deg_2 / 2
        w_deg, wp_deg = safe_wilcoxon(diff_deg, alternative="less")

        result["degradation"] = {
            "abrupt_mean": float(np.mean(a_d)),
            "abrupt_std": float(np.std(a_d, ddof=1)),
            "smooth_mean": float(np.mean(s_d)),
            "smooth_std": float(np.std(s_d, ddof=1)),
            "mean_diff": float(np.mean(diff_deg)),
            "reduction_pct": (
                float((1 - np.mean(s_d) / np.mean(a_d)) * 100)
                if np.mean(a_d) > 0
                else np.nan
            ),
            "paired_ttest_2sided": {"t": float(t_deg), "p": float(p_deg_2)},
            "paired_ttest_1sided": {"t": float(t_deg), "p": float(p_deg_1)},
            "wilcoxon_less": {"W": w_deg, "p": wp_deg},
            "cohens_d": cohens_d(s_d, a_d),
            "bootstrap_ci_95": bootstrap_ci(diff_deg),
        }

    # ── Degradation components ───────────────────────────────────────────
    components = {}
    for label, arr_a, arr_s in [
        ("entropy_diff", a_ent, s_ent),
        ("scale_diff", a_scale, s_scale),
        ("groove_diff", a_groove, s_groove),
    ]:
        if len(arr_a) >= 3:
            xa, xs = np.array(arr_a), np.array(arr_s)
            t, p2 = stats.ttest_rel(xs, xa)
            p1 = p2 / 2 if t < 0 else 1 - p2 / 2
            components[label] = {
                "abrupt_mean": float(np.mean(xa)),
                "smooth_mean": float(np.mean(xs)),
                "mean_diff": float(np.mean(xs - xa)),
                "paired_ttest_1sided_p": float(p1),
            }
    if components:
        result["degradation_components"] = components

    # ── Success rate ─────────────────────────────────────────────────────
    sr_a = float(np.mean(a_succ))
    sr_s = float(np.mean(s_succ))
    b = int(np.sum((a_succ == 1) & (s_succ == 0)))  # only abrupt succeeded
    c = int(np.sum((a_succ == 0) & (s_succ == 1)))  # only smooth succeeded
    if b + c > 0:
        chi2 = (b - c) ** 2 / (b + c)
        p_mcnemar = float(1 - stats.chi2.cdf(chi2, df=1))
    else:
        chi2, p_mcnemar = 0.0, 1.0

    result["success_rate"] = {
        "abrupt": sr_a,
        "smooth": sr_s,
        "diff": sr_s - sr_a,
        "mcnemar": {
            "discordant_ab_only": b,
            "discordant_sm_only": c,
            "chi2": float(chi2),
            "p": p_mcnemar,
        },
    }

    # ── Per-strength breakdown (descriptive only) ────────────────────────
    by_str = defaultdict(lambda: {"a_mag": [], "s_mag": [], "a_deg": [], "s_deg": []})
    for a, s, song, strength in matched:
        if abs(strength) < 0.01:
            continue
        by_str[strength]["a_mag"].append(abs(a.get(change_key, 0)))
        by_str[strength]["s_mag"].append(abs(s.get(change_key, 0)))
        ad = a.get("degradation", {}).get("total_degradation", np.nan)
        sd = s.get("degradation", {}).get("total_degradation", np.nan)
        if not (np.isnan(ad) or np.isnan(sd)):
            by_str[strength]["a_deg"].append(ad)
            by_str[strength]["s_deg"].append(sd)

    per_strength = {}
    for strength in sorted(by_str.keys()):
        d = by_str[strength]
        n = len(d["a_mag"])
        am_arr = np.array(d["a_mag"])
        sm_arr = np.array(d["s_mag"])
        row = {
            "n_pairs": n,
            "abrupt_mag": float(np.mean(am_arr)),
            "smooth_mag": float(np.mean(sm_arr)),
            "retention_pct": (
                float(np.mean(sm_arr) / np.mean(am_arr) * 100)
                if np.mean(am_arr) > 0
                else np.nan
            ),
        }
        if d["a_deg"]:
            ad_arr = np.array(d["a_deg"])
            sd_arr = np.array(d["s_deg"])
            row["abrupt_deg"] = float(np.mean(ad_arr))
            row["smooth_deg"] = float(np.mean(sd_arr))
            row["deg_reduction_pct"] = (
                float((1 - np.mean(sd_arr) / np.mean(ad_arr)) * 100)
                if np.mean(ad_arr) > 0
                else np.nan
            )
        per_strength[f"{strength:+.2f}"] = row

    result["per_strength"] = per_strength
    return result


# ── Reporting ────────────────────────────────────────────────────────────────


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_report(all_analyses: Dict[str, dict]) -> str:
    """Format a human-readable significance report."""
    lines = []
    sep = "=" * 100

    lines.append(sep)
    lines.append("SMOOTH vs ABRUPT STEERING — PAIRED SIGNIFICANCE TESTS")
    lines.append(sep)
    lines.append("")

    for label, analysis in all_analyses.items():
        smooth_lbl = analysis.get("_smooth_label", "?")
        method = analysis.get("_method", "?").upper()
        concept = analysis.get("_concept", "?")
        n = analysis.get("n_pairs", 0)

        lines.append(f"── {method} × {concept} ── ({smooth_lbl} vs abrupt)")
        lines.append(f"   Paired samples (excl. baseline): {n}")
        lines.append("")

        if n < 3:
            lines.append("   ⚠ Too few pairs for significance testing\n")
            lines.append("-" * 100 + "\n")
            continue

        # Magnitude
        m = analysis.get("magnitude", {})
        if m:
            ci = m.get("bootstrap_ci_95", [0, 0])
            p_w = m.get("wilcoxon", {}).get("p", 1)
            lines.append("   STEERING MAGNITUDE (|change|):")
            lines.append(f"     Abrupt: {m['abrupt_mean']:.2f} ± {m['abrupt_std']:.2f}")
            lines.append(f"     Smooth: {m['smooth_mean']:.2f} ± {m['smooth_std']:.2f}")
            lines.append(f"     Retention: {m['retention_pct']:.1f}%")
            lines.append(
                f"     Mean diff: {m['mean_diff']:+.2f}  "
                f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
            )
            lines.append(
                f"     Paired t: t={m['paired_ttest']['t']:.3f}, "
                f"p={m['paired_ttest']['p']:.4f} "
                f"{sig_stars(m['paired_ttest']['p'])}"
            )
            lines.append(
                f"     Wilcoxon: W={m['wilcoxon']['W']:.0f}, "
                f"p={p_w:.4f} {sig_stars(p_w)}"
            )
            lines.append(f"     Cohen's d: {m['cohens_d']:.3f}")
            lines.append("")

        # Degradation
        d = analysis.get("degradation", {})
        if d:
            ci = d.get("bootstrap_ci_95", [0, 0])
            p1 = d.get("paired_ttest_1sided", {}).get("p", 1)
            pw = d.get("wilcoxon_less", {}).get("p", 1)
            lines.append("   QUALITY DEGRADATION:")
            lines.append(f"     Abrupt: {d['abrupt_mean']:.2f} ± {d['abrupt_std']:.2f}")
            lines.append(f"     Smooth: {d['smooth_mean']:.2f} ± {d['smooth_std']:.2f}")
            lines.append(f"     Reduction: {d['reduction_pct']:.1f}%")
            lines.append(
                f"     Mean diff: {d['mean_diff']:+.2f}  "
                f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
            )
            lines.append(
                f"     Paired t (1-sided): t={d['paired_ttest_1sided']['t']:.3f}, "
                f"p={p1:.4f} {sig_stars(p1)}"
            )
            lines.append(
                f"     Wilcoxon (H1: smooth < abrupt): "
                f"W={d['wilcoxon_less']['W']:.0f}, p={pw:.4f} {sig_stars(pw)}"
            )
            lines.append(f"     Cohen's d: {d['cohens_d']:.3f}")
            lines.append("")

        # Degradation components
        dc = analysis.get("degradation_components", {})
        if dc:
            lines.append("   DEGRADATION COMPONENTS (1-sided p: smooth < abrupt):")
            for comp, vals in dc.items():
                p1c = vals["paired_ttest_1sided_p"]
                lines.append(
                    f"     {comp:<14} abrupt={vals['abrupt_mean']:.2f}  "
                    f"smooth={vals['smooth_mean']:.2f}  "
                    f"diff={vals['mean_diff']:+.2f}  "
                    f"p={p1c:.4f} {sig_stars(p1c)}"
                )
            lines.append("")

        # Success rate
        sr = analysis.get("success_rate", {})
        if sr:
            mc = sr.get("mcnemar", {})
            lines.append("   STEERING SUCCESS RATE:")
            lines.append(f"     Abrupt: {sr['abrupt']*100:.1f}%")
            lines.append(f"     Smooth: {sr['smooth']*100:.1f}%")
            lines.append(f"     Diff:   {sr['diff']*100:+.1f}pp")
            lines.append(
                f"     McNemar: chi2={mc.get('chi2', 0):.2f}, "
                f"p={mc.get('p', 1):.4f} {sig_stars(mc.get('p', 1))}"
            )
            lines.append("")

        # Per-strength summary
        ps = analysis.get("per_strength", {})
        if ps:
            lines.append("   PER-STRENGTH BREAKDOWN:")
            lines.append(
                f"   {'Str':>7} {'N':>5} {'Abr |Δ|':>10} {'Sm |Δ|':>10} "
                f"{'Retain':>8} {'Abr Deg':>9} {'Sm Deg':>9} {'Deg↓':>8}"
            )
            lines.append(f"   {'-'*75}")
            for s_str, row in sorted(ps.items(), key=lambda x: float(x[0])):
                ret = (
                    f"{row['retention_pct']:.0f}%"
                    if not np.isnan(row.get("retention_pct", np.nan))
                    else "N/A"
                )
                a_d = f"{row['abrupt_deg']:.1f}" if "abrupt_deg" in row else "N/A"
                s_d = f"{row['smooth_deg']:.1f}" if "smooth_deg" in row else "N/A"
                dr = (
                    f"{row['deg_reduction_pct']:.0f}%"
                    if "deg_reduction_pct" in row
                    and not np.isnan(row.get("deg_reduction_pct", np.nan))
                    else "N/A"
                )
                lines.append(
                    f"   {s_str:>7} {row['n_pairs']:>5} "
                    f"{row['abrupt_mag']:>10.2f} {row['smooth_mag']:>10.2f} "
                    f"{ret:>8} {a_d:>9} {s_d:>9} {dr:>8}"
                )
            lines.append("")

        lines.append("-" * 100)
        lines.append("")

    # ── Summary table ────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("SUMMARY TABLE")
    lines.append(sep)
    lines.append(
        f"{'Config':<35} {'N':>5} {'Retain':>8} {'Deg↓':>8} "
        f"{'p(mag)':>10} {'p(deg)':>10} {'Verdict'}"
    )
    lines.append("-" * 100)

    for label, analysis in all_analyses.items():
        method = analysis.get("_method", "?").upper()
        concept_short = analysis.get("_concept", "?").replace("average_", "")
        smooth_lbl = analysis.get("_smooth_label", "?")
        config_str = f"{method} {concept_short} {smooth_lbl}"

        n = analysis.get("n_pairs", 0)
        m = analysis.get("magnitude", {})
        d = analysis.get("degradation", {})

        ret = f"{m.get('retention_pct', 0):.0f}%" if m else "N/A"
        dr = (
            f"{d.get('reduction_pct', 0):.0f}%"
            if d and not np.isnan(d.get("reduction_pct", np.nan))
            else "N/A"
        )
        p_m = m.get("wilcoxon", {}).get("p", 1) if m else 1
        p_d = d.get("wilcoxon_less", {}).get("p", 1) if d else 1

        # Verdict logic
        deg_reduced = d and d.get("reduction_pct", 0) > 0 and p_d < 0.05
        retain_high = m and m.get("retention_pct", 0) > 70
        retain_mid = m and m.get("retention_pct", 0) > 50

        if deg_reduced and retain_high:
            verdict = "EFFECTIVE"
        elif deg_reduced and retain_mid:
            verdict = "PARTIAL"
        elif deg_reduced:
            verdict = "TOO COSTLY"
        elif m and p_m > 0.05:
            verdict = "NO SIG DIFF"
        else:
            verdict = "NOT EFFECTIVE"

        lines.append(
            f"{config_str:<35} {n:>5} {ret:>8} {dr:>8} "
            f"{p_m:>9.4f}{sig_stars(p_m)} "
            f"{p_d:>9.4f}{sig_stars(p_d)} "
            f"{verdict}"
        )

    lines.append(sep)
    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    lines.append("p(mag): Wilcoxon two-sided (H0: no difference in |change|)")
    lines.append("p(deg): Wilcoxon one-sided (H1: smooth degradation < abrupt)")
    lines.append(
        "Verdict: EFFECTIVE = deg↓ sig + retain>70% | "
        "PARTIAL = deg↓ sig + retain>50% | "
        "TOO COSTLY = deg↓ sig + retain<50%"
    )
    lines.append("")

    return "\n".join(lines)


# ── Plotting ─────────────────────────────────────────────────────────────────


def generate_plots(all_analyses: Dict[str, dict], output_dir: pathlib.Path):
    """Generate comparison bar charts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    # Filter to analyses with data
    valid = {k: v for k, v in all_analyses.items() if v.get("n_pairs", 0) >= 3}
    if not valid:
        return

    n = len(valid)
    _, axes = plt.subplots(1, 2, figsize=(6 + 2 * n, 6))

    labels = []
    abrupt_mags, smooth_mags = [], []
    abrupt_degs, smooth_degs = [], []

    for label, analysis in valid.items():
        method = analysis.get("_method", "?").upper()
        concept_short = analysis.get("_concept", "?").replace("average_", "")
        smooth_lbl = analysis.get("_smooth_label", "?")
        labels.append(f"{method}\n{concept_short}\n{smooth_lbl}")

        m = analysis.get("magnitude", {})
        abrupt_mags.append(m.get("abrupt_mean", 0))
        smooth_mags.append(m.get("smooth_mean", 0))

        d = analysis.get("degradation", {})
        abrupt_degs.append(d.get("abrupt_mean", 0))
        smooth_degs.append(d.get("smooth_mean", 0))

    x = np.arange(len(labels))
    width = 0.35

    # Magnitude panel
    ax = axes[0]
    ax.bar(x - width / 2, abrupt_mags, width, label="Abrupt", color="#FF9800")
    ax.bar(x + width / 2, smooth_mags, width, label="Smooth", color="#4CAF50")

    # Add significance markers
    for i, (label_key, analysis) in enumerate(valid.items()):
        p = analysis.get("magnitude", {}).get("wilcoxon", {}).get("p", 1)
        stars = sig_stars(p)
        if stars != "ns":
            max_h = max(abrupt_mags[i], smooth_mags[i])
            ax.text(i, max_h * 1.05, stars, ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("|Change| (steering magnitude)")
    ax.set_title("Steering Magnitude: Abrupt vs Smooth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Degradation panel
    ax = axes[1]
    ax.bar(x - width / 2, abrupt_degs, width, label="Abrupt", color="#FF9800")
    ax.bar(x + width / 2, smooth_degs, width, label="Smooth", color="#4CAF50")

    for i, (label_key, analysis) in enumerate(valid.items()):
        p = analysis.get("degradation", {}).get("wilcoxon_less", {}).get("p", 1)
        stars = sig_stars(p)
        if stars != "ns":
            max_h = max(abrupt_degs[i], smooth_degs[i])
            ax.text(i, max_h * 1.05, stars, ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Total Degradation")
    ax.set_title("Quality Degradation: Abrupt vs Smooth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Smooth vs Abrupt Steering — Significance Comparison",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / "significance_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {plot_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Smooth vs Abrupt — final significance experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_songs", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT / "exp" / "sod" / "smooth_final",
    )
    parser.add_argument(
        "--skip_wav",
        action="store_true",
        default=True,
        help="Skip WAV generation (default: True)",
    )
    parser.add_argument(
        "--with_wav",
        action="store_true",
        help="Generate WAV files (overrides --skip_wav)",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Skip generation, only analyze existing results",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Run generation only, skip analysis",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=DM_ALPHAS,
        help="DiffMean alpha values",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default=SAS_LAMBDAS,
        help="SAS lambda values",
    )
    args = parser.parse_args()

    skip_wav = not args.with_wav
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Generate ────────────────────────────────────────────────
    if not args.analyze_only:
        logger.info("=" * 60)
        logger.info("PHASE 1: GENERATING EXPERIMENTS")
        logger.info(f"  n_songs={args.n_songs}, output_dir={args.output_dir}")
        logger.info(f"  Configs: {len(SMOOTH_CONFIGS)} smooth + 4 abrupt")
        logger.info("=" * 60)

        generate_experiments(
            SMOOTH_CONFIGS,
            args.n_songs,
            args.output_dir,
            args.gpu,
            skip_wav,
            args.alphas,
            args.lambdas,
        )
        logger.info("\nGeneration complete!")

    if args.generate_only:
        return

    # ── Phase 2: Analyze ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: STATISTICAL SIGNIFICANCE ANALYSIS")
    logger.info(f"  Scanning: {args.output_dir}")
    logger.info("=" * 60)

    pairs = auto_detect_pairs(args.output_dir)
    if not pairs:
        logger.error("No experiment pairs found!")
        return

    logger.info(f"  Found {len(pairs)} (abrupt, smooth) pair(s):\n")
    for p in pairs:
        logger.info(
            f"    {p['method'].upper()} × {p['concept']} : "
            f"abrupt vs {p['smooth_label']}"
        )

    all_analyses = {}
    for p in pairs:
        label = f"{p['method']}_{p['concept']}_{p['smooth_label']}"

        abrupt_data = load_json(p["abrupt_file"])
        smooth_data = load_json(p["smooth_file"])

        matched = match_samples(abrupt_data, smooth_data, p["method"])
        logger.info(
            f"\n  {label}: {len(matched)} total matched pairs "
            f"({len([m for m in matched if abs(m[3]) > 0.01])} non-baseline)"
        )

        analysis = run_significance_tests(matched, p["concept"])
        # Attach metadata for reporting
        analysis["_method"] = p["method"]
        analysis["_concept"] = p["concept"]
        analysis["_smooth_label"] = p["smooth_label"]
        all_analyses[label] = analysis

    # Report
    report = format_report(all_analyses)
    print("\n" + report)

    # Save JSON (strip metadata keys starting with _)
    save_data = {}
    for k, v in all_analyses.items():
        save_data[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}

    json_path = args.output_dir / "significance_results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"Saved JSON: {json_path}")

    report_path = args.output_dir / "significance_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")

    # Plot
    generate_plots(all_analyses, args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
