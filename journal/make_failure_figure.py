#!/usr/bin/env python3
"""Failure-case piano-roll figure for the journal paper (Sect. "Failure-case analysis").

Renders side-by-side piano rolls of a failed steering generation and a matched
successful one from the same conditioning prefix, annotated with the attribute
trajectories (per-bar mean pitch / mean duration), so the reader can see *what*
went wrong at the score level.

Pick the MIDIs after syncing/rerunning the conditioned dual-steering
experiments (see journal/EXPERIMENTS_RUNBOOK.md, E6). Typical sources:
  sparse H/L->L/S failures :  exp/sod/sparse_steering/dual_steering/conditioned_*/
  dense  H/S->L/L high-delta: steering_interventions/dual_steering/outputs/diffmean_conditioned/

Usage:
    python journal/make_failure_figure.py \
        --pairs fail=path/to/failed.mid,ok=path/to/success.mid \
        --pairs fail=path/to/failed2.mid,ok=path/to/success2.mid \
        --labels "Sparse H/L→L/S" "Dense H/S→L/L" \
        --prefix-beats 16 --out journal/figs/failure_cases.png
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import muspy
except ImportError:
    sys.exit("muspy is required: pip install muspy")

RESOLUTION = 12  # MMT ticks per beat
BAR_BEATS = 4

_ENCODING = None


def _mmt_encoding():
    """Load the MMT representation encoding once (for decoding .npy outputs)."""
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in (root, root / "mmt"):
        sys.path.insert(0, str(p))
    import representation  # noqa: E402  (repo module)
    for enc in (root / "mmt" / "encoding.json",
                root / "data" / "sod" / "processed" / "notes" / "encoding.json"):
        if enc.exists():
            _ENCODING = (representation, representation.load_encoding(enc))
            return _ENCODING
    raise SystemExit("could not locate encoding.json to decode .npy files")


def load_notes(path: pathlib.Path):
    path = pathlib.Path(path)
    if path.suffix == ".npy":
        representation, encoding = _mmt_encoding()
        seq = np.load(path)
        music = representation.decode(seq, encoding)  # muspy Music at res 12
        notes = [n for tr in music.tracks for n in tr.notes]
        if not notes:
            raise ValueError(f"No notes decoded from {path}")
        return [(n.time, n.duration, n.pitch) for n in notes]
    music = muspy.read_midi(str(path))
    notes = [n for track in music.tracks for n in track.notes]
    if not notes:
        raise ValueError(f"No notes in {path}")
    scale = RESOLUTION / music.resolution  # normalize to MMT tick resolution
    return [(n.time * scale, n.duration * scale, n.pitch) for n in notes]


def per_bar_stats(notes, stat):
    end = max(t + d for t, d, _ in notes)
    n_bars = int(np.ceil(end / (RESOLUTION * BAR_BEATS)))
    xs, ys = [], []
    for b in range(n_bars):
        lo, hi = b * RESOLUTION * BAR_BEATS, (b + 1) * RESOLUTION * BAR_BEATS
        bar = [(t, d, p) for t, d, p in notes if lo <= t < hi]
        if not bar:
            continue
        xs.append((lo + hi) / 2)
        ys.append(np.mean([p for _, _, p in bar]) if stat == "pitch"
                  else np.mean([d for _, d, _ in bar]))
    return np.array(xs), np.array(ys)


def draw_roll(ax, notes, prefix_ticks, title):
    for t, d, p in notes:
        in_prefix = t < prefix_ticks
        ax.add_patch(plt.Rectangle(
            (t, p - 0.4), max(d, 1), 0.8,
            facecolor="#8896ab" if in_prefix else "#3b6ea5",
            edgecolor="none", alpha=0.55 if in_prefix else 0.9))
    if prefix_ticks > 0:
        ax.axvline(prefix_ticks, color="#b3433b", lw=1.2, ls="--")
    pitches = [p for _, _, p in notes]
    ends = [t + d for t, d, _ in notes]
    ax.set_xlim(0, max(ends) * 1.02)
    ax.set_ylim(min(pitches) - 4, max(pitches) + 4)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("MIDI pitch", fontsize=8)
    ax.tick_params(labelsize=7)


import json

# Record field fallbacks (schemas differ between the dense and sparse runs).
_SUCCESS_KEYS = ("success", "both_success", "overall_success", "dual_success",
                 "both", "is_success")
_PATH_KEYS = ("mid_path", "midi_path", "filepath", "npy_path", "path", "file")
_SCENARIO_KEYS = ("scenario", "scenario_name", "category", "direction")


def _records(data):
    if isinstance(data, list):
        return data
    for k in ("results", "samples", "all_results", "records"):
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, list):
            return v
        if isinstance(v, dict):  # results keyed by scenario -> flatten
            out = []
            for scen, lst in v.items():
                if isinstance(lst, list):
                    for r in lst:
                        r = dict(r)
                        r.setdefault("scenario", scen)
                        out.append(r)
            if out:
                return out
    raise SystemExit(f"could not find a record list; top-level keys: "
                     f"{list(data) if isinstance(data, dict) else type(data)}")


def _get(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return None


def _to_gen(path_str):
    """Resolve a record's stored path to an existing generation file.

    Prefers a rendered .mid, but falls back to the raw .npy (which load_notes
    can decode), so no manual decoding step is required.
    """
    if not path_str:
        return None
    p = pathlib.Path(str(path_str))
    for cand in (p.with_suffix(".mid"), p.with_suffix(".midi"), p,
                 p.with_suffix(".npy")):
        if cand.exists():
            return cand
    return None


def _is_success(r):
    v = _get(r, _SUCCESS_KEYS)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v >= 0.5
    if isinstance(v, str):
        return v.lower() in ("true", "success", "yes", "1")
    return None


def auto_pick(results_path, scenario, strategy=None):
    """Pick (failure_midi, success_midi) from a conditioned results JSON."""
    data = json.load(open(results_path))
    recs = _records(data)
    def keep(r):
        s = str(_get(r, _SCENARIO_KEYS) or "")
        if scenario and scenario not in s:
            return False
        if strategy and str(r.get("strategy", "")) != strategy:
            return False
        return True
    recs = [r for r in recs if keep(r)]
    if not recs:
        raise SystemExit(f"no records for scenario='{scenario}' strategy='{strategy}' "
                         f"in {results_path}")
    fails = [r for r in recs if _is_success(r) is False]
    oks = [r for r in recs if _is_success(r) is True]
    if not fails or not oks:
        raise SystemExit(f"need both a failure and a success in scenario "
                         f"'{scenario}' ({len(fails)} fail / {len(oks)} ok found); "
                         f"success field candidates checked: {_SUCCESS_KEYS}")
    fail_gen = next((m for r in fails if (m := _to_gen(_get(r, _PATH_KEYS)))), None)
    ok_gen = next((m for r in oks if (m := _to_gen(_get(r, _PATH_KEYS)))), None)
    if not fail_gen or not ok_gen:
        sample = _get(fails[0], _PATH_KEYS) if fails else "?"
        raise SystemExit("found records but no generation file exists on disk; "
                         f"example stored path was '{sample}'. Check that the run "
                         "outputs (.npy or .mid) are present at those paths.")
    return fail_gen, ok_gen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", action="append", default=[],
                    help="manual mode: fail=<midi>,ok=<midi> (repeatable)")
    ap.add_argument("--auto", action="store_true",
                    help="auto-select fail/ok MIDIs from conditioned results JSONs")
    ap.add_argument("--sparse-results", type=pathlib.Path,
                    help="[auto] sparse conditioned_results.json")
    ap.add_argument("--dense-results", type=pathlib.Path,
                    help="[auto] dense conditioned_results.json")
    ap.add_argument("--sparse-scenario", default="high_pitch_long_duration_to_low_short")
    ap.add_argument("--dense-scenario", default="high_pitch_short_duration_to_low_long")
    ap.add_argument("--sparse-strategy", default="gram_schmidt_ek2")
    ap.add_argument("--dense-strategy", default="gram_schmidt_pitch")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one scenario label per pair")
    ap.add_argument("--prefix-beats", type=int, default=16)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("journal/figs/failure_cases.png"))
    args = ap.parse_args()

    pairs, labels = [], args.labels
    if args.auto:
        if not args.sparse_results or not args.dense_results:
            ap.error("--auto requires --sparse-results and --dense-results")
        pairs.append(auto_pick(args.sparse_results, args.sparse_scenario,
                               args.sparse_strategy))
        pairs.append(auto_pick(args.dense_results, args.dense_scenario,
                               args.dense_strategy))
        labels = labels or ["Sparse H/L->L/S", "Dense H/S->L/L"]
        for (f, o), lab in zip(pairs, labels):
            print(f"[{lab}] fail={f}  ok={o}")
    else:
        if not args.pairs:
            ap.error("provide --pairs (manual) or --auto with results files")
        for spec in args.pairs:
            d = dict(part.split("=", 1) for part in spec.split(","))
            pairs.append((pathlib.Path(d["fail"]), pathlib.Path(d["ok"])))
    labels = labels or [f"Scenario {i+1}" for i in range(len(pairs))]
    prefix_ticks = args.prefix_beats * RESOLUTION

    fig, axes = plt.subplots(len(pairs), 2,
                             figsize=(10, 3.1 * len(pairs)), squeeze=False)
    for row, ((fail_p, ok_p), label) in enumerate(zip(pairs, labels)):
        for col, (path, tag) in enumerate([(fail_p, "failed"), (ok_p, "successful")]):
            notes = load_notes(path)
            ax = axes[row][col]
            draw_roll(ax, notes, prefix_ticks, f"{label} — {tag}")
            # attribute trajectory overlay (per-bar mean pitch)
            xs, ys = per_bar_stats(notes, "pitch")
            ax2 = ax.twinx()
            ax2.plot(xs, ys, color="#c96a1f", lw=1.4)
            ax2.set_yticks([])
        axes[row][0].annotate("prefix", xy=(prefix_ticks * 0.5, 1.01),
                              xycoords=("data", "axes fraction"),
                              fontsize=7, color="#b3433b", ha="center")
    axes[-1][0].set_xlabel("time (ticks)", fontsize=8)
    axes[-1][1].set_xlabel("time (ticks)", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
