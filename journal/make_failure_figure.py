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


def load_notes(path: pathlib.Path):
    music = muspy.read_midi(str(path))
    notes = [n for track in music.tracks for n in track.notes]
    if not notes:
        raise ValueError(f"No notes in {path}")
    # normalize to MMT tick resolution
    scale = RESOLUTION / music.resolution
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", action="append", required=True,
                    help="fail=<midi>,ok=<midi> (repeatable)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one scenario label per pair")
    ap.add_argument("--prefix-beats", type=int, default=16)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("journal/figs/failure_cases.png"))
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        d = dict(part.split("=", 1) for part in spec.split(","))
        pairs.append((pathlib.Path(d["fail"]), pathlib.Path(d["ok"])))
    labels = args.labels or [f"Scenario {i+1}" for i in range(len(pairs))]
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
