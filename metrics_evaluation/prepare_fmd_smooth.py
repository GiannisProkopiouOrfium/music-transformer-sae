#!/usr/bin/env python3
"""Prepare & compute FMD for smooth steering comparison experiments.

Collects MIDIs from ``smooth_final_verification/`` experiment directories,
builds a flat workspace, creates SOD reference MIDIs, and computes FMD for
each experiment against the SOD reference.

Comparisons performed:
  - FMD(SOD, each_experiment)  — absolute distributional quality
  - FMD(baseline, smooth_variant) — perturbation cost of smooth steering

Outputs:
  - ``fmd_workspace_smooth/reference_sod/`` — reference MIDIs
  - ``fmd_workspace_smooth/<exp_name>/`` — flat MIDI dirs per experiment
  - ``fmd_workspace_smooth/manifest.json`` — dir manifest
  - ``fmd_workspace_smooth/fmd_results.json`` — all FMD scores
  - ``fmd_workspace_smooth/fmd_summary.txt`` — human-readable table

Usage:
    python metrics_evaluation/prepare_fmd_smooth.py \\
        --experiment_dir exp/sod/smooth_final_verification \\
        --gpu 0
"""

import argparse
import glob
import json
import logging
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def prepare_sod_reference(
    workspace: pathlib.Path, n_samples: int = 0, seed: int = 42
) -> pathlib.Path:
    """Convert SOD ground-truth JSONs to MIDI.

    Args:
        n_samples: 0 = use ALL SOD files (~5710).
    """
    import muspy

    out_dir = workspace / "reference_sod"
    json_dir = PROJECT_ROOT / "data" / "sod" / "processed" / "json"
    all_jsons = sorted(glob.glob(str(json_dir / "**" / "*.json"), recursive=True))
    logger.info(f"Found {len(all_jsons)} SOD JSON files")

    if n_samples > 0:
        random.seed(seed)
        selected = random.sample(all_jsons, min(n_samples, len(all_jsons)))
    else:
        selected = all_jsons

    target = len(selected)
    if out_dir.exists() and len(list(out_dir.glob("*.mid"))) >= target:
        existing = len(list(out_dir.glob("*.mid")))
        logger.info(f"SOD reference already prepared ({existing} MIDIs)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for jp in selected:
        name = pathlib.Path(jp).stem
        midi_path = out_dir / f"{name}.mid"
        if midi_path.exists():
            success += 1
            continue
        try:
            music = muspy.load_json(jp)
            music.write(str(midi_path))
            success += 1
        except Exception as e:
            logger.debug(f"Failed SOD {jp}: {e}")

    logger.info(f"SOD reference: {success}/{len(selected)} converted -> {out_dir}")
    return out_dir


def collect_experiment_midis(
    experiment_dir: pathlib.Path,
    workspace: pathlib.Path,
) -> dict:
    """Symlink/copy MIDIs from each experiment subdirectory into flat dirs.

    Returns: {exp_name: flat_midi_dir_path}
    """
    dirs = {}
    for exp_subdir in sorted(experiment_dir.iterdir()):
        if not exp_subdir.is_dir():
            continue
        exp_name = exp_subdir.name

        # Collect all .mid files recursively
        mid_files = list(exp_subdir.rglob("*.mid"))
        if not mid_files:
            logger.warning(f"No MIDIs in {exp_subdir}")
            continue

        flat_dir = workspace / exp_name
        flat_dir.mkdir(parents=True, exist_ok=True)

        linked = 0
        for mid in mid_files:
            # Unique name: category__lambda__song.mid
            rel = mid.relative_to(exp_subdir)
            flat_name = str(rel).replace("/", "__")
            target = flat_dir / flat_name
            if not target.exists():
                try:
                    target.symlink_to(mid.resolve())
                    linked += 1
                except OSError:
                    # Fallback: copy
                    import shutil
                    shutil.copy2(mid, target)
                    linked += 1

        dirs[exp_name] = flat_dir
        logger.info(f"  {exp_name}: {len(mid_files)} MIDIs ({linked} new)")

    return dirs


def compute_fmd(ref_path: str, test_path: str, metric) -> dict:
    """Compute FMD between two directories."""
    ref_dir = pathlib.Path(ref_path)
    test_dir = pathlib.Path(test_path)

    n_ref = len(list(ref_dir.glob("*.mid")))
    n_test = len(list(test_dir.glob("*.mid")))

    if n_ref < 2 or n_test < 2:
        return {
            "fmd": None,
            "n_ref": n_ref,
            "n_test": n_test,
            "error": f"Too few MIDIs (ref={n_ref}, test={n_test})",
        }

    try:
        t0 = time.time()
        score = metric.score(
            reference_path=str(ref_dir),
            test_path=str(test_dir),
        )
        elapsed = time.time() - t0
        return {
            "fmd": float(score),
            "n_ref": n_ref,
            "n_test": n_test,
            "time_sec": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "fmd": None,
            "n_ref": n_ref,
            "n_test": n_test,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare & compute FMD for smooth steering experiments"
    )
    parser.add_argument(
        "--experiment_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "smooth_final_verification",
        help="Directory with experiment subdirectories",
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        default=None,
        help="FMD workspace (default: <experiment_dir>/fmd_workspace)",
    )
    parser.add_argument(
        "--n_sod_samples",
        type=int,
        default=0,
        help="SOD reference samples (0 = all ~5710)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--model",
        choices=["clamp2", "clamp"],
        default="clamp2",
        help="CLaMP model for FMD",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare workspace, skip FMD computation",
    )

    args = parser.parse_args()

    if args.workspace_dir is None:
        args.workspace_dir = args.experiment_dir / "fmd_workspace"
    args.workspace_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. SOD reference ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Preparing SOD reference MIDIs...")
    ref_dir = prepare_sod_reference(args.workspace_dir, args.n_sod_samples)
    n_ref = len(list(ref_dir.glob("*.mid")))
    logger.info(f"SOD reference: {n_ref} MIDIs")

    # ── 2. Collect experiment MIDIs ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Collecting experiment MIDIs...")
    exp_dirs = collect_experiment_midis(args.experiment_dir, args.workspace_dir)

    # ── 3. Save manifest ─────────────────────────────────────────────────
    manifest = {"reference_sod": {"path": str(ref_dir), "n_midis": n_ref}}
    for name, d in exp_dirs.items():
        n = len(list(d.glob("*.mid")))
        manifest[name] = {"path": str(d), "n_midis": n}

    manifest_path = args.workspace_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_path}")

    # Summary
    logger.info("\nDirectory summary:")
    for label, info in sorted(manifest.items()):
        logger.info(f"  {label:<50} {info['n_midis']:>5} MIDIs")

    if args.prepare_only:
        logger.info("Preparation complete (--prepare_only). Exiting.")
        return

    # ── 4. Compute FMD ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Computing FMD scores...")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from frechet_music_distance import FrechetMusicDistance

    metric = FrechetMusicDistance(
        feature_extractor=args.model,
        gaussian_estimator="mle",
        verbose=True,
    )

    comparisons = []

    # Identify baselines and smooth variants
    # Group by (method, concept): find abrupt vs smooth
    exp_groups = {}  # (method, concept) -> {mode: dir_path}
    for name, d in exp_dirs.items():
        # Parse: sas_average_pitch_abrupt, dm_average_duration_ramp_up_32, etc.
        parts = name.split("_", 1)
        method = parts[0]  # sas or dm
        rest = parts[1] if len(parts) > 1 else ""

        # Find concept and mode
        for concept in ("average_pitch", "average_duration"):
            if concept in rest:
                mode = rest.replace(concept + "_", "")
                key = (method, concept)
                if key not in exp_groups:
                    exp_groups[key] = {}
                exp_groups[key][mode] = name
                break

    # A. FMD(SOD, experiment) for each experiment
    for name, d in sorted(exp_dirs.items()):
        comparisons.append(
            (f"SOD vs {name}", str(ref_dir), str(d))
        )

    # B. FMD(abrupt, smooth) for each (method, concept) pair
    for (method, concept), modes in sorted(exp_groups.items()):
        abrupt_name = modes.get("abrupt")
        if not abrupt_name:
            continue
        abrupt_dir = exp_dirs[abrupt_name]
        for mode, name in sorted(modes.items()):
            if mode == "abrupt":
                continue
            comparisons.append(
                (
                    f"Baseline({method}_{concept}) vs {mode}",
                    str(abrupt_dir),
                    str(exp_dirs[name]),
                )
            )

    logger.info(f"\nRunning {len(comparisons)} FMD comparisons...\n")

    results = []
    for label, ref, test in comparisons:
        logger.info(f"Computing: {label}")
        result = compute_fmd(ref, test, metric)
        result["comparison"] = label
        result["reference"] = ref
        result["test"] = test
        results.append(result)

        fmd_str = f"{result['fmd']:.4f}" if result["fmd"] is not None else "FAILED"
        logger.info(
            f"  FMD = {fmd_str}  (ref={result['n_ref']}, test={result['n_test']})"
        )

    # ── 5. Save results ──────────────────────────────────────────────────
    results_path = args.workspace_dir / "fmd_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved FMD results: {results_path}")

    # ── 6. Print summary ─────────────────────────────────────────────────
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append(" SMOOTH STEERING FMD RESULTS")
    lines.append("=" * 90)
    lines.append(f"\n{'Comparison':<55} {'FMD':>8} {'#Ref':>5} {'#Test':>5}")
    lines.append("-" * 80)

    lines.append("\n  --- Absolute quality (SOD vs experiment) ---")
    for r in results:
        if not r["comparison"].startswith("SOD vs"):
            continue
        fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
        lines.append(
            f"  {r['comparison']:<53} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}"
        )

    lines.append("\n  --- Perturbation cost (abrupt vs smooth) ---")
    for r in results:
        if not r["comparison"].startswith("Baseline"):
            continue
        fmd = f"{r['fmd']:.4f}" if r["fmd"] is not None else "N/A"
        lines.append(
            f"  {r['comparison']:<53} {fmd:>8} {r['n_ref']:>5} {r['n_test']:>5}"
        )

    lines.append("=" * 90)

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = args.workspace_dir / "fmd_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
