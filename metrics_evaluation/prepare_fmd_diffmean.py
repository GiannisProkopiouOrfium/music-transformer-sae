#!/usr/bin/env python3
"""Prepare DiffMean (dense steering) MIDI directories for FMD evaluation.

Organises DiffMean experiment outputs into the same FMD workspace structure
used by evaluate_fmd.py, so DiffMean and SAS results can be directly compared.

Directory structure created:
    fmd_workspace/
    ├── diffmean_conditioned/
    │   ├── gram_schmidt_pitch/       # All non-zero-alpha conditioned (pooled)
    │   ├── per_lambda/               # Per (λ_p, λ_d) pair
    │   │   └── gram_schmidt_pitch__lp+X.XX_ld+Y.YY/
    │   ├── per_lambda_pitch/         # Marginal per λ_p
    │   │   └── gram_schmidt_pitch__lp+X.XX/
    │   └── per_lambda_duration/      # Marginal per λ_d
    │       └── gram_schmidt_pitch__ld+Y.YY/
    └── diffmean_unconditioned/
        ├── gram_schmidt_pitch/       # All non-zero-alpha unconditioned (pooled)
        ├── per_lambda/
        │   └── gram_schmidt_pitch__p+X.XX_d+Y.YY/
        ├── per_lambda_pitch/
        │   └── gram_schmidt_pitch__p+X.XX/
        └── per_lambda_duration/
            └── gram_schmidt_pitch__d+Y.YY/

DiffMean source layouts:
  Unconditioned: {uncond_dir}/midi/{strategy}/p+X.XX_d+Y.YY_sN.npy
  Conditioned:   {cond_dir}/{scenario}/{strategy}/ap+X.X_ad+Y.Y/{song}.npy

Usage:
    python metrics_evaluation/prepare_fmd_diffmean.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace
"""

import argparse
import json
import logging
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def npy_to_midi(
    npy_path: pathlib.Path, midi_path: pathlib.Path, encoding: dict
) -> bool:
    """Convert a .npy token file to MIDI."""
    try:
        tokens = np.load(npy_path)
        music = representation.decode(tokens, encoding)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        music.write(str(midi_path))
        return True
    except Exception as e:
        logger.debug(f"Failed to convert {npy_path}: {e}")
        return False


def prepare_diffmean_unconditioned(
    workspace: pathlib.Path,
    encoding: dict,
    strategies: list,
    uncond_dir: pathlib.Path,
) -> dict:
    """Prepare unconditioned DiffMean MIDIs.

    Source layout: {uncond_dir}/midi/{strategy}/p+X.XX_d+Y.YY_sN.npy
    (Same naming as SAS after the test_multi_steering.py fix)
    """
    dirs = {}

    baseline_dir = workspace / "diffmean_baseline_unconditioned"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_count = 0

    for strategy in strategies:
        midi_subdir = uncond_dir / "midi" / strategy
        if not midi_subdir.exists():
            logger.warning(f"DiffMean uncond midi dir not found: {midi_subdir}")
            continue

        steered_dir = workspace / "diffmean_unconditioned" / strategy
        steered_dir.mkdir(parents=True, exist_ok=True)

        all_npy = sorted(midi_subdir.glob("*.npy"))
        steered_count = 0
        logger.info(f"DiffMean uncond {strategy}: found {len(all_npy)} .npy files")

        for npy in all_npy:
            name = npy.stem  # e.g. "p+0.00_d+0.00_s0"

            is_baseline = name.startswith("p+0.00_d+0.00") or name.startswith(
                "p-0.00_d-0.00"
            )

            if is_baseline:
                midi_name = f"dm_{strategy}_{npy.stem}.mid"
                midi_path = baseline_dir / midi_name
                if not midi_path.exists():
                    if npy_to_midi(npy, midi_path, encoding):
                        baseline_count += 1
            else:
                # Parse lambda values
                try:
                    parts = name.split("_")
                    ap = float(parts[0].replace("p", ""))
                    ad = float(parts[1].replace("d", ""))
                except (ValueError, IndexError):
                    continue

                is_dual = abs(ap) > 1e-6 and abs(ad) > 1e-6
                is_single_pitch = abs(ap) > 1e-6 and abs(ad) < 1e-6
                is_single_duration = abs(ap) < 1e-6 and abs(ad) > 1e-6

                if not (is_dual or is_single_pitch or is_single_duration):
                    continue

                # Pooled steered dir (dual only)
                if is_dual:
                    midi_name = f"{npy.stem}.mid"
                    if not (steered_dir / midi_name).exists():
                        if npy_to_midi(npy, steered_dir / midi_name, encoding):
                            steered_count += 1

                # Per-lambda-pair dir
                lambda_label = f"{parts[0]}_{parts[1]}"
                per_lam_dir = (
                    workspace
                    / "diffmean_unconditioned"
                    / "per_lambda"
                    / f"{strategy}__{lambda_label}"
                )
                per_lam_dir.mkdir(parents=True, exist_ok=True)
                per_lam_midi = f"{npy.stem}.mid"
                if not (per_lam_dir / per_lam_midi).exists():
                    npy_to_midi(npy, per_lam_dir / per_lam_midi, encoding)

                # Marginal per-λ_pitch
                if abs(ap) > 1e-6:
                    pitch_dir = (
                        workspace
                        / "diffmean_unconditioned"
                        / "per_lambda_pitch"
                        / f"{strategy}__p{ap:+.2f}"
                    )
                    pitch_dir.mkdir(parents=True, exist_ok=True)
                    pitch_midi = f"{npy.stem}.mid"
                    if not (pitch_dir / pitch_midi).exists():
                        npy_to_midi(npy, pitch_dir / pitch_midi, encoding)

                # Marginal per-λ_duration
                if abs(ad) > 1e-6:
                    dur_dir = (
                        workspace
                        / "diffmean_unconditioned"
                        / "per_lambda_duration"
                        / f"{strategy}__d{ad:+.2f}"
                    )
                    dur_dir.mkdir(parents=True, exist_ok=True)
                    dur_midi = f"{npy.stem}.mid"
                    if not (dur_dir / dur_midi).exists():
                        npy_to_midi(npy, dur_dir / dur_midi, encoding)

        dirs[f"diffmean_unconditioned/{strategy}"] = steered_dir
        logger.info(f"DiffMean uncond {strategy}: {steered_count} steered MIDIs")

    dirs["diffmean_baseline_unconditioned"] = baseline_dir
    logger.info(f"DiffMean uncond baselines: {baseline_count} MIDIs")

    # Collect per-lambda dirs
    for subtype in ("per_lambda", "per_lambda_pitch", "per_lambda_duration"):
        base = workspace / "diffmean_unconditioned" / subtype
        if base.exists():
            for d in sorted(base.iterdir()):
                if d.is_dir():
                    dirs[f"diffmean_unconditioned/{subtype}/{d.name}"] = d

    return dirs


def prepare_diffmean_conditioned(
    workspace: pathlib.Path,
    encoding: dict,
    strategies: list,
    cond_dir: pathlib.Path,
) -> dict:
    """Prepare conditioned DiffMean MIDIs.

    Source layout: {cond_dir}/{scenario}/{strategy}/ap+X.X_ad+Y.Y/{song}.npy
    """
    dirs = {}

    baseline_dir = workspace / "diffmean_baseline_conditioned"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_count = 0

    for strategy in strategies:
        steered_dir = workspace / "diffmean_conditioned" / strategy
        steered_dir.mkdir(parents=True, exist_ok=True)
        steered_count = 0

        # Scan all scenarios
        if not cond_dir.exists():
            logger.warning(f"DiffMean cond dir not found: {cond_dir}")
            continue

        for scenario_dir in sorted(cond_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario = scenario_dir.name

            strat_dir = scenario_dir / strategy
            if not strat_dir.exists():
                continue

            for alpha_dir in sorted(strat_dir.iterdir()):
                if not alpha_dir.is_dir():
                    continue

                # Parse alphas from dir name: ap+X.X_ad+Y.Y
                m = re.match(r"ap([+-]?\d+\.?\d*)_ad([+-]?\d+\.?\d*)", alpha_dir.name)
                if not m:
                    continue

                lp = float(m.group(1))
                ld = float(m.group(2))

                is_baseline = abs(lp) < 1e-6 and abs(ld) < 1e-6
                is_dual = abs(lp) > 1e-6 and abs(ld) > 1e-6
                is_single_pitch = abs(lp) > 1e-6 and abs(ld) < 1e-6
                is_single_duration = abs(lp) < 1e-6 and abs(ld) > 1e-6

                for npy in sorted(alpha_dir.glob("*.npy")):
                    if is_baseline:
                        midi_name = f"dm_{strategy}_{scenario}_{npy.stem}.mid"
                        midi_path = baseline_dir / midi_name
                        if not midi_path.exists():
                            if npy_to_midi(npy, midi_path, encoding):
                                baseline_count += 1
                    else:
                        if not (is_dual or is_single_pitch or is_single_duration):
                            continue

                        # Pooled steered dir (dual only)
                        if is_dual:
                            midi_name = f"{scenario}_{alpha_dir.name}_{npy.stem}.mid"
                            if not (steered_dir / midi_name).exists():
                                if npy_to_midi(npy, steered_dir / midi_name, encoding):
                                    steered_count += 1

                        # Per-lambda-pair dir
                        per_lam_dir = (
                            workspace
                            / "diffmean_conditioned"
                            / "per_lambda"
                            / f"{strategy}__lp{lp:+.2f}_ld{ld:+.2f}"
                        )
                        per_lam_dir.mkdir(parents=True, exist_ok=True)
                        per_lam_midi = f"{scenario}_{npy.stem}.mid"
                        if not (per_lam_dir / per_lam_midi).exists():
                            npy_to_midi(npy, per_lam_dir / per_lam_midi, encoding)

                        # Marginal per-λ_pitch
                        if abs(lp) > 1e-6:
                            pitch_dir = (
                                workspace
                                / "diffmean_conditioned"
                                / "per_lambda_pitch"
                                / f"{strategy}__lp{lp:+.2f}"
                            )
                            pitch_dir.mkdir(parents=True, exist_ok=True)
                            pitch_midi = f"{scenario}_{alpha_dir.name}_{npy.stem}.mid"
                            if not (pitch_dir / pitch_midi).exists():
                                npy_to_midi(npy, pitch_dir / pitch_midi, encoding)

                        # Marginal per-λ_duration
                        if abs(ld) > 1e-6:
                            dur_dir = (
                                workspace
                                / "diffmean_conditioned"
                                / "per_lambda_duration"
                                / f"{strategy}__ld{ld:+.2f}"
                            )
                            dur_dir.mkdir(parents=True, exist_ok=True)
                            dur_midi = f"{scenario}_{alpha_dir.name}_{npy.stem}.mid"
                            if not (dur_dir / dur_midi).exists():
                                npy_to_midi(npy, dur_dir / dur_midi, encoding)

        dirs[f"diffmean_conditioned/{strategy}"] = steered_dir
        logger.info(f"DiffMean cond {strategy}: {steered_count} steered MIDIs")

    dirs["diffmean_baseline_conditioned"] = baseline_dir
    logger.info(f"DiffMean cond baselines: {baseline_count} MIDIs")

    # Collect per-lambda dirs
    for subtype in ("per_lambda", "per_lambda_pitch", "per_lambda_duration"):
        base = workspace / "diffmean_conditioned" / subtype
        if base.exists():
            for d in sorted(base.iterdir()):
                if d.is_dir():
                    dirs[f"diffmean_conditioned/{subtype}/{d.name}"] = d

    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DiffMean MIDI directories for FMD evaluation"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "fmd_workspace",
    )
    parser.add_argument(
        "--diffmean_uncond_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "steering_interventions"
        / "dual_steering"
        / "outputs"
        / "diffmean_unconditioned",
        help="DiffMean unconditioned experiment dir (contains midi/<strategy>/*.npy)",
    )
    parser.add_argument(
        "--diffmean_cond_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "steering_interventions"
        / "dual_steering"
        / "outputs"
        / "diffmean_conditioned",
        help="DiffMean conditioned experiment dir (contains <scenario>/<strategy>/...)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["gram_schmidt_pitch"],
        help="DiffMean strategies to include",
    )
    args = parser.parse_args()
    args.workspace_dir.mkdir(parents=True, exist_ok=True)

    # Load encoding
    encoding_path = (
        PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
    )
    encoding = representation.load_encoding(str(encoding_path))
    logger.info(f"Loaded encoding from {encoding_path}")

    # Load existing manifest (to merge with SAS data)
    manifest_path = args.workspace_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info(f"Loaded existing manifest with {len(manifest)} entries")
    else:
        manifest = {}
        logger.info("No existing manifest found — creating new one")

    # Prepare unconditioned
    logger.info("=" * 60)
    logger.info("Preparing DiffMean unconditioned MIDIs...")
    uncond_dirs = prepare_diffmean_unconditioned(
        args.workspace_dir, encoding, args.strategies, args.diffmean_uncond_dir
    )

    # Prepare conditioned
    logger.info("=" * 60)
    logger.info("Preparing DiffMean conditioned MIDIs...")
    cond_dirs = prepare_diffmean_conditioned(
        args.workspace_dir, encoding, args.strategies, args.diffmean_cond_dir
    )

    # Merge into manifest
    all_new_dirs = {**uncond_dirs, **cond_dirs}

    logger.info("=" * 60)
    logger.info("DiffMean MIDI preparation complete:")
    for label, d in sorted(all_new_dirs.items()):
        n_mid = (
            len(list(pathlib.Path(d).glob("*.mid"))) if pathlib.Path(d).exists() else 0
        )
        logger.info(f"  {label:<60} {n_mid:>4} MIDIs")
        manifest[label] = {
            "path": str(d),
            "n_midis": n_mid,
        }

    # Save merged manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest with {len(manifest)} total entries: {manifest_path}")


if __name__ == "__main__":
    main()
