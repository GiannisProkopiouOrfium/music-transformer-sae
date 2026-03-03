#!/usr/bin/env python3
"""Prepare MIDI directories for FMD evaluation.

Organises all necessary MIDI files into a clean directory structure:

    fmd_workspace/
    ├── reference_sod/            # 200 random SOD ground-truth MIDIs
    ├── baseline_conditioned/     # Unsteered conditioned generations (pooled)
    ├── baseline_unconditioned/   # Unsteered unconditioned generations (pooled)
    ├── conditioned/
    │   ├── expanded_k_2x/        # All non-zero-alpha conditioned steered
    │   ├── gram_schmidt_ek2/
    │   ├── per_scenario/
    │   │   └── ...
    │   └── per_lambda/           # Per-(λ_p,λ_d) conditioned dirs
    │       └── <strategy>__lp+X.XX_ld+Y.YY/
    └── unconditioned/
        ├── expanded_k_2x/        # All non-zero-alpha unconditioned steered
        ├── gram_schmidt_ek2/
        └── per_lambda/           # Per-(λ_p,λ_d) unconditioned dirs
            └── <strategy>__p+X.XX_d+Y.YY/

Usage:
    python metrics_evaluation/prepare_fmd_midis.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace \
        --n_sod_samples 200
"""

import argparse
import glob
import json
import logging
import pathlib
import random
import sys

import muspy
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
    """Convert a 6-dim .npy token file to MIDI."""
    try:
        tokens = np.load(npy_path)
        music = representation.decode(tokens, encoding)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        music.write(str(midi_path))
        return True
    except Exception as e:
        logger.debug(f"Failed to convert {npy_path}: {e}")
        return False


def prepare_sod_reference(
    workspace: pathlib.Path, n_samples: int = 200, seed: int = 42
) -> pathlib.Path:
    """Sample SOD ground-truth JSONs and convert to MIDI."""
    out_dir = workspace / "reference_sod"
    if out_dir.exists() and len(list(out_dir.glob("*.mid"))) >= n_samples:
        logger.info(
            f"SOD reference already prepared ({len(list(out_dir.glob('*.mid')))} MIDIs)"
        )
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = PROJECT_ROOT / "data" / "sod" / "processed" / "json"
    all_jsons = sorted(glob.glob(str(json_dir / "**" / "*.json"), recursive=True))
    logger.info(f"Found {len(all_jsons)} SOD JSON files")

    random.seed(seed)
    selected = random.sample(all_jsons, min(n_samples, len(all_jsons)))

    success = 0
    for jp in selected:
        try:
            music = muspy.load_json(jp)
            name = pathlib.Path(jp).stem
            music.write(str(out_dir / f"{name}.mid"))
            success += 1
        except Exception as e:
            logger.debug(f"Failed SOD {jp}: {e}")

    logger.info(f"SOD reference: {success}/{len(selected)} converted → {out_dir}")
    return out_dir


def prepare_conditioned(
    workspace: pathlib.Path,
    encoding: dict,
    strategies: list,
    experiment_dirs: dict,
) -> dict:
    """Prepare conditioned MIDI dirs from experiment results.

    Args:
        experiment_dirs: {strategy_name: path_to_experiment_output_dir}
            e.g. {"expanded_k_2x": "exp/sod/.../conditioned_ek2"}

    Returns:
        Dict of {label: midi_dir_path}
    """
    dirs = {}

    baseline_dir = workspace / "baseline_conditioned"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_count = 0

    for strategy in strategies:
        exp_dir = pathlib.Path(experiment_dirs[strategy])
        if not exp_dir.exists():
            logger.warning(f"Conditioned dir not found: {exp_dir}")
            continue

        # Steered (non-zero alphas) — pooled across all scenarios
        steered_dir = workspace / "conditioned" / strategy
        steered_dir.mkdir(parents=True, exist_ok=True)

        # Find all .npy files — only under the matching strategy subfolder
        # Structure: <exp_dir>/<scenario>/<strategy_folder>/<lambda_dir>/<song>.npy
        all_npy = sorted(exp_dir.rglob("*.npy"))
        # Filter to only files whose strategy-level parent matches
        all_npy = [
            npy
            for npy in all_npy
            if npy.parent.parent.name == strategy  # parent.parent = strategy folder
        ]
        steered_count = 0

        for npy in all_npy:
            # Parse the folder name to determine if baseline
            parent_name = npy.parent.name  # e.g. "lp+0.00_ld+0.00"

            is_baseline = (
                "lp+0.00_ld+0.00" in parent_name or "lp-0.00_ld-0.00" in parent_name
            )

            if is_baseline:
                # Pool baselines across strategies (same song = same baseline)
                midi_name = f"{npy.stem}_{npy.parent.parent.parent.name}.mid"
                midi_path = baseline_dir / midi_name
                if not midi_path.exists():
                    if npy_to_midi(npy, midi_path, encoding):
                        baseline_count += 1
            else:
                # Check both lambdas are non-zero from folder name
                parts = parent_name.split("_")
                try:
                    lp = float(parts[0].replace("lp", ""))
                    ld = float(parts[1].replace("ld", ""))
                    if lp == 0.0 or ld == 0.0:
                        continue
                except (ValueError, IndexError):
                    continue

                # Unique name to avoid collisions
                scenario = npy.parent.parent.parent.name
                midi_name = f"{scenario}_{parent_name}_{npy.stem}.mid"
                if npy_to_midi(npy, steered_dir / midi_name, encoding):
                    steered_count += 1

                # Also per-scenario dir
                per_sc_dir = (
                    workspace
                    / "conditioned"
                    / "per_scenario"
                    / f"{strategy}__{scenario}"
                )
                per_sc_dir.mkdir(parents=True, exist_ok=True)
                per_sc_midi = f"{parent_name}_{npy.stem}.mid"
                if not (per_sc_dir / per_sc_midi).exists():
                    npy_to_midi(npy, per_sc_dir / per_sc_midi, encoding)

                # Also per-lambda dir (pooled across scenarios)
                per_lam_dir = (
                    workspace
                    / "conditioned"
                    / "per_lambda"
                    / f"{strategy}__lp{lp:+.2f}_ld{ld:+.2f}"
                )
                per_lam_dir.mkdir(parents=True, exist_ok=True)
                per_lam_midi = f"{scenario}_{npy.stem}.mid"
                if not (per_lam_dir / per_lam_midi).exists():
                    npy_to_midi(npy, per_lam_dir / per_lam_midi, encoding)

        dirs[f"conditioned/{strategy}"] = steered_dir
        logger.info(f"Conditioned {strategy}: {steered_count} steered MIDIs")

    dirs["baseline_conditioned"] = baseline_dir
    logger.info(f"Conditioned baselines: {baseline_count} MIDIs (pooled)")

    # Collect per-scenario dirs
    per_sc_base = workspace / "conditioned" / "per_scenario"
    if per_sc_base.exists():
        for d in sorted(per_sc_base.iterdir()):
            if d.is_dir():
                dirs[f"conditioned/per_scenario/{d.name}"] = d

    # Collect per-lambda dirs
    per_lam_base = workspace / "conditioned" / "per_lambda"
    if per_lam_base.exists():
        for d in sorted(per_lam_base.iterdir()):
            if d.is_dir():
                dirs[f"conditioned/per_lambda/{d.name}"] = d

    return dirs


def prepare_unconditioned(
    workspace: pathlib.Path,
    encoding: dict,
    strategies: list,
    experiment_dirs: dict,
) -> dict:
    """Prepare unconditioned MIDI dirs.

    experiment_dirs: {strategy_name: path_to_experiment_dir}
        The .npy files are at <dir>/midi/<strategy>/p+X.XX_d+Y.YY_sN.npy
    """
    dirs = {}

    baseline_dir = workspace / "baseline_unconditioned"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_count = 0

    for strategy in strategies:
        exp_dir = pathlib.Path(experiment_dirs[strategy])
        midi_subdir = exp_dir / "midi" / strategy

        if not midi_subdir.exists():
            logger.warning(f"Unconditioned midi dir not found: {midi_subdir}")
            continue

        steered_dir = workspace / "unconditioned" / strategy
        steered_dir.mkdir(parents=True, exist_ok=True)

        all_npy = sorted(midi_subdir.glob("*.npy"))
        steered_count = 0

        for npy in all_npy:
            name = npy.stem  # e.g. "p+0.00_d+0.00_s0"

            is_baseline = name.startswith("p+0.00_d+0.00") or name.startswith(
                "p-0.00_d-0.00"
            )

            if is_baseline:
                midi_name = f"{strategy}_{npy.stem}.mid"
                midi_path = baseline_dir / midi_name
                if not midi_path.exists():
                    if npy_to_midi(npy, midi_path, encoding):
                        baseline_count += 1
            else:
                # Check both alphas are non-zero
                try:
                    parts = name.split("_")
                    ap = float(parts[0].replace("p", ""))
                    ad = float(parts[1].replace("d", ""))
                    if ap == 0.0 or ad == 0.0:
                        continue
                except (ValueError, IndexError):
                    continue

                midi_name = f"{npy.stem}.mid"
                if npy_to_midi(npy, steered_dir / midi_name, encoding):
                    steered_count += 1

                # Also per-lambda dir
                lambda_label = f"{parts[0]}_{parts[1]}"  # e.g. p+0.50_d+1.00
                per_lam_dir = (
                    workspace
                    / "unconditioned"
                    / "per_lambda"
                    / f"{strategy}__{lambda_label}"
                )
                per_lam_dir.mkdir(parents=True, exist_ok=True)
                per_lam_midi = f"{npy.stem}.mid"
                if not (per_lam_dir / per_lam_midi).exists():
                    npy_to_midi(npy, per_lam_dir / per_lam_midi, encoding)

        dirs[f"unconditioned/{strategy}"] = steered_dir
        logger.info(f"Unconditioned {strategy}: {steered_count} steered MIDIs")

    dirs["baseline_unconditioned"] = baseline_dir
    logger.info(f"Unconditioned baselines: {baseline_count} MIDIs (pooled)")

    # Collect per-lambda dirs
    per_lam_base = workspace / "unconditioned" / "per_lambda"
    if per_lam_base.exists():
        for d in sorted(per_lam_base.iterdir()):
            if d.is_dir():
                dirs[f"unconditioned/per_lambda/{d.name}"] = d

    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIDI directories for FMD evaluation"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "fmd_workspace",
    )
    parser.add_argument("--n_sod_samples", type=int, default=200)

    # Conditioned experiment dirs (one per strategy)
    parser.add_argument(
        "--cond_expanded_k_2x",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "dual_steering"
        / "conditioned_ek2",
    )
    parser.add_argument(
        "--cond_gram_schmidt_ek2",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "dual_steering"
        / "conditioned_gs_both",
    )

    # Unconditioned experiment dirs
    parser.add_argument(
        "--uncond_expanded_k_2x",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "dual_steering"
        / "unconditioned_triplet_fixed",
    )
    parser.add_argument(
        "--uncond_gram_schmidt_ek2",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "dual_steering"
        / "unconditioned_triplet_fixed",
    )

    args = parser.parse_args()
    args.workspace_dir.mkdir(parents=True, exist_ok=True)

    # Load encoding
    encoding_path = (
        PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
    )
    encoding = representation.load_encoding(str(encoding_path))
    logger.info(f"Loaded encoding from {encoding_path}")

    strategies = ["expanded_k_2x", "gram_schmidt_ek2"]

    # 1. SOD reference
    logger.info("=" * 60)
    logger.info("Preparing SOD reference...")
    prepare_sod_reference(args.workspace_dir, args.n_sod_samples)

    # 1b. Dedicated baselines (from generate_fmd_baselines.py)
    baseline_gen_dir = args.workspace_dir / "baselines"
    if baseline_gen_dir.exists():
        logger.info("=" * 60)
        logger.info("Converting dedicated baseline .npy files to MIDI...")
        for mode in ("conditioned", "unconditioned"):
            src = baseline_gen_dir / mode
            if not src.exists():
                continue
            dst = args.workspace_dir / f"baseline_{mode}"
            dst.mkdir(parents=True, exist_ok=True)
            n_converted = 0
            for npy in sorted(src.glob("*.npy")):
                midi_path = dst / f"{npy.stem}.mid"
                if not midi_path.exists():
                    if npy_to_midi(npy, midi_path, encoding):
                        n_converted += 1
            existing = len(list(dst.glob("*.mid")))
            logger.info(
                f"  {mode} baselines: {n_converted} new + {existing - n_converted} existing = {existing} total"
            )

    # 2. Conditioned
    logger.info("=" * 60)
    logger.info("Preparing conditioned MIDIs...")
    cond_dirs = prepare_conditioned(
        args.workspace_dir,
        encoding,
        strategies,
        {
            "expanded_k_2x": str(args.cond_expanded_k_2x),
            "gram_schmidt_ek2": str(args.cond_gram_schmidt_ek2),
        },
    )

    # 3. Unconditioned
    logger.info("=" * 60)
    logger.info("Preparing unconditioned MIDIs...")
    uncond_dirs = prepare_unconditioned(
        args.workspace_dir,
        encoding,
        strategies,
        {
            "expanded_k_2x": str(args.uncond_expanded_k_2x),
            "gram_schmidt_ek2": str(args.uncond_gram_schmidt_ek2),
        },
    )

    # 4. Summary
    all_dirs = {**cond_dirs, **uncond_dirs}
    all_dirs["reference_sod"] = args.workspace_dir / "reference_sod"

    logger.info("=" * 60)
    logger.info("MIDI preparation complete. Directory summary:")
    for label, d in sorted(all_dirs.items()):
        n_mid = (
            len(list(pathlib.Path(d).glob("*.mid"))) if pathlib.Path(d).exists() else 0
        )
        logger.info(f"  {label:<55} {n_mid:>4} MIDIs")

    # Save manifest
    manifest = {
        k: {"path": str(v), "n_midis": len(list(pathlib.Path(v).glob("*.mid")))}
        for k, v in all_dirs.items()
    }
    manifest_path = args.workspace_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
