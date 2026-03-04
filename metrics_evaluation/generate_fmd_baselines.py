#!/usr/bin/env python3
"""Generate baseline (unsteered) MIDI samples for FMD evaluation.

Produces additional baseline samples to make the FMD comparison
trustworthy. For unconditioned, generates from scratch with no hooks.
For conditioned, uses random SOD songs as primers with no hooks.

Usage:
    python metrics_evaluation/generate_fmd_baselines.py \
        --n_uncond 50 --n_cond 50 \
        --output_dir exp/sod/sparse_steering/fmd_workspace/baselines \
        --gpu 0
"""

import argparse
import glob
import logging
import pathlib
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mmt"))
sys.path.insert(0, str(PROJECT_ROOT / "sparse_steering" / "dual_steering"))

import music_x_transformers
import representation
import utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, train_args_path, encoding_path, device):
    """Load MMT model (same as test_dual_conditioned)."""
    from config_dual import DEFAULT_CHECKPOINT, DEFAULT_TRAIN_ARGS, DEFAULT_ENCODING

    cp = str(checkpoint_path or DEFAULT_CHECKPOINT)
    ta = str(train_args_path or DEFAULT_TRAIN_ARGS)
    en = str(encoding_path or DEFAULT_ENCODING)

    train_args = utils.load_json(ta)
    encoding = representation.load_encoding(en)

    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    ckpt = torch.load(cp, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    return model, encoding


def get_sod_json_paths():
    """Get all SOD JSON paths."""
    json_dir = PROJECT_ROOT / "data" / "sod" / "processed" / "json"
    return sorted(glob.glob(str(json_dir / "**" / "*.json"), recursive=True))


def load_primer_from_json(json_path, encoding, n_beats=16):
    """Load a SOD JSON, encode, extract first n_beats as primer."""
    import muspy

    music = muspy.load_json(json_path)
    tokens = representation.encode(music, encoding)
    if tokens is None or len(tokens) < 4:
        return None

    # Find tokens covering first n_beats
    cond_len = 0
    for i, tok in enumerate(tokens):
        beat = tok[1]  # beat column
        if beat >= n_beats:
            cond_len = i
            break
    else:
        cond_len = len(tokens)

    if cond_len < 2:
        cond_len = min(16, len(tokens))

    return tokens[:cond_len]


def generate_unconditioned_baselines(
    model, encoding, n_samples, seq_len, device, output_dir
):
    """Generate N unconditioned baseline samples (no hooks)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eos = encoding["type_code_map"]["end-of-song"]

    # Count existing to allow incremental runs
    existing = len(list(output_dir.glob("*.npy")))
    if existing >= n_samples:
        logger.info(
            f"Already have {existing}/{n_samples} unconditioned baselines, skipping"
        )
        return existing

    start_idx = existing
    success = existing

    for i in range(start_idx, n_samples):
        npy_path = output_dir / f"baseline_uncond_s{i:04d}.npy"
        if npy_path.exists():
            success += 1
            continue
        try:
            with torch.no_grad():
                start = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
                generated = model.generate(start, seq_len, eos_token=eos)

            tokens = generated[0].cpu().numpy()
            np.save(npy_path, tokens)
            success += 1

            if (success - existing) % 50 == 0:
                logger.info(f"  Unconditioned baselines: {success}/{n_samples}")
        except Exception as e:
            logger.warning(f"Unconditioned sample {i} failed: {e}")

    logger.info(f"Unconditioned baselines: {success}/{n_samples} saved → {output_dir}")
    return success


def generate_conditioned_baselines(
    model, encoding, n_samples, seq_len, device, output_dir, n_beats=16, seed=42
):
    """Generate N conditioned baseline samples from random SOD primers (no hooks).

    Args:
        n_beats: Number of beats to use as primer (16 = paper default).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eos = encoding["type_code_map"]["end-of-song"]

    # Count existing to allow incremental runs
    existing = len(list(output_dir.glob("*.npy")))
    if existing >= n_samples:
        logger.info(
            f"Already have {existing}/{n_samples} conditioned baselines, skipping"
        )
        return existing

    all_jsons = get_sod_json_paths()
    random.seed(seed)
    random.shuffle(all_jsons)

    success = existing
    json_idx = 0

    while success < n_samples and json_idx < len(all_jsons):
        jp = all_jsons[json_idx]
        json_idx += 1

        try:
            primer_tokens = load_primer_from_json(jp, encoding, n_beats=n_beats)
            if primer_tokens is None:
                continue

            cond_tensor = torch.from_numpy(primer_tokens).unsqueeze(0).long().to(device)

            with torch.no_grad():
                generated = model.generate(cond_tensor, seq_len, eos_token=eos)

            continuation = generated[0].cpu().numpy()
            full_seq = np.concatenate([primer_tokens, continuation], axis=0)

            song_name = pathlib.Path(jp).stem
            np.save(
                output_dir / f"baseline_cond_{song_name}_s{success:04d}.npy", full_seq
            )
            success += 1

            if (success - existing) % 50 == 0 and success > existing:
                logger.info(f"  Conditioned baselines: {success}/{n_samples}")
        except Exception as e:
            logger.debug(f"Failed on {jp}: {e}")

    logger.info(f"Conditioned baselines: {success}/{n_samples} saved → {output_dir}")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline MIDI samples for FMD"
    )
    parser.add_argument(
        "--n_uncond",
        type=int,
        default=1000,
        help="Number of unconditioned baselines (paper uses 1000)",
    )
    parser.add_argument(
        "--n_cond",
        type=int,
        default=1000,
        help="Number of 16-beat conditioned baselines (paper uses 1000)",
    )
    parser.add_argument(
        "--n_beats",
        type=int,
        default=16,
        help="Conditioning beats for conditioned mode (paper uses 16)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "fmd_workspace"
        / "baselines",
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--train_args", type=pathlib.Path, default=None)
    parser.add_argument("--encoding_path", type=pathlib.Path, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model, encoding = load_model(
        args.checkpoint, args.train_args, args.encoding_path, device
    )

    # 1. Unconditioned baselines
    if args.n_uncond > 0:
        logger.info(f"Generating {args.n_uncond} unconditioned baselines...")
        uncond_dir = args.output_dir / "unconditioned"
        generate_unconditioned_baselines(
            model, encoding, args.n_uncond, args.seq_len, device, uncond_dir
        )

    # 2. Conditioned baselines (16-beat continuation, matching paper)
    if args.n_cond > 0:
        logger.info(
            f"Generating {args.n_cond} conditioned baselines ({args.n_beats}-beat continuation)..."
        )
        cond_dir = args.output_dir / "conditioned"
        generate_conditioned_baselines(
            model,
            encoding,
            args.n_cond,
            args.seq_len,
            device,
            cond_dir,
            n_beats=args.n_beats,
            seed=args.seed,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
