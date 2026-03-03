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

import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, train_args_path, encoding_path, device):
    """Load MMT model."""
    from config_dual import DEFAULT_CHECKPOINT, DEFAULT_TRAIN_ARGS, DEFAULT_ENCODING

    cp = checkpoint_path or DEFAULT_CHECKPOINT
    ta = train_args_path or DEFAULT_TRAIN_ARGS
    en = encoding_path or DEFAULT_ENCODING

    encoding = representation.load_encoding(str(en))

    import json

    with open(ta) as f:
        train_args = json.load(f)

    from x_transformers import TransformerWrapper, Decoder

    dim = train_args.get("dim", 512)
    depth = train_args.get("layers", 12)
    heads = train_args.get("heads", 8)
    max_seq_len = train_args.get("max_seq_len", 1024)
    abs_pos_emb = train_args.get("abs_pos_emb", True)

    n_tokens = [len(v) for v in encoding["code_type_map"].values()]

    model = TransformerWrapper(
        num_tokens=n_tokens,
        max_seq_len=max_seq_len,
        attn_layers=Decoder(
            dim=dim, depth=depth, heads=heads, rotary_pos_emb=not abs_pos_emb,
            attn_flash=True,
        ),
    ).to(device)

    ckpt = torch.load(str(cp), map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
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


def generate_unconditioned_baselines(model, encoding, n_samples, seq_len, device, output_dir):
    """Generate N unconditioned baseline samples (no hooks)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eos = encoding["type_code_map"]["end-of-song"]
    success = 0

    for i in range(n_samples):
        try:
            with torch.no_grad():
                start = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
                generated = model.generate(start, seq_len, eos_token=eos)

            tokens = generated[0].cpu().numpy()
            np.save(output_dir / f"baseline_uncond_s{i:03d}.npy", tokens)
            success += 1
        except Exception as e:
            logger.warning(f"Unconditioned sample {i} failed: {e}")

    logger.info(f"Unconditioned baselines: {success}/{n_samples} saved → {output_dir}")
    return success


def generate_conditioned_baselines(model, encoding, n_samples, seq_len, device, output_dir, seed=42):
    """Generate N conditioned baseline samples from random SOD primers (no hooks)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eos = encoding["type_code_map"]["end-of-song"]

    all_jsons = get_sod_json_paths()
    random.seed(seed)
    random.shuffle(all_jsons)

    success = 0
    json_idx = 0

    while success < n_samples and json_idx < len(all_jsons):
        jp = all_jsons[json_idx]
        json_idx += 1

        try:
            primer_tokens = load_primer_from_json(jp, encoding)
            if primer_tokens is None:
                continue

            cond_tensor = torch.from_numpy(primer_tokens).unsqueeze(0).long().to(device)

            with torch.no_grad():
                generated = model.generate(cond_tensor, seq_len, eos_token=eos)

            continuation = generated[0].cpu().numpy()
            full_seq = np.concatenate([primer_tokens, continuation], axis=0)

            song_name = pathlib.Path(jp).stem
            np.save(output_dir / f"baseline_cond_{song_name}_s{success:03d}.npy", full_seq)
            success += 1

            if success % 10 == 0:
                logger.info(f"  Conditioned baselines: {success}/{n_samples}")
        except Exception as e:
            logger.debug(f"Failed on {jp}: {e}")

    logger.info(f"Conditioned baselines: {success}/{n_samples} saved → {output_dir}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Generate baseline MIDI samples for FMD")
    parser.add_argument("--n_uncond", type=int, default=50, help="Number of unconditioned baselines")
    parser.add_argument("--n_cond", type=int, default=50, help="Number of conditioned baselines")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "fmd_workspace" / "baselines",
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
    model, encoding = load_model(args.checkpoint, args.train_args, args.encoding_path, device)

    # 1. Unconditioned baselines
    if args.n_uncond > 0:
        logger.info(f"Generating {args.n_uncond} unconditioned baselines...")
        uncond_dir = args.output_dir / "unconditioned"
        generate_unconditioned_baselines(model, encoding, args.n_uncond, args.seq_len, device, uncond_dir)

    # 2. Conditioned baselines
    if args.n_cond > 0:
        logger.info(f"Generating {args.n_cond} conditioned baselines...")
        cond_dir = args.output_dir / "conditioned"
        generate_conditioned_baselines(model, encoding, args.n_cond, args.seq_len, device, cond_dir, args.seed)

    logger.info("Done!")


if __name__ == "__main__":
    main()
