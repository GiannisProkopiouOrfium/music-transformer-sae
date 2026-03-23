#!/usr/bin/env python3
"""
Step 1: Get Music Flamingo descriptions for all audio files.

Supports two backends:
  - API mode (default): Uses the HuggingFace Space via Gradio client
  - Local mode (--local): Runs nvidia/music-flamingo-2601-hf locally on GPU

Usage:
    # API mode (HuggingFace Space)
    python -m flamingo_evaluation.run_describe \
        --audio_dir "AUDIO EVAL TRIM/ALL" \
        --output_dir flamingo_evaluation/results

    # Local mode (GPU required)
    python -m flamingo_evaluation.run_describe \
        --audio_dir "AUDIO EVAL TRIM/ALL" \
        --output_dir flamingo_evaluation/results \
        --local
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from flamingo_evaluation.audio_catalog import build_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Comprehensive prompt that extracts all dimensions we need for comparison.
FLAMINGO_PROMPT = (
    "Analyze this music track in detail. Describe:\n"
    "(1) PITCH: the overall pitch register (high/mid/low), pitch range and variety, "
    "melodic movement and contour.\n"
    "(2) RHYTHM & DURATION: note lengths (short/long/mixed), rhythmic complexity "
    "and variety, tempo and groove stability.\n"
    "(3) QUALITY: musical coherence, naturalness (does it sound human-composed?), "
    "production quality, any artifacts or glitches.\n"
    "(4) OVERALL: genre, mood, instrumentation, and general character."
)


def _make_client(cache_dir: Path, local: bool = False, hf_token: str = None):
    """Create the appropriate Flamingo client."""
    if local:
        from flamingo_evaluation.flamingo_local import FlamingoLocalClient

        logger.info("Using LOCAL Music Flamingo model (GPU)")
        return FlamingoLocalClient(cache_dir=cache_dir)
    else:
        from flamingo_evaluation.flamingo_client import FlamingoClient

        logger.info("Using API Music Flamingo (HuggingFace Space)")
        return FlamingoClient(cache_dir=cache_dir, hf_token=hf_token)


def run_describe(
    audio_dir: Path, output_dir: Path, hf_token: str = None, local: bool = False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "flamingo_cache"

    client = _make_client(cache_dir, local=local, hf_token=hf_token)
    catalog = build_catalog(audio_dir)
    logger.info("Found %d audio files to describe", len(catalog))

    descriptions = {}
    failed = []

    # Incremental CSV — append each success so nothing is lost on crash
    csv_path = output_dir / "descriptions.csv"
    csv_existed = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_existed:
        csv_writer.writerow(
            [
                "filename",
                "method",
                "concept",
                "direction",
                "song_id",
                "song_num",
                "role",
                "params",
                "mode",
                "description",
            ]
        )

    # Load already-described filenames from CSV to skip them
    already_done = set()
    if csv_existed:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(row["filename"])
        logger.info("Resuming: %d files already in CSV", len(already_done))

    for i, entry in enumerate(catalog, 1):
        if entry.filename in already_done:
            logger.debug(
                "[%d/%d] SKIP (already in CSV): %s", i, len(catalog), entry.filename
            )
            # Still load into descriptions dict from cache
            try:
                desc = client.describe(entry.path, FLAMINGO_PROMPT)  # hits cache
                descriptions[entry.filename] = {
                    "method": entry.method,
                    "concept": entry.concept,
                    "direction": entry.direction,
                    "song_id": entry.song_id,
                    "song_num": entry.song_num,
                    "role": entry.role,
                    "params": entry.params,
                    "mode": entry.mode,
                    "description": desc,
                }
            except Exception:
                pass
            continue

        logger.info("[%d/%d] %s", i, len(catalog), entry.filename)
        try:
            desc = client.describe(entry.path, FLAMINGO_PROMPT)
            descriptions[entry.filename] = {
                "method": entry.method,
                "concept": entry.concept,
                "direction": entry.direction,
                "song_id": entry.song_id,
                "song_num": entry.song_num,
                "role": entry.role,
                "params": entry.params,
                "mode": entry.mode,
                "description": desc,
            }
            # Write to CSV immediately
            csv_writer.writerow(
                [
                    entry.filename,
                    entry.method,
                    entry.concept,
                    entry.direction,
                    entry.song_id,
                    entry.song_num,
                    entry.role,
                    entry.params,
                    entry.mode,
                    desc,
                ]
            )
            csv_file.flush()
            logger.info("  ✓ Got description (%d chars)", len(desc))
        except RuntimeError as e:
            logger.error("  ✗ Failed: %s", e)
            failed.append(entry.filename)

    csv_file.close()

    # Save consolidated results
    out_file = output_dir / "descriptions.json"
    out_file.write_text(json.dumps(descriptions, indent=2))
    logger.info("Saved %d descriptions to %s", len(descriptions), out_file)

    if failed:
        logger.warning(
            "%d files failed after all retries: %s",
            len(failed),
            ", ".join(failed),
        )
        (output_dir / "failed_files.txt").write_text("\n".join(failed))

    return descriptions, failed


def main():
    parser = argparse.ArgumentParser(description="Get Music Flamingo descriptions")
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path("AUDIO EVAL TRIM/ALL"),
        help="Directory with processed MP3 files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("flamingo_evaluation/results"),
        help="Output directory for descriptions and cache",
    )
    args = parser.parse_args()
    run_describe(args.audio_dir, args.output_dir)


if __name__ == "__main__":
    main()
