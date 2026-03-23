#!/usr/bin/env python3
"""
Music Flamingo Evaluation — Full Pipeline

Two-step evaluation:
  1. Describe: Send each audio to Music Flamingo → get text descriptions
  2. Compare:  Feed description pairs to GPT-4  → get evaluation judgments

Usage:
    # Full pipeline
    python -m flamingo_evaluation.run_all

    # Step 1 only (Flamingo descriptions)
    python -m flamingo_evaluation.run_all --step describe

    # Step 2 only (LLM comparison, reuses existing descriptions.json)
    python -m flamingo_evaluation.run_all --step compare

    # Dry run (show pairs without calling APIs)
    python -m flamingo_evaluation.run_all --dry_run
"""

import argparse
import logging
from pathlib import Path

from flamingo_evaluation.audio_catalog import build_catalog, build_all_pairs
from flamingo_evaluation.run_describe import run_describe
from flamingo_evaluation.run_compare import run_compare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def dry_run(audio_dir: Path):
    """Show what would be evaluated without calling any APIs."""
    catalog = build_catalog(audio_dir)
    print(f"\n📁 Found {len(catalog)} audio files\n")

    # Show catalog breakdown
    methods = {}
    for e in catalog:
        methods.setdefault(e.method, []).append(e)
    for method, entries in sorted(methods.items()):
        baselines = sum(1 for e in entries if e.role == "baseline")
        steered = sum(1 for e in entries if e.role == "steered")
        print(
            f"  {method}: {len(entries)} files ({baselines} baselines, {steered} steered)"
        )

    # Show pairs per block
    all_pairs = build_all_pairs(catalog)
    print(f"\n📊 Comparison pairs per block:\n")
    for block, pairs in all_pairs.items():
        print(f"  {block}: {len(pairs)} pairs")
        if pairs:
            concepts = {}
            for p in pairs:
                concepts.setdefault(p.concept, []).append(p)
            for concept, cpairs in sorted(concepts.items()):
                print(f"    [{concept}] {len(cpairs)} pairs")
                for p in cpairs[:3]:  # show first 3
                    print(
                        f"      {p.label_a} ({p.audio_a.filename}) "
                        f"vs {p.label_b} ({p.audio_b.filename})"
                    )
                if len(cpairs) > 3:
                    print(f"      ... and {len(cpairs)-3} more")

    total_flamingo = len(catalog)
    total_llm = sum(len(pairs) for pairs in all_pairs.values())
    print(f"\n📡 API calls needed:")
    print(
        f"  Flamingo: {total_flamingo} (with retries, ~{total_flamingo*1.25:.0f} expected)"
    )
    print(f"  LLM comparisons: ~{total_llm * 2} (multiple questions per pair)")


def main():
    parser = argparse.ArgumentParser(
        description="Music Flamingo Evaluation Pipeline",
    )
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path("AUDIO EVAL TRIM/ALL"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("flamingo_evaluation/results"),
    )
    parser.add_argument(
        "--step",
        choices=["all", "describe", "compare"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for comparison step",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show pairs and counts without calling APIs",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for higher GPU quota (API mode only)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run Music Flamingo locally on GPU instead of HF Space API",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.audio_dir)
        return

    if args.step in ("all", "describe"):
        logger.info("=" * 60)
        logger.info("STEP 1: Getting Music Flamingo descriptions")
        logger.info("=" * 60)
        run_describe(
            args.audio_dir, args.output_dir, hf_token=args.hf_token, local=args.local
        )

    if args.step in ("all", "compare"):
        desc_path = args.output_dir / "descriptions.json"
        if not desc_path.exists():
            logger.error(
                "descriptions.json not found at %s. Run step 'describe' first.",
                desc_path,
            )
            return
        logger.info("=" * 60)
        logger.info("STEP 2: LLM-based comparison")
        logger.info("=" * 60)
        run_compare(desc_path, args.audio_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()
