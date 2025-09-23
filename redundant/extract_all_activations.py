#!/usr/bin/env python3
"""
Script to extract activations from all data splits (train, valid, test) efficiently.
"""

import argparse
import logging
import pathlib
import subprocess
import sys


def run_extraction(dataset, split, layer, n_samples, max_seq_len, output_dir, gpu):
    """Run activation extraction for a specific split."""

    cmd = [
        sys.executable,
        "mmt/extract_activations.py",
        "-d",
        dataset,
        "-l",
        str(layer),
        "-ns",
        str(n_samples),
        "--max_seq_len",
        str(max_seq_len),
        "-o",
        output_dir,
        "-g",
        str(gpu),
    ]

    logging.info(f"Running extraction for {split} split: {' '.join(cmd)}")

    # Modify the script temporarily to use the correct split
    script_path = pathlib.Path("mmt/extract_activations.py")

    # Read current content
    with open(script_path, "r") as f:
        content = f.read()

    # Replace the dataset path
    old_line = 'pathlib.Path("data/sod/processed/train-names.txt")'
    new_line = f'pathlib.Path("data/{dataset}/processed/{split}-names.txt")'

    modified_content = content.replace(old_line, new_line)

    # Write modified content
    with open(script_path, "w") as f:
        f.write(modified_content)

    try:
        # Run the extraction
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Extraction for {split} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Extraction for {split} failed: {e}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        return False
    finally:
        # Restore original content
        with open(script_path, "w") as f:
            f.write(content)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Extract activations from all data splits"
    )
    parser.add_argument("-d", "--dataset", default="sod", help="Dataset name")
    parser.add_argument(
        "-l", "--layer", type=int, default=3, help="Layer to extract from"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Max sequence length"
    )
    parser.add_argument(
        "-o", "--output_dir", default="exp/sod/ape", help="Output directory"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU number")

    args = parser.parse_args()

    # Define splits and their maximum samples
    splits_config = {"train": 4594, "valid": 574, "test": 574}

    total_extracted = 0

    for split, max_samples in splits_config.items():
        logging.info(f"\n=== Processing {split} split ({max_samples} samples) ===")

        # Use all available samples for each split
        success = run_extraction(
            dataset=args.dataset,
            split=split,
            layer=args.layer,
            n_samples=max_samples,
            max_seq_len=args.max_seq_len,
            output_dir=f"{args.output_dir}/{split}",
            gpu=args.gpu,
        )

        if success:
            total_extracted += max_samples
            logging.info(f"Successfully extracted from {split} split")
        else:
            logging.error(f"Failed to extract from {split} split")

    logging.info("\n=== Extraction Summary ===")
    logging.info(f"Total samples processed: {total_extracted}")
    logging.info(f"Results saved in: {args.output_dir}/{{train,valid,test}}/")


if __name__ == "__main__":
    main()
