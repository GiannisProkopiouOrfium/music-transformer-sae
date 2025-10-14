#!/usr/bin/env python3
"""Debug script to check token contents in intervention files."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Now import - use full path like run_llm_batch_evaluation.py does
import baseline.representation_remi as representation_remi


def main():
    # Load encoding
    encoding = representation_remi.get_encoding()
    vocabulary = encoding["code_event_map"]

    # Load a test file
    test_file = "batch_extractions_interventions/interventions/feature1323/baseline.pt"
    print(f"Loading: {test_file}")
    data = torch.load(test_file, map_location="cpu")
    tokens = data.cpu().numpy().flatten()

    print(f"\nTotal tokens: {len(tokens)}")

    print("\n" + "=" * 80)
    print("TOKENS 128-158 (skipping prefix, showing post-generation):")
    print("=" * 80)
    for i in range(128, min(158, len(tokens))):
        token = tokens[i]
        event = vocabulary.get(token, "UNKNOWN")
        print(f"  {i:3d}: token={token:4d} -> {event}")

    print("\n" + "=" * 80)
    print("TOKENS 0-30 (prefix region):")
    print("=" * 80)
    for i in range(0, min(30, len(tokens))):
        token = tokens[i]
        event = vocabulary.get(token, "UNKNOWN")
        print(f"  {i:3d}: token={token:4d} -> {event}")

    # Test the dump function
    print("\n" + "=" * 80)
    print("TESTING REMI dump() on tokens 128-178:")
    print("=" * 80)
    excerpt = tokens[128:178]

    # Filter out end-of-track
    filtered = []
    for token in excerpt:
        event = vocabulary.get(token, "")
        if event != "end-of-track":
            filtered.append(token)

    print(f"Tokens before filtering: {len(excerpt)}")
    print(f"Tokens after filtering: {len(filtered)}")

    if filtered:
        try:
            txt = representation_remi.dump(np.array(filtered), vocabulary)
            print("\nDumped text:")
            print(txt[:500])
        except Exception as e:
            print(f"\nError during dump: {e}")

    # Check event type distribution
    print("\n" + "=" * 80)
    print("EVENT TYPE DISTRIBUTION (tokens 128-528):")
    print("=" * 80)
    event_counts = {}
    for i in range(128, min(528, len(tokens))):
        token = tokens[i]
        event = vocabulary.get(token, "UNKNOWN")
        event_type = event.split("_")[0] if "_" in event else event
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type:20s}: {count:4d}")


if __name__ == "__main__":
    main()
