#!/usr/bin/env python3
"""
Simple Pipeline Runner for Music Transformer SAE Analysis

This script provides an easy-to-use interface for running the SAE pipeline.
"""

import pathlib
import sys

# Add mmt module to path
sys.path.append(str(pathlib.Path(__file__).parent / "mmt"))

from mmt.pipeline.main import main

if __name__ == "__main__":
    main()
