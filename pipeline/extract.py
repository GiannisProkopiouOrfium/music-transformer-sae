#!/usr/bin/env python3
"""
Pipeline extraction module - unified interface for cross-platform SAE analysis.
"""

import sys
from pathlib import Path

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the actual implementation
from pipeline.sae_pipeline import UnifiedSAEPipeline

# Re-export for compatibility
__all__ = ["UnifiedSAEPipeline"]
