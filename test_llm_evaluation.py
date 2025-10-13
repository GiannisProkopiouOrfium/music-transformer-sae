#!/usr/bin/env python3
"""
Quick Test: LLM Evaluation

Tests LLM evaluation on a single feature intervention to verify setup.

Usage:
    python test_llm_evaluation.py
"""

import sys
import json
from pathlib import Path

# Check if OpenAI is installed
try:
    import openai

    print("✅ OpenAI package installed")
except ImportError:
    print("❌ OpenAI package not installed")
    print("   Install with: pip install openai python-dotenv")
    sys.exit(1)

# Check if API key is available
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY not found")
    print("   Create .env file with: OPENAI_API_KEY=your_key_here")
    sys.exit(1)
else:
    print(f"✅ API key found: {api_key[:10]}...")

# Check if intervention files exist
interventions_dir = Path("batch_extractions_interventions/interventions")
if not interventions_dir.exists():
    print(f"❌ Interventions directory not found: {interventions_dir}")
    print("   Run feature extraction first")
    sys.exit(1)
else:
    print(f"✅ Interventions directory found")

# Find a test feature
test_feature = None
for layer_dir in interventions_dir.glob("layer*"):
    for feature_dir in layer_dir.glob("feature*"):
        baseline_files = list(feature_dir.glob("baseline_*.pt"))
        intervention_files = [
            f for f in feature_dir.glob("*.pt") if not f.name.startswith("baseline_")
        ]

        if baseline_files and intervention_files:
            test_feature = {
                "feature_dir": feature_dir,
                "baseline": baseline_files[0],
                "intervention": intervention_files[0],
            }
            break
    if test_feature:
        break

if not test_feature:
    print("❌ No test feature found")
    sys.exit(1)

print(f"✅ Found test feature: {test_feature['feature_dir'].name}")
print(f"   Baseline: {test_feature['baseline'].name}")
print(f"   Intervention: {test_feature['intervention'].name}")

print("\n" + "=" * 60)
print("Running LLM Evaluation Test")
print("=" * 60)

# Import and run evaluator
from llm_text_evaluation import TextBasedLLMEvaluator
from deterministic_analysis.midi_feature_extractors import MIDIFeatureExtractor
import baseline.representation_remi as representation
import torch

print("\n1. Loading representation...")
# Load the encoding
encoding = representation.get_encoding()
vocabulary = encoding["code_event_map"]
print(f"   ✅ Loaded encoding with {len(vocabulary)} tokens")

print("\n2. Extracting baseline metrics...")
feature_extractor = MIDIFeatureExtractor()

# Load baseline
baseline_data = torch.load(
    test_feature["baseline"], map_location="cpu", weights_only=False
)
baseline_tokens = (
    baseline_data["generated"] if "generated" in baseline_data else baseline_data
)
# Convert tokens to numpy array (expected format)
import numpy as np

if torch.is_tensor(baseline_tokens):
    baseline_tokens = baseline_tokens.cpu().numpy()
elif isinstance(baseline_tokens, list):
    baseline_tokens = np.array(baseline_tokens)

# Squeeze to 1D if needed (remove batch dimension)
if baseline_tokens.ndim > 1:
    baseline_tokens = baseline_tokens.squeeze()

# Decode tokens to MusPy Music object
baseline_midi = representation.decode(baseline_tokens, encoding, vocabulary)
baseline_metrics = feature_extractor.extract_all_features(baseline_midi)
print(f"   ✅ Extracted {len(baseline_metrics)} metric categories")

print("\n3. Extracting intervention metrics...")
intervention_data = torch.load(
    test_feature["intervention"], map_location="cpu", weights_only=False
)
intervention_tokens = (
    intervention_data["generated"]
    if "generated" in intervention_data
    else intervention_data
)
# Convert tokens to numpy array (expected format)
if torch.is_tensor(intervention_tokens):
    intervention_tokens = intervention_tokens.cpu().numpy()
elif isinstance(intervention_tokens, list):
    intervention_tokens = np.array(intervention_tokens)

# Squeeze to 1D if needed (remove batch dimension)
if intervention_tokens.ndim > 1:
    intervention_tokens = intervention_tokens.squeeze()

# Decode tokens to MusPy Music object
intervention_midi = representation.decode(intervention_tokens, encoding, vocabulary)
intervention_metrics = feature_extractor.extract_all_features(intervention_midi)
print(f"   ✅ Extracted {len(intervention_metrics)} metric categories")

print("\n4. Calling OpenAI API...")
evaluator = TextBasedLLMEvaluator(model="gpt-4-turbo-preview", temperature=0.3)

# Extract feature ID from directory name
import re

feature_match = re.search(r"feature(\d+)", test_feature["feature_dir"].name)
feature_id = feature_match.group(1) if feature_match else "unknown"

# Determine intervention type and strength
intervention_name = test_feature["intervention"].stem
if "ablation" in intervention_name:
    intervention_type = "ablation"
    strength = 0.0
elif "add_+" in intervention_name:
    strength_match = re.search(r"add_\+([0-9.]+)", intervention_name)
    strength = float(strength_match.group(1)) if strength_match else 1.0
    intervention_type = "addition"
elif "add_-" in intervention_name:
    strength_match = re.search(r"add_-([0-9.]+)", intervention_name)
    strength = -float(strength_match.group(1)) if strength_match else -1.0
    intervention_type = "addition"
else:
    intervention_type = "unknown"
    strength = 0.0

print(f"   Feature ID: {feature_id}")
print(f"   Intervention: {intervention_type} (strength: {strength})")

try:
    result = evaluator.evaluate_intervention(
        feature_id=feature_id,
        baseline_metrics=baseline_metrics,
        intervention_metrics=intervention_metrics,
        strength=strength,
        intervention_type=intervention_type,
        include_deterministic=False,
    )

    print("\n4. ✅ Evaluation complete!")
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Effectiveness Score: {result.effectiveness_score:.3f}")
    print(f"Quality Score:       {result.quality_score:.3f}")
    print(f"Coherence Score:     {result.coherence_score:.3f}")
    print(f"Musicality Score:    {result.musicality_score:.3f}")
    print(f"Overall Score:       {result.overall_score:.3f}")
    print(f"\nTokens Used:         {result.tokens_used}")
    estimated_cost = (result.tokens_used / 1_000_000) * 20
    print(f"Estimated Cost:      ${estimated_cost:.4f}")

    print("\n" + "=" * 60)
    print("Effectiveness Reasoning:")
    print("=" * 60)
    print(result.effectiveness_reasoning)

    print("\n" + "=" * 60)
    print("Key Observations:")
    print("=" * 60)
    for i, obs in enumerate(result.key_observations, 1):
        print(f"{i}. {obs}")

    if result.concerns:
        print("\n" + "=" * 60)
        print("Concerns:")
        print("=" * 60)
        for i, concern in enumerate(result.concerns, 1):
            print(f"{i}. {concern}")

    print("\n" + "=" * 60)
    print("✅ Test Successful!")
    print("=" * 60)
    print("\nYou can now run batch evaluation:")
    print(
        "python run_llm_batch_evaluation.py batch_extractions_interventions/interventions \\"
    )
    print("    --output-dir llm_evaluation_results")

except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
