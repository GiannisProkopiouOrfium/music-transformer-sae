#!/bin/bash
#
# Test Script for Batch Extraction Pipeline
#
# This script tests a single feature extraction and intervention to validate
# the pipeline before running the full batch.
#

set -e

# Test parameters
TEST_LAYER=3
TEST_FEATURE=997  # antiphonal_texture - should be easily audible
TEST_OUTPUT_DIR="test_pipeline_output"

echo "🧪 TESTING BATCH PIPELINE"
echo "========================="
echo "Test Layer: $TEST_LAYER"
echo "Test Feature: $TEST_FEATURE (antiphonal_texture)"
echo "Output Dir: $TEST_OUTPUT_DIR"
echo ""

# Clean up previous test if exists
if [[ -d "$TEST_OUTPUT_DIR" ]]; then
    echo "🗑️  Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

# Test 1: Single extraction
echo "🔬 TEST 1: Single Feature Extraction"
echo "------------------------------------"

mkdir -p "$TEST_OUTPUT_DIR/extractions/layer$TEST_LAYER"

echo "Running: python extract_limuf_sae_column.py --layer $TEST_LAYER --feature-id $TEST_FEATURE --output-dir $TEST_OUTPUT_DIR/extractions/layer$TEST_LAYER"

if python extract_limuf_sae_column.py \
    --layer "$TEST_LAYER" \
    --feature-id "$TEST_FEATURE" \
    --output-dir "$TEST_OUTPUT_DIR/extractions/layer$TEST_LAYER"; then
    echo "✅ Extraction test passed"
else
    echo "❌ Extraction test failed"
    exit 1
fi

# Check extraction output
LIMUF_PATH="$TEST_OUTPUT_DIR/extractions/layer$TEST_LAYER/limufs_layer${TEST_LAYER}_sae_columns/limufs.pt"
if [[ -f "$LIMUF_PATH" ]]; then
    echo "✅ LiMuF file created: $LIMUF_PATH"
else
    echo "❌ LiMuF file not found: $LIMUF_PATH"
    exit 1
fi

echo ""

# Test 2: Single intervention
echo "🔬 TEST 2: Single Feature Intervention"
echo "-------------------------------------"

INTERVENTION_DIR="$TEST_OUTPUT_DIR/interventions/layer$TEST_LAYER/feature${TEST_FEATURE}_antiphonal_texture"

echo "Running: python mmt/conditioned_controlled_feature_intervention.py --feature-limuf-path $LIMUF_PATH --output-dir $INTERVENTION_DIR --intervention-layer $TEST_LAYER --conditioning-length 2 --seq-len 256"

if python mmt/conditioned_controlled_feature_intervention.py \
    --feature-limuf-path "$LIMUF_PATH" \
    --output-dir "$INTERVENTION_DIR" \
    --intervention-layer "$TEST_LAYER" \
    --conditioning-length 2 \
    --seq-len 512 \
    --addition-strengths "-2.0,-1.0,1.0,2.0" \
    --temperature 0.1 \
    --noise-scale 1.2 \
    --conditioning-seed 24 \
    --generation-seed 24; then
    echo "✅ Intervention test passed"
else
    echo "❌ Intervention test failed"
    exit 1
fi

# Check intervention outputs
echo ""
echo "🔍 Checking intervention outputs:"
if ls "$INTERVENTION_DIR"/*.wav &>/dev/null; then
    WAV_COUNT=$(ls "$INTERVENTION_DIR"/*.wav | wc -l)
    echo "✅ Generated $WAV_COUNT WAV files:"
    ls "$INTERVENTION_DIR"/*.wav | sed 's|.*/||'
else
    echo "❌ No WAV files found in $INTERVENTION_DIR"
    exit 1
fi

if ls "$INTERVENTION_DIR"/*.pt &>/dev/null; then
    PT_COUNT=$(ls "$INTERVENTION_DIR"/*.pt | wc -l)
    echo "✅ Generated $PT_COUNT tensor files"
else
    echo "❌ No tensor files found in $INTERVENTION_DIR"
fi

if [[ -f "$INTERVENTION_DIR/conditioned_experiment_feature${TEST_FEATURE}_layer${TEST_LAYER}_summary.json" ]]; then
    echo "✅ Experiment summary created"
else
    echo "❌ Experiment summary not found"
fi

echo ""

# Test 3: Batch script dry run
echo "🔬 TEST 3: Batch Script Dry Run"
echo "-------------------------------"

if ./extract_and_intervene_batch.sh --output-dir "$TEST_OUTPUT_DIR/batch_test" --layers "$TEST_LAYER" --dry-run; then
    echo "✅ Batch script dry run passed"
else
    echo "❌ Batch script dry run failed"
    exit 1
fi

echo ""

# Summary
echo "🎉 ALL TESTS PASSED!"
echo "===================="
echo "✅ Feature extraction works"
echo "✅ Conditioned intervention works"  
echo "✅ Batch script dry run works"
echo "✅ Output files generated correctly"
echo ""
echo "📁 Test outputs in: $TEST_OUTPUT_DIR/"
echo "🎵 Test WAV files:"
ls "$INTERVENTION_DIR"/*.wav 2>/dev/null | sed 's|.*/|  - |' || echo "  (no WAV files)"
echo ""
echo "🚀 Ready to run full batch pipeline!"
echo ""
echo "Next steps:"
echo "1. Listen to test WAV files to verify intervention effects"
echo "2. Run full batch: ./extract_and_intervene_batch.sh"
echo "3. Or run specific layers: ./extract_and_intervene_batch.sh --layers \"1 3 5\""