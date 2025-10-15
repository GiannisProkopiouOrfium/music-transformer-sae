#!/bin/bash
# Complete Contrasting Interventions Pipeline Runner
# 
# This script runs the complete pipeline:
# 1. Find contrasting songs
# 2. Run interventions
# 3. Open listening guide
#
# Usage:
#   ./run_complete_pipeline.sh 3 182           # Layer 3, Feature 182 (test mode)
#   ./run_complete_pipeline.sh 3 "182 855"     # Layer 3, Multiple features
#   ./run_complete_pipeline.sh 1 325 10        # Layer 1, Feature 325, 10 songs

set -e  # Exit on error

# Configuration
LAYER=${1:-3}
FEATURES=${2:-"182"}
NUM_SONGS=${3:-1}
CONDITIONING_LENGTH=${4:-4}
TOP_N=${5:-20}

# Paths (adjust if running from different location)
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAE_PATH="${PARENT_DIR}/exp/sod/ape/sae_models/sae_layer_2048d.pt"
DIVERSITY_REPORT="${PARENT_DIR}/layer_interpretations/enhanced_diversity_report_layer${LAYER}.json"
DATA_DIR_JSON="${PARENT_DIR}/data/sod/processed/json/Kunstderfuge"
DATA_DIR_NOTES="${PARENT_DIR}/data/sod/processed/notes"

# Output directories
OUTPUT_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRASTING_SONGS_DIR="${OUTPUT_BASE}/contrasting_songs_results"
INTERVENTION_DIR="${OUTPUT_BASE}/contrasting_intervention_results_layer${LAYER}"

echo "=========================================="
echo "Contrasting Interventions Pipeline"
echo "=========================================="
echo "Layer: ${LAYER}"
echo "Features: ${FEATURES}"
echo "Number of songs: ${NUM_SONGS}"
echo "Conditioning length: ${CONDITIONING_LENGTH} beats"
echo "Top N contrasting: ${TOP_N}"
echo ""

# Step 1: Find contrasting songs
echo "=========================================="
echo "STEP 1: Finding Contrasting Songs"
echo "=========================================="

CONTRASTING_SONGS_FILE="${CONTRASTING_SONGS_DIR}/contrasting_songs_layer${LAYER}_top${TOP_N}.txt"

if [ ! -f "$CONTRASTING_SONGS_FILE" ]; then
    echo "Running contrasting song finder..."
    python find_contrasting_songs.py \
        --layer ${LAYER} \
        --features ${FEATURES} \
        --sae-path "${SAE_PATH}" \
        --diversity-report "${DIVERSITY_REPORT}" \
        --data-dir "${DATA_DIR_JSON}" \
        --output-dir "${CONTRASTING_SONGS_DIR}" \
        --top-n ${TOP_N}
    
    echo ""
    echo "✅ Contrasting songs identified!"
else
    echo "⏭️  Contrasting songs already found: ${CONTRASTING_SONGS_FILE}"
fi

# Check if file was created
if [ ! -f "$CONTRASTING_SONGS_FILE" ]; then
    echo "❌ Error: Contrasting songs file not created!"
    exit 1
fi

# Show top 5 songs
echo ""
echo "Top 5 contrasting songs:"
head -5 "$CONTRASTING_SONGS_FILE"
echo ""

# Step 2: Run interventions
echo "=========================================="
echo "STEP 2: Running Interventions"
echo "=========================================="

python run_contrasting_interventions.py \
    --layer ${LAYER} \
    --features ${FEATURES} \
    --contrasting-songs "${CONTRASTING_SONGS_FILE}" \
    --data-dir "${DATA_DIR_NOTES}" \
    --output-dir "${INTERVENTION_DIR}" \
    --num-songs ${NUM_SONGS} \
    --conditioning-length ${CONDITIONING_LENGTH} \
    --addition-strengths -2.0 -1.0 0.0 1.0 2.0

echo ""
echo "✅ Interventions complete!"

# Step 3: Open listening guide
echo ""
echo "=========================================="
echo "STEP 3: Opening Listening Guide"
echo "=========================================="

LISTENING_GUIDE="${INTERVENTION_DIR}/listening_guide_layer${LAYER}.html"

if [ -f "$LISTENING_GUIDE" ]; then
    echo "Opening: ${LISTENING_GUIDE}"
    
    # Open in browser (platform-specific)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "${LISTENING_GUIDE}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "${LISTENING_GUIDE}"
    else
        echo "Please open manually: ${LISTENING_GUIDE}"
    fi
else
    echo "⚠️  Warning: Listening guide not found at ${LISTENING_GUIDE}"
fi

echo ""
echo "=========================================="
echo "🎉 Pipeline Complete!"
echo "=========================================="
echo "Results saved to: ${INTERVENTION_DIR}"
echo ""
echo "Next steps:"
echo "1. Listen to the generated audio in the HTML guide"
echo "2. Compare baseline (0.0) vs +2.0 for maximum contrast"
echo "3. Try with more songs: ./run_complete_pipeline.sh ${LAYER} \"${FEATURES}\" 5"
echo ""
