#!/bin/bash
#
# Extended Strength Range Benchmark Configuration
#
# This benchmark tests a wider range of intervention strengths
# to find optimal intervention levels and detect saturation points
#

# Benchmark identification
BENCHMARK_NAME="extended_strengths_v1"
BENCHMARK_DESCRIPTION="Testing wider intervention strength range to find optimal intervention levels"

# Output configuration
OUTPUT_BASE_DIR="benchmarks"
OUTPUT_SUBDIR="${BENCHMARK_NAME}_$(date +%Y%m%d_%H%M%S)"

# Processing parameters - EXTENDED STRENGTH RANGE
LAYERS="3 5"
CONDITIONING_LENGTH=2
SEQ_LEN=512
TEMPERATURE=0.1  # Keep same as baseline for comparison
NOISE_SCALE=1.2
SEEDS=24
STRENGTHS="-0.75,-0.5,0.5,0.75"  # Much wider range

# Focus on strongest-effect features from original evaluation
declare -A LAYER_1_FEATURES=(
    ["325"]="rhythmic_displacement"  # Strong intervention effects
    ["256"]="dynamic_contrasts"     # Clear interpretability
)

declare -A LAYER_3_FEATURES=(
    ["182"]="steady_pulse"          # Consistent effects
    ["997"]="antiphonal_texture"    # Strong layer 3 effects
)

declare -A LAYER_5_FEATURES=(
    ["471"]="unison_doubling"       # Strong layer 5 effects
    ["904"]="rhythmic_augmentation" # Clear musical impact
)

# Execution options
SKIP_EXTRACTION=false
SKIP_INTERVENTION=false
DRY_RUN=false
RESUME=false

# Benchmark metadata
EVALUATION_NOTES="Testing extended strength range (-0.75 to +0.75) to identify optimal intervention levels and saturation points"
EXPECTED_DIFFERENCES="Should see linear effects at low strengths, possible saturation at high strengths, asymmetric positive/negative effects"
RESEARCH_QUESTIONS="What are optimal intervention strengths? Do effects saturate? Are positive/negative interventions symmetric?"

echo "📋 LOADING BENCHMARK CONFIG: $BENCHMARK_NAME"
echo "Description: $BENCHMARK_DESCRIPTION"
echo "Output: $OUTPUT_BASE_DIR/$OUTPUT_SUBDIR"