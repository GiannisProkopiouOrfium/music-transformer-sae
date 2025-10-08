#!/bin/bash
#
# Benchmark Configuration Template
#
# Copy this file and modify it to create custom benchmarks with different:
# - Features selections
# - Generation parameters  
# - Intervention strengths
# - Output configurations
#

# Benchmark identification
BENCHMARK_NAME="custom_benchmark_v1"
BENCHMARK_DESCRIPTION="Custom benchmark with different temperature and features"

# Output configuration
OUTPUT_BASE_DIR="benchmarks"
OUTPUT_SUBDIR="${BENCHMARK_NAME}_$(date +%Y%m%d_%H%M%S)"

# Processing parameters
LAYERS="1 3 5"
CONDITIONING_LENGTH=3
SEQ_LEN=1024
TEMPERATURE=0.5
NOISE_SCALE=1.5
SEEDS=42
STRENGTHS="-3.0,-1.5,-0.5,0.5,1.5,3.0"

# Feature selections (easily customizable)
declare -A LAYER_1_FEATURES=(
    ["256"]="dynamic_contrasts"
    ["325"]="rhythmic_displacement"
    ["500"]="custom_feature_name"  # Add your custom features here
)

declare -A LAYER_3_FEATURES=(
    ["182"]="steady_pulse"
    ["855"]="dynamic_contrasts"
    ["1000"]="another_custom_feature"
)

declare -A LAYER_5_FEATURES=(
    ["471"]="unison_doubling"
    ["904"]="rhythmic_augmentation"
    ["1500"]="third_custom_feature"
)

# Execution options
SKIP_EXTRACTION=false
SKIP_INTERVENTION=false
DRY_RUN=false
RESUME=false

# Additional benchmark-specific parameters
EVALUATION_NOTES="Testing higher temperature (0.5) with longer sequences (1024) and extended strength range"
EXPECTED_DIFFERENCES="Higher temperature should increase pitch entropy, longer sequences may improve musical coherence"

echo "📋 BENCHMARK CONFIGURATION: $BENCHMARK_NAME"
echo "Description: $BENCHMARK_DESCRIPTION"
echo "Output will be saved to: $OUTPUT_BASE_DIR/$OUTPUT_SUBDIR"
echo "Notes: $EVALUATION_NOTES"