#!/bin/bash
#
# Enhanced Batch Feature Extraction and Intervention Pipeline
#
# This enhanced version supports:
# - Configuration file-based benchmarks
# - Easy parameter customization
# - Multiple benchmark management
# - Automated result organization
#
# Usage:
#   ./extract_and_intervene_batch_enhanced.sh [--config CONFIG_FILE] [OPTIONS]
#   ./extract_and_intervene_batch_enhanced.sh --create-config my_benchmark
#

set -e  # Exit on any error

# Default parameters (fallback if no config file)
DEFAULT_OUTPUT_DIR="batch_extractions_interventions"
DEFAULT_LAYERS="1 3 5"
DEFAULT_CONDITIONING_LENGTH=2
DEFAULT_SEQ_LEN=512
DEFAULT_TEMPERATURE=0.1
DEFAULT_NOISE_SCALE=1.2
DEFAULT_SEEDS=24
DEFAULT_STRENGTHS="-2.0,-1.0,1.0,2.0"

# Default features (same as original)
declare -A DEFAULT_LAYER_1_FEATURES=(
    ["325"]="rhythmic_displacement"
    ["256"]="dynamic_contrasts" 
    ["1323"]="wide_pitch_range"
)

declare -A DEFAULT_LAYER_3_FEATURES=(
    ["997"]="antiphonal_texture"
    ["855"]="dynamic_contrasts"
    ["182"]="steady_pulse"
)

declare -A DEFAULT_LAYER_5_FEATURES=(
    ["471"]="unison_doubling"
    ["1950"]="dynamic_contrasts_layer5"
    ["904"]="rhythmic_augmentation"
)

# Initialize with defaults
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
LAYERS="$DEFAULT_LAYERS"
CONDITIONING_LENGTH="$DEFAULT_CONDITIONING_LENGTH"
SEQ_LEN="$DEFAULT_SEQ_LEN"
TEMPERATURE="$DEFAULT_TEMPERATURE"
NOISE_SCALE="$DEFAULT_NOISE_SCALE"
SEEDS="$DEFAULT_SEEDS"
STRENGTHS="$DEFAULT_STRENGTHS"

SKIP_EXTRACTION=false
SKIP_INTERVENTION=false
RESUME=false
DRY_RUN=false
CONFIG_FILE=""
CREATE_CONFIG=""

# Copy default features
declare -A LAYER_1_FEATURES
declare -A LAYER_3_FEATURES  
declare -A LAYER_5_FEATURES

for key in "${!DEFAULT_LAYER_1_FEATURES[@]}"; do
    LAYER_1_FEATURES[$key]="${DEFAULT_LAYER_1_FEATURES[$key]}"
done

for key in "${!DEFAULT_LAYER_3_FEATURES[@]}"; do
    LAYER_3_FEATURES[$key]="${DEFAULT_LAYER_3_FEATURES[$key]}"
done

for key in "${!DEFAULT_LAYER_5_FEATURES[@]}"; do
    LAYER_5_FEATURES[$key]="${DEFAULT_LAYER_5_FEATURES[$key]}"
done

# Function to create a new config file
create_config_file() {
    local config_name="$1"
    local config_file="benchmark_configs/${config_name}_config.sh"
    
    mkdir -p benchmark_configs
    
    cat > "$config_file" << 'EOF'
#!/bin/bash
#
# Benchmark Configuration: BENCHMARK_NAME_PLACEHOLDER
#
# Customize this file to create your specific benchmark
#

# Benchmark identification
BENCHMARK_NAME="BENCHMARK_NAME_PLACEHOLDER"
BENCHMARK_DESCRIPTION="Custom benchmark - modify this description"

# Output configuration
OUTPUT_BASE_DIR="benchmarks"
OUTPUT_SUBDIR="${BENCHMARK_NAME}_$(date +%Y%m%d_%H%M%S)"

# Processing parameters
LAYERS="1 3 5"
CONDITIONING_LENGTH=2
SEQ_LEN=512
TEMPERATURE=0.1
NOISE_SCALE=1.2
SEEDS=24
STRENGTHS="-2.0,-1.0,1.0,2.0"

# Feature selections (modify these arrays for your benchmark)
declare -A LAYER_1_FEATURES=(
    ["325"]="rhythmic_displacement"
    ["256"]="dynamic_contrasts" 
    ["1323"]="wide_pitch_range"
    # Add your custom features:
    # ["XXX"]="your_feature_name"
)

declare -A LAYER_3_FEATURES=(
    ["997"]="antiphonal_texture"
    ["855"]="dynamic_contrasts"
    ["182"]="steady_pulse"
    # Add your custom features:
    # ["XXX"]="your_feature_name"
)

declare -A LAYER_5_FEATURES=(
    ["471"]="unison_doubling"
    ["1950"]="dynamic_contrasts_layer5"
    ["904"]="rhythmic_augmentation"
    # Add your custom features:
    # ["XXX"]="your_feature_name"
)

# Execution options
SKIP_EXTRACTION=false
SKIP_INTERVENTION=false
DRY_RUN=false
RESUME=false

# Benchmark metadata
EVALUATION_NOTES="Add notes about what you expect from this benchmark"
EXPECTED_DIFFERENCES="Describe expected differences from baseline"
RESEARCH_QUESTIONS="What questions does this benchmark address?"

echo "📋 LOADING BENCHMARK CONFIG: $BENCHMARK_NAME"
echo "Description: $BENCHMARK_DESCRIPTION"
echo "Output: $OUTPUT_BASE_DIR/$OUTPUT_SUBDIR"
EOF

    # Replace placeholder with actual name
    sed -i.bak "s/BENCHMARK_NAME_PLACEHOLDER/$config_name/g" "$config_file"
    rm "$config_file.bak"
    
    echo "✅ Created new benchmark config: $config_file"
    echo "📝 Edit this file to customize your benchmark parameters"
    echo "🚀 Run with: $0 --config $config_file"
    
    return 0
}

# Function to load config file
load_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        echo "❌ Config file not found: $config_file"
        exit 1
    fi
    
    echo "📋 Loading configuration from: $config_file"
    
    # Source the config file to load variables
    source "$config_file"
    
    # Override defaults with config values
    if [[ -n "$OUTPUT_BASE_DIR" && -n "$OUTPUT_SUBDIR" ]]; then
        OUTPUT_DIR="$OUTPUT_BASE_DIR/$OUTPUT_SUBDIR"
    fi
    
    echo "✅ Configuration loaded successfully"
    echo "📊 Benchmark: ${BENCHMARK_NAME:-unnamed}"  
    echo "📝 Description: ${BENCHMARK_DESCRIPTION:-none}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --create-config)
            CREATE_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --skip-intervention)
            SKIP_INTERVENTION=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Enhanced Batch Feature Extraction and Intervention Pipeline"
            echo ""
            echo "Usage:"
            echo "  $0 [OPTIONS]                    # Run with default or overridden parameters"
            echo "  $0 --config CONFIG_FILE        # Run with configuration file"
            echo "  $0 --create-config NAME         # Create new benchmark configuration"
            echo ""
            echo "Configuration Mode:"
            echo "  --config FILE                   Load parameters from configuration file"
            echo "  --create-config NAME            Create new benchmark config template"
            echo ""
            echo "Direct Options (override config):"
            echo "  --output-dir DIR                Output directory"
            echo "  --layers \"1 3 5\"                Layers to process"
            echo "  --skip-extraction               Skip extraction step"
            echo "  --skip-intervention             Skip intervention step"
            echo "  --resume                        Resume from previous run"
            echo "  --dry-run                       Show what would be done"
            echo "  --temperature F                 Generation temperature"
            echo "  --seq-len N                     Sequence length"
            echo ""
            echo "Examples:"
            echo "  $0 --create-config high_temp_benchmark"
            echo "  $0 --config benchmark_configs/high_temp_benchmark_config.sh"
            echo "  $0 --config my_config.sh --dry-run"
            echo "  $0 --temperature 0.5 --seq-len 1024  # Override defaults"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Handle config creation
if [[ -n "$CREATE_CONFIG" ]]; then
    create_config_file "$CREATE_CONFIG"
    exit 0
fi

# Load configuration if specified
if [[ -n "$CONFIG_FILE" ]]; then
    load_config "$CONFIG_FILE"
fi

# [REST OF THE ORIGINAL SCRIPT FUNCTIONS GO HERE - UNCHANGED]
# ... (all the logging functions, create_output_structure, get_layer_features, etc.)

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*" | tee -a "$OUTPUT_DIR/batch_extraction.log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$OUTPUT_DIR/batch_extraction.log" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $*" | tee -a "$OUTPUT_DIR/batch_extraction.log"
}

# Create output directory structure
create_output_structure() {
    local base_dir="$1"
    
    mkdir -p "$base_dir"/{extractions,interventions,progress,logs}
    
    for layer in $LAYERS; do
        mkdir -p "$base_dir/extractions/layer$layer"
        mkdir -p "$base_dir/interventions/layer$layer"
    done
    
    # Save benchmark metadata if available
    if [[ -n "$BENCHMARK_NAME" ]]; then
        cat > "$base_dir/benchmark_metadata.json" << EOF
{
    "benchmark_name": "${BENCHMARK_NAME}",
    "benchmark_description": "${BENCHMARK_DESCRIPTION:-N/A}",
    "evaluation_notes": "${EVALUATION_NOTES:-N/A}",
    "expected_differences": "${EXPECTED_DIFFERENCES:-N/A}",
    "research_questions": "${RESEARCH_QUESTIONS:-N/A}",
    "parameters": {
        "layers": "${LAYERS}",
        "conditioning_length": ${CONDITIONING_LENGTH},
        "seq_len": ${SEQ_LEN},
        "temperature": ${TEMPERATURE},
        "noise_scale": ${NOISE_SCALE},
        "seeds": ${SEEDS},
        "strengths": "${STRENGTHS}"
    },
    "timestamp": "$(date -Iseconds)"
}
EOF
        log_info "Saved benchmark metadata to $base_dir/benchmark_metadata.json"
    fi
    
    log_info "Created output directory structure in $base_dir"
}

# Get features for a layer
get_layer_features() {
    local layer="$1"
    
    case $layer in
        1)
            for feature_id in "${!LAYER_1_FEATURES[@]}"; do
                echo "$feature_id:${LAYER_1_FEATURES[$feature_id]}"
            done
            ;;
        3)
            for feature_id in "${!LAYER_3_FEATURES[@]}"; do
                echo "$feature_id:${LAYER_3_FEATURES[$feature_id]}"
            done
            ;;
        5)
            for feature_id in "${!LAYER_5_FEATURES[@]}"; do
                echo "$feature_id:${LAYER_5_FEATURES[$feature_id]}"
            done
            ;;
        *)
            log_error "Unknown layer: $layer"
            return 1
            ;;
    esac
}

# [Continue with rest of original functions...]
# Check if extraction completed
extraction_completed() {
    local layer="$1"
    local feature_id="$2"
    local progress_file="$OUTPUT_DIR/progress/completed_extractions.txt"
    
    if [[ -f "$progress_file" ]]; then
        grep -q "layer${layer}_feature${feature_id}" "$progress_file" 2>/dev/null
    else
        return 1
    fi
}

# Mark extraction completed
mark_extraction_completed() {
    local layer="$1"
    local feature_id="$2"
    local progress_file="$OUTPUT_DIR/progress/completed_extractions.txt"
    
    echo "layer${layer}_feature${feature_id}" >> "$progress_file"
}

# Check if intervention completed
intervention_completed() {
    local layer="$1"
    local feature_id="$2"
    local progress_file="$OUTPUT_DIR/progress/completed_interventions.txt"
    
    if [[ -f "$progress_file" ]]; then
        grep -q "layer${layer}_feature${feature_id}" "$progress_file" 2>/dev/null
    else
        return 1
    fi
}

# Mark intervention completed
mark_intervention_completed() {
    local layer="$1" 
    local feature_id="$2"
    local progress_file="$OUTPUT_DIR/progress/completed_interventions.txt"
    
    echo "layer${layer}_feature${feature_id}" >> "$progress_file"
}

# Run extraction for a feature
run_extraction() {
    local layer="$1"
    local feature_id="$2"
    local extraction_dir="$OUTPUT_DIR/extractions/layer$layer"
    
    log_info "🔄 Extracting Layer $layer, Feature $feature_id"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 [DRY RUN] Would extract Layer $layer, Feature $feature_id"
        return 0
    fi
    
    # Run extraction command
    if python extract_limuf_sae_column.py \
        --layer "$layer" \
        --feature-id "$feature_id" \
        --output-dir "$extraction_dir" 2>&1 | tee -a "$OUTPUT_DIR/logs/extraction_layer${layer}_feature${feature_id}.log"; then
        
        log_success "Extraction completed for Layer $layer, Feature $feature_id"
        mark_extraction_completed "$layer" "$feature_id"
        return 0
    else
        log_error "Extraction failed for Layer $layer, Feature $feature_id"
        return 1
    fi
}

# Run intervention for a feature
run_intervention() {
    local layer="$1"
    local feature_id="$2"
    local feature_name="$3"
    
    local limuf_path="$OUTPUT_DIR/extractions/layer$layer/limufs_layer${layer}_sae_columns/limufs.pt"
    local intervention_dir="$OUTPUT_DIR/interventions/layer$layer/feature${feature_id}_${feature_name}"
    
    log_info "🎵 Running interventions for Layer $layer, Feature $feature_id ($feature_name)"
    
    if [[ ! -f "$limuf_path" ]]; then
        log_error "LiMuF file not found: $limuf_path"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🎵 [DRY RUN] Would run interventions for Layer $layer, Feature $feature_id"
        return 0
    fi
    
    # Run intervention command
    if python mmt/conditioned_controlled_feature_intervention.py \
        --feature-limuf-path "$limuf_path" \
        --output-dir "$intervention_dir" \
        --intervention-layer "$layer" \
        --addition-strengths "$STRENGTHS" \
        --conditioning-length "$CONDITIONING_LENGTH" \
        --seq-len "$SEQ_LEN" \
        --temperature "$TEMPERATURE" \
        --noise-scale "$NOISE_SCALE" \
        --conditioning-seed "$SEEDS" \
        --generation-seed "$SEEDS" 2>&1 | tee -a "$OUTPUT_DIR/logs/intervention_layer${layer}_feature${feature_id}.log"; then
        
        log_success "Interventions completed for Layer $layer, Feature $feature_id"
        mark_intervention_completed "$layer" "$feature_id"
        return 0
    else
        log_error "Intervention failed for Layer $layer, Feature $feature_id"
        return 1
    fi
}

# Enhanced main execution with benchmark context
main() {
    log_info "🚀 ENHANCED BATCH FEATURE EXTRACTION AND INTERVENTION PIPELINE"
    log_info "================================================================================"
    
    if [[ -n "$BENCHMARK_NAME" ]]; then
        log_info "📋 BENCHMARK: $BENCHMARK_NAME"
        log_info "📝 DESCRIPTION: ${BENCHMARK_DESCRIPTION:-N/A}"
        if [[ -n "$RESEARCH_QUESTIONS" ]]; then
            log_info "❓ RESEARCH QUESTIONS: $RESEARCH_QUESTIONS"
        fi
    fi
    
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Processing layers: $LAYERS"
    log_info "Skip extraction: $SKIP_EXTRACTION"
    log_info "Skip intervention: $SKIP_INTERVENTION"
    log_info "Resume mode: $RESUME"
    log_info "Dry run: $DRY_RUN"
    log_info "Parameters: conditioning_length=$CONDITIONING_LENGTH, seq_len=$SEQ_LEN, temperature=$TEMPERATURE, noise_scale=$NOISE_SCALE"
    
    # Create output structure (with metadata)
    create_output_structure "$OUTPUT_DIR"
    
    # [Rest of main function same as original...]
    # Count total features
    local total_features=0
    for layer in $LAYERS; do
        local layer_feature_count=$(get_layer_features "$layer" | wc -l)
        total_features=$((total_features + layer_feature_count))
    done
    
    log_info "📊 PROCESSING PLAN:"
    for layer in $LAYERS; do
        local features=$(get_layer_features "$layer")
        local feature_count=$(echo "$features" | wc -l)
        log_info "Layer $layer: $feature_count features"
        
        while IFS=':' read -r feature_id feature_name; do
            log_info "  - Feature $feature_id: $feature_name"
        done <<< "$features"
    done
    
    log_info "🎯 STARTING PROCESSING ($total_features total features)"
    
    local current_feature=0
    local successful_extractions=0
    local failed_extractions=0
    local successful_interventions=0
    local failed_interventions=0
    
    # Process each layer and feature
    for layer in $LAYERS; do
        local features=$(get_layer_features "$layer")
        local layer_feature_count=$(echo "$features" | wc -l)
        
        log_info ""
        log_info "🔄 PROCESSING LAYER $layer ($layer_feature_count features)"
        
        while IFS=':' read -r feature_id feature_name; do
            current_feature=$((current_feature + 1))
            
            log_info ""
            log_info "📍 [$current_feature/$total_features] Layer $layer, Feature $feature_id ($feature_name)"
            
            # Step 1: Extraction
            if [[ "$SKIP_EXTRACTION" != "true" ]]; then
                if [[ "$RESUME" == "true" ]] && extraction_completed "$layer" "$feature_id"; then
                    log_info "⏭️  Extraction already completed: layer${layer}_feature${feature_id}"
                else
                    if run_extraction "$layer" "$feature_id"; then
                        successful_extractions=$((successful_extractions + 1))
                    else
                        failed_extractions=$((failed_extractions + 1))
                        log_error "❌ Skipping intervention for failed extraction: layer${layer}_feature${feature_id}"
                        continue
                    fi
                fi
            fi
            
            # Step 2: Intervention
            if [[ "$SKIP_INTERVENTION" != "true" ]]; then
                if [[ "$RESUME" == "true" ]] && intervention_completed "$layer" "$feature_id"; then
                    log_info "⏭️  Intervention already completed: layer${layer}_feature${feature_id}"
                else
                    if run_intervention "$layer" "$feature_id" "$feature_name"; then
                        successful_interventions=$((successful_interventions + 1))
                    else
                        failed_interventions=$((failed_interventions + 1))
                    fi
                fi
            fi
            
        done <<< "$features"
    done
    
    # Enhanced final results with benchmark context
    log_info ""
    log_info "================================================================================"
    if [[ -n "$BENCHMARK_NAME" ]]; then
        log_info "🎉 BENCHMARK '$BENCHMARK_NAME' PROCESSING COMPLETE!"
    else
        log_info "🎉 BATCH PROCESSING COMPLETE!"
    fi
    log_info "✅ Successful extractions: $successful_extractions"
    log_info "❌ Failed extractions: $failed_extractions" 
    log_info "✅ Successful interventions: $successful_interventions"
    log_info "❌ Failed interventions: $failed_interventions"
    log_info ""
    log_info "📁 Output structure created in: $OUTPUT_DIR"
    
    if [[ -n "$EXPECTED_DIFFERENCES" ]]; then
        log_info "🔍 Expected differences: $EXPECTED_DIFFERENCES"
    fi
    
    log_info "🎵 Ready for systematic evaluation!"
    
    # Return success if no failures
    local total_failures=$((failed_extractions + failed_interventions))
    return $total_failures
}

# Run main function
main
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    log_success "Processing completed successfully!"
else
    log_error "Processing completed with $exit_code failures"
fi

exit $exit_code