#!/bin/bash
#
# Batch Feature Extraction and Intervention Pipeline (Shell Version)
#
# This script automates the extraction of Linear Musical Features (LiMuFs) from SAE
# columns and performs conditioned controlled interventions for systematic evaluation.
#
# Usage:
#   ./extract_and_intervene_batch.sh [--output-dir DIR] [--layers "1 3 5"] [--resume]
#

set -e  # Exit on any error

# Default parameters
DEFAULT_OUTPUT_DIR="batch_extractions_interventions"
DEFAULT_LAYERS="1 3 5"
DEFAULT_CONDITIONING_LENGTH=2
DEFAULT_SEQ_LEN=512
DEFAULT_TEMPERATURE=0.1
DEFAULT_NOISE_SCALE=1.2
DEFAULT_STRENGTHS="-2.0,-1.0,1.0,2.0"

# Note: Seeds are auto-generated per feature as (layer * 1000 + feature_id)
# This ensures different musical content per feature while maintaining reproducibility

# Selected features per layer (based on diversity reports)
declare -A LAYER_1_FEATURES=(
    ["325"]="rhythmic_displacement"
    ["256"]="dynamic_contrasts" 
    ["1323"]="wide_pitch_range"
)

declare -A LAYER_3_FEATURES=(
    ["997"]="antiphonal_texture"
    ["855"]="dynamic_contrasts"
    ["182"]="steady_pulse"
)

declare -A LAYER_5_FEATURES=(
    ["471"]="unison_doubling"
    ["1950"]="dynamic_contrasts_layer5"
    ["904"]="rhythmic_augmentation"
)

# Parse arguments
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
LAYERS="$DEFAULT_LAYERS"
SKIP_EXTRACTION=false
SKIP_INTERVENTION=false
RESUME=false
DRY_RUN=false
CONDITIONING_LENGTH="$DEFAULT_CONDITIONING_LENGTH"
SEQ_LEN="$DEFAULT_SEQ_LEN"
TEMPERATURE="$DEFAULT_TEMPERATURE"
NOISE_SCALE="$DEFAULT_NOISE_SCALE"

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --conditioning-length)
            CONDITIONING_LENGTH="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --noise-scale)
            NOISE_SCALE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR          Output directory (default: $DEFAULT_OUTPUT_DIR)"
            echo "  --layers \"1 3 5\"          Layers to process (default: $DEFAULT_LAYERS)"
            echo "  --skip-extraction         Skip extraction step"
            echo "  --skip-intervention       Skip intervention step"
            echo "  --resume                  Resume from previous run"
            echo "  --dry-run                 Show what would be done"
            echo "  --conditioning-length N   Conditioning length (default: $DEFAULT_CONDITIONING_LENGTH)"
            echo "  --seq-len N               Sequence length (default: $DEFAULT_SEQ_LEN)"
            echo "  --temperature F           Temperature (default: $DEFAULT_TEMPERATURE)"
            echo "  --noise-scale F           Noise scale (default: $DEFAULT_NOISE_SCALE)"
            echo "  -h, --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
    local feature_specific_dir="$OUTPUT_DIR/extractions/layer$layer/feature${feature_id}"
    
    log_info "🔄 Extracting Layer $layer, Feature $feature_id"
    log_info "   Output dir: $feature_specific_dir"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 [DRY RUN] Would extract Layer $layer, Feature $feature_id to $feature_specific_dir"
        return 0
    fi
    
    # Create feature-specific extraction directory
    mkdir -p "$feature_specific_dir"
    
    # Run extraction command with feature-specific output directory
    if python extract_limuf_sae_column.py \
        --layer "$layer" \
        --feature-id "$feature_id" \
        --output-dir "$feature_specific_dir/limufs_layer${layer}_feature${feature_id}_sae_columns" 2>&1 | tee -a "$OUTPUT_DIR/logs/extraction_layer${layer}_feature${feature_id}.log"; then
        
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
    
    local limuf_path="$OUTPUT_DIR/extractions/layer$layer/feature${feature_id}/limufs_layer${layer}_feature${feature_id}_sae_columns/limufs.pt"
    local intervention_dir="$OUTPUT_DIR/interventions/layer$layer/feature${feature_id}_${feature_name}"
    
    # Generate feature-specific seeds based on layer and feature_id
    # This ensures each feature gets different musical content but maintains reproducibility
    local feature_seed=$((layer * 1000 + feature_id))
    
    log_info "🎵 Running interventions for Layer $layer, Feature $feature_id ($feature_name)"
    log_info "   Using feature-specific seed: $feature_seed"
    
    if [[ ! -f "$limuf_path" ]]; then
        log_error "LiMuF file not found: $limuf_path"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🎵 [DRY RUN] Would run interventions for Layer $layer, Feature $feature_id (seed: $feature_seed)"
        return 0
    fi
    
    # Run intervention command with feature-specific seeds
    if python mmt/conditioned_controlled_feature_intervention.py \
        --feature-limuf-path "$limuf_path" \
        --output-dir "$intervention_dir" \
        --intervention-layer "$layer" \
        --conditioning-length "$CONDITIONING_LENGTH" \
        --seq-len "$SEQ_LEN" \
        --temperature "$TEMPERATURE" \
        --noise-scale "$NOISE_SCALE" \
        --conditioning-seed "$feature_seed" \
        --generation-seed "$feature_seed" 2>&1 | tee -a "$OUTPUT_DIR/logs/intervention_layer${layer}_feature${feature_id}.log"; then
        
        log_success "Interventions completed for Layer $layer, Feature $feature_id (seed: $feature_seed)"
        mark_intervention_completed "$layer" "$feature_id"
        return 0
    else
        log_error "Intervention failed for Layer $layer, Feature $feature_id"
        return 1
    fi
}

# Main execution
main() {
    # Create output directory first so logging works
    mkdir -p "$OUTPUT_DIR"
    
    log_info "🚀 STARTING BATCH FEATURE EXTRACTION AND INTERVENTION PIPELINE"
    log_info "================================================================================"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Processing layers: $LAYERS"
    log_info "Skip extraction: $SKIP_EXTRACTION"
    log_info "Skip intervention: $SKIP_INTERVENTION"
    log_info "Resume mode: $RESUME"
    log_info "Dry run: $DRY_RUN"
    log_info "Parameters: conditioning_length=$CONDITIONING_LENGTH, seq_len=$SEQ_LEN, temperature=$TEMPERATURE, noise_scale=$NOISE_SCALE"
    log_info "Seeds: Auto-generated per feature as (layer * 1000 + feature_id) for diversity"
    
    # Create full output structure
    create_output_structure "$OUTPUT_DIR"
    
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
    
    # Final results
    log_info ""
    log_info "================================================================================"
    log_info "🎉 BATCH PROCESSING COMPLETE!"
    log_info "✅ Successful extractions: $successful_extractions"
    log_info "❌ Failed extractions: $failed_extractions"
    log_info "✅ Successful interventions: $successful_interventions"
    log_info "❌ Failed interventions: $failed_interventions"
    log_info ""
    log_info "📁 Output structure created in: $OUTPUT_DIR"
    log_info "🎵 Ready for systematic evaluation!"
    
    # Return success if no failures
    local total_failures=$((failed_extractions + failed_interventions))
    return $total_failures
}

# Run main function
main
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    log_success "Batch processing completed successfully!"
else
    log_error "Batch processing completed with $exit_code failures"
fi

exit $exit_code