#!/bin/bash
#
# Benchmark Manager - Easy benchmark creation and execution
#
# Usage:
#   ./benchmark_manager.sh list                    # List available benchmarks
#   ./benchmark_manager.sh create my_benchmark     # Create new benchmark config
#   ./benchmark_manager.sh run high_temperature    # Run specific benchmark
#   ./benchmark_manager.sh run all                 # Run all benchmarks
#

set -e

BENCHMARK_DIR="benchmark_configs"
ENHANCED_SCRIPT="./extract_and_intervene_batch_enhanced.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# List available benchmarks
list_benchmarks() {
    log_info "Available Benchmark Configurations:"
    echo ""
    
    if [[ ! -d "$BENCHMARK_DIR" ]]; then
        log_warning "No benchmark configs directory found. Create one with: $0 create <name>"
        return
    fi
    
    local count=0
    for config_file in "$BENCHMARK_DIR"/*_config.sh; do
        if [[ -f "$config_file" ]]; then
            local basename=$(basename "$config_file" _config.sh)
            local description="N/A"
            
            # Try to extract description
            if grep -q "BENCHMARK_DESCRIPTION=" "$config_file"; then
                description=$(grep "BENCHMARK_DESCRIPTION=" "$config_file" | cut -d'"' -f2)
            fi
            
            echo "  📋 $basename"
            echo "     Description: $description"
            echo "     Config: $config_file"
            echo ""
            count=$((count + 1))
        fi
    done
    
    if [[ $count -eq 0 ]]; then
        log_warning "No benchmark configs found in $BENCHMARK_DIR/"
        echo "Create one with: $0 create <benchmark_name>"
    else
        log_success "Found $count benchmark configurations"
    fi
}

# Create new benchmark
create_benchmark() {
    local benchmark_name="$1"
    
    if [[ -z "$benchmark_name" ]]; then
        log_error "Please provide a benchmark name"
        echo "Usage: $0 create <benchmark_name>"
        exit 1
    fi
    
    log_info "Creating new benchmark configuration: $benchmark_name"
    
    if [[ ! -f "$ENHANCED_SCRIPT" ]]; then
        log_error "Enhanced script not found: $ENHANCED_SCRIPT"
        exit 1
    fi
    
    "$ENHANCED_SCRIPT" --create-config "$benchmark_name"
    
    log_success "Benchmark created! Edit the config file to customize parameters."
    echo "Run with: $0 run $benchmark_name"
}

# Run specific benchmark
run_benchmark() {
    local benchmark_name="$1"
    local extra_args="${@:2}"  # Pass through any additional arguments
    
    if [[ -z "$benchmark_name" ]]; then
        log_error "Please provide a benchmark name"
        echo "Usage: $0 run <benchmark_name> [extra_args]"
        exit 1
    fi
    
    if [[ "$benchmark_name" == "all" ]]; then
        run_all_benchmarks
        return
    fi
    
    local config_file="$BENCHMARK_DIR/${benchmark_name}_config.sh"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Benchmark config not found: $config_file"
        echo "Available benchmarks:"
        list_benchmarks
        exit 1
    fi
    
    log_info "🚀 Running benchmark: $benchmark_name"
    log_info "📋 Config file: $config_file"
    
    if [[ ! -f "$ENHANCED_SCRIPT" ]]; then
        log_error "Enhanced script not found: $ENHANCED_SCRIPT"
        exit 1
    fi
    
    # Run the benchmark
    if "$ENHANCED_SCRIPT" --config "$config_file" $extra_args; then
        log_success "✅ Benchmark '$benchmark_name' completed successfully!"
    else
        log_error "❌ Benchmark '$benchmark_name' failed!"
        exit 1
    fi
}

# Run all benchmarks
run_all_benchmarks() {
    log_info "🚀 Running ALL benchmarks sequentially..."
    
    if [[ ! -d "$BENCHMARK_DIR" ]]; then
        log_error "No benchmark configs directory found"
        exit 1
    fi
    
    local total=0
    local successful=0
    local failed=0
    
    for config_file in "$BENCHMARK_DIR"/*_config.sh; do
        if [[ -f "$config_file" ]]; then
            local basename=$(basename "$config_file" _config.sh)
            total=$((total + 1))
            
            log_info ""
            log_info "📍 [$((successful + failed + 1))/$total] Running benchmark: $basename"
            
            if "$ENHANCED_SCRIPT" --config "$config_file"; then
                successful=$((successful + 1))
                log_success "✅ Benchmark '$basename' completed"
            else
                failed=$((failed + 1))
                log_error "❌ Benchmark '$basename' failed"
            fi
        fi
    done
    
    log_info ""
    log_info "==============================================="
    log_success "🎉 ALL BENCHMARKS COMPLETE!"
    log_info "✅ Successful: $successful"
    log_info "❌ Failed: $failed"
    log_info "📊 Total: $total"
}

# Show help
show_help() {
    echo "Benchmark Manager - Easy benchmark creation and execution"
    echo ""
    echo "Usage:"
    echo "  $0 list                       List available benchmark configurations"
    echo "  $0 create <name>              Create new benchmark configuration template"
    echo "  $0 run <name> [extra_args]    Run specific benchmark"
    echo "  $0 run all                    Run all available benchmarks"
    echo "  $0 help                       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 create my_custom_benchmark"
    echo "  $0 run high_temperature"
    echo "  $0 run extended_strengths --dry-run"
    echo "  $0 run all"
    echo ""
    echo "Pre-configured benchmarks:"
    echo "  - high_temperature: Tests effects with temperature=0.8"
    echo "  - extended_strengths: Tests wider intervention strength range"
    echo "  - layer5_deep_dive: Focus on Layer 5 with optimized parameters"
}

# Main execution
case "$1" in
    list|ls)
        list_benchmarks
        ;;
    create|new)
        create_benchmark "$2"
        ;;
    run|execute)
        run_benchmark "${@:2}"
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac