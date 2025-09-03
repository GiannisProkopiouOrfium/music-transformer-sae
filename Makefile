# Music Transformer SAE Pipeline Makefile

.PHONY: help install test quick-test extract train analyze pipeline clean

help:
	@echo "Music Transformer SAE Pipeline"
	@echo "=============================="
	@echo
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  quick-test   - Quick extraction test"
	@echo "  extract      - Extract activations only"
	@echo "  pipeline     - Run full pipeline"
	@echo "  configs      - Create default config files"
	@echo "  clean        - Clean temporary files"
	@echo
	@echo "Examples:"
	@echo "  make extract DATASET=sod REP=ape LAYERS='2 3 4' SAMPLES=1000 GPU=0"
	@echo "  make pipeline CONFIG=mmt/configs/gpu.json"
	@echo "  make quick-test"

# Variables with defaults
DATASET ?= sod
REP ?= ape
LAYERS ?= 3
SAMPLES ?= 500
GPU ?= -1
CONFIG ?= mmt/configs/default.json
OUTPUT ?= 

install:
	@echo "Installing dependencies..."
	pip install -r requirements_macos.txt
	@echo "✓ Dependencies installed"

quick-test:
	@echo "Running quick extraction test..."
	python run_pipeline.py -c mmt/configs/quick_test.json --extract-only

extract:
	@echo "Extracting activations..."
	@echo "  Dataset: $(DATASET)"
	@echo "  Representation: $(REP)"
	@echo "  Layers: $(LAYERS)"
	@echo "  Samples: $(SAMPLES)"
	@echo "  GPU: $(GPU)"
	@if [ -n "$(OUTPUT)" ]; then \
		python run_pipeline.py -d $(DATASET) -r $(REP) --extract-only --layers $(LAYERS) --n-samples $(SAMPLES) -g $(GPU) --output_dir $(OUTPUT); \
	else \
		python run_pipeline.py -d $(DATASET) -r $(REP) --extract-only --layers $(LAYERS) --n-samples $(SAMPLES) -g $(GPU); \
	fi

train:
	@echo "Training SAE..."
	python run_pipeline.py --train-only --skip-extraction

analyze:
	@echo "Analyzing SAE..."
	python run_pipeline.py --analyze-only --skip-extraction --skip-training

pipeline:
	@echo "Running full pipeline with config: $(CONFIG)"
	python run_pipeline.py -c $(CONFIG)

configs:
	@echo "Creating default configuration files..."
	python run_pipeline.py --create-configs
	@echo "✓ Configuration files created in mmt/configs/"

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "✓ Temporary files cleaned"

# Development helpers
dev-extract:
	@echo "Development extraction (small dataset)..."
	python run_pipeline.py -d sod -r ape --extract-only --layers 3 --n-samples 50 -g -1 --output_dir dev_test

dev-gpu:
	@echo "Development extraction with GPU..."
	python run_pipeline.py -d sod -r ape --extract-only --layers 2 3 4 --n-samples 200 -g 0 --output_dir dev_gpu_test
