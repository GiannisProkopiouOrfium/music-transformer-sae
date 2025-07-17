# 🎵 Music Transformer SAE Pipeline

A comprehensive pipeline for applying Sparse Autoencoders (SAEs) to music transformer interpretability.

## 🚀 Quick Start

### 1. All-in-One Pipeline
```bash
# Run complete pipeline (extract → train → analyze)
python pipeline/main.py --config configs/macos_config.yaml --output-dir ./results --experiment-name my_experiment

# On AWS EC2
python pipeline/main.py --config configs/aws_ec2_config.yaml --output-dir ./results --experiment-name production_run
```

### 2. Individual Stages
```bash
# Extract activations only
python pipeline/main.py --config configs/macos_config.yaml --output-dir ./results --experiment-name test --stage extract

# Train SAE only (requires extracted activations)
python pipeline/main.py --config configs/macos_config.yaml --output-dir ./results --experiment-name test --stage train

# Analyze SAE only (requires trained model)
python pipeline/main.py --config configs/macos_config.yaml --output-dir ./results --experiment-name test --stage analyze
```

### 3. Individual Module Usage
```bash
# Use individual pipeline modules
python pipeline/extract.py --config configs/macos_config.yaml --output results/extract
python pipeline/train.py --config configs/macos_config.yaml --activations results/extract/activations.h5
python pipeline/analyze.py --config configs/macos_config.yaml --model results/train/sae_model.pt
```

## 🎯 What Each Stage Does

### Stage 1: Extract Activations
- **Input**: Music transformer model + dataset
- **Process**: Extract hidden states from specific layers during inference
- **Output**: `activations.h5` with activation vectors + metadata
- **Fallback**: Generates synthetic activations if model unavailable

### Stage 2: Train SAE
- **Input**: Extracted activations
- **Process**: Train sparse autoencoder with sparsity constraints
- **Output**: `sae_model.pt` with trained SAE + training metrics
- **Design**: k-sparse or L1-regularized autoencoder

### Stage 3: Analyze & Interpret
- **Input**: Trained SAE + original activations  
- **Process**: Feature analysis, musical interpretation, visualizations
- **Output**: Analysis reports, feature interpretations, visualizations
- **Features**: 
  - Feature activation patterns
  - Musical context analysis
  - Top-activating musical segments
  - Feature clustering and relationships

## 📁 Output Structure

```
results/
├── experiment_name/
│   ├── activations.h5          # Extracted activations
│   ├── sae_model.pt           # Trained SAE model
│   ├── analysis_results.h5    # Analysis metrics
│   ├── sae_analysis.png       # Feature visualization
│   ├── musical_features.json  # Musical interpretations
│   └── experiment.log         # Execution log
```

## ⚙️ Configuration

### macOS (Local Testing)
```yaml
# configs/macos_config.yaml
extraction:
  batch_size: 4
  max_samples: 2000
sae:
  expansion_factor: 2.0
  num_epochs: 30
platform:
  use_mps: true
  synthetic_fallback: true
```

### AWS EC2 (Production)
```yaml
# configs/aws_ec2_config.yaml  
extraction:
  batch_size: 32
  max_samples: 50000
sae:
  expansion_factor: 4.0
  num_epochs: 100
platform:
  use_cuda: true
  synthetic_fallback: false
```

## 🔬 Key Features

1. **Cross-Platform**: Automatic platform detection and optimization
2. **Graceful Fallback**: Uses synthetic data when models unavailable
3. **Modular Design**: Run individual stages or complete pipeline
4. **Musical Interpretation**: Correlates SAE features with musical concepts
5. **Comprehensive Analysis**: Feature statistics, patterns, and visualizations

## 📊 Expected Insights

The pipeline discovers:
- **Chord Features**: SAE units responding to specific harmonic progressions
- **Rhythm Features**: Units activating on drum patterns or rhythmic motifs
- **Melodic Features**: Units capturing scale degrees or melodic contours
- **Instrument Features**: Units specialized for specific instruments
- **Structural Features**: Units responding to phrase boundaries or musical form

## 🛠️ Platform Optimizations

| Platform | Batch Size | Samples | Memory | Acceleration | Use Case |
|----------|------------|---------|---------|--------------|----------|
| macOS M2 | 4-16 | 2K | ~4GB | MPS | Local testing |
| EC2 g4dn | 32-64 | 50K | ~12GB | CUDA | Small production |
| EC2 p3 | 128+ | 500K+ | ~40GB | CUDA | Large production |
