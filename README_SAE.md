# Music Transformer SAE Analysis

This repository contains Sparse Autoencoder (SAE) analysis for music transformer interpretability.

## Quick Start
```bash
# 1. Extract activations
python mmt/extract_activations.py -d sod -o exp/sod/ape -g -1 -ns 500 -l 3

# 2. Train SAE  
python sae/train_sae.py

# 3. Analyze features
python sae/analyze_sae.py
python interpretation/interpret_music_sae.py
```

## Key Discoveries
- ✅ Discovered 3 musical feature families: Harmony, Rhythm, Melody
- ✅ Achieved 50% sparsity with 988 interpretable features
- ✅ Identified musical complexity patterns (1-9 active features)
- ✅ Found climactic musical moments (samples with 9+ active features)

## Files Added
- `mmt/extract_activations.py` - Extract transformer activations
- `sae/train_sae.py` - SAE training pipeline
- `sae/sae_data.py` - Data utilities
- `interpretation/*.py` - Feature analysis scripts

