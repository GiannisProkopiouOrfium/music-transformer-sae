#!/usr/bin/env python3
"""
Quick SAE Analysis - Get key metrics without complex visualizations
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path

def quick_sae_analysis(model_path, activations_path):
    """Quick analysis of trained SAE without complex visualizations."""
    
    print("🎼 QUICK SAE ANALYSIS")
    print("=" * 50)
    
    # Load model
    print(f"📦 Loading SAE model: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    print(f"✅ SAE Architecture: {checkpoint['input_dim']} -> {checkpoint['config']['hidden_dim']}")
    print(f"   Sparsity coefficient: {checkpoint['config']['sparsity_coeff']}")
    print(f"   Best validation loss: {checkpoint['best_val_loss']:.6f}")
    print(f"   Final sparsity ratio: {checkpoint['final_sparsity']:.6f}")
    
    # Load model weights
    model_weights = checkpoint['model_state_dict']
    encoder_weights = model_weights['encoder.weight'].numpy()
    decoder_weights = model_weights['decoder.weight'].numpy()
    
    # Analyze decoder norms (key metric for SAE quality)
    decoder_norms = np.linalg.norm(decoder_weights, axis=0)
    print(f"\n📊 DECODER ANALYSIS:")
    print(f"   Mean norm: {decoder_norms.mean():.8f}")
    print(f"   Std norm: {decoder_norms.std():.8f}")
    print(f"   Min norm: {decoder_norms.min():.8f}")
    print(f"   Max norm: {decoder_norms.max():.8f}")
    print(f"   Range: {decoder_norms.max() - decoder_norms.min():.8f}")
    print(f"   Unique values: {len(np.unique(decoder_norms.round(8)))}")
    
    # Quick activation analysis
    print(f"\n🎯 ACTIVATION ANALYSIS:")
    print(f"📂 Loading activations: {activations_path}")
    
    with h5py.File(activations_path, 'r') as f:
        layer_key = list(f.keys())[0]
        print(f"   Using layer: {layer_key}")
        
        # Sample activations for quick analysis
        activations = f[layer_key][:5000]  # First 5000 samples
        print(f"   Loaded {activations.shape[0]} samples, {activations.shape[1]} features")
    
    # Simulate SAE forward pass
    device = torch.device('cpu')
    model_state = checkpoint['model_state_dict']
    
    # Create simple SAE model for analysis
    class SimpleSAE:
        def __init__(self, encoder_weight, decoder_weight):
            self.encoder_weight = torch.tensor(encoder_weight)
            self.decoder_weight = torch.tensor(decoder_weight)
        
        def forward(self, x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            hidden = torch.relu(torch.matmul(x_tensor, self.encoder_weight.T))
            return hidden.numpy()
    
    sae = SimpleSAE(encoder_weights, decoder_weights)
    
    # Analyze feature activations
    print("   Computing feature activations...")
    all_hidden = []
    batch_size = 500
    
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        hidden = sae.forward(batch)
        all_hidden.append(hidden)
    
    all_hidden = np.vstack(all_hidden)
    
    # Feature statistics
    activation_frequencies = np.mean(all_hidden > 0, axis=0)
    mean_activations = np.mean(all_hidden, axis=0)
    
    # Count feature types
    dead_features = np.sum(activation_frequencies == 0)
    rare_features = np.sum(activation_frequencies < 0.01)
    moderate_features = np.sum((activation_frequencies >= 0.01) & (activation_frequencies < 0.1))
    active_features = np.sum(activation_frequencies >= 0.1)
    
    print(f"\n📈 FEATURE STATISTICS:")
    print(f"   Total features: {len(activation_frequencies)}")
    print(f"   Dead features (0%): {dead_features} ({dead_features/len(activation_frequencies)*100:.1f}%)")
    print(f"   Rare features (<1%): {rare_features} ({rare_features/len(activation_frequencies)*100:.1f}%)")
    print(f"   Moderate features (1-10%): {moderate_features} ({moderate_features/len(activation_frequencies)*100:.1f}%)")
    print(f"   Active features (>10%): {active_features} ({active_features/len(activation_frequencies)*100:.1f}%)")
    
    print(f"\n🎯 ACTIVATION FREQUENCY DISTRIBUTION:")
    print(f"   Min frequency: {activation_frequencies.min():.6f}")
    print(f"   Max frequency: {activation_frequencies.max():.6f}")
    print(f"   Mean frequency: {activation_frequencies.mean():.6f}")
    print(f"   Median frequency: {np.median(activation_frequencies):.6f}")
    
    # Sparsity analysis
    overall_sparsity = np.mean(all_hidden == 0)
    print(f"\n💎 SPARSITY ANALYSIS:")
    print(f"   Overall sparsity ratio: {overall_sparsity:.6f}")
    print(f"   Average activations per sample: {np.mean(np.sum(all_hidden > 0, axis=1)):.1f}")
    
    # Feature correlation analysis (sample)
    print(f"\n🔗 CORRELATION ANALYSIS (sample):")
    sample_features = min(500, all_hidden.shape[1])  # Sample for speed
    sample_hidden = all_hidden[:, :sample_features]
    
    # Compute correlations
    corr_matrix = np.corrcoef(sample_hidden.T)
    high_corr_mask = (np.abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)
    high_corr_count = np.sum(high_corr_mask) // 2  # Divide by 2 for symmetry
    
    print(f"   High correlations (>0.7) in sample: {high_corr_count}")
    print(f"   Estimated total high correlations: {high_corr_count * (len(activation_frequencies)/sample_features)**2:.0f}")
    
    # Quality assessment
    print(f"\n🏆 QUALITY ASSESSMENT:")
    quality_score = 0
    
    # Good signs
    if dead_features > 0:
        print("   ✅ Has dead features (good for interpretability)")
        quality_score += 1
    if rare_features > len(activation_frequencies) * 0.1:
        print("   ✅ Has many rare features (good specialization)")
        quality_score += 1
    if overall_sparsity > 0.8:
        print("   ✅ High sparsity (good feature selectivity)")
        quality_score += 1
    if high_corr_count < sample_features * 0.05:
        print("   ✅ Low feature correlations (good independence)")
        quality_score += 1
    if decoder_norms.std() < 0.01:
        print("   ✅ Very uniform decoder norms (excellent training)")
        quality_score += 1
    
    # Areas for improvement
    if dead_features == 0:
        print("   ⚠️ No dead features (could increase sparsity)")
    if rare_features < len(activation_frequencies) * 0.05:
        print("   ⚠️ Few rare features (could increase specialization)")
    if overall_sparsity < 0.5:
        print("   ⚠️ Low sparsity (features not selective enough)")
    
    print(f"\n🎯 Overall Quality Score: {quality_score}/5")
    
    if quality_score >= 4:
        print("   🏆 EXCELLENT SAE! Ready for interpretation.")
    elif quality_score >= 3:
        print("   ✅ GOOD SAE! Minor improvements possible.")
    elif quality_score >= 2:
        print("   ⚠️ FAIR SAE. Consider increasing sparsity.")
    else:
        print("   ❌ POOR SAE. Needs significant improvement.")
    
    return {
        'dead_features': int(dead_features),
        'rare_features': int(rare_features),
        'active_features': int(active_features),
        'overall_sparsity': float(overall_sparsity),
        'high_correlations': int(high_corr_count),
        'quality_score': int(quality_score),
        'decoder_norm_std': float(decoder_norms.std()),
        'best_val_loss': float(checkpoint['best_val_loss'])
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python quick_sae_analysis.py <model_path> <activations_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    activations_path = sys.argv[2]
    
    results = quick_sae_analysis(model_path, activations_path)
    
    # Save results
    output_file = Path(model_path).parent / "quick_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
