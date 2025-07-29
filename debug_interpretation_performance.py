#!/usr/bin/env python3
"""
Debug script to check interpretation performance bottlenecks
"""

import torch
import time
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sae_loading_and_processing():
    """Test SAE model loading and basic processing speed"""
    
    print("🔍 DEBUGGING INTERPRETATION PERFORMANCE")
    print("=" * 50)
    
    # Test parameters
    model_path = "exp/sod/ape/sae_models/sae_layer_2048d.pt"
    activations_path = "exp/sod/ape/activations/activations_train_layers_3.h5"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Test 1: Model loading speed
    print("\n📦 Test 1: SAE Model Loading Speed")
    start_time = time.time()
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from sae.train_sae import SparseAutoencoder

        model = SparseAutoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            sparsity_coeff=checkpoint["config"]["sparsity_coeff"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        print(f"   📏 Model size: {checkpoint['input_dim']} -> {checkpoint['config']['hidden_dim']}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 2: Data loading speed
    print("\n📊 Test 2: Data Loading Speed")
    start_time = time.time()
    
    try:
        from sae.sae_data import create_sae_dataloader
        
        layer_key = checkpoint.get("layer_key", "layer_3")
        
        # Test with small subsample first
        test_subsample = 5000
        dataloader, data_info = create_sae_dataloader(
            activations_path,
            layer_key=layer_key,
            batch_size=1024,  # Smaller batch for testing
            shuffle=False,
            normalize=True,
            subsample=test_subsample,
            num_workers=2,  # Fewer workers for testing
            memory_efficient=True,
            cache_size=500,
        )
        
        load_time = time.time() - start_time
        print(f"   ✅ Dataloader created in {load_time:.2f}s")
        print(f"   📊 Data shape: {data_info['shape']}")
        print(f"   🔢 Batches: {len(dataloader)}, Batch size: {dataloader.batch_size}")
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test 3: Single batch processing speed
    print("\n⚡ Test 3: Single Batch Processing Speed")
    
    try:
        # Get first batch
        start_time = time.time()
        first_batch = next(iter(dataloader))
        batch_get_time = time.time() - start_time
        print(f"   ✅ First batch retrieved in {batch_get_time:.3f}s")
        print(f"   📦 Batch shape: {first_batch.shape}")
        
        # Move to GPU
        start_time = time.time()
        first_batch = first_batch.to(device, non_blocking=True)
        gpu_move_time = time.time() - start_time
        print(f"   ✅ Moved to {device} in {gpu_move_time:.3f}s")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    _, hidden = model(first_batch)
            else:
                _, hidden = model(first_batch)
        forward_time = time.time() - start_time
        print(f"   ✅ Forward pass completed in {forward_time:.3f}s")
        print(f"   🔍 Hidden shape: {hidden.shape}")
        
        # Test feature activation detection
        start_time = time.time()
        min_threshold = 0.1
        activated_mask = hidden > min_threshold
        num_activations = torch.sum(activated_mask).item()
        total_possible = hidden.numel()
        activation_detection_time = time.time() - start_time
        
        print(f"   ✅ Activation detection in {activation_detection_time:.3f}s")
        print(f"   🎯 Activations: {num_activations}/{total_possible} ({num_activations/total_possible:.1%})")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    # Test 4: Estimate full processing time
    print("\n⏱️  Test 4: Full Processing Time Estimate")
    
    total_batches = len(dataloader)
    time_per_batch = batch_get_time + gpu_move_time + forward_time + activation_detection_time
    estimated_total_time = total_batches * time_per_batch
    
    print(f"   📊 Time per batch: {time_per_batch:.3f}s")
    print(f"   📈 Total batches: {total_batches}")
    print(f"   ⏰ Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} minutes)")
    
    if estimated_total_time > 300:  # More than 5 minutes
        print(f"   ⚠️  Warning: Processing may take a long time!")
        
        print("\n🚀 Optimization Suggestions:")
        print("   1. Increase batch_size to 4096 for GPU")
        print("   2. Reduce subsample size (e.g., 10000 instead of 50000)")
        print("   3. Increase min_activation_threshold to reduce feature processing")
        
        # Test with optimized settings
        print("\n🔧 Testing with optimized settings...")
        try:
            opt_dataloader, opt_data_info = create_sae_dataloader(
                activations_path,
                layer_key=layer_key,
                batch_size=4096 if device == "cuda" else 1024,
                shuffle=False,
                normalize=True,
                subsample=10000,  # Smaller subsample
                num_workers=4,
                memory_efficient=True,
                cache_size=2000,
            )
            
            opt_batches = len(opt_dataloader)
            opt_estimated_time = opt_batches * time_per_batch * 0.8  # Assume 20% speedup
            
            print(f"   ✅ Optimized batches: {opt_batches}")
            print(f"   ⏰ Optimized estimated time: {opt_estimated_time:.1f}s ({opt_estimated_time/60:.1f} minutes)")
        except Exception as e:
            print(f"   ❌ Optimization test failed: {e}")
    
    print("\n✅ Performance debug complete!")
    return True

if __name__ == "__main__":
    test_sae_loading_and_processing()
