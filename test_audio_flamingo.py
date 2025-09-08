#!/usr/bin/env python3
"""
Simple test script to verify Audio Flamingo model loading.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_audio_flamingo_model(model_name: str = "nvidia/audio-flamingo-2-0.5B"):
    """
    Test loading the Audio Flamingo model.
    
    Args:
        model_name: HuggingFace model name to test
    """
    print(f"🧪 Testing Audio Flamingo model: {model_name}")
    print("=" * 50)
    
    try:
        print("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✅ Processor loaded successfully")
        
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Information:")
        print(f"   Total parameters: {num_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model device: {model.device}")
        print(f"   Model dtype: {model.dtype}")
        
        # Test basic functionality
        print(f"\n🔧 Testing basic functionality...")
        
        # Create dummy audio input
        dummy_audio = torch.randn(16000)  # 1 second of dummy audio at 16kHz
        question = "Describe this audio."
        
        # Prepare inputs
        inputs = processor(audio=dummy_audio, text=question, return_tensors="pt")
        print(f"   Input keys: {list(inputs.keys())}")
        
        # Test generation (very short)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"   Generated response: '{response.strip()}'")
        print("✅ Basic functionality test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_multiple_models():
    """Test multiple Audio Flamingo models."""
    models_to_test = [
        "nvidia/audio-flamingo-3",  # Latest version - try this first
        "nvidia/audio-flamingo-2",  # Previous stable version
        "nvidia/audio-flamingo-2-1.5B", 
        "nvidia/audio-flamingo-2-0.5B",
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n" + "="*60)
        success = test_audio_flamingo_model(model_name)
        results[model_name] = success
        
        if not success:
            print(f"   ⚠️  Skipping to next model...")
        else:
            print(f"   🎉 {model_name} works!")
            
    print(f"\n" + "="*60)
    print("📋 SUMMARY OF RESULTS:")
    print("="*60)
    
    for model_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {model_name:35s}: {status}")
    
    working_models = [name for name, success in results.items() if success]
    if working_models:
        print(f"\n🎯 RECOMMENDATION: Use {working_models[0]} (first working model)")
    else:
        print(f"\n⚠️  No models worked. Check your environment setup.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Audio Flamingo models")
    parser.add_argument(
        "--model", 
        default="nvidia/audio-flamingo-3",  # Try latest version first
        help="Specific model to test"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all available models"
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        test_multiple_models()
    else:
        test_audio_flamingo_model(args.model)
