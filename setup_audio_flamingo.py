#!/usr/bin/env python3
"""
Simple script to install and test Audio Flamingo models.
"""

import subprocess
import sys
import os


def install_audio_flamingo_dependencies():
    """Install required dependencies for Audio Flamingo."""
    print("📦 Installing Audio Flamingo dependencies...")
    
    required_packages = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "accelerate>=0.20.0",
    ]
    
    for package in required_packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True


def test_simple_loading():
    """Test simple model loading without full initialization."""
    print("\n🧪 Testing simple model loading...")
    
    try:
        from transformers import AutoConfig
        
        # Test models in order of preference
        models_to_try = [
            "nvidia/audio-flamingo-3",
            "nvidia/audio-flamingo-2", 
            "nvidia/audio-flamingo-2-1.5B",
        ]
        
        for model_name in models_to_try:
            print(f"\n   Testing {model_name}...")
            try:
                # Just try to load the config first
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                print(f"   ✅ Config loaded for {model_name}")
                print(f"      Model type: {config.model_type}")
                
                # If config works, this model should work
                return model_name
                
            except Exception as e:
                print(f"   ❌ Failed to load config for {model_name}: {e}")
                continue
        
        print("   ⚠️  No Audio Flamingo models could be loaded")
        return None
        
    except ImportError as e:
        print(f"   ❌ Cannot import transformers: {e}")
        return None


def test_audio_flamingo_basic():
    """Test basic Audio Flamingo functionality."""
    print("\n🎵 Testing basic Audio Flamingo functionality...")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        import numpy as np
        
        # Use the model that worked in simple loading
        working_model = test_simple_loading()
        if not working_model:
            print("   ❌ No working model found")
            return False
        
        print(f"\n   Using model: {working_model}")
        
        # Load processor
        print("   Loading processor...")
        processor = AutoProcessor.from_pretrained(working_model, trust_remote_code=True)
        
        # Load model with CPU fallback
        print("   Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                working_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                working_model,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.to(device)
        
        print("   ✅ Model loaded successfully")
        
        # Test with dummy audio
        print("   Testing with dummy audio...")
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        question = "What do you hear in this audio?"
        
        # Process inputs
        inputs = processor(audio=dummy_audio, text=question, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, 'to')}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"   Generated response: '{response.strip()}'")
        print("   ✅ Basic functionality test passed!")
        
        return working_model
        
    except Exception as e:
        print(f"   ❌ Error in basic test: {e}")
        return False


def create_audio_flamingo_config():
    """Create a configuration file for the working Audio Flamingo model."""
    working_model = test_audio_flamingo_basic()
    
    if working_model:
        config_content = f'''# Audio Flamingo Configuration
# Generated automatically by setup script

# Working model identified: {working_model}
AUDIO_FLAMINGO_MODEL = "{working_model}"

# Usage in your scripts:
# from audio_flamingo_config import AUDIO_FLAMINGO_MODEL
# analyzer = AudioFlamingoAnalyzer(AUDIO_FLAMINGO_MODEL)
'''
        
        with open("audio_flamingo_config.py", "w") as f:
            f.write(config_content)
        
        print(f"\n📄 Created audio_flamingo_config.py with working model: {working_model}")
        print("   You can now import this in your scripts!")
        
        return working_model
    
    return None


def main():
    """Main setup function."""
    print("🔥 AUDIO FLAMINGO SETUP & TEST")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_audio_flamingo_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Step 2: Test model loading
    working_model = create_audio_flamingo_config()
    
    if working_model:
        print(f"\n🎉 SUCCESS! Audio Flamingo is ready to use")
        print(f"✅ Working model: {working_model}")
        print(f"✅ Configuration saved to: audio_flamingo_config.py")
        print(f"\nNext steps:")
        print(f"1. Update your audio analysis scripts to use: {working_model}")
        print(f"2. Run: python mmt/audio_flamingo_analysis.py --model-name {working_model}")
    else:
        print(f"\n⚠️  Audio Flamingo setup failed")
        print(f"   Falling back to librosa-based audio analysis")
        print(f"   Your scripts will still work but without Audio Flamingo")


if __name__ == "__main__":
    main()
