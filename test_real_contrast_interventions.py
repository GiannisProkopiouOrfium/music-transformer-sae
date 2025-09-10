#!/usr/bin/env python3
"""
Test Real Contrast Interventions using your existing model infrastructure.

This script uses:
- Your trained music transformer: exp/sod/ape/checkpoints/best_model.pt
- Real contrast LiMuF extracted from SAE analysis
- Intervention strengths: [-2, -1, 0, +1, +2]
"""

import torch
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Add your project path
sys.path.append(str(Path(__file__).parent))

# Import your existing model components
from baseline.train import TransformerModel
from baseline.representation_ape import APERepresentation


def load_music_transformer(model_path: str, device: str = "cuda"):
    """Load your trained music transformer."""
    print(f"📦 Loading music transformer from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Get model configuration (adjust based on your model)
    config = checkpoint.get('args', {})
    
    model = TransformerModel(
        vocab_size=config.get('vocab_size', 388),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8), 
        n_layers=config.get('n_layers', 12),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def generate_with_intervention(
    model,
    contrast_vector: torch.Tensor,
    strength: float,
    intervention_layer: int = 3,
    seq_len: int = 256,
    num_sequences: int = 3,
    device: str = "cuda"
):
    """Generate sequences with contrast intervention."""
    
    # Hook for intervention
    def intervention_hook(module, input, output):
        if hasattr(intervention_hook, 'active') and intervention_hook.active:
            # Apply intervention to last token
            # output shape: [batch_size, seq_len, d_model]
            if len(output.shape) == 3:
                output[:, -1, :] += strength * contrast_vector.unsqueeze(0)
            else:
                output += strength * contrast_vector.unsqueeze(0)
        return output
    
    # Register hook on intervention layer
    hook_handle = None
    for name, module in model.named_modules():
        if f"layers.{intervention_layer}" in name and "ln" not in name and "mlp" not in name:
            hook_handle = module.register_forward_hook(intervention_hook)
            print(f"   Registered hook on: {name}")
            break
    
    if hook_handle is None:
        print(f"⚠️  Could not find layer {intervention_layer} for intervention")
    
    # Generate sequences
    generated_sequences = []
    
    with torch.no_grad():
        for seq_idx in tqdm(range(num_sequences), desc=f"Generating (strength {strength:+.1f})"):
            # Create simple prompt (adjust based on your representation)
            prompt = torch.tensor([[1, 2]], device=device)  # Start tokens
            
            # Activate intervention
            intervention_hook.active = True
            
            # Generate sequence
            sequence = generate_sequence(
                model=model,
                prompt=prompt,
                max_length=seq_len,
                device=device
            )
            
            # Deactivate intervention
            intervention_hook.active = False
            
            generated_sequences.append(sequence.cpu())
    
    # Remove hook
    if hook_handle:
        hook_handle.remove()
    
    return generated_sequences


def generate_sequence(model, prompt, max_length: int, device: str = "cuda", temperature: float = 1.0):
    """Generate a single sequence using the model."""
    model.eval()
    
    with torch.no_grad():
        sequence = prompt.clone()
        
        for _ in range(max_length - prompt.shape[1]):
            # Forward pass
            output = model(sequence)
            
            # Get next token logits
            if isinstance(output, tuple):
                logits = output[0]  # If model returns (logits, ...)
            else:
                logits = output
            
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            sequence = torch.cat([sequence, next_token], dim=1)
            
            # Check for end token (adjust based on your representation)
            if next_token.item() == 0:  # Assuming 0 is end token
                break
    
    return sequence.squeeze(0)


def test_real_contrast_interventions(
    model_path: str,
    contrast_limuf_path: str,
    output_dir: str,
    strengths: list = [-2.0, -1.0, 0.0, 1.0, 2.0],
    intervention_layer: int = 3,
    seq_len: int = 256,
    num_sequences: int = 3,
    device: str = "cuda"
):
    """Test real contrast interventions."""
    
    print("🎼 TESTING REAL CONTRAST INTERVENTIONS")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Contrast LiMuF: {contrast_limuf_path}")
    print(f"Strengths: {strengths}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_music_transformer(model_path, device)
    
    # Load contrast LiMuF
    print("📁 Loading contrast LiMuF...")
    contrast_data = torch.load(contrast_limuf_path, map_location="cpu")
    
    # Get the contrast vector (should be only one)
    contrast_name = list(contrast_data["limufs"].keys())[0]
    contrast_vector = contrast_data["limufs"][contrast_name].to(device)
    metadata = contrast_data["metadata"][contrast_name]
    
    print(f"✅ Loaded contrast: {contrast_name}")
    print(f"   Vector shape: {contrast_vector.shape}")
    print(f"   Feature A: {metadata['feature_a_id']} - {metadata['feature_a_description'][:60]}...")
    print(f"   Feature B: {metadata['feature_b_id']} - {metadata['feature_b_description'][:60]}...")
    print()
    
    # Test each strength
    all_results = {}
    
    for strength in strengths:
        print(f"🎵 Testing strength {strength:+.1f}")
        
        sequences = generate_with_intervention(
            model=model,
            contrast_vector=contrast_vector,
            strength=strength,
            intervention_layer=intervention_layer,
            seq_len=seq_len,
            num_sequences=num_sequences,
            device=device
        )
        
        # Save sequences
        strength_name = f"strength_{strength:+.1f}".replace(".", "_").replace("+", "plus").replace("-", "minus")
        output_file = output_path / f"{contrast_name}_{strength_name}.pt"
        
        torch.save({
            "sequences": sequences,
            "strength": strength,
            "contrast_name": contrast_name,
            "metadata": metadata,
            "generation_params": {
                "num_sequences": num_sequences,
                "seq_len": seq_len,
                "intervention_layer": intervention_layer,
                "model_path": model_path,
            }
        }, output_file)
        
        all_results[strength] = {
            "file": str(output_file),
            "num_sequences": len(sequences),
            "avg_length": float(torch.stack(sequences).shape[1]) if sequences else 0,
        }
        
        print(f"   ✅ Saved {len(sequences)} sequences to {output_file}")
    
    # Save summary
    summary = {
        "contrast_name": contrast_name,
        "feature_a_id": metadata['feature_a_id'],
        "feature_b_id": metadata['feature_b_id'],
        "strengths_tested": strengths,
        "results": all_results,
        "metadata": metadata,
        "generation_params": {
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "intervention_layer": intervention_layer,
            "model_path": model_path,
        }
    }
    
    summary_file = output_path / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ INTERVENTION TESTING COMPLETE!")
    print(f"📊 Summary: {summary_file}")
    print(f"📁 Generated files: {output_dir}/")
    print()
    print("🎯 EXPECTED RESULTS:")
    print("   • Negative strengths: More melodic sequences (Feature 1743 direction)")
    print("   • Positive strengths: More steady pulse (Feature 182 direction)")
    print("   • Baseline (0.0): Normal model behavior")
    
    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Test real contrast interventions")
    parser.add_argument(
        "--model-path",
        default="exp/sod/ape/checkpoints/best_model.pt",
        help="Path to trained music transformer"
    )
    parser.add_argument(
        "--contrast-limuf-path",
        default="real_contrast_limufs_182_vs_1743/limufs.pt",
        help="Path to extracted contrast LiMuF"
    )
    parser.add_argument(
        "--output-dir",
        default="real_contrast_interventions_182_vs_1743",
        help="Output directory"
    )
    parser.add_argument(
        "--strengths",
        default="-2.0,-1.0,0.0,1.0,2.0",
        help="Comma-separated intervention strengths"
    )
    parser.add_argument(
        "--intervention-layer",
        type=int,
        default=3,
        help="Layer to apply intervention"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length to generate"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=3,
        help="Number of sequences per strength"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Parse strengths
    strengths = [float(s.strip()) for s in args.strengths.split(",")]
    
    # Run intervention testing
    summary = test_real_contrast_interventions(
        model_path=args.model_path,
        contrast_limuf_path=args.contrast_limuf_path,
        output_dir=args.output_dir,
        strengths=strengths,
        intervention_layer=args.intervention_layer,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        device=args.device
    )
    
    print(f"\n🎉 SUCCESS! Check results in {args.output_dir}/")
    
    return True


if __name__ == "__main__":
    main()
