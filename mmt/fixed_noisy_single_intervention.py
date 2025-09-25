#!/usr/bin/env python3
"""
FIXED Noisy Single Feature Interventions with Reproducible Creativity.

This script fixes the noise injection issue by properly implementing manual generation
that bypasses model.generate() and applies noise at the right stage.

FIXES:
1. Manual token-by-token generation loop
2. Proper noise injection before temperature scaling
3. Correct hook placement for intervention effects
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import pathlib
import argparse
import json
from tqdm import tqdm

# Add your project path
sys.path.append(str(Path(__file__).parent))

# Import your existing model components
from music_x_transformers import MusicXTransformer, sample
import representation
import utils
from generate import save_result


def create_start_tokens(encoding, device: str = "cuda") -> torch.Tensor:
    """Create start tokens for generation."""
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    # Simple start-of-song token - adapt based on your tokenization
    start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    start_tokens[:, 0, 0] = sos  # Start-of-song type

    return start_tokens, eos


def load_music_transformer(model_path: str, device: str = "cuda"):
    """Load your trained MusicXTransformer."""
    print(f"📦 Loading MusicXTransformer from: {model_path}")

    # Load training arguments
    exp_dir = pathlib.Path(model_path).parent.parent
    train_args = utils.load_json(exp_dir / "train-args.json")
    print(f"📄 Loaded training args from: {exp_dir / 'train-args.json'}")

    # Load encoding
    encoding = representation.load_encoding("data/sod/processed/notes/encoding.json")
    print("📄 Loaded encoding from: data/sod/processed/notes/encoding.json")

    # Create MusicXTransformer with proper parameters
    model = MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, encoding


def manual_generate_with_noise(
    model,
    encoding,
    feature_vector: torch.Tensor,
    strength: float,
    intervention_type: str = "addition",
    intervention_layer: int = 3,
    seq_len: int = 256,
    noise_seed: int = 42,
    temperature: float = 1.2,
    noise_scale: float = 0.1,
    filter_logits_fn: str = "top_k",
    filter_thres: float = 0.9,
    min_p_pow: float = 2.0,
    min_p_ratio: float = 0.02,
    device: str = "cuda",
):
    """
    Manual generation with proper noise injection.
    
    This bypasses model.generate() and implements the generation loop manually
    so we can inject noise at the correct stage.
    """
    
    # Set seed for reproducible noise
    torch.manual_seed(noise_seed)
    
    # Create start tokens
    start_tokens, eos_token = create_start_tokens(encoding, device)
    
    # Get model components
    decoder_wrapper = model.decoder  # MusicAutoregressiveWrapper
    net = decoder_wrapper.net  # MusicTransformerWrapper
    
    # Normalize feature vector for ablation
    feature_unit = feature_vector / torch.norm(feature_vector, dim=0)
    
    # Hook for intervention  
    def intervention_hook(module, input, output):
        if hasattr(intervention_hook, "active") and intervention_hook.active:
            # Apply intervention to last token
            if len(output.shape) == 3:
                target_activations = output[:, -1, :]  # [batch_size, d_model]
            else:
                target_activations = output  # [batch_size, d_model]

            if intervention_type == "addition":
                # Feature addition: h'(x) = h(x) + α * r
                if len(output.shape) == 3:
                    output[:, -1, :] += strength * feature_vector.unsqueeze(0)
                else:
                    output += strength * feature_vector.unsqueeze(0)

            elif intervention_type == "ablation":
                # Feature ablation: h'(x) = h(x) - r̂r̂ᵀh(x)
                projection_coeff = torch.matmul(target_activations, feature_unit)
                projection = feature_unit.unsqueeze(0) * projection_coeff.unsqueeze(1)

                if len(output.shape) == 3:
                    output[:, -1, :] -= projection  
                else:
                    output -= projection

            else:
                raise ValueError(f"Unknown intervention_type: {intervention_type}")

        return output
    
    # Register intervention hook
    intervention_handle = None
    possible_layer_patterns = [
        f"decoder.net.attn_layers.layers.{intervention_layer}",
        f"decoder.net.attn_layers.layers.{intervention_layer}.ff", 
        f"decoder.net.attn_layers.layers.{intervention_layer}.attn",
    ]

    for pattern in possible_layer_patterns:
        for name, module in model.named_modules():
            if pattern in name and ("ln" not in name.lower() and "norm" not in name.lower()):
                intervention_handle = module.register_forward_hook(intervention_hook)
                print(f"   Registered intervention hook on: {name}")
                break
        if intervention_handle:
            break
    
    if intervention_handle is None:
        print(f"⚠️  Could not find layer {intervention_layer} for intervention")
        return None
    
    # Manual generation loop (based on MusicAutoregressiveWrapper.generate)
    with torch.no_grad():
        # Initialize generation state
        out = start_tokens
        mask = torch.ones((out.shape[0], out.shape[1]), dtype=torch.bool, device=out.device)
        
        # Set up temperature and sampling parameters
        dim = 6  # number of dimensions 
        if isinstance(temperature, (float, int)):
            temperatures = [temperature] * dim
        else:
            temperatures = temperature
            
        if isinstance(filter_logits_fn, str):
            filter_fns = [filter_logits_fn] * dim
        else:
            filter_fns = filter_logits_fn
            
        if isinstance(filter_thres, (float, int)):
            filter_thresholds = [filter_thres] * dim
        else:
            filter_thresholds = filter_thres
        
        # Get dimension mappings
        instrument_dim = decoder_wrapper.dimensions["instrument"]
        
        # Type codes
        sos_type_code = decoder_wrapper.sos_type_code
        eos_type_code = decoder_wrapper.eos_type_code  
        son_type_code = decoder_wrapper.son_type_code
        instrument_type_code = decoder_wrapper.instrument_type_code
        note_type_code = decoder_wrapper.note_type_code
        
        # Generation loop
        for _ in range(seq_len):
            # Truncate to max sequence length
            x = out[:, -net.max_seq_len:]
            mask_truncated = mask[:, -net.max_seq_len:]
            
            # Activate intervention hook
            intervention_hook.active = True
            
            # Forward pass through the network
            logits = net(x, mask=mask_truncated)
            
            # Deactivate intervention hook
            intervention_hook.active = False
            
            # Extract last token logits
            logits = [logit_tensor[:, -1, :] for logit_tensor in logits]
            
            # INJECT NOISE HERE - This is where your noise should go!
            for i, logit_tensor in enumerate(logits):
                if noise_scale > 0:
                    noise = torch.randn_like(logit_tensor) * noise_scale
                    logits[i] = logit_tensor + noise
            
            # Filter out start-of-song token
            logits[0][:, sos_type_code] = -float("inf")
            
            # Sample the type token first
            sample_type = sample(
                logits[0],
                filter_fns[0], 
                filter_thresholds[0],
                temperatures[0],
                min_p_pow,
                min_p_ratio,
            )
            
            # Build the complete token based on type
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                # Special tokens (sos, eos, son) - pad with zeros
                if s_type in (sos_type_code, eos_type_code, son_type_code):
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 1)
                    
                # Instrument code  
                elif s_type == instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 2)
                    # Sample instrument (avoid 'none' = 0)
                    logits[instrument_dim][:, 0] = -float("inf")
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_fns[instrument_dim],
                        filter_thresholds[instrument_dim], 
                        temperatures[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                    
                # Note code - sample all dimensions
                elif s_type == note_type_code:
                    for d in range(1, dim):
                        # Avoid 'none' = 0
                        logits[d][:, 0] = -float("inf")
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_fns[d],
                            filter_thresholds[d],
                            temperatures[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")
            
            # Stack and append the new token
            stacked = torch.stack([torch.cat(s).expand(1, -1) for s in samples], 0)
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)
            
            # Check for end-of-song
            if eos_token is not None:
                is_eos_tokens = out[..., 0] == eos_token
                if is_eos_tokens.any(dim=1).all():
                    for i, is_eos_token in enumerate(is_eos_tokens):
                        idx = torch.argmax(is_eos_token.byte())
                        out[i, idx + 1:] = decoder_wrapper.pad_value
                    break
    
    # Remove intervention hook
    if intervention_handle:
        intervention_handle.remove()
    
    # Return only the newly generated tokens (excluding start tokens)
    generated_tokens = out[:, start_tokens.shape[1]:]
    return generated_tokens


def test_fixed_noisy_interventions(
    model_path: str,
    feature_limuf_path: str,
    output_dir: str,
    strengths: list = [-2.0, -1.0, 0.0, 1.0, 2.0],
    intervention_type: str = "addition",
    intervention_layer: int = 3,
    seq_len: int = 256,
    num_sequences: int = 3,
    noise_seed: int = 42,
    temperature: float = 1.2,
    noise_scale: float = 0.1,
    device: str = "cuda",
):
    """Test fixed noisy single feature interventions."""

    print("🎼 TESTING FIXED NOISY SINGLE FEATURE INTERVENTIONS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Feature LiMuF: {feature_limuf_path}")
    print(f"Intervention Type: {intervention_type}")
    print(f"Strengths: {strengths}")
    print(f"Noise Seed: {noise_seed}")
    print(f"Temperature: {temperature}")
    print(f"Noise Scale: {noise_scale}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, encoding = load_music_transformer(model_path, device)

    # Load feature LiMuF
    print("📁 Loading feature LiMuF...")
    feature_data = torch.load(feature_limuf_path, map_location="cpu")

    # Get the feature vector
    feature_name = list(feature_data["limufs"].keys())[0]
    feature_vector = feature_data["limufs"][feature_name].to(device)
    metadata = feature_data["metadata"][feature_name]

    print(f"✅ Loaded feature: {feature_name}")
    print(f"   Vector shape: {feature_vector.shape}")
    print(f"   Feature ID: {metadata['feature_id']} - {metadata['feature_description'][:80]}...")
    print(f"   Active samples: {metadata['active_samples']}")
    print()

    # Test each strength
    all_results = {}

    for strength in strengths:
        print(f"🎵 Testing strength {strength:+.1f} ({intervention_type}) with FIXED controlled noise")

        sequences = []
        
        for seq_idx in tqdm(range(num_sequences), desc=f"Generating (strength {strength:+.1f})"):
            try:
                # Generate with unique seed per sequence
                sequence = manual_generate_with_noise(
                    model=model,
                    encoding=encoding,
                    feature_vector=feature_vector,
                    strength=strength,
                    intervention_type=intervention_type,
                    intervention_layer=intervention_layer,
                    seq_len=seq_len,
                    noise_seed=noise_seed + seq_idx,  # Different seed per sequence
                    temperature=temperature,
                    noise_scale=noise_scale,
                    device=device,
                )
                
                if sequence is not None:
                    sequences.append(sequence)
                    
            except Exception as e:
                print(f"     Generation error for sequence {seq_idx}: {e}")
                continue

        # Save sequences
        strength_name = f"strength_{strength:+.1f}".replace(".", "_").replace("+", "plus").replace("-", "minus")
        intervention_prefix = "fixed_noisy_add" if intervention_type == "addition" else "fixed_noisy_abl"

        # Save the results 
        if sequences and len(sequences) > 0:
            save_seqs = [seq.cpu().numpy() for seq in sequences]

            first_seq = save_seqs[0]
            if len(first_seq.shape) == 3:
                first_seq = first_seq[0]

            if first_seq.shape[0] > 1:
                try:
                    save_result(
                        f"{intervention_prefix}_{strength_name}_temp{temperature}_noise{noise_scale}_seed{noise_seed}",
                        first_seq,
                        output_dir,
                        encoding,
                    )
                except Exception as e:
                    print(f"Warning: Could not save MIDI result: {e}")

        for seq_idx, sequence in enumerate(sequences):
            output_file = output_path / f"{feature_name}_{intervention_prefix}_{strength_name}_temp{temperature}_noise{noise_scale}_seed{noise_seed}_{seq_idx}.pt"

            torch.save({
                "generated": sequence.cpu(),
                "strength": strength,
                "feature_name": feature_name,
                "metadata": metadata,
                "sequence_index": seq_idx,
                "shape": list(sequence.shape),
                "generation_params": {
                    "num_sequences": num_sequences,
                    "seq_len": seq_len,
                    "intervention_layer": intervention_layer,
                    "intervention_type": intervention_type,
                    "noise_seed": noise_seed,
                    "temperature": temperature,
                    "noise_scale": noise_scale,
                    "model_path": model_path,
                    "method": "fixed_manual_generation_with_noise",
                },
            }, output_file)

        all_results[strength] = {
            "files": [str(output_path / f"{feature_name}_{intervention_prefix}_{strength_name}_temp{temperature}_noise{noise_scale}_seed{noise_seed}_{i}.pt")
                     for i in range(len(sequences))],
            "num_sequences": len(sequences),
            "avg_length": float(torch.stack(sequences).shape[1]) if sequences else 0,
        }

        print(f"   ✅ Saved {len(sequences)} sequences")

    # Save summary
    summary = {
        "feature_name": feature_name,
        "feature_id": metadata["feature_id"],
        "feature_description": metadata["feature_description"],
        "strengths_tested": strengths,
        "results": all_results,
        "metadata": metadata,
        "generation_params": {
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "intervention_layer": intervention_layer,
            "intervention_type": intervention_type,
            "noise_seed": noise_seed,
            "temperature": temperature,
            "noise_scale": noise_scale,
            "model_path": model_path,
            "method": "fixed_manual_generation_with_noise",
        },
    }

    summary_file = output_path / "fixed_noisy_generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ FIXED NOISY INTERVENTION TESTING COMPLETE!")
    print(f"📊 Summary: {summary_file}")
    print(f"📁 Generated files: {output_dir}/")
    print()
    print("🎯 FIXED NOISE INJECTION:")
    print("   • Noise now applied BEFORE temperature scaling")
    print("   • Manual generation bypasses model.generate() limitations")
    print("   • Each sequence gets different seed for true variation")
    print(f"   • Noise scale {noise_scale} should now create audible differences")

    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Test FIXED single feature interventions with controlled noise")
    parser.add_argument("--model-path", default="exp/sod/ape/checkpoints/best_model.pt", help="Path to trained music transformer")
    parser.add_argument("--feature-limuf-path", default="real_limufs_layer3/limufs.pt", help="Path to extracted feature LiMuF")
    parser.add_argument("--output-dir", default="fixed_noisy_single_feature_interventions", help="Output directory")
    parser.add_argument("--intervention-type", default="addition", choices=["addition", "ablation"], help="Type of intervention")
    parser.add_argument("--strengths", default="-2.0,-1.0,0.0,1.0,2.0", help="Comma-separated intervention strengths", type=str)
    parser.add_argument("--intervention-layer", type=int, default=3, help="Layer to apply intervention")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length to generate")
    parser.add_argument("--num-sequences", type=int, default=3, help="Number of sequences per strength")
    parser.add_argument("--noise-seed", type=int, default=42, help="Base seed for reproducible noise")
    parser.add_argument("--temperature", type=float, default=1.2, help="Sampling temperature")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Scale of noise injection")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    # Parse strengths
    strengths = [float(s.strip()) for s in args.strengths.split(",")]

    # Run fixed noisy intervention testing
    test_fixed_noisy_interventions(
        model_path=args.model_path,
        feature_limuf_path=args.feature_limuf_path,
        output_dir=args.output_dir,
        strengths=strengths,
        intervention_type=args.intervention_type,
        intervention_layer=args.intervention_layer,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        noise_seed=args.noise_seed,
        temperature=args.temperature,
        noise_scale=args.noise_scale,
        device=args.device,
    )

    print(f"\n🎉 SUCCESS! Check FIXED reproducible creative results in {args.output_dir}/")
    return True


if __name__ == "__main__":
    main()