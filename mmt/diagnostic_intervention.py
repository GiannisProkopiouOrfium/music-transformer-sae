#!/usr/bin/env python3
"""
DIAGNOSTIC Noisy Single Feature Interventions - Debug Version.

This script adds extensive debugging to identify why interventions aren't working.
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
    start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    start_tokens[:, 0, 0] = sos
    return start_tokens, eos


def load_music_transformer(model_path: str, device: str = "cuda"):
    """Load your trained MusicXTransformer."""
    print(f"📦 Loading MusicXTransformer from: {model_path}")

    exp_dir = pathlib.Path(model_path).parent.parent
    train_args = utils.load_json(exp_dir / "train-args.json")
    encoding = representation.load_encoding("data/sod/processed/notes/encoding.json")

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


def diagnostic_generate_with_interventions(
    model,
    encoding,
    feature_vector: torch.Tensor,
    strength: float,
    intervention_type: str = "addition",
    intervention_layer: int = 3,
    seq_len: int = 64,  # Shorter for debugging
    noise_seed: int = 42,
    temperature: float = 1.2,
    noise_scale: float = 0.0,  # Start with no noise to isolate intervention effects
    device: str = "cuda",
):
    """
    Diagnostic generation with extensive debugging.
    """
    
    print(f"\n🔬 DIAGNOSTIC GENERATION")
    print(f"   Intervention: {intervention_type}, Strength: {strength}")
    print(f"   Feature vector shape: {feature_vector.shape}")
    print(f"   Feature vector norm: {torch.norm(feature_vector):.6f}")
    print(f"   Noise scale: {noise_scale}")
    
    torch.manual_seed(noise_seed)
    start_tokens, eos_token = create_start_tokens(encoding, device)
    
    decoder_wrapper = model.decoder
    net = decoder_wrapper.net
    
    # Store intervention statistics
    intervention_stats = {
        "hook_calls": 0,
        "interventions_applied": 0,
        "activation_norms_before": [],
        "activation_norms_after": [],
        "feature_projections": [],
    }
    
    # Normalize feature vector
    feature_unit = feature_vector / (torch.norm(feature_vector) + 1e-8)
    
    def diagnostic_intervention_hook(module, inputs, output):
        if hasattr(diagnostic_intervention_hook, "active") and diagnostic_intervention_hook.active:
            intervention_stats["hook_calls"] += 1
            
            if len(output.shape) == 3:
                _, seq_len_current, d_model = output.shape
                print(f"      🎯 Hook triggered: output shape {output.shape}")
                
                # Check dimension compatibility
                if feature_vector.shape[0] != d_model:
                    print(f"      ❌ DIMENSION MISMATCH: feature {feature_vector.shape[0]} vs output {d_model}")
                    return output
                
                # Get last token activations
                last_token_before = output[:, -1, :].clone()
                activation_norm_before = torch.norm(last_token_before).item()
                intervention_stats["activation_norms_before"].append(activation_norm_before)
                
                if intervention_type == "addition" and strength != 0.0:
                    print(f"      ➕ Applying ADDITION with strength {strength}")
                    output[:, -1, :] = last_token_before + strength * feature_vector.unsqueeze(0)
                    intervention_stats["interventions_applied"] += 1
                    
                elif intervention_type == "ablation" and strength != 0.0:
                    print(f"      ➖ Applying ABLATION")
                    projection_coeff = torch.matmul(last_token_before, feature_unit)
                    projection = feature_unit.unsqueeze(0) * projection_coeff.unsqueeze(1)
                    output[:, -1, :] = last_token_before - projection
                    intervention_stats["interventions_applied"] += 1
                    intervention_stats["feature_projections"].append(projection_coeff.item())
                else:
                    print(f"      ⏸️  No intervention (strength={strength})")
                
                # Measure change
                last_token_after = output[:, -1, :]
                activation_norm_after = torch.norm(last_token_after).item()
                intervention_stats["activation_norms_after"].append(activation_norm_after)
                
                change_magnitude = torch.norm(last_token_after - last_token_before).item()
                print(f"      📊 Activation norm: {activation_norm_before:.6f} → {activation_norm_after:.6f}")
                print(f"      📊 Change magnitude: {change_magnitude:.6f}")
                
            else:
                print(f"      ⚠️  Unexpected output shape: {output.shape}")
        
        return output
    
    # Find and register hook with better diagnostics
    print(f"\n🔧 Looking for intervention target at layer {intervention_layer}...")
    intervention_handle = None
    target_layer_name = None
    
    # List all available layers for debugging
    available_layers = []
    for name, module in model.named_modules():
        if f"layers.{intervention_layer}" in name:
            available_layers.append(name)
    
    print(f"   Available layers for layer {intervention_layer}:")
    for layer_name in available_layers:
        print(f"      - {layer_name}")
    
    # Try to hook the most relevant layer
    layer_patterns = [
        f"decoder.net.attn_layers.layers.{intervention_layer}",
        f"decoder.net.attn_layers.layers.{intervention_layer}.ff",
        f"decoder.net.attn_layers.layers.{intervention_layer}.attn.to_out",
    ]
    
    for pattern in layer_patterns:
        for name, module in model.named_modules():
            if name == pattern:
                intervention_handle = module.register_forward_hook(diagnostic_intervention_hook)
                target_layer_name = name
                print(f"   ✅ Hooked: {name}")
                break
        if intervention_handle:
            break
    
    if intervention_handle is None:
        print(f"   ❌ Could not find suitable hook target!")
        return None, intervention_stats
    
    # Manual generation with detailed logging
    print(f"\n🎵 Starting generation...")
    with torch.no_grad():
        out = start_tokens
        mask = torch.ones((out.shape[0], out.shape[1]), dtype=torch.bool, device=out.device)
        
        dim = 6
        temperatures = [temperature] * dim
        filter_fns = ["top_k"] * dim
        filter_thresholds = [0.9] * dim
        
        instrument_dim = decoder_wrapper.dimensions["instrument"]
        sos_type_code = decoder_wrapper.sos_type_code
        eos_type_code = decoder_wrapper.eos_type_code
        son_type_code = decoder_wrapper.son_type_code
        instrument_type_code = decoder_wrapper.instrument_type_code
        note_type_code = decoder_wrapper.note_type_code
        
        for step in range(seq_len):
            print(f"\n   Step {step+1}/{seq_len}")
            
            # Prepare input
            x = out[:, -net.max_seq_len:]
            mask_truncated = mask[:, -net.max_seq_len:]
            
            # Activate hook
            diagnostic_intervention_hook.active = True
            
            # Forward pass
            print(f"      Forward pass through {target_layer_name}")
            logits = net(x, mask=mask_truncated)
            
            # Deactivate hook
            diagnostic_intervention_hook.active = False
            
            # Extract logits
            logits = [logit_tensor[:, -1, :] for logit_tensor in logits]
            
            # Apply noise AFTER intervention
            if noise_scale > 0:
                for i, logit_tensor in enumerate(logits):
                    noise = torch.randn_like(logit_tensor) * noise_scale
                    logits[i] = logit_tensor + noise
                print(f"      Applied noise with scale {noise_scale}")
            
            # Sample
            logits[0][:, sos_type_code] = -float("inf")
            sample_type = sample(logits[0], filter_fns[0], filter_thresholds[0], temperatures[0], 2.0, 0.02)
            
            # Build token (simplified for debugging)
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                if s_type in (sos_type_code, eos_type_code, son_type_code):
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 1)
                elif s_type == instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 2)
                    logits[instrument_dim][:, 0] = -float("inf")
                    sampled = sample(logits[instrument_dim][idx:idx+1], filter_fns[instrument_dim], 
                                   filter_thresholds[instrument_dim], temperatures[instrument_dim], 2.0, 0.02)[0]
                    samples[idx].append(sampled)
                elif s_type == note_type_code:
                    for d in range(1, dim):
                        logits[d][:, 0] = -float("inf")
                        sampled = sample(logits[d][idx:idx+1], filter_fns[d], 
                                       filter_thresholds[d], temperatures[d], 2.0, 0.02)[0]
                        samples[idx].append(sampled)
                else:
                    raise ValueError(f"Unknown event type: {s_type}")
            
            # Update sequence
            stacked = torch.stack([torch.cat(s).expand(1, -1) for s in samples], 0)
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)
            
            print(f"      Generated token type: {sample_type.item()}")
            
            # Early termination for debugging
            if step >= 10:  # Only generate a few tokens for debugging
                break
    
    # Remove hook
    if intervention_handle:
        intervention_handle.remove()
    
    generated_tokens = out[:, start_tokens.shape[1]:]
    
    # Print intervention statistics
    print(f"\n📈 INTERVENTION STATISTICS:")
    print(f"   Hook calls: {intervention_stats['hook_calls']}")
    print(f"   Interventions applied: {intervention_stats['interventions_applied']}")
    if intervention_stats['activation_norms_before']:
        avg_norm_before = sum(intervention_stats['activation_norms_before']) / len(intervention_stats['activation_norms_before'])
        avg_norm_after = sum(intervention_stats['activation_norms_after']) / len(intervention_stats['activation_norms_after'])
        print(f"   Avg activation norm: {avg_norm_before:.6f} → {avg_norm_after:.6f}")
    if intervention_stats['feature_projections']:
        avg_projection = sum(intervention_stats['feature_projections']) / len(intervention_stats['feature_projections'])
        print(f"   Avg feature projection: {avg_projection:.6f}")
    
    return generated_tokens, intervention_stats


def test_diagnostic_interventions(
    model_path: str,
    feature_limuf_path: str,
    output_dir: str = "diagnostic_interventions",
    device: str = "cuda",
):
    """Test interventions with extensive diagnostics."""
    print("🔬 DIAGNOSTIC INTERVENTION TESTING")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and feature
    model, encoding = load_music_transformer(model_path, device)
    feature_data = torch.load(feature_limuf_path, map_location="cpu")
    feature_name = list(feature_data["limufs"].keys())[0]
    feature_vector = feature_data["limufs"][feature_name].to(device)
    metadata = feature_data["metadata"][feature_name]
    
    print(f"\n✅ Loaded feature: {feature_name}")
    print(f"   Feature description: {metadata['feature_description']}")
    print(f"   Vector shape: {feature_vector.shape}")
    print(f"   Vector norm: {torch.norm(feature_vector):.6f}")
    
    # Test different conditions
    test_conditions = [
        {"strength": 0.0, "intervention_type": "addition", "noise_scale": 0.0, "name": "baseline"},
        {"strength": 1.0, "intervention_type": "addition", "noise_scale": 0.0, "name": "addition_1.0"},
        {"strength": -1.0, "intervention_type": "addition", "noise_scale": 0.0, "name": "addition_-1.0"},
        {"strength": 1.0, "intervention_type": "ablation", "noise_scale": 0.0, "name": "ablation"},
        {"strength": 1.0, "intervention_type": "addition", "noise_scale": 0.1, "name": "addition_with_noise"},
    ]
    
    all_results = {}
    all_stats = {}
    
    for condition in test_conditions:
        print(f"\n{'='*60}")
        print(f"TESTING: {condition['name'].upper()}")
        print(f"{'='*60}")
        
        sequence, stats = diagnostic_generate_with_interventions(
            model=model,
            encoding=encoding,
            feature_vector=feature_vector,
            strength=condition["strength"],
            intervention_type=condition["intervention_type"],
            seq_len=20,  # Short sequences for debugging
            noise_seed=42,
            temperature=1.2,
            noise_scale=condition["noise_scale"],
            device=device,
        )
        
        if sequence is not None:
            all_results[condition["name"]] = sequence
            all_stats[condition["name"]] = stats
            
            # Save sequence
            torch.save({
                "generated": sequence.cpu(),
                "condition": condition,
                "stats": stats,
                "feature_name": feature_name,
                "metadata": metadata,
            }, output_path / f"diagnostic_{condition['name']}.pt")
            
            print(f"\n✅ Saved: diagnostic_{condition['name']}.pt")
        else:
            print(f"\n❌ Failed to generate sequence for {condition['name']}")
    
    # Compare sequences
    print(f"\n🔍 SEQUENCE COMPARISON:")
    baseline_seq = all_results.get("baseline")
    if baseline_seq is not None:
        for name, sequence in all_results.items():
            if name != "baseline":
                if torch.equal(baseline_seq, sequence):
                    print(f"   ❌ {name}: IDENTICAL to baseline")
                else:
                    diff_tokens = torch.sum(baseline_seq != sequence).item()
                    total_tokens = sequence.numel()
                    print(f"   ✅ {name}: {diff_tokens}/{total_tokens} tokens different ({100*diff_tokens/total_tokens:.1f}%)")
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY:")
    for name, stats in all_stats.items():
        print(f"   {name}:")
        print(f"      Hook calls: {stats['hook_calls']}")
        print(f"      Interventions: {stats['interventions_applied']}")
    
    return all_results, all_stats


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Diagnostic intervention testing")
    parser.add_argument("--model-path", default="exp/sod/ape/checkpoints/best_model.pt")
    parser.add_argument("--feature-limuf-path", default="real_limufs_layer3/limufs.pt")
    parser.add_argument("--output-dir", default="diagnostic_interventions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    results, stats = test_diagnostic_interventions(
        model_path=args.model_path,
        feature_limuf_path=args.feature_limuf_path,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    print(f"\n🎉 Diagnostic complete! Check results in {args.output_dir}/")
    return True


if __name__ == "__main__":
    main()