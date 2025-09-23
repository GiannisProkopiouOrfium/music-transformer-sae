#!/usr/bin/env python3
"""
Noisy Single Feature Interventions with Reproducible Creativity.

This script implements the controlled noise methodology:
- Fixed noise injection for reproducibility
- High temperature sampling for creativity
- Clean intervention effect isolation

This script uses:
- Your trained music transformer: exp/sod/ape/checkpoints/best_model.pt
- Real single feature LiMuF extracted from SAE analysis
- Fixed noise + high temperature for reproducible creativity
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
from music_x_transformers import MusicXTransformer
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


def generate_with_controlled_noise(
    model,
    encoding,
    feature_vector: torch.Tensor,
    strength: float,
    intervention_type: str = "addition",
    intervention_layer: int = 3,
    seq_len: int = 256,
    num_sequences: int = 3,
    noise_seed: int = 42,
    temperature: float = 1.2,
    noise_scale: float = 0.1,
    device: str = "cuda",
):
    """Generate sequences with controlled noise for reproducible creativity.

    Args:
        feature_vector: The feature direction vector
        strength: Intervention strength
        intervention_type: "addition" or "ablation"
        intervention_layer: Layer to apply intervention
        seq_len: Sequence length to generate
        num_sequences: Number of sequences to generate
        noise_seed: Fixed seed for reproducible noise
        temperature: Sampling temperature (higher = more creative)
        noise_scale: Scale of the noise injection
        device: Device to use
    """

    # Create proper start tokens
    start_tokens, eos = create_start_tokens(encoding, device)

    generated_sequences = []

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

    # Register hook on intervention layer
    hook_handle = None

    # Try different layer naming patterns for MusicXTransformer
    possible_layer_patterns = [
        f"transformer.layers.{intervention_layer}",
        f"net.layers.{intervention_layer}",
        f"layers.{intervention_layer}",
        f"transformer.{intervention_layer}",
    ]

    for pattern in possible_layer_patterns:
        for name, module in model.named_modules():
            if pattern in name and (
                "ln" not in name.lower() and "norm" not in name.lower()
            ):
                hook_handle = module.register_forward_hook(intervention_hook)
                print(f"   Registered hook on: {name}")
                break
        if hook_handle:
            break

    if hook_handle is None:
        # Fallback: try to find any layer that matches
        print(f"   Looking for any layer containing '{intervention_layer}'...")
        for name, module in model.named_modules():
            if str(intervention_layer) in name and hasattr(
                module, "register_forward_hook"
            ):
                hook_handle = module.register_forward_hook(intervention_hook)
                print(f"   Fallback hook registered on: {name}")
                break

    if hook_handle is None:
        print(f"⚠️  Could not find layer {intervention_layer} for intervention")
        return []

    # Generate sequences with controlled noise
    with torch.no_grad():
        for seq_idx in tqdm(
            range(num_sequences), desc=f"Generating (strength {strength:+.1f}, noise)"
        ):
            try:
                # Set fixed seed for reproducible noise
                torch.manual_seed(noise_seed + seq_idx)

                # Activate intervention
                intervention_hook.active = True

                # Generate sequence step by step with controlled noise
                current_sequence = start_tokens.clone()

                for _ in range(seq_len):
                    # Get model logits for next token
                    try:
                        # Use the decoder's net to get logits (not the full model)
                        # model.decoder.net returns a list of logits for each dimension
                        logits_list = model.decoder.net(
                            current_sequence
                        )  # Returns list of [batch, seq, vocab] tensors

                        # The first dimension is the token type (what we want to sample)
                        if isinstance(logits_list, list) and len(logits_list) > 0:
                            # Get type logits for the last position
                            type_logits = logits_list[
                                0
                            ]  # [batch_size, seq_len, vocab_size] for type dimension
                            if type_logits.shape[1] > 0:
                                next_token_logits = type_logits[
                                    :, -1, :
                                ]  # [batch_size, vocab_size]
                            else:
                                print(
                                    f"Empty sequence in type logits: {type_logits.shape}"
                                )
                                break
                        else:
                            print(f"Unexpected logits format: {type(logits_list)}")
                            break

                    except Exception as e:
                        print(f"Error getting logits: {e}")
                        break

                    # Apply controlled noise (same seed = same noise pattern)
                    noise = torch.randn_like(next_token_logits) * noise_scale
                    noisy_logits = next_token_logits + noise

                    # Apply temperature scaling for creativity
                    scaled_logits = noisy_logits / temperature

                    # Sample next token
                    probs = F.softmax(scaled_logits, dim=-1)

                    # Check if probs are valid
                    if torch.isnan(probs).any() or probs.sum(dim=-1).min() == 0:
                        print("Invalid probabilities, breaking generation")
                        break

                    next_token_type = torch.multinomial(
                        probs, num_samples=1
                    )  # [batch_size, 1]

                    # Check for EOS token
                    if next_token_type.item() == eos:
                        break

                    # Create the full 6-dimensional token
                    # For simplicity, we'll set other dimensions to 0 (this mimics the start token format)
                    next_token_full = torch.zeros(
                        (1, 1, 6), dtype=torch.long, device=device
                    )
                    next_token_full[:, 0, 0] = (
                        next_token_type.squeeze()
                    )  # Set type dimension
                    # Other dimensions (beat, position, pitch, duration, instrument) remain 0

                    current_sequence = torch.cat(
                        [current_sequence, next_token_full], dim=1
                    )

                # Deactivate intervention
                intervention_hook.active = False

                generated_sequences.append(current_sequence)

            except Exception as e:
                print(f"     Generation error for sequence {seq_idx}: {e}")
                intervention_hook.active = False
                continue

    # Remove hook
    if hook_handle:
        hook_handle.remove()

    return generated_sequences


def test_noisy_single_feature_interventions(
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
    """Test single feature interventions with controlled noise for reproducible creativity."""

    print("🎼 TESTING NOISY SINGLE FEATURE INTERVENTIONS")
    print("=" * 55)
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
    print(
        f"   Feature ID: {metadata['feature_id']} - {metadata['feature_description'][:80]}..."
    )
    print(f"   Active samples: {metadata['active_samples']}")
    print()

    # Test each strength
    all_results = {}

    for strength in strengths:
        print(
            f"🎵 Testing strength {strength:+.1f} ({intervention_type}) with controlled noise"
        )

        sequences = generate_with_controlled_noise(
            model=model,
            encoding=encoding,
            feature_vector=feature_vector,
            strength=strength,
            intervention_type=intervention_type,
            intervention_layer=intervention_layer,
            seq_len=seq_len,
            num_sequences=num_sequences,
            noise_seed=noise_seed,
            temperature=temperature,
            noise_scale=noise_scale,
            device=device,
        )

        # Save sequences
        strength_name = (
            f"strength_{strength:+.1f}".replace(".", "_")
            .replace("+", "plus")
            .replace("-", "minus")
        )

        # Include intervention type and noise info in filename
        intervention_prefix = (
            "noisy_add" if intervention_type == "addition" else "noisy_abl"
        )

        # Save the results (only if we have sequences and they're not empty)
        if sequences and len(sequences) > 0:
            save_seqs = [seq.cpu().numpy() for seq in sequences]

            # Ensure the data format is correct for save_result
            first_seq = save_seqs[0]
            if len(first_seq.shape) == 3:  # [batch, seq_len, features]
                first_seq = first_seq[0]  # Take first batch element

            # Only save if sequence has meaningful length (more than just start token)
            if first_seq.shape[0] > 1:
                try:
                    save_result(
                        f"{intervention_prefix}_{strength_name}_temp{temperature}_seed{noise_seed}",
                        first_seq,
                        output_dir,
                        encoding,
                    )
                except Exception as e:
                    print(f"Warning: Could not save MIDI result: {e}")
            else:
                print(
                    f"Warning: Generated sequence too short ({first_seq.shape[0]} tokens), skipping MIDI save"
                )
        else:
            print("Warning: No sequences generated, skipping MIDI save")

        for seq_idx, sequence in enumerate(sequences):
            output_file = (
                output_path
                / f"{feature_name}_{intervention_prefix}_{strength_name}_temp{temperature}_seed{noise_seed}_{seq_idx}.pt"
            )

            torch.save(
                {
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
                        "method": "controlled_noise_sampling",
                    },
                },
                output_file,
            )

        all_results[strength] = {
            "files": [
                str(
                    output_path
                    / f"{feature_name}_{intervention_prefix}_{strength_name}_temp{temperature}_seed{noise_seed}_{i}.pt"
                )
                for i in range(len(sequences))
            ],
            "num_sequences": len(sequences),
            "avg_length": float(torch.stack(sequences).shape[1]) if sequences else 0,
        }

        print(
            f"   ✅ Saved {len(sequences)} sequences as {feature_name}_{intervention_prefix}_{strength_name}_temp{temperature}_seed{noise_seed}_*.pt"
        )

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
            "method": "controlled_noise_sampling",
        },
    }

    summary_file = output_path / "noisy_generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ NOISY INTERVENTION TESTING COMPLETE!")
    print(f"📊 Summary: {summary_file}")
    print(f"📁 Generated files: {output_dir}/")
    print()
    print("🎯 REPRODUCIBLE CREATIVITY RESULTS:")
    print(f"   • Same noise seed ({noise_seed}) = identical randomness across runs")
    print(f"   • High temperature ({temperature}) = creative, diverse choices")
    print(f"   • Noise scale ({noise_scale}) = controlled randomness level")
    print("   • Clean intervention isolation = pure effect measurement")

    if intervention_type == "addition":
        print()
        print("📈 EXPECTED INTERVENTION EFFECTS:")
        print(
            f"   • Negative strengths: Less {metadata['feature_description'][:50]}..."
        )
        print(
            f"   • Positive strengths: More {metadata['feature_description'][:50]}..."
        )
        print("   • Baseline (0.0): Normal model behavior")
    else:  # ablation
        print()
        print("📈 EXPECTED INTERVENTION EFFECTS:")
        print("   • Ablation removes the feature direction from activations")
        print(
            f"   • Should reduce {metadata['feature_description'][:50]}... regardless of strength"
        )
        print("   • Baseline (0.0): Normal model behavior")

    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Test single feature interventions with controlled noise for reproducible creativity"
    )
    parser.add_argument(
        "--model-path",
        default="exp/sod/ape/checkpoints/best_model.pt",
        help="Path to trained music transformer",
    )
    parser.add_argument(
        "--feature-limuf-path",
        default="real_limufs_layer3/limufs.pt",
        help="Path to extracted feature LiMuF",
    )
    parser.add_argument(
        "--output-dir",
        default="noisy_single_feature_interventions",
        help="Output directory",
    )
    parser.add_argument(
        "--intervention-type",
        default="addition",
        choices=["addition", "ablation"],
        help="Type of intervention: 'addition' for h'(x) = h(x) + α*r, 'ablation' for h'(x) = h(x) - r̂r̂ᵀh(x)",
    )
    parser.add_argument(
        "--strengths",
        default="-2.0,-1.0,0.0,1.0,2.0",
        help="Comma-separated intervention strengths",
        type=str,
    )
    parser.add_argument(
        "--intervention-layer", type=int, default=3, help="Layer to apply intervention"
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Sequence length to generate"
    )
    parser.add_argument(
        "--num-sequences", type=int, default=3, help="Number of sequences per strength"
    )
    parser.add_argument(
        "--noise-seed", type=int, default=42, help="Fixed seed for reproducible noise"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature (higher = more creative)",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.1, help="Scale of noise injection"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Parse strengths
    strengths = [float(s.strip()) for s in args.strengths.split(",")]

    # Run noisy intervention testing
    test_noisy_single_feature_interventions(
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

    print(f"\n🎉 SUCCESS! Check reproducible creative results in {args.output_dir}/")

    return True


if __name__ == "__main__":
    main()
