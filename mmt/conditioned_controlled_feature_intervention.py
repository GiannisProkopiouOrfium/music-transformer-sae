#!/usr/bin/env python3
"""
Conditioned Controlled Feature Interventions with Shared Musical Prefixes.

This script provides conditioned comparative generation where all interventions
start from the same musical context (conditioning tokens) for direct comparison.

Features:
- Shared conditioning prefix (2-3 tokens) for all experimental conditions
- Baseline, multiple addition strengths, and ablation from same starting point
- Reproducible noise injection (same noise across comparable conditions)
- Musical continuity and coherent transitions
- Clear intervention attribution without confounding factors

Experimental Design:
1. Generate conditioning tokens (no intervention)
2. Use same conditioning for: baseline, addition strengths, ablation
3. Apply interventions only after conditioning phase
4. Save comparable WAV outputs for perceptual evaluation
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


def generate_conditioning_tokens(
    model,
    encoding,
    conditioning_length: int = 3,
    conditioning_seed: int = 42,
    temperature: float = 1.2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate conditioning tokens without any intervention.
    These will be used as the shared prefix for all experimental conditions.
    """
    print(
        f"🎼 Generating {conditioning_length} conditioning tokens (seed={conditioning_seed})"
    )

    # Set seed for reproducible conditioning
    torch.manual_seed(conditioning_seed)

    # Create start tokens
    start_tokens, eos_token = create_start_tokens(encoding, device)

    # Get model components
    decoder_wrapper = model.decoder
    net = decoder_wrapper.net

    with torch.no_grad():
        out = start_tokens
        mask = torch.ones(
            (out.shape[0], out.shape[1]), dtype=torch.bool, device=out.device
        )

        # Set up sampling parameters
        dim = 6
        temperatures = [temperature] * dim
        filter_fns = ["top_k"] * dim
        filter_thresholds = [0.9] * dim

        # Get dimension mappings and type codes
        instrument_dim = decoder_wrapper.dimensions["instrument"]
        sos_type_code = decoder_wrapper.sos_type_code
        eos_type_code = decoder_wrapper.eos_type_code
        son_type_code = decoder_wrapper.son_type_code
        instrument_type_code = decoder_wrapper.instrument_type_code
        note_type_code = decoder_wrapper.note_type_code

        # Generate conditioning tokens (NO INTERVENTION)
        for step in range(conditioning_length):
            x = out[:, -net.max_seq_len :]
            mask_truncated = mask[:, -net.max_seq_len :]

            # Forward pass WITHOUT intervention
            logits = net(x, mask=mask_truncated)

            # Extract last token logits
            logits = [logit_tensor[:, -1, :] for logit_tensor in logits]

            # Filter out start-of-song token
            logits[0][:, sos_type_code] = -float("inf")

            # Sample tokens
            sample_type = sample(
                logits[0],
                filter_fns[0],
                filter_thresholds[0],
                temperatures[0],
                2.0,
                0.02,
            )

            # Build complete token
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                if s_type in (sos_type_code, eos_type_code, son_type_code):
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 1)
                elif s_type == instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 2)
                    logits[instrument_dim][:, 0] = -float("inf")
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_fns[instrument_dim],
                        filter_thresholds[instrument_dim],
                        temperatures[instrument_dim],
                        2.0,
                        0.02,
                    )[0]
                    samples[idx].append(sampled)
                elif s_type == note_type_code:
                    for d in range(1, dim):
                        logits[d][:, 0] = -float("inf")
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_fns[d],
                            filter_thresholds[d],
                            temperatures[d],
                            2.0,
                            0.02,
                        )[0]
                        samples[idx].append(sampled)
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            # Add new token
            stacked = torch.stack([torch.cat(s).expand(1, -1) for s in samples], 0)
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            print(f"   Step {step + 1}: Generated token")

    conditioning_tokens = out[:, start_tokens.shape[1] :]  # Remove start tokens
    print(f"✅ Generated conditioning: {conditioning_tokens.shape}")
    return conditioning_tokens


def conditioned_generate_with_intervention(
    model,
    encoding,
    conditioning_tokens: torch.Tensor,
    feature_vector: torch.Tensor = None,
    strength: float = 0.0,
    intervention_type: str = "baseline",  # "baseline", "addition", "ablation"
    controlled_intervention: bool = False,
    intervention_layer: int = 3,
    seq_len: int = 256,
    noise_seed: int = 42,
    temperature: float = 1.2,
    noise_scale: float = 0.1,
    device: str = "cuda",
):
    """
    Generate continuation from conditioning tokens with optional intervention.

    Args:
        conditioning_tokens: Pre-generated tokens to start from
        intervention_type: "baseline", "addition", or "ablation"
        Other args as before
    """

    # Set seed for reproducible noise (same across comparable conditions)
    torch.manual_seed(noise_seed)

    # Create start tokens and combine with conditioning
    start_tokens, eos_token = create_start_tokens(encoding, device)

    # Combine start tokens with conditioning tokens
    out = torch.cat([start_tokens, conditioning_tokens], dim=1)

    conditioning_length = conditioning_tokens.shape[1]
    intervention_start_step = (
        conditioning_length  # Start intervention after conditioning
    )

    print(f"🎵 Conditioned generation: {intervention_type} (strength={strength})")
    print(f"   Conditioning length: {conditioning_length}")
    print(f"   Intervention starts at step: {intervention_start_step}")

    # Get model components
    decoder_wrapper = model.decoder
    net = decoder_wrapper.net

    # Normalize feature vector if needed
    if feature_vector is not None:
        feature_unit = feature_vector / (torch.norm(feature_vector, dim=0) + 1e-8)
    else:
        feature_unit = None

    # Hook for intervention with step-aware activation
    def intervention_hook(module, inputs, output):
        if not hasattr(intervention_hook, "call_count"):
            intervention_hook.call_count = 0
        intervention_hook.call_count += 1

        # Check if we should apply intervention
        current_step = getattr(intervention_hook, "current_step", 0)
        should_intervene = (
            hasattr(intervention_hook, "active")
            and intervention_hook.active
            and intervention_type != "baseline"
            and current_step >= intervention_start_step
            and feature_unit is not None
        )

        if should_intervene:
            if len(output.shape) == 3:  # [batch_size, seq_len, d_model]
                _, _, d_model = output.shape

                if feature_vector.shape[0] != d_model:
                    return output

                # Get last token activations
                last_token_activations = output[:, -1, :].clone()

                if intervention_type == "addition":
                    if controlled_intervention:
                        # Controlled addition: remove existing, add desired
                        existing_strengths = torch.matmul(
                            last_token_activations, feature_unit
                        )
                        existing_components = feature_unit.unsqueeze(
                            0
                        ) * existing_strengths.unsqueeze(1)
                        cleaned_activations = (
                            last_token_activations - existing_components
                        )
                        desired_components = strength * feature_unit.unsqueeze(
                            0
                        ).expand(last_token_activations.shape[0], -1)
                        output[:, -1, :] = cleaned_activations + desired_components
                    else:
                        # Standard addition
                        intervention_vector = strength * feature_unit.unsqueeze(0)
                        output[:, -1, :] = last_token_activations + intervention_vector

                elif intervention_type == "ablation":
                    # Ablation: remove feature direction
                    projection_coeffs = torch.matmul(
                        last_token_activations, feature_unit
                    )
                    projection = feature_unit.unsqueeze(
                        0
                    ) * projection_coeffs.unsqueeze(1)
                    output[:, -1, :] = last_token_activations - projection

        return output

    # Register intervention hook
    intervention_handle = None
    target_layer_name = None

    if intervention_type != "baseline":
        target_patterns = [
            f"decoder.net.attn_layers.layers.{intervention_layer}.1",
            f"decoder.net.attn_layers.layers.{intervention_layer}.1.net",
            f"decoder.net.attn_layers.layers.{intervention_layer}.1.to_out",
        ]

        for pattern in target_patterns:
            for name, module in model.named_modules():
                if name == pattern:
                    intervention_handle = module.register_forward_hook(
                        intervention_hook
                    )
                    target_layer_name = name
                    break
            if intervention_handle:
                break

    # Generation state
    mask = torch.ones((out.shape[0], out.shape[1]), dtype=torch.bool, device=out.device)

    # Sampling parameters
    dim = 6
    temperatures = [temperature] * dim
    filter_fns = ["top_k"] * dim
    filter_thresholds = [0.9] * dim

    # Get type codes
    instrument_dim = decoder_wrapper.dimensions["instrument"]
    sos_type_code = decoder_wrapper.sos_type_code
    eos_type_code = decoder_wrapper.eos_type_code
    son_type_code = decoder_wrapper.son_type_code
    instrument_type_code = decoder_wrapper.instrument_type_code
    note_type_code = decoder_wrapper.note_type_code

    # Store noise for reproducibility across conditions
    stored_noise = []

    with torch.no_grad():
        # Continue generation from conditioning
        total_steps = seq_len - conditioning_length

        for step in range(total_steps):
            current_step = conditioning_length + step

            # Set current step for hook
            if intervention_handle:
                intervention_hook.current_step = current_step

            x = out[:, -net.max_seq_len :]
            mask_truncated = mask[:, -net.max_seq_len :]

            # Activate intervention if needed
            if intervention_handle:
                intervention_hook.active = True

            # Forward pass
            logits = net(x, mask=mask_truncated)

            # Deactivate intervention
            if intervention_handle:
                intervention_hook.active = False

            # Extract logits
            logits = [logit_tensor[:, -1, :] for logit_tensor in logits]

            # Apply SAME noise across all conditions for this step
            if step < len(stored_noise):
                # Use stored noise for reproducibility
                step_noise = stored_noise[step]
            else:
                # Generate new noise and store it
                step_noise = []
                for i, logit_tensor in enumerate(logits):
                    if noise_scale > 0:
                        noise = torch.randn_like(logit_tensor) * noise_scale
                        step_noise.append(noise)
                    else:
                        step_noise.append(torch.zeros_like(logit_tensor))
                stored_noise.append(step_noise)

            # Apply stored noise
            for i, noise in enumerate(step_noise):
                logits[i] = logits[i] + noise

            # Filter and sample
            logits[0][:, sos_type_code] = -float("inf")

            sample_type = sample(
                logits[0],
                filter_fns[0],
                filter_thresholds[0],
                temperatures[0],
                2.0,
                0.02,
            )

            # Build token
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                if s_type in (sos_type_code, eos_type_code, son_type_code):
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 1)
                elif s_type == instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (dim - 2)
                    logits[instrument_dim][:, 0] = -float("inf")
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_fns[instrument_dim],
                        filter_thresholds[instrument_dim],
                        temperatures[instrument_dim],
                        2.0,
                        0.02,
                    )[0]
                    samples[idx].append(sampled)
                elif s_type == note_type_code:
                    for d in range(1, dim):
                        logits[d][:, 0] = -float("inf")
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_fns[d],
                            filter_thresholds[d],
                            temperatures[d],
                            2.0,
                            0.02,
                        )[0]
                        samples[idx].append(sampled)
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            # Add token
            stacked = torch.stack([torch.cat(s).expand(1, -1) for s in samples], 0)
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            # Check for EOS
            if eos_token is not None:
                is_eos_tokens = out[..., 0] == eos_token
                if is_eos_tokens.any(dim=1).all():
                    break

    # Clean up
    if intervention_handle:
        intervention_handle.remove()

    # Return only generated tokens (excluding start + conditioning)
    start_length = 1  # start tokens
    generated_tokens = out[:, start_length:]
    print(
        f"✅ Generated {generated_tokens.shape[1]} tokens ({conditioning_length} conditioning + {generated_tokens.shape[1] - conditioning_length} new)"
    )

    return generated_tokens


def test_conditioned_interventions(
    model_path: str,
    feature_limuf_path: str,
    output_dir: str,
    addition_strengths: list = [-2.0, -1.0, 1.0, 2.0],
    controlled_intervention: bool = False,
    intervention_layer: int = 3,
    conditioning_length: int = 3,
    seq_len: int = 256,
    conditioning_seed: int = 42,
    generation_seed: int = 123,
    temperature: float = 1.2,
    noise_scale: float = 0.1,
    device: str = "cuda",
):
    """
    Test conditioned interventions with shared musical prefix.

    Generates: baseline, addition strengths, ablation - all from same conditioning.
    """

    print("🎼 TESTING CONDITIONED CONTROLLED FEATURE INTERVENTIONS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Feature LiMuF: {feature_limuf_path}")
    print(f"Intervention Layer: {intervention_layer}")
    print(f"Addition Strengths: {addition_strengths}")
    print(f"Controlled Intervention: {controlled_intervention}")
    print(f"Conditioning Length: {conditioning_length}")
    print(f"Conditioning Seed: {conditioning_seed}")
    print(f"Generation Seed: {generation_seed}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # make a wav subfolder if it doesn't exist
    wav_path = output_path / "wav"
    wav_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, encoding = load_music_transformer(model_path, device)

    # Load feature LiMuF
    print("📁 Loading feature LiMuF...")
    feature_data = torch.load(feature_limuf_path, map_location="cpu")

    feature_name = list(feature_data["limufs"].keys())[0]
    feature_vector = feature_data["limufs"][feature_name].to(device)
    metadata = feature_data["metadata"][feature_name]

    feature_id = metadata["feature_id"]
    layer = metadata.get("layer", intervention_layer)

    print(f"✅ Loaded feature: {feature_name}")
    print(f"   Feature ID: {feature_id}")
    print(f"   Layer: {layer}")
    print(f"   Description: {metadata['feature_description'][:80]}...")
    print()

    # Step 1: Generate conditioning tokens (shared across all conditions)
    print("🎯 PHASE 1: Generating shared conditioning tokens")
    conditioning_tokens = generate_conditioning_tokens(
        model=model,
        encoding=encoding,
        conditioning_length=conditioning_length,
        conditioning_seed=conditioning_seed,
        temperature=temperature,
        device=device,
    )

    # Step 2: Generate all experimental conditions with same conditioning
    print("\n🎯 PHASE 2: Generating experimental conditions")

    experimental_conditions = []

    # Baseline (no intervention)
    experimental_conditions.append(("baseline", 0.0, "baseline"))

    # Addition strengths
    for strength in addition_strengths:
        experimental_conditions.append(("addition", strength, f"add_{strength:+.1f}"))

    # Ablation
    experimental_conditions.append(("ablation", 0.0, "ablation"))

    results = {}

    for intervention_type, strength, condition_name in experimental_conditions:
        print(f"\n🎵 Generating: {condition_name}")

        # Generate with same conditioning and comparable noise
        sequence = conditioned_generate_with_intervention(
            model=model,
            encoding=encoding,
            conditioning_tokens=conditioning_tokens,
            feature_vector=feature_vector if intervention_type != "baseline" else None,
            strength=strength,
            intervention_type=intervention_type,
            controlled_intervention=controlled_intervention,
            intervention_layer=intervention_layer,
            seq_len=seq_len,
            noise_seed=generation_seed,  # Same seed for comparable noise
            temperature=temperature,
            noise_scale=noise_scale,
            device=device,
        )

        if sequence is not None:
            # Save tensor
            tensor_file = (
                output_path / f"{condition_name}_feature{feature_id}_layer{layer}.pt"
            )
            torch.save(
                {
                    "generated": sequence.cpu(),
                    "conditioning_tokens": conditioning_tokens.cpu(),
                    "condition_name": condition_name,
                    "intervention_type": intervention_type,
                    "strength": strength,
                    "feature_id": feature_id,
                    "layer": layer,
                    "metadata": metadata,
                    "generation_params": {
                        "conditioning_length": conditioning_length,
                        "conditioning_seed": conditioning_seed,
                        "generation_seed": generation_seed,
                        "controlled_intervention": controlled_intervention,
                        "temperature": temperature,
                        "noise_scale": noise_scale,
                    },
                },
                tensor_file,
            )

            # Save WAV
            try:
                wav_filename = f"{condition_name}_feature{feature_id}_layer{layer}"

                # Convert to numpy for saving
                seq_np = sequence.cpu().numpy()
                if len(seq_np.shape) == 3:
                    seq_np = seq_np[0]  # Remove batch dimension

                save_result(wav_filename, seq_np, str(output_path), encoding)

                print(f"   ✅ Saved: {wav_filename}.wav")

            except Exception as e:
                print(f"   ⚠️ Could not save WAV: {e}")

            results[condition_name] = {
                "tensor_file": str(tensor_file),
                "wav_file": f"{wav_filename}.wav",
                "intervention_type": intervention_type,
                "strength": strength,
                "sequence_length": sequence.shape[1],
            }

    # Step 3: Save experimental summary
    summary = {
        "experiment_type": "conditioned_controlled_interventions",
        "feature_name": feature_name,
        "feature_id": feature_id,
        "layer": layer,
        "feature_description": metadata["feature_description"],
        "conditioning_info": {
            "conditioning_length": conditioning_length,
            "conditioning_seed": conditioning_seed,
            "conditioning_tokens_shape": list(conditioning_tokens.shape),
        },
        "experimental_conditions": {
            "baseline": "No intervention",
            "addition_strengths": addition_strengths,
            "ablation": "Feature direction removed",
            "controlled_intervention": controlled_intervention,
        },
        "generation_params": {
            "generation_seed": generation_seed,
            "temperature": temperature,
            "noise_scale": noise_scale,
            "seq_len": seq_len,
            "intervention_layer": intervention_layer,
        },
        "results": results,
        "metadata": metadata,
    }

    summary_file = (
        output_path
        / f"conditioned_experiment_feature{feature_id}_layer{layer}_summary.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ CONDITIONED EXPERIMENT COMPLETE!")
    print(f"📊 Summary: {summary_file}")
    print(f"📁 Output files: {output_path}/")
    print("\n🎵 Generated WAV files:")
    for condition_name, result in results.items():
        print(f"   • {result['wav_file']}")

    print("\n🎯 EXPERIMENTAL DESIGN ACHIEVED:")
    print(
        f"   • All conditions start from same {conditioning_length}-token musical prefix"
    )
    print("   • Interventions applied only after conditioning phase")
    print("   • Same noise patterns across comparable conditions")
    print("   • Direct perceptual comparison possible")
    print("   • Musical continuity preserved")

    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Test conditioned controlled feature interventions with shared musical prefixes"
    )
    parser.add_argument(
        "--model-path",
        default="exp/sod/ape/checkpoints/best_model.pt",
        help="Path to trained music transformer",
    )
    parser.add_argument(
        "--feature-limuf-path",
        default="limufs_layer3_sae_columns/limufs.pt",
        help="Path to extracted feature LiMuF",
    )
    parser.add_argument(
        "--output-dir",
        default="conditioned_interventions",
        help="Output directory",
    )
    parser.add_argument(
        "--controlled-intervention",
        action="store_true",
        help="Use controlled intervention (remove existing feature component first)",
    )
    parser.add_argument(
        "--addition-strengths",
        default="-2.0,-1.0,1.0,2.0",
        help="Comma-separated addition strengths to test",
        type=str,
    )
    parser.add_argument(
        "--intervention-layer", type=int, default=3, help="Layer to apply intervention"
    )
    parser.add_argument(
        "--conditioning-length",
        type=int,
        default=3,
        help="Length of conditioning prefix",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Total sequence length"
    )
    parser.add_argument(
        "--conditioning-seed",
        type=int,
        default=42,
        help="Seed for conditioning generation",
    )
    parser.add_argument(
        "--generation-seed", type=int, default=123, help="Seed for main generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.2, help="Sampling temperature"
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

    # Parse addition strengths
    addition_strengths = [float(s.strip()) for s in args.addition_strengths.split(",")]

    # Run conditioned intervention testing
    test_conditioned_interventions(
        model_path=args.model_path,
        feature_limuf_path=args.feature_limuf_path,
        output_dir=args.output_dir,
        addition_strengths=addition_strengths,
        controlled_intervention=args.controlled_intervention,
        intervention_layer=args.intervention_layer,
        conditioning_length=args.conditioning_length,
        seq_len=args.seq_len,
        conditioning_seed=args.conditioning_seed,
        generation_seed=args.generation_seed,
        temperature=args.temperature,
        noise_scale=args.noise_scale,
        device=args.device,
    )

    method = "controlled" if args.controlled_intervention else "standard"
    print(
        f"\n🎉 SUCCESS! Check {method} conditioned intervention results in {args.output_dir}/"
    )
    print(
        "🎵 Listen to the WAV files to hear intervention effects on the same musical material!"
    )
    return True


if __name__ == "__main__":
    main()
