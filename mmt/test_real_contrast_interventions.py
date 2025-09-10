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


def load_music_transformer(model_path: str, device: str = "cuda"):
    """Load your trained MusicXTransformer."""
    print(f"📦 Loading MusicXTransformer from: {model_path}")

    # Load training arguments
    exp_dir = pathlib.Path(model_path).parent.parent
    train_args = utils.load_json(exp_dir / "train-args.json")
    print(f"📄 Loaded training args from: {exp_dir / 'train-args.json'}")

    # Load encoding
    encoding = representation.load_encoding("data/sod/processed/notes/encoding.json")
    print(f"📄 Loaded encoding from: data/sod/processed/notes/encoding.json")

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
    return model


def generate_with_intervention(
    model,
    contrast_vector: torch.Tensor,
    strength: float,
    intervention_layer: int = 3,
    seq_len: int = 256,
    num_sequences: int = 3,
    device: str = "cuda",
):
    """Generate sequences with contrast intervention."""

    # Hook for intervention
    def intervention_hook(module, input, output):
        if hasattr(intervention_hook, "active") and intervention_hook.active:
            # Apply intervention to last token
            # output shape: [batch_size, seq_len, d_model]
            if len(output.shape) == 3:
                output[:, -1, :] += strength * contrast_vector.unsqueeze(0)
            else:
                output += strength * contrast_vector.unsqueeze(0)
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
        print("   Available modules:")
        for name, _ in list(model.named_modules())[:10]:  # Show first 10
            print(f"     {name}")
        print("     ...")

    # Generate sequences
    generated_sequences = []

    with torch.no_grad():
        for seq_idx in tqdm(
            range(num_sequences), desc=f"Generating (strength {strength:+.1f})"
        ):
            # Create simple prompt (adjust based on your representation)
            prompt = torch.tensor([[1, 2]], device=device)  # Start tokens

            # Activate intervention
            intervention_hook.active = True

            # Generate sequence
            sequence = generate_sequence(
                model=model, prompt=prompt, max_length=seq_len, device=device
            )

            # Deactivate intervention
            intervention_hook.active = False

            generated_sequences.append(sequence.cpu())

    # Remove hook
    if hook_handle:
        hook_handle.remove()

    return generated_sequences


def generate_sequence(
    model, prompt, max_length: int, device: str = "cuda", temperature: float = 1.0
):
    """Generate a single sequence using MusicXTransformer."""
    model.eval()

    with torch.no_grad():
        sequence = prompt.clone()

        for _ in range(max_length - prompt.shape[1]):
            try:
                # Forward pass with MusicXTransformer
                # MusicXTransformer returns logits directly
                logits = model(sequence)

                # Handle different output formats
                if isinstance(logits, tuple):
                    logits = logits[0]  # Take first element if tuple

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Append to sequence
                sequence = torch.cat([sequence, next_token], dim=1)

                # Check for end token or max length
                if next_token.item() == 0:  # Assuming 0 is end token
                    break

            except Exception as e:
                print(f"     Generation error at step {sequence.shape[1]}: {e}")
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
    device: str = "cuda",
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
    print(
        f"   Feature A: {metadata['feature_a_id']} - {metadata['feature_a_description'][:60]}..."
    )
    print(
        f"   Feature B: {metadata['feature_b_id']} - {metadata['feature_b_description'][:60]}..."
    )
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
            device=device,
        )

        # Save sequences
        strength_name = (
            f"strength_{strength:+.1f}".replace(".", "_")
            .replace("+", "plus")
            .replace("-", "minus")
        )
        output_file = output_path / f"{contrast_name}_{strength_name}.pt"

        torch.save(
            {
                "sequences": sequences,
                "strength": strength,
                "contrast_name": contrast_name,
                "metadata": metadata,
                "generation_params": {
                    "num_sequences": num_sequences,
                    "seq_len": seq_len,
                    "intervention_layer": intervention_layer,
                    "model_path": model_path,
                },
            },
            output_file,
        )

        all_results[strength] = {
            "file": str(output_file),
            "num_sequences": len(sequences),
            "avg_length": float(torch.stack(sequences).shape[1]) if sequences else 0,
        }

        print(f"   ✅ Saved {len(sequences)} sequences to {output_file}")

    # Save summary
    summary = {
        "contrast_name": contrast_name,
        "feature_a_id": metadata["feature_a_id"],
        "feature_b_id": metadata["feature_b_id"],
        "strengths_tested": strengths,
        "results": all_results,
        "metadata": metadata,
        "generation_params": {
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "intervention_layer": intervention_layer,
            "model_path": model_path,
        },
    }

    summary_file = output_path / "generation_summary.json"
    with open(summary_file, "w") as f:
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
        help="Path to trained music transformer",
    )
    parser.add_argument(
        "--contrast-limuf-path",
        default="real_contrast_182_vs_1743/limufs.pt",
        help="Path to extracted contrast LiMuF",
    )
    parser.add_argument(
        "--output-dir",
        default="real_contrast_interventions_182_vs_1743",
        help="Output directory",
    )
    parser.add_argument(
        "--strengths",
        default="-2.0,-1.0,0.0,1.0,2.0",
        help="Comma-separated intervention strengths",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
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
        device=args.device,
    )

    print(f"\n🎉 SUCCESS! Check results in {args.output_dir}/")

    return True


if __name__ == "__main__":
    main()
