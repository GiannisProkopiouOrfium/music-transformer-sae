"""Simple example demonstrating how to use steering interventions.

This script shows how to:
1. Load steering vectors
2. Generate music with different steering strengths
3. Compare results
"""

import pathlib
import sys

import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors


def main():
    """Simple example usage."""
    print("=" * 60)
    print("Steering Interventions - Simple Example")
    print("=" * 60)
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")
    
    model = music_x_transformers.MusicXTransformer(
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
    
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✓ Model loaded")
    
    # Load steering vectors
    print("\nLoading steering vectors...")
    concept = "velocity"
    steering_path = config.OUTPUT_DIR / "steering_vectors" / f"{concept}_steering_vectors.pt"
    
    if not steering_path.exists():
        print(f"\nError: Steering vectors not found at {steering_path}")
        print("Please run the pipeline first:")
        print("  python run_pipeline.py --concept velocity")
        return
    
    steering_vectors, metadata = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}
    print(f"✓ Loaded steering vectors for concept: {metadata.get('concept', 'unknown')}")
    
    # Create steered generator
    generator = SteeredGenerator(model, steering_vectors, encoding)
    
    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    
    # Generate with different alpha values
    alpha_values = [-1.0, 0.0, 1.0]
    seq_len = 256
    
    output_dir = pathlib.Path(__file__).parent / "outputs" / "example"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(alpha_values)} samples...")
    
    for alpha in alpha_values:
        print(f"\n  Alpha = {alpha:+.1f}")
        
        # Create start tokens
        tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
        tgt_start[:, 0, 0] = sos
        
        # Generate
        generated = generator.generate(
            tgt_start,
            seq_len,
            alpha=alpha,
            target_layers=None,  # Use all layers
            eos_token=eos,
            temperature=1.0,
            filter_logits_fn="top_k",
            filter_thres=0.9,
            monotonicity_dim=("type", "beat"),
        )
        
        # Combine with start
        full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]
        
        # Decode and measure velocity
        music = representation.decode(full_seq, encoding)
        
        velocities = []
        for track in music.tracks:
            for note in track.notes:
                velocities.append(note.velocity)
        
        if velocities:
            mean_velocity = sum(velocities) / len(velocities)
            print(f"    Mean velocity: {mean_velocity:.1f} ({len(velocities)} notes)")
        else:
            print("    No notes generated")
        
        # Save
        output_file = output_dir / f"example_alpha_{alpha:+.1f}.mid"
        music.write(output_file)
        print(f"    Saved: {output_file}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print(f"Check {output_dir} for generated MIDI files")
    print("=" * 60)


if __name__ == "__main__":
    main()
