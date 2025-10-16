#!/usr/bin/env python3
"""
Song-Conditioned Feature Interventions

This script performs feature interventions using actual song prefixes as conditioning,
specifically designed for contrasting songs where interventions create dramatic effects.

Key difference from conditioned_controlled_feature_intervention.py:
- Uses REAL SONG DATA as conditioning prefix (first N tokens of contrasting songs)
- Automatically includes start-of-notes token for proper generation
- Ensures musical continuity from existing compositions
- Creates comparable interventions starting from same musical material

Usage:
    python song_conditioned_interventions.py \
        --song-path data/sod/processed/notes/bach_bwv001.pt \
        --feature-limuf-path limufs.pt \
        --output-dir output \
        --conditioning-length 3 \
        --addition-strengths -2.0,-1.0,1.0,2.0

Notes:
    - Baseline (strength 0.0) and ablation are ALWAYS generated automatically
    - addition-strengths should NOT include 0.0 (it will be skipped if present)
    - conditioning-length specifies minimum tokens; script extends to include start-of-notes
    
Output: all_wavs/ folder containing:
    - baseline.wav (always)
    - ablation.wav (always)
    - add_-2.0.wav, add_-1.0.wav, add_+1.0.wav, add_+2.0.wav (one per strength)
"""

import argparse
import json
import logging
import pathlib
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "mmt"))
sys.path.insert(0, str(parent_dir / "baseline"))

# Import model components
from mmt.music_x_transformers import MusicXTransformer, sample
from mmt import representation
from mmt import utils


def load_song_data(song_path: Path, encoding: dict) -> torch.Tensor:
    """
    Load song data from .pt file.

    Args:
        song_path: Path to .pt file containing note tokens
        encoding: Encoding dictionary

    Returns:
        Tensor of shape (1, seq_len, 6) containing note tokens
    """
    print(f"📁 Loading song from: {song_path}")

    data = torch.load(song_path, map_location="cpu")

    # Handle different possible formats
    if isinstance(data, dict):
        # Might have keys like 'tokens', 'notes', etc.
        if "tokens" in data:
            tokens = data["tokens"]
        elif "notes" in data:
            tokens = data["notes"]
        else:
            # Try first key
            tokens = data[list(data.keys())[0]]
    else:
        tokens = data

    # Ensure correct shape: (1, seq_len, 6)
    if len(tokens.shape) == 2:
        tokens = tokens.unsqueeze(0)  # Add batch dimension
    elif len(tokens.shape) == 3 and tokens.shape[0] != 1:
        tokens = tokens[:1]  # Take first batch item

    print(f"✅ Loaded song data: shape {tokens.shape}")

    # Debug: Show first few tokens from the original song
    print(f"   🔍 Debug: First 5 tokens from song file:")
    code_type_map = encoding["code_type_map"]
    for i in range(min(5, tokens.shape[1])):
        token = tokens[0, i]
        event_type = code_type_map.get(int(token[0]), f"unknown_{int(token[0])}")
        print(f"      Token {i}: type={event_type}, data={token.tolist()}")

    return tokens


def extract_conditioning_prefix(
    song_tokens: torch.Tensor, conditioning_length: int, encoding: dict
) -> torch.Tensor:
    """
    Extract first N tokens of song as conditioning prefix, ensuring we include start-of-notes.

    Args:
        song_tokens: Full song tokens (1, seq_len, 6)
        conditioning_length: Number of TOKENS to use as minimum
        encoding: Encoding dictionary

    Returns:
        Conditioning tokens (1, conditioning_seq_len, 6)
    """
    print(
        f"✂️  Extracting conditioning prefix (minimum {conditioning_length} tokens)..."
    )

    # Get event type codes
    son_type_code = encoding["type_code_map"]["start-of-notes"]

    # Start with requested length
    conditioning_end_idx = min(conditioning_length, song_tokens.shape[1])

    # Check if we have start-of-notes token in the conditioning
    has_start_of_notes = False
    for i in range(conditioning_end_idx):
        if song_tokens[0, i, 0] == son_type_code:
            has_start_of_notes = True
            break

    # If not, extend conditioning to include it
    if not has_start_of_notes:
        print(f"   ⚠️  Conditioning doesn't include start-of-notes, extending...")
        # Search for start-of-notes in next tokens
        for i in range(
            conditioning_end_idx, min(song_tokens.shape[1], conditioning_length + 10)
        ):
            if song_tokens[0, i, 0] == son_type_code:
                conditioning_end_idx = i + 1  # Include the start-of-notes token
                print(
                    f"   ✓ Found start-of-notes at token {i}, extending to {conditioning_end_idx} tokens"
                )
                break

    conditioning_tokens = song_tokens[:, :conditioning_end_idx, :]

    # Verify we have proper structure
    token_types = []
    for i in range(min(5, conditioning_tokens.shape[1])):
        token_type = conditioning_tokens[0, i, 0].item()
        token_type_name = encoding["code_type_map"].get(
            token_type, f"unknown_{token_type}"
        )
        token_types.append(token_type_name)

    print(f"✅ Conditioning prefix: {conditioning_tokens.shape[1]} tokens")
    print(f"   Token sequence: {' → '.join(token_types)}")

    return conditioning_tokens


def load_music_transformer(model_path: str, device: str = "cuda"):
    """Load trained MusicXTransformer."""
    print(f"📦 Loading MusicXTransformer from: {model_path}")

    # Load training arguments
    exp_dir = pathlib.Path(model_path).parent.parent
    train_args = utils.load_json(exp_dir / "train-args.json")

    # Load encoding
    encoding = representation.load_encoding("data/sod/processed/notes/encoding.json")

    # Create model
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
    return model, encoding, train_args


def song_conditioned_generate(
    model,
    encoding,
    conditioning_tokens: torch.Tensor,
    feature_vector: Optional[torch.Tensor] = None,
    strength: float = 0.0,
    intervention_type: str = "baseline",
    intervention_layer: int = 3,
    seq_len: int = 512,
    temperature: float = 0.1,
    noise_scale: float = 1.2,
    generation_seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate continuation from song conditioning with optional intervention.

    Args:
        conditioning_tokens: Real song tokens to condition on (1, cond_len, 6)
        feature_vector: Feature direction for intervention
        strength: Intervention strength
        intervention_type: "baseline", "addition", or "ablation"
        Other standard generation parameters

    Returns:
        Complete sequence including conditioning + generation
    """
    # Set seed for reproducibility
    torch.manual_seed(generation_seed)

    conditioning_tokens = conditioning_tokens.to(device)
    conditioning_length = conditioning_tokens.shape[1]

    # Check if conditioning already has start-of-song token
    sos = encoding["type_code_map"]["start-of-song"]
    has_start_token = (conditioning_tokens[0, 0, 0] == sos).item()

    if has_start_token:
        # Conditioning already includes start-of-song, use as-is
        out = conditioning_tokens.clone()
        conditioning_length_with_start = conditioning_length
        print(f"   ℹ️  Conditioning already includes start-of-song token")
    else:
        # Need to add start-of-song token
        start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
        start_tokens[:, 0, 0] = sos
        out = torch.cat([start_tokens, conditioning_tokens], dim=1)
        conditioning_length_with_start = conditioning_length + 1
        print(f"   ℹ️  Added start-of-song token to conditioning")

    mask = torch.ones((out.shape[0], out.shape[1]), dtype=torch.bool, device=device)
    intervention_start_step = (
        conditioning_length_with_start  # Intervention starts AFTER conditioning
    )

    print(f"🎵 Song-conditioned generation: {intervention_type} (strength={strength})")
    print(f"   Conditioning: {conditioning_tokens.shape[1]} tokens from real song")
    print(f"   Intervention starts after token: {intervention_start_step}")
    print(f"   Target length: {seq_len} tokens")
    print(f"   Will generate: {seq_len - conditioning_length_with_start} new tokens")

    # Get model components
    decoder_wrapper = model.decoder
    net = decoder_wrapper.net

    # Normalize feature vector
    if feature_vector is not None:
        feature_unit = feature_vector / (torch.norm(feature_vector, dim=0) + 1e-8)
    else:
        feature_unit = None

    # Intervention hook - only applies to NEW tokens after conditioning
    def intervention_hook(module, inputs, output):
        if not hasattr(intervention_hook, "current_step"):
            return output

        current_step = intervention_hook.current_step
        should_intervene = (
            hasattr(intervention_hook, "active")
            and intervention_hook.active
            and intervention_type != "baseline"
            and current_step >= intervention_start_step  # Only intervene to NEW tokens
            and feature_unit is not None
        )

        if should_intervene:
            if len(output.shape) == 3:
                batch, seq, dim = output.shape

                # Get current token activation
                current_activation = output[:, -1:, :]  # (1, 1, dim)

                if intervention_type == "addition":
                    # Add feature direction
                    intervention = strength * feature_unit.unsqueeze(0).unsqueeze(0)
                    output[:, -1:, :] = current_activation + intervention

                elif intervention_type == "ablation":
                    # Remove feature component
                    projection = torch.sum(
                        current_activation * feature_unit.unsqueeze(0).unsqueeze(0),
                        dim=-1,
                        keepdim=True,
                    )
                    output[:, -1:, :] = (
                        current_activation
                        - projection * feature_unit.unsqueeze(0).unsqueeze(0)
                    )

        return output

    # Register hook
    intervention_handle = None
    if intervention_type != "baseline":
        target_patterns = [
            f"decoder.net.attn_layers.layers.{intervention_layer}.1",
            f"decoder.net.attn_layers.layers.{intervention_layer}.1.net",
            f"decoder.net.attn_layers.layers.{intervention_layer}.1.to_out",
        ]

        for pattern in target_patterns:
            for name, module in model.named_modules():
                if pattern in name and "to_out" in name:
                    intervention_handle = module.register_forward_hook(
                        intervention_hook
                    )
                    print(f"   Registered intervention hook at: {name}")
                    break
            if intervention_handle:
                break

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

    # Generate continuation
    total_steps = seq_len - conditioning_length_with_start

    with torch.no_grad():
        for step in range(total_steps):
            current_step = conditioning_length_with_start + step

            # Update hook's current step
            if intervention_handle:
                intervention_hook.current_step = current_step

            # Truncate to max sequence length
            x = out[:, -net.max_seq_len :]
            mask_truncated = mask[:, -net.max_seq_len :]

            # Activate intervention
            if intervention_handle:
                intervention_hook.active = True

            # Forward pass
            logits = net(x, mask=mask_truncated)

            # Deactivate intervention
            if intervention_handle:
                intervention_hook.active = False

            # Extract last token logits
            logits = [logit_tensor[:, -1, :] for logit_tensor in logits]

            # Filter start-of-song token
            logits[0][:, sos_type_code] = -float("inf")

            # Sample type
            sample_type = sample(
                logits[0],
                filter_fns[0],
                filter_thresholds[0],
                temperatures[0],
                noise_scale,
                0.02,
            )

            # Build complete token based on type
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                if s_type == eos_type_code:
                    # End of song
                    samples[idx] += [torch.tensor([0] * (dim - 1), device=device)]
                elif s_type == son_type_code:
                    # Start of notes - sample instrument
                    instrument = sample(
                        logits[instrument_dim][[idx]],
                        filter_fns[instrument_dim],
                        filter_thresholds[instrument_dim],
                        temperatures[instrument_dim],
                        noise_scale,
                        0.02,
                    )
                    samples[idx].append(instrument[0])
                    samples[idx] += [torch.tensor([0] * (dim - 2), device=device)]
                elif s_type == instrument_type_code:
                    # Instrument change
                    instrument = sample(
                        logits[1][[idx]],
                        filter_fns[1],
                        filter_thresholds[1],
                        temperatures[1],
                        noise_scale,
                        0.02,
                    )
                    samples[idx].append(instrument[0])
                    samples[idx] += [torch.tensor([0] * (dim - 2), device=device)]
                elif s_type == note_type_code:
                    # Note event - sample all dimensions
                    for i in range(1, dim):
                        value = sample(
                            logits[i][[idx]],
                            filter_fns[i],
                            filter_thresholds[i],
                            temperatures[i],
                            noise_scale,
                            0.02,
                        )
                        samples[idx].append(value[0])
                else:
                    # Other event types
                    samples[idx] += [torch.tensor([0] * (dim - 1), device=device)]

            # Stack and append
            stacked = torch.stack([torch.cat(s).expand(1, -1) for s in samples], 0)
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            # Check for end-of-song
            if (stacked[0, 0, 0] == eos_type_code).any():
                print(f"   Reached end-of-song at step {step + 1}")
                break

            if (step + 1) % 50 == 0:
                print(f"   Generated {step + 1}/{total_steps} tokens...")

    # Clean up
    if intervention_handle:
        intervention_handle.remove()

    # Return complete sequence INCLUDING start token (needed for proper decoding)
    # The sequence should be: [start-of-song] + [conditioning tokens] + [new generated tokens]
    generated_tokens = out  # Keep everything including start token
    print(
        f"✅ Generation complete: {generated_tokens.shape[1]} total tokens (1 start + {conditioning_tokens.shape[1]} conditioning + {generated_tokens.shape[1] - conditioning_length_with_start} new)"
    )
    return generated_tokens


def save_result(filename: str, tokens: torch.Tensor, encoding: dict, output_dir: Path):
    """Save generated tokens as MIDI and WAV."""
    # Convert tokens to numpy
    tokens_np = tokens[0].cpu().numpy()  # Remove batch dimension

    # Save as numpy
    np.save(output_dir / f"{filename}.npy", tokens_np)

    # Decode to MusPy Music object
    try:
        # Debug: Check token shape and first few tokens
        print(f"   🔍 Debug: tokens_np shape = {tokens_np.shape}")
        print(f"   🔍 Debug: first 5 tokens = {tokens_np[:5]}")

        # Decode notes first to check if we get any
        notes = representation.decode_notes(tokens_np, encoding)
        print(f"   🔍 Debug: decoded {len(notes)} notes")

        if len(notes) == 0:
            print(f"   ⚠️  Warning: No notes were decoded from tokens!")
            print(f"   🔍 Debug: Checking event types in first 10 tokens...")
            code_type_map = encoding["code_type_map"]
            for i, row in enumerate(tokens_np[:10]):
                event_type = code_type_map.get(int(row[0]), f"unknown_{int(row[0])}")
                print(f"      Token {i}: type={event_type}, full={row}")
            return None, None

        # Now decode full music
        music = representation.decode(tokens_np, encoding)

        print(f"   🔍 Debug: Music object has {len(music.tracks)} tracks")
        for i, track in enumerate(music.tracks):
            print(f"      Track {i}: program={track.program}, {len(track.notes)} notes")

        # Save MIDI
        midi_path = output_dir / f"{filename}.mid"
        music.write(str(midi_path))
        print(f"   💾 Saved MIDI: {filename}.mid")

        # Save WAV (using MusPy's audio synthesis)
        wav_path = output_dir / f"{filename}.wav"
        music.write_audio(str(wav_path))
        print(f"   💾 Saved WAV: {filename}.wav")

        return str(wav_path), str(midi_path)

    except Exception as e:
        import traceback

        print(f"   ⚠️  Could not save audio: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None, None


def run_song_conditioned_interventions(
    song_path: Path,
    feature_limuf_path: Path,
    output_dir: Path,
    model_path: Path = None,
    conditioning_length: int = 4,
    seq_len: int = 512,
    addition_strengths: List[float] = None,
    intervention_layer: int = 3,
    temperature: float = 0.1,
    noise_scale: float = 1.2,
    generation_seed: int = 42,
    device: str = "cuda",
):
    """
    Run full intervention pipeline on one song.

    Args:
        song_path: Path to song .pt file
        feature_limuf_path: Path to feature LiMuF
        output_dir: Where to save results
        conditioning_length: Number of beats to use as prefix
        addition_strengths: List of strengths for additions (baseline and ablation always generated)
        Other generation parameters
    """
    if addition_strengths is None:
        addition_strengths = [
            -2.0,
            -1.0,
            1.0,
            2.0,
        ]  # Baseline and ablation generated automatically

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SONG-CONDITIONED FEATURE INTERVENTIONS")
    print("=" * 80)
    print(f"Song: {song_path.name}")
    print(f"Feature LiMuF: {feature_limuf_path}")
    print(f"Conditioning: {conditioning_length} beats")
    print(f"Strengths: {addition_strengths}")
    print(f"Output: {output_dir}")
    print()

    # Default model path if not provided
    if model_path is None:
        model_path = Path("exp/sod/ape/checkpoints/best_model.pt")

    # Load model
    model, encoding, train_args = load_music_transformer(str(model_path), device)

    # Load song data
    song_tokens = load_song_data(song_path, encoding)

    # Extract conditioning prefix
    conditioning_tokens = extract_conditioning_prefix(
        song_tokens, conditioning_length, encoding
    )

    # Load feature LiMuF
    print("📁 Loading feature LiMuF...")
    feature_data = torch.load(feature_limuf_path, map_location="cpu")
    feature_name = list(feature_data["limufs"].keys())[0]
    feature_vector = feature_data["limufs"][feature_name].to(device)
    metadata = feature_data["metadata"][feature_name]

    feature_id = metadata["feature_id"]
    print(f"✅ Feature: {feature_name} (ID: {feature_id})")
    print()

    # Generate interventions
    results = {}

    experimental_conditions = []

    # Baseline first (only once, even if 0.0 is in addition_strengths)
    experimental_conditions.append(("baseline", 0.0, "baseline"))

    # Addition strengths (skip 0.0 if present since we did baseline)
    for strength in addition_strengths:
        if abs(strength) > 0.001:  # Skip 0.0
            experimental_conditions.append(
                ("addition", strength, f"add_{strength:+.1f}")
            )

    # Ablation last (only once)
    experimental_conditions.append(("ablation", 0.0, "ablation"))

    for intervention_type, strength, condition_name in experimental_conditions:
        print(f"\n{'='*80}")
        print(f"Generating: {condition_name}")
        print(f"{'='*80}")

        sequence = song_conditioned_generate(
            model=model,
            encoding=encoding,
            conditioning_tokens=conditioning_tokens,
            feature_vector=feature_vector if intervention_type != "baseline" else None,
            strength=strength,
            intervention_type=intervention_type,
            intervention_layer=intervention_layer,
            seq_len=seq_len,
            temperature=temperature,
            noise_scale=noise_scale,
            generation_seed=generation_seed,
            device=device,
        )

        # Save result
        song_name = song_path.stem
        filename = f"{song_name}_{condition_name}"
        wav_path, midi_path = save_result(filename, sequence, encoding, output_dir)

        if wav_path:
            results[condition_name] = {
                "intervention_type": intervention_type,
                "strength": strength,
                "wav_path": wav_path,
                "midi_path": midi_path,
                "sequence_length": sequence.shape[1],
                "conditioning_length": conditioning_tokens.shape[1],
            }

    # Save summary
    summary = {
        "song": str(song_path),
        "song_name": song_path.stem,
        "feature_id": feature_id,
        "feature_name": feature_name,
        "conditioning_length_beats": conditioning_length,
        "conditioning_length_tokens": conditioning_tokens.shape[1],
        "total_sequence_length": seq_len,
        "intervention_layer": intervention_layer,
        "results": results,
    }

    summary_file = output_dir / f"{song_path.stem}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Centralize all WAVs in a single folder for easy upload
    wav_central_dir = output_dir / "all_wavs"
    wav_central_dir.mkdir(exist_ok=True)

    print("\n📦 Centralizing WAV files...")
    import shutil

    for condition_name, result_data in results.items():
        if "wav_path" in result_data:
            wav_src = Path(result_data["wav_path"])
            if wav_src.exists():
                wav_dst = wav_central_dir / f"{song_path.stem}_{condition_name}.wav"
                shutil.copy2(wav_src, wav_dst)
                print(f"   ✅ Copied: {wav_dst.name}")

    print(f"✅ All WAVs centralized in: {wav_central_dir}")

    print("\n" + "=" * 80)
    print("✅ SONG-CONDITIONED INTERVENTIONS COMPLETE!")
    print("=" * 80)
    print(f"Generated {len(results)} conditions")
    print(f"Summary: {summary_file}")
    print(f"Central WAV folder: {wav_central_dir}")
    print()

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Song-conditioned feature interventions using real song prefixes"
    )
    parser.add_argument(
        "--song-path",
        type=Path,
        required=True,
        help="Path to song .pt file (e.g., data/sod/processed/notes/song.pt)",
    )
    parser.add_argument(
        "--feature-limuf-path",
        type=Path,
        required=True,
        help="Path to feature LiMuF .pt file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: exp/sod/ape/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--conditioning-length",
        type=int,
        default=3,
        help="Minimum number of tokens for conditioning prefix (default: 3). Will automatically extend to include start-of-notes token if needed.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Total sequence length to generate (default: 512)",
    )
    parser.add_argument(
        "--addition-strengths",
        type=str,
        default="-2.0,-1.0,1.0,2.0",
        help="Comma-separated intervention strengths (default: -2.0,-1.0,1.0,2.0). Baseline (0.0) and ablation are always generated automatically.",
    )
    parser.add_argument(
        "--intervention-layer",
        type=int,
        default=3,
        help="Layer to apply intervention (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.2,
        help="Noise scale for sampling (default: 1.2)",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    # Parse addition strengths
    addition_strengths = [float(s.strip()) for s in args.addition_strengths.split(",")]

    # Run interventions
    run_song_conditioned_interventions(
        song_path=args.song_path,
        feature_limuf_path=args.feature_limuf_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        conditioning_length=args.conditioning_length,
        seq_len=args.seq_len,
        addition_strengths=addition_strengths,
        intervention_layer=args.intervention_layer,
        temperature=args.temperature,
        noise_scale=args.noise_scale,
        generation_seed=args.generation_seed,
        device=args.device,
    )

    print("🎉 Success! Check the output directory for WAV and MIDI files.")


if __name__ == "__main__":
    main()
