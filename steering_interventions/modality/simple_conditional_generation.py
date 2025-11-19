"""Simple Conditional Generation with Steering.

A streamlined script for quick one-shot generations with specific conditioning,
alpha values, and steering vectors.

Usage:
    python simple_conditional_generation.py \\
        --song_path data/sod/processed/notes/Kunstderfuge/Kunstderfuge-0.npy \\
        --alpha 2.0 \\
        --conditioning_beats 8 \\
        --continuation_len 256 \\
        --gpu 0
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import muspy
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors

# Import music21
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("WARNING: music21 not available. Modality detection will be skipped.")
    print("Install with: pip install music21")


# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


def load_song_tokens(filepath: pathlib.Path, encoding: Dict) -> np.ndarray:
    """Load song tokens from .npy file.

    Args:
        filepath: Path to .npy file
        encoding: Encoding dictionary

    Returns:
        Token array (seq_len, 6)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load notes in 5D format: [beat, position, pitch, duration, program]
    notes = np.load(filepath)

    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")

    # Convert to 6D codes
    codes = representation.encode_notes(notes, encoding)
    return codes


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 8) -> torch.Tensor:
    """Extract first N beats as conditioning.

    Args:
        tokens: Full song tokens (seq_len, 6)
        n_beats: Number of beats to extract

    Returns:
        Conditioning tokens (1, cond_len, 6)
    """
    max_beat = n_beats
    cond_len = 0

    for i, token in enumerate(tokens):
        beat = token[1]
        if beat >= max_beat:
            cond_len = i
            break

    if cond_len == 0:
        cond_len = len(tokens)

    cond_tokens = tokens[:cond_len]
    cond_tensor = torch.from_numpy(cond_tokens).long().unsqueeze(0)

    return cond_tensor


def detect_modality_from_tokens(
    tokens: np.ndarray, encoding: Dict
) -> Tuple[str, float]:
    """Detect modality from tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        (mode, confidence): ("major"/"minor"/"unknown", correlation_coefficient)
    """
    if not MUSIC21_AVAILABLE:
        return ("unknown", 0.0)

    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        for token in tokens:
            if token[0] == note_type:
                pitch = token[3]
                pitches.append(pitch)

        if len(pitches) < 10:
            return ("unknown", 0.0)

        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")
        return (key.mode, key.correlationCoefficient)

    except Exception as e:
        logging.warning(f"Modality detection failed: {e}")
        return ("unknown", 0.0)


def evaluate_quality_metrics(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Evaluate objective quality metrics.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with quality metrics
    """
    try:
        music = representation.decode(tokens, encoding)
        music.trim(music.resolution * 64)

        if not music.tracks:
            return {
                "pitch_class_entropy": np.nan,
                "scale_consistency": np.nan,
                "groove_consistency": np.nan,
            }

        return {
            "pitch_class_entropy": muspy.pitch_class_entropy(music),
            "scale_consistency": muspy.scale_consistency(music) * 100,
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution)
            * 100,
        }
    except Exception as e:
        logging.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
            "error": str(e),
        }


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline.

    Args:
        metrics: Current quality metrics
        baseline: Baseline quality metrics

    Returns:
        Degradation scores
    """
    entropy_diff = abs(metrics["pitch_class_entropy"] - baseline["pitch_class_entropy"])
    scale_diff = max(0, baseline["scale_consistency"] - metrics["scale_consistency"])
    groove_diff = max(0, baseline["groove_consistency"] - metrics["groove_consistency"])

    total_degradation = entropy_diff + scale_diff + groove_diff

    return {
        "entropy_diff": float(entropy_diff),
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": float(total_degradation),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simple conditional generation with steering"
    )
    parser.add_argument(
        "--song_path",
        type=pathlib.Path,
        required=True,
        help="Path to .npy file (e.g., data/sod/processed/notes/Kunstderfuge/Kunstderfuge-0.npy)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="modality",
        help="Concept name (default: modality)",
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None, help="Model checkpoint path"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="Steering strength (default: 0.0)"
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: all layers)",
    )
    parser.add_argument(
        "--conditioning_beats", type=int, default=8, help="Beats for conditioning"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=256, help="Tokens to generate"
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: steering_interventions/modality/outputs/simple_generation)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = pathlib.Path(
            "steering_interventions/modality/outputs/simple_generation"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA/MPS not available, using CPU")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load encoding
    logging.info("Loading encoding...")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    # Load song tokens
    logging.info(f"Loading song: {args.song_path}")
    tokens = load_song_tokens(args.song_path, encoding)
    logging.info(f"Loaded {len(tokens)} tokens")

    # Extract conditioning prefix
    logging.info(f"Extracting conditioning prefix ({args.conditioning_beats} beats)...")
    conditioning = extract_conditioning_prefix(tokens, args.conditioning_beats)
    logging.info(f"Conditioning: {conditioning.shape[1]} tokens")

    # Detect modality in conditioning
    cond_mode, cond_confidence = detect_modality_from_tokens(
        conditioning.numpy()[0], encoding
    )
    logging.info(
        f"Conditioning modality: {cond_mode} (confidence: {cond_confidence:.3f})"
    )

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = pathlib.Path(
            f"steering_interventions/{args.concept}/outputs/steering_vectors/{args.concept}_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    logging.info(f"Loading steering vectors from: {steering_path}")
    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Load model
    logging.info("Loading model...")
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")

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

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully")

    # Parse target layers
    target_layers = None
    if args.target_layers is not None:
        target_layers = [int(layer.strip()) for layer in args.target_layers.split(",")]
        logging.info(f"Applying steering to layers: {target_layers}")
    else:
        logging.info("Applying steering to all layers")

    # Generate with steering
    logging.info(
        f"\nGenerating continuation with α = {args.alpha} (steering strength)..."
    )

    generator = SteeredGenerator(model, steering_vectors, encoding)
    eos = encoding["type_code_map"]["end-of-song"]

    conditioning = conditioning.to(device)

    generated = generator.generate(
        conditioning,
        args.continuation_len,
        alpha=args.alpha,
        target_layers=target_layers,
        eos_token=eos,
        temperature=config.GENERATION_TEMPERATURE,
        filter_logits_fn=config.GENERATION_FILTER,
        filter_thres=config.GENERATION_FILTER_THRESHOLD,
        monotonicity_dim=("type", "beat"),
    )

    logging.info("Generation complete")

    # Extract generated portion only
    generated_only = generated.cpu().numpy()[0]

    # Detect modality in generated portion
    gen_mode, gen_confidence = detect_modality_from_tokens(generated_only, encoding)
    logging.info(f"Generated modality: {gen_mode} (confidence: {gen_confidence:.3f})")

    # Evaluate quality metrics
    logging.info("Evaluating quality metrics...")
    quality_metrics = evaluate_quality_metrics(generated_only, encoding)
    degradation = calculate_degradation(quality_metrics, GROUND_TRUTH_METRICS)

    # Print results
    print("\n" + "=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)
    print(f"\nSong: {args.song_path.stem}")
    print(
        f"Conditioning: {args.conditioning_beats} beats ({conditioning.shape[1]} tokens)"
    )
    print(f"  Modality: {cond_mode} (confidence: {cond_confidence:.3f})")
    print(f"\nSteering: α = {args.alpha}")
    if target_layers:
        print(f"  Layers: {target_layers}")
    else:
        print(f"  Layers: all")
    print(f"\nGeneration: {args.continuation_len} tokens requested")
    print(f"  Generated: {len(generated_only)} tokens")
    print(f"  Modality: {gen_mode} (confidence: {gen_confidence:.3f})")
    print(f"  Mode shift: {'YES' if gen_mode != cond_mode else 'NO'}")

    print("\nQuality Metrics:")
    print(
        f"  Pitch class entropy: {quality_metrics.get('pitch_class_entropy', np.nan):.3f}"
    )
    print(
        f"  Scale consistency:   {quality_metrics.get('scale_consistency', np.nan):.1f}%"
    )
    print(
        f"  Groove consistency:  {quality_metrics.get('groove_consistency', np.nan):.1f}%"
    )

    print("\nDegradation from baseline:")
    print(f"  Entropy diff:  {degradation['entropy_diff']:.3f}")
    print(f"  Scale diff:    {degradation['scale_diff']:.1f}%")
    print(f"  Groove diff:   {degradation['groove_diff']:.1f}%")
    print(f"  Total:         {degradation['total_degradation']:.2f}")

    # Save outputs
    song_name = args.song_path.stem
    alpha_str = f"alpha_{args.alpha:+.1f}".replace(".", "p")
    output_name = f"{song_name}_{alpha_str}"

    # Combine conditioning + generated for full sequence
    full_seq = torch.cat((conditioning.cpu(), generated.cpu()), 1).numpy()[0]

    # Save tokens
    npy_path = args.output_dir / f"{output_name}.npy"
    np.save(npy_path, full_seq)
    logging.info(f"\nSaved tokens: {npy_path}")

    # Save MIDI
    try:
        music = representation.decode(full_seq, encoding)
        midi_path = args.output_dir / f"{output_name}.mid"
        music.write(str(midi_path))
        logging.info(f"Saved MIDI: {midi_path}")

        # Save audio
        wav_path = args.output_dir / f"{output_name}.wav"
        music.write_audio(str(wav_path))
        logging.info(f"Saved audio: {wav_path}")
    except Exception as e:
        logging.error(f"Error saving MIDI/audio: {e}")

    # Save metadata
    import json

    metadata = {
        "song_path": str(args.song_path),
        "song_name": song_name,
        "conditioning": {
            "beats": args.conditioning_beats,
            "tokens": int(conditioning.shape[1]),
            "modality": cond_mode,
            "confidence": float(cond_confidence),
        },
        "steering": {
            "alpha": args.alpha,
            "concept": args.concept,
            "target_layers": target_layers if target_layers else "all",
        },
        "generation": {
            "continuation_len_requested": args.continuation_len,
            "tokens_generated": len(generated_only),
            "modality": gen_mode,
            "confidence": float(gen_confidence),
            "mode_shift": gen_mode != cond_mode,
        },
        "quality_metrics": quality_metrics,
        "degradation": degradation,
    }

    metadata_path = args.output_dir / f"{output_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
