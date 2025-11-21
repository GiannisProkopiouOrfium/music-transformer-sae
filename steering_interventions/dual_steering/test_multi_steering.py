#!/usr/bin/env python3
"""Phase 2: Small-Scale Dual-Steering Strategy Validation.

Tests both composition strategies (direct and gram_schmidt) with a focused
parameter grid to identify which performs best before running full grid search.

Test matrix:
- Strategies: direct, gram_schmidt (2 strategies)
- Alpha pitch: [0, 1.0, 1.5, 2.0] (4 values)
- Alpha modality: [0, 2.0, 2.5, 3.0] (4 values)
- Total configs: 2 * 4 * 4 = 32 configs
- Samples per config: 5
- Total generations: 32 * 5 = 160 samples (~30-60 min on single GPU)

Output:
- Generated MIDI files
- JSON with metrics per config
- Summary comparison between strategies

Usage:
    python dual_steering/test_multi_steering.py \\
        --model_checkpoint path/to/model.ckpt \\
        --pitch_vectors steering_interventions/outputs/steering_vectors/average_pitch_steering_vectors.pt \\
        --modality_vectors steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt \\
        --output_dir steering_interventions/dual_steering/outputs/phase2_validation \\
        --n_samples 5
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import representation
import utils
from vector_composition import VectorComposer


def extract_music_features(tokens: np.ndarray, encoding: dict) -> Dict:
    """Extract pitch and modality features from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with extracted features
    """
    try:
        music = representation.decode(tokens, encoding)

        pitches = []
        note_events = 0

        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
                note_events += 1

        if len(pitches) == 0:
            return {
                "valid": False,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_range": 0,
                "num_notes": 0,
            }

        # Pitch statistics
        pitch_mean = float(np.mean(pitches))
        pitch_std = float(np.std(pitches))
        pitch_range = int(max(pitches) - min(pitches))

        return {
            "valid": True,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "num_notes": note_events,
        }

    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return {
            "valid": False,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0,
            "num_notes": 0,
        }


def detect_modality_from_pitches(pitches: List[int]) -> Dict:
    """Simple heuristic modality detection from pitch content.

    Args:
        pitches: List of MIDI pitch values

    Returns:
        Dictionary with modality estimates
    """
    if len(pitches) < 10:
        return {"confidence": 0.0, "major_likelihood": 0.5}

    # Count pitch class occurrences
    pitch_classes = [p % 12 for p in pitches]
    pc_counts = np.bincount(pitch_classes, minlength=12)
    pc_probs = pc_counts / pc_counts.sum()

    # Major scale profile (C major as reference)
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    major_profile = major_profile / major_profile.sum()

    # Minor scale profile (A minor as reference)
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    minor_profile = minor_profile / minor_profile.sum()

    # Compute correlations
    major_corr = float(np.corrcoef(pc_probs, major_profile)[0, 1])
    minor_corr = float(np.corrcoef(pc_probs, minor_profile)[0, 1])

    # Estimate likelihood (normalize to 0-1)
    total = max(abs(major_corr) + abs(minor_corr), 1e-8)
    major_likelihood = abs(major_corr) / total
    confidence = max(abs(major_corr), abs(minor_corr))

    return {"confidence": confidence, "major_likelihood": major_likelihood}


def generate_and_evaluate(
    model,
    composer: VectorComposer,
    strategy: str,
    alpha_pitch: float,
    alpha_modality: float,
    encoding: dict,
    device: torch.device,
    n_samples: int = 5,
    seq_len: int = 512,
) -> Dict:
    """Generate samples and evaluate for one configuration.

    Args:
        model: Transformer model
        composer: VectorComposer instance
        strategy: Composition strategy ("direct" or "gram_schmidt")
        alpha_pitch: Pitch scaling factor
        alpha_modality: Modality scaling factor
        encoding: Encoding dictionary
        device: Torch device
        n_samples: Number of samples to generate
        seq_len: Target sequence length

    Returns:
        Dictionary with aggregated metrics
    """
    # Create generator directly with composer (avoid reloading vectors)
    from multi_steered_generator import MultiSteeringGenerator

    generator = MultiSteeringGenerator(
        model=model,
        composer=composer,
        alpha_pitch=alpha_pitch,
        alpha_modality=alpha_modality,
        strategy=strategy,
        intervention_position="last",
    )

    # Generate samples
    all_pitches = []
    all_modalities = []
    valid_count = 0
    tokens_list = []

    for i in range(n_samples):
        # Create empty primer
        primer = torch.zeros((1, 1, 6), dtype=torch.long, device=device)

        # Generate
        try:
            output = generator.generate(
                primer=primer,
                target_seq_length=seq_len,
                beam=0,
                beam_chance=1.0,
            )

            # Convert to numpy
            tokens = output[0].cpu().numpy()
            tokens_list.append(tokens)

            # Extract features
            features = extract_music_features(tokens, encoding)

            if features["valid"]:
                valid_count += 1
                all_pitches.append(features["pitch_mean"])

                # Detect modality
                music = representation.decode(tokens, encoding)
                pitches = []
                for track in music.tracks:
                    for note in track.notes:
                        pitches.append(note.pitch)

                if len(pitches) >= 10:
                    modality = detect_modality_from_pitches(pitches)
                    all_modalities.append(modality["major_likelihood"])

        except Exception as e:
            logging.error(f"Generation failed for sample {i}: {e}")
            continue

    # Aggregate metrics
    if valid_count == 0:
        return {
            "strategy": strategy,
            "alpha_pitch": alpha_pitch,
            "alpha_modality": alpha_modality,
            "valid_samples": 0,
            "pitch_control": {"mean": 0.0, "std": 0.0},
            "modality_control": {"mean_major_likelihood": 0.5, "std": 0.0},
            "success_rate": 0.0,
        }

    return {
        "strategy": strategy,
        "alpha_pitch": alpha_pitch,
        "alpha_modality": alpha_modality,
        "valid_samples": valid_count,
        "pitch_control": {
            "mean": float(np.mean(all_pitches)),
            "std": float(np.std(all_pitches)),
        },
        "modality_control": {
            "mean_major_likelihood": (
                float(np.mean(all_modalities)) if all_modalities else 0.5
            ),
            "std": float(np.std(all_modalities)) if all_modalities else 0.0,
        },
        "success_rate": valid_count / n_samples,
        "tokens": [t.tolist() for t in tokens_list],  # For later analysis
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Test dual-steering strategies"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to model checkpoint (defaults to best_model.pt)",
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "outputs/steering_vectors/average_pitch_steering_vectors.pt"
        ),
        help="Path to pitch steering vectors",
    )
    parser.add_argument(
        "--modality_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        ),
        help="Path to modality steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase2_validation"
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Samples per configuration"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--alphas_pitch",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 1.5, 2.0],
        help="Pitch alpha values to test",
    )
    parser.add_argument(
        "--alphas_modality",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 2.5, 3.0],
        help="Modality alpha values to test",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["direct", "gram_schmidt"],
        help="Strategies to test",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (0, 1, etc.)",
    )

    args = parser.parse_args()

    # Create output directory first (needed for log file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "phase2_validation.log"),
            logging.StreamHandler(),
        ],
    )

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

    # Load model
    logging.info(f"Loading model from {args.model_checkpoint}")
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

    if args.model_checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.model_checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Create composer
    logging.info("Creating vector composer")
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(
        str(args.pitch_vectors), str(args.modality_vectors)
    )

    # Generate test matrix
    test_configs = []
    for strategy in args.strategies:
        for alpha_pitch in args.alphas_pitch:
            for alpha_modality in args.alphas_modality:
                test_configs.append((strategy, alpha_pitch, alpha_modality))

    logging.info(f"Testing {len(test_configs)} configurations")
    logging.info(
        f"Total generations: {len(test_configs)} configs × {args.n_samples} samples = {len(test_configs) * args.n_samples}"
    )

    # Run experiments
    results = []
    start_time = time.time()

    for strategy, alpha_pitch, alpha_modality in tqdm(
        test_configs, desc="Testing configurations"
    ):
        logging.info(
            f"\nTesting: strategy={strategy}, α_pitch={alpha_pitch}, α_modality={alpha_modality}"
        )

        result = generate_and_evaluate(
            model=model,
            composer=composer,
            strategy=strategy,
            alpha_pitch=alpha_pitch,
            alpha_modality=alpha_modality,
            encoding=encoding,
            device=device,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
        )

        results.append(result)

        # Log immediate results
        logging.info(
            f"  Valid samples: {result['valid_samples']}/{args.n_samples} "
            f"({result['success_rate']*100:.1f}%)"
        )
        if result["valid_samples"] > 0:
            logging.info(
                f"  Pitch mean: {result['pitch_control']['mean']:.2f} ± {result['pitch_control']['std']:.2f}"
            )
            logging.info(
                f"  Major likelihood: {result['modality_control']['mean_major_likelihood']:.3f}"
            )

    elapsed_time = time.time() - start_time

    # Save results
    results_file = args.output_dir / "phase2_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "n_samples": args.n_samples,
                    "seq_len": args.seq_len,
                    "alphas_pitch": args.alphas_pitch,
                    "alphas_modality": args.alphas_modality,
                    "strategies": args.strategies,
                    "total_configs": len(test_configs),
                    "elapsed_time_seconds": elapsed_time,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    logging.info(f"\nSaved results to: {results_file}")
    logging.info(
        f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/len(test_configs):.1f}s per config)"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 2 VALIDATION COMPLETE")
    print("=" * 80)
    print(
        f"\nTested {len(test_configs)} configurations in {elapsed_time/60:.1f} minutes"
    )
    print(f"\nResults saved to: {results_file}")
    print("\nNext step: Run analyze_strategy_results.py to compare strategies")
    print("=" * 80)


if __name__ == "__main__":
    main()
