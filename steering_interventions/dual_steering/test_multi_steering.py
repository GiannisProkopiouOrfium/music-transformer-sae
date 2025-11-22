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
import muspy
import representation
import utils
from multi_steered_generator import MultiSteeringGenerator
from vector_composition import VectorComposer

# Import music21
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)

# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        List of pitch values
    """
    try:
        music = representation.decode(tokens, encoding)

        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)

        return pitches
    except Exception as e:
        logging.error(f"Error extracting pitches: {e}")
        return []


def detect_key_from_tokens(tokens: np.ndarray, encoding: dict) -> tuple:
    """Detect full key (tonic + mode) from tokens using music21.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        (tonic, mode, confidence): e.g., ("A", "minor", 0.85)
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        for token in tokens:
            if token[0] == note_type:
                pitch_value = token[3]
                pitches.append(pitch_value)

        if len(pitches) < 10:
            return ("unknown", "unknown", 0.0)

        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")

        # Extract tonic name (e.g., "A", "C#", "Bb")
        tonic = key.tonic.name
        mode = key.mode
        confidence = key.correlationCoefficient

        return (tonic, mode, confidence)

    except Exception as e:
        logging.warning(f"Key detection failed: {e}")
        return ("unknown", "unknown", 0.0)


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> Dict:
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
        }


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline.

    Args:
        metrics: Current quality metrics
        baseline: Baseline quality metrics (GROUND_TRUTH_METRICS)

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
    all_pitch_means = []
    all_modes = []  # "major" or "minor"
    all_mode_confidences = []
    all_quality_metrics = []
    all_degradations = []
    valid_count = 0
    tokens_list = []

    for i in range(n_samples):
        # Create empty primer
        primer = torch.zeros((1, 1, 6), dtype=torch.long, device=device)

        # Generate
        try:
            output = generator.generate(
                primer,
                seq_len,
                eos_token=None,
                temperature=1.0,
                filter_logits_fn="top_k",
                filter_thres=0.9,
            )

            # Convert to numpy
            tokens = output[0].cpu().numpy()
            tokens_list.append(tokens)

            # Extract pitches
            pitches = extract_pitches_from_tokens(tokens, encoding)

            if len(pitches) >= 10:
                valid_count += 1

                # Pitch statistics
                pitch_mean = float(np.mean(pitches))
                all_pitch_means.append(pitch_mean)

                # Detect key/modality using music21
                tonic, mode, confidence = detect_key_from_tokens(tokens, encoding)
                all_modes.append(mode)
                all_mode_confidences.append(confidence)

                # Evaluate quality metrics
                quality = evaluate_quality_metrics(tokens, encoding)
                all_quality_metrics.append(quality)

                # Calculate degradation from ground truth
                if not np.isnan(quality["pitch_class_entropy"]):
                    degradation = calculate_degradation(quality, GROUND_TRUTH_METRICS)
                    all_degradations.append(degradation)

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
            "modality_control": {
                "major_count": 0,
                "minor_count": 0,
                "major_percentage": 0.0,
                "avg_confidence": 0.0,
            },
            "quality_metrics": {
                "pitch_class_entropy": {"mean": np.nan, "std": np.nan},
                "scale_consistency": {"mean": np.nan, "std": np.nan},
                "groove_consistency": {"mean": np.nan, "std": np.nan},
            },
            "degradation": {
                "entropy_diff": {"mean": np.nan, "std": np.nan},
                "scale_diff": {"mean": np.nan, "std": np.nan},
                "groove_diff": {"mean": np.nan, "std": np.nan},
                "total_degradation": {"mean": np.nan, "std": np.nan},
            },
            "success_rate": 0.0,
        }

    # Calculate modality statistics
    confident_modes = [m for m, c in zip(all_modes, all_mode_confidences) if c >= 0.5]
    major_count = sum(1 for m in confident_modes if m == "major")
    minor_count = sum(1 for m in confident_modes if m == "minor")
    total_confident = major_count + minor_count

    major_percentage = (
        100 * major_count / total_confident if total_confident > 0 else 0.0
    )
    avg_confidence = (
        float(np.mean([c for c in all_mode_confidences if c >= 0.5]))
        if confident_modes
        else 0.0
    )

    # Aggregate quality metrics
    quality_summary = {}
    for metric in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        values = [
            q[metric]
            for q in all_quality_metrics
            if not np.isnan(q.get(metric, np.nan))
        ]
        quality_summary[metric] = {
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
        }

    # Aggregate degradation
    degradation_summary = {}
    for metric in ["entropy_diff", "scale_diff", "groove_diff", "total_degradation"]:
        values = [d[metric] for d in all_degradations if not np.isnan(d[metric])]
        degradation_summary[metric] = {
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
        }

    return {
        "strategy": strategy,
        "alpha_pitch": alpha_pitch,
        "alpha_modality": alpha_modality,
        "valid_samples": valid_count,
        "pitch_control": {
            "mean": float(np.mean(all_pitch_means)),
            "std": float(np.std(all_pitch_means)),
        },
        "modality_control": {
            "major_count": major_count,
            "minor_count": minor_count,
            "major_percentage": major_percentage,
            "avg_confidence": avg_confidence,
        },
        "quality_metrics": quality_summary,
        "degradation": degradation_summary,
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
                f"  Major percentage: {result['modality_control']['major_percentage']:.1f}% (conf: {result['modality_control']['avg_confidence']:.2f})"
            )
            logging.info(
                f"  Quality degradation: {result['degradation']['total_degradation']['mean']:.2f}"
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
