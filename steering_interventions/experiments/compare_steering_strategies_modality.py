"""Compare All-to-All vs One-to-All vs Some-to-Some Steering Strategies for Modality.

This script compares three steering approaches for major/minor modality:
1. All-to-All: Each layer receives its own steering vector (always tested)
2. One-to-All: A single "best" layer's vector is applied to all layers
3. Some-to-Some: Only specific layers receive their corresponding steering vectors

Usage:
    # Test both One-to-All and Some-to-Some
    python experiments/compare_steering_strategies_modality.py \\
        --best_layers 10,11 \\
        --layer_groups "10,11|8,9,10,11|0,1,2,3" \\
        --alphas "-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0" \\
        --n_samples 30 \\
        --gpu 0
    
    # Test only Some-to-Some (no One-to-All)
    python experiments/compare_steering_strategies_modality.py \\
        --layer_groups "0,1,2,3|4,5,6,7|8,9,10,11" \\
        --n_samples 30 \\
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import muspy
import numpy as np
import torch
import tqdm

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import utils
import music_x_transformers
import representation
from steered_generator import SteeredGenerator, load_steering_vectors

# Import music21 for modality detection
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "compare_strategies_modality.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare steering strategies for modality"
    )
    parser.add_argument(
        "--best_layers",
        type=str,
        default="",
        help="Best layers to test for One-to-All (comma-separated). Leave empty to skip One-to-All.",
    )
    parser.add_argument(
        "--layer_groups",
        type=str,
        default="",
        help="Layer groups for Some-to-Some strategy (pipe-separated groups, e.g., '0,1,2|3,4|8,9,10,11'). Leave empty to skip Some-to-Some.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0",
        help="Alpha values to test (comma-separated)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=30, help="Number of samples per configuration"
    )
    parser.add_argument(
        "--output_len", type=int, default=512, help="Length of generated sequences"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/experiments/results"),
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    return parser.parse_args()


def detect_modality_from_tokens(
    tokens: np.ndarray, encoding: Dict
) -> Tuple[str, float]:
    """Detect modality from tokens."""
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
    """Evaluate objective quality metrics."""
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
            "scale_consistency": muspy.scale_consistency(music),
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution),
        }
    except Exception as e:
        logging.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
            "error": str(e),
        }


def generate_with_strategy(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    strategy: str,
    best_layer: int | None,
    active_layers: List[int] | None,
    alpha: float,
    n_samples: int,
    output_len: int,
    device: torch.device,
) -> List[np.ndarray]:
    """Generate samples with specified strategy."""
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    # Prepare steering vectors based on strategy
    if strategy == "all_to_all":
        steering_dict = steering_vectors
        target_layers = None  # Use all layers
    elif strategy == "one_to_all":
        # Use only the best layer's vector for ALL layers
        if best_layer is None:
            raise ValueError("best_layer must be specified for one_to_all strategy")
        best_vector = steering_vectors[best_layer]
        steering_dict = {layer: best_vector for layer in range(12)}
        target_layers = None  # Apply to all
    elif strategy == "some_to_some":
        # Use only specified layers with their own vectors
        if active_layers is None:
            raise ValueError(
                "active_layers must be specified for some_to_some strategy"
            )
        steering_dict = {layer: steering_vectors[layer] for layer in active_layers}
        target_layers = active_layers  # Only apply to these layers
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create generator
    generator = SteeredGenerator(model, steering_dict, encoding)

    samples = []
    for _ in range(n_samples):
        # Start token
        tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
        tgt_start[:, 0, 0] = sos

        # Generate
        generated = generator.generate(
            tgt_start,
            output_len,
            alpha=alpha,
            target_layers=target_layers,
            eos_token=eos,
            temperature=config.GENERATION_TEMPERATURE,
            filter_logits_fn=config.GENERATION_FILTER,
            filter_thres=config.GENERATION_FILTER_THRESHOLD,
            monotonicity_dim=("type", "beat"),
        )

        # Combine
        full_seq = torch.cat((tgt_start, generated), 1).cpu().numpy()[0]
        samples.append(full_seq)

    return samples


def evaluate_samples(samples: List[np.ndarray], encoding: Dict) -> Dict:
    """Evaluate a list of samples for modality and quality."""
    # Detect modality
    modality_results = []
    for sample in samples:
        mode, confidence = detect_modality_from_tokens(sample, encoding)
        modality_results.append({"mode": mode, "confidence": confidence})

    # Quality metrics
    quality_metrics = [evaluate_quality_metrics(s, encoding) for s in samples]

    # Calculate modality statistics
    confident_modes = [r for r in modality_results if r["confidence"] >= 0.5]

    if confident_modes:
        major_count = sum(1 for r in confident_modes if r["mode"] == "major")
        minor_count = sum(1 for r in confident_modes if r["mode"] == "minor")
        total = len(confident_modes)

        major_pct = 100 * major_count / total if total > 0 else 0
        minor_pct = 100 * minor_count / total if total > 0 else 0
        avg_confidence = np.mean([r["confidence"] for r in confident_modes])
    else:
        major_pct = 0
        minor_pct = 0
        avg_confidence = 0

    return {
        "modality": {
            "major_percentage": float(major_pct),
            "minor_percentage": float(minor_pct),
            "avg_confidence": float(avg_confidence),
            "n_confident": len(confident_modes),
        },
        "quality": {
            "pitch_class_entropy": float(
                np.nanmean([q["pitch_class_entropy"] for q in quality_metrics])
            ),
            "scale_consistency": float(
                np.nanmean([q["scale_consistency"] for q in quality_metrics])
            ),
            "groove_consistency": float(
                np.nanmean([q["groove_consistency"] for q in quality_metrics])
            ),
        },
        "n_samples": len(samples),
    }


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("STEERING STRATEGY COMPARISON - MODALITY")
    logging.info("=" * 80)
    if args.best_layers:
        logging.info(f"Best layers for One-to-All: {args.best_layers}")
    if args.layer_groups:
        logging.info(f"Layer groups for Some-to-Some: {args.layer_groups}")
    logging.info(f"Alphas: {args.alphas}")
    logging.info(f"Samples per config: {args.n_samples}")

    # Parse parameters
    best_layers = (
        [int(layer) for layer in args.best_layers.split(",")]
        if args.best_layers
        else []
    )
    alphas = [float(a) for a in args.alphas.split(",")]

    # Parse layer groups for Some-to-Some strategy
    layer_groups = []
    if args.layer_groups:
        for group_str in args.layer_groups.split("|"):
            group = [int(layer) for layer in group_str.split(",")]
            layer_groups.append(group)

    if layer_groups:
        logging.info(f"Parsed layer groups: {layer_groups}")

    # Validate that at least one strategy is selected
    if not best_layers and not layer_groups:
        raise ValueError(
            "Must specify at least one of --best_layers or --layer_groups. "
            "Use --best_layers for One-to-All, --layer_groups for Some-to-Some, or both."
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

    # Load encoding
    logging.info(f"Loading encoding from {config.NOTES_DIR / 'encoding.json'}")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")

    # Load model
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

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    logging.info(f"Loading steering vectors from {steering_path}")
    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Storage for results
    results = {
        "all_to_all": {},
        "one_to_all": {layer: {} for layer in best_layers} if best_layers else {},
        "some_to_some": (
            {str(group): {} for group in layer_groups} if layer_groups else {}
        ),
    }

    # Test All-to-All strategy
    logging.info("\n" + "=" * 80)
    logging.info("STRATEGY: ALL-TO-ALL")
    logging.info("=" * 80)

    for alpha in tqdm.tqdm(alphas, desc="All-to-All"):
        logging.info(f"\nAlpha: {alpha}")

        samples = generate_with_strategy(
            model,
            steering_vectors,
            encoding,
            "all_to_all",
            None,
            None,
            alpha,
            args.n_samples,
            args.output_len,
            device,
        )

        eval_results = evaluate_samples(samples, encoding)
        results["all_to_all"][alpha] = eval_results

        logging.info(
            f"  Major: {eval_results['modality']['major_percentage']:.1f}%, Minor: {eval_results['modality']['minor_percentage']:.1f}%"
        )
        logging.info(
            f"  Avg confidence: {eval_results['modality']['avg_confidence']:.3f}"
        )
        logging.info(
            f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
        )
        logging.info(
            f"  Scale consistency: {eval_results['quality']['scale_consistency']:.3f}"
        )

    # Test One-to-All strategy for each best layer
    if best_layers:
        for best_layer in best_layers:
            logging.info("\n" + "=" * 80)
            logging.info(f"STRATEGY: ONE-TO-ALL (Layer {best_layer})")
            logging.info("=" * 80)

            for alpha in tqdm.tqdm(alphas, desc=f"One-to-All (L{best_layer})"):
                logging.info(f"\nAlpha: {alpha}")

                samples = generate_with_strategy(
                    model,
                    steering_vectors,
                    encoding,
                    "one_to_all",
                    best_layer,
                    None,
                    alpha,
                    args.n_samples,
                    args.output_len,
                    device,
                )

                eval_results = evaluate_samples(samples, encoding)
                results["one_to_all"][best_layer][alpha] = eval_results

                logging.info(
                    f"  Major: {eval_results['modality']['major_percentage']:.1f}%, Minor: {eval_results['modality']['minor_percentage']:.1f}%"
                )
                logging.info(
                    f"  Avg confidence: {eval_results['modality']['avg_confidence']:.3f}"
                )
                logging.info(
                    f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
                )
                logging.info(
                    f"  Scale consistency: {eval_results['quality']['scale_consistency']:.3f}"
                )
    else:
        logging.info("\n" + "=" * 80)
        logging.info("SKIPPING ONE-TO-ALL STRATEGY (no --best_layers specified)")
        logging.info("=" * 80)

    # Test Some-to-Some strategy for each layer group
    if layer_groups:
        for layer_group in layer_groups:
            group_str = str(layer_group)
            logging.info("\n" + "=" * 80)
            logging.info(f"STRATEGY: SOME-TO-SOME (Layers {layer_group})")
            logging.info("=" * 80)

            for alpha in tqdm.tqdm(alphas, desc=f"Some-to-Some ({group_str})"):
                logging.info(f"\nAlpha: {alpha}")

                samples = generate_with_strategy(
                    model,
                    steering_vectors,
                    encoding,
                    "some_to_some",
                    None,
                    layer_group,
                    alpha,
                    args.n_samples,
                    args.output_len,
                    device,
                )

                eval_results = evaluate_samples(samples, encoding)
                results["some_to_some"][group_str][alpha] = eval_results

                logging.info(
                    f"  Major: {eval_results['modality']['major_percentage']:.1f}%, Minor: {eval_results['modality']['minor_percentage']:.1f}%"
                )
                logging.info(
                    f"  Avg confidence: {eval_results['modality']['avg_confidence']:.3f}"
                )
                logging.info(
                    f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
                )
                logging.info(
                    f"  Scale consistency: {eval_results['quality']['scale_consistency']:.3f}"
                )
    else:
        logging.info("\n" + "=" * 80)
        logging.info("SKIPPING SOME-TO-SOME STRATEGY (no --layer_groups specified)")
        logging.info("=" * 80)

    # Save results
    output_file = args.output_dir / "strategy_comparison_modality.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "concept": "modality",
                "best_layers": best_layers,
                "layer_groups": layer_groups,
                "alphas": alphas,
                "n_samples": args.n_samples,
                "results": results,
            },
            f,
            indent=2,
        )
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to {output_file}")

    # Print summary comparison
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY COMPARISON")
    logging.info("=" * 80)

    # Get baseline (alpha=0) for comparison
    baseline_major = results["all_to_all"][0.0]["modality"]["major_percentage"]
    logging.info(f"Baseline (All-to-All, α=0.0): {baseline_major:.1f}% major\n")

    for alpha in alphas:
        if abs(alpha) < 0.001:  # Skip baseline (alpha ~ 0.0)
            continue

        logging.info(f"\nAlpha = {alpha:+.1f}:")

        # All-to-All
        ata_major = results["all_to_all"][alpha]["modality"]["major_percentage"]
        ata_shift = ata_major - baseline_major
        ata_entropy = results["all_to_all"][alpha]["quality"]["pitch_class_entropy"]
        ata_scale = results["all_to_all"][alpha]["quality"]["scale_consistency"]
        logging.info(
            f"  All-to-All:        major={ata_major:5.1f}% (shift={ata_shift:+6.1f}%), entropy={ata_entropy:.3f}, scale={ata_scale:.3f}"
        )

        # One-to-All for each layer
        if best_layers:
            for layer in best_layers:
                ota_major = results["one_to_all"][layer][alpha]["modality"][
                    "major_percentage"
                ]
                ota_shift = ota_major - baseline_major
                ota_entropy = results["one_to_all"][layer][alpha]["quality"][
                    "pitch_class_entropy"
                ]
                ota_scale = results["one_to_all"][layer][alpha]["quality"][
                    "scale_consistency"
                ]
                logging.info(
                    f"  One-to-All (L{layer:2d}): major={ota_major:5.1f}% (shift={ota_shift:+6.1f}%), entropy={ota_entropy:.3f}, scale={ota_scale:.3f}"
                )

        # Some-to-Some for each layer group
        if layer_groups:
            for layer_group in layer_groups:
                group_str = str(layer_group)
                sts_major = results["some_to_some"][group_str][alpha]["modality"][
                    "major_percentage"
                ]
                sts_shift = sts_major - baseline_major
                sts_entropy = results["some_to_some"][group_str][alpha]["quality"][
                    "pitch_class_entropy"
                ]
                sts_scale = results["some_to_some"][group_str][alpha]["quality"][
                    "scale_consistency"
                ]
                logging.info(
                    f"  Some-to-Some {group_str:12s}: major={sts_major:5.1f}% (shift={sts_shift:+6.1f}%), entropy={sts_entropy:.3f}, scale={sts_scale:.3f}"
                )

    logging.info("\nComparison complete!")


if __name__ == "__main__":
    main()
