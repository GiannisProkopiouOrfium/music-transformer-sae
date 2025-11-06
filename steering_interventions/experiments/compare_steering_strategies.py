"""Compare All-to-All vs One-to-All vs Some-to-Some Steering Strategies.

This script compares three steering approaches:
1. All-to-All: Each layer receives its own steering vector (always tested)
2. One-to-All: A single "best" layer's vector is applied to all layers
3. Some-to-Some: Only specific layers receive their corresponding steering vectors

Usage:
    # Test only Some-to-Some (no One-to-All)
    python experiments/compare_steering_strategies.py \\
        --concept average_pitch \\
        --layer_groups "0,1,2,3|4,5,6,7|8,9,10,11" \\
        --n_samples 30 \\
        --gpu 0
    
    # Test only One-to-All (no Some-to-Some)
    python experiments/compare_steering_strategies.py \\
        --concept average_pitch \\
        --best_layers 10,11 \\
        --n_samples 30 \\
        --gpu 0
    
    # Test both One-to-All and Some-to-Some
    python experiments/compare_steering_strategies.py \\
        --concept average_pitch \\
        --best_layers 10,11 \\
        --layer_groups "10,11|8,9,10,11" \\
        --alphas "-1.0,-0.5,0.0,0.5,1.0" \\
        --n_samples 50 \\
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

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


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "compare_strategies.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare steering strategies")
    parser.add_argument(
        "--concept", type=str, required=True, help="Concept name (e.g., average_pitch)"
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
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Alpha values to test (comma-separated)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50, help="Number of samples per configuration"
    )
    parser.add_argument(
        "--output_len", type=int, default=512, help="Length of generated sequences"
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default=pathlib.Path("../exp/sod/model_best.pt"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--steering_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_vectors"),
        help="Directory containing steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("experiments/results"),
        help="Output directory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    return parser.parse_args()


def measure_pitch_from_tokens(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Measure pitch statistics from tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)

        if not pitches:
            return {"mean": 0.0, "std": 0.0, "n_notes": 0, "unique_pitches": 0}

        return {
            "mean": float(np.mean(pitches)),
            "std": float(np.std(pitches)),
            "min": int(np.min(pitches)),
            "max": int(np.max(pitches)),
            "n_notes": len(pitches),
            "unique_pitches": len(set(pitches)),
        }
    except Exception as e:
        logging.error(f"Error measuring pitch: {e}")
        return {
            "mean": 0.0,
            "std": 0.0,
            "n_notes": 0,
            "unique_pitches": 0,
            "error": str(e),
        }


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
    """Generate samples with specified strategy.

    Args:
        model: The model
        steering_vectors: Dict of steering vectors per layer
        encoding: Encoding dictionary
        strategy: "all_to_all", "one_to_all", or "some_to_some"
        best_layer: Layer index for one_to_all (ignored for other strategies)
        active_layers: List of layers for some_to_some (ignored for other strategies)
        alpha: Steering strength
        n_samples: Number of samples to generate
        output_len: Length of generated sequences
        device: Device to use
    """
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
    """Evaluate a list of samples."""
    pitch_stats = [measure_pitch_from_tokens(s, encoding) for s in samples]
    quality_metrics = [evaluate_quality_metrics(s, encoding) for s in samples]

    return {
        "pitch": {
            "mean": float(np.nanmean([p["mean"] for p in pitch_stats])),
            "std": float(np.nanstd([p["mean"] for p in pitch_stats])),
            "diversity": float(np.nanmean([p["unique_pitches"] for p in pitch_stats])),
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
    logging.info("STEERING STRATEGY COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Concept: {args.concept}")
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
    device = torch.device(f"cuda:{args.gpu}")
    logging.info(f"Using device: {device}")

    # Load encoding
    logging.info(f"Loading encoding from {config.NOTES_DIR / 'encoding.json'}")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")

    # Load model
    logging.info(f"Loading model from {args.model_path}")
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
    logging.info(f"Loading steering vectors from {args.steering_dir}")
    steering_path = (
        config.OUTPUT_DIR / "steering_vectors" / f"{args.concept}_steering_vectors.pt"
    )
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

        logging.info(f"  Mean pitch: {eval_results['pitch']['mean']:.2f}")
        logging.info(
            f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
        )
        logging.info(
            f"  Groove consistency: {eval_results['quality']['groove_consistency']:.3f}"
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

                logging.info(f"  Mean pitch: {eval_results['pitch']['mean']:.2f}")
                logging.info(
                    f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
                )
                logging.info(
                    f"  Groove consistency: {eval_results['quality']['groove_consistency']:.3f}"
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

                logging.info(f"  Mean pitch: {eval_results['pitch']['mean']:.2f}")
                logging.info(
                    f"  Pitch class entropy: {eval_results['quality']['pitch_class_entropy']:.3f}"
                )
                logging.info(
                    f"  Groove consistency: {eval_results['quality']['groove_consistency']:.3f}"
                )
                logging.info(
                    f"  Scale consistency: {eval_results['quality']['scale_consistency']:.3f}"
                )
    else:
        logging.info("\n" + "=" * 80)
        logging.info("SKIPPING SOME-TO-SOME STRATEGY (no --layer_groups specified)")
        logging.info("=" * 80)

    # Save results
    output_file = args.output_dir / f"strategy_comparison_{args.concept}.json"
    with open(output_file, "w") as f:
        # Convert layer group lists to strings for JSON serialization
        results_for_json = {
            "all_to_all": results["all_to_all"],
            "one_to_all": results["one_to_all"],
            "some_to_some": results["some_to_some"],
        }
        json.dump(
            {
                "concept": args.concept,
                "best_layers": best_layers,
                "layer_groups": layer_groups,
                "alphas": alphas,
                "n_samples": args.n_samples,
                "results": results_for_json,
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

    for alpha in alphas:
        if abs(alpha) < 0.001:  # Skip baseline (alpha ~ 0.0)
            continue

        logging.info(f"\nAlpha = {alpha:+.1f}:")

        # All-to-All
        ata_pitch = results["all_to_all"][alpha]["pitch"]["mean"]
        ata_entropy = results["all_to_all"][alpha]["quality"]["pitch_class_entropy"]
        ata_groove = results["all_to_all"][alpha]["quality"]["groove_consistency"]
        ata_scale = results["all_to_all"][alpha]["quality"]["scale_consistency"]
        logging.info(
            f"  All-to-All:        pitch={ata_pitch:6.2f}, entropy={ata_entropy:.3f}, groove={ata_groove:.3f}, scale={ata_scale:.3f}"
        )

        # One-to-All for each layer
        if best_layers:
            for layer in best_layers:
                ota_pitch = results["one_to_all"][layer][alpha]["pitch"]["mean"]
                ota_entropy = results["one_to_all"][layer][alpha]["quality"][
                    "pitch_class_entropy"
                ]
                ota_groove = results["one_to_all"][layer][alpha]["quality"][
                    "groove_consistency"
                ]
                ota_scale = results["one_to_all"][layer][alpha]["quality"][
                    "scale_consistency"
                ]
                logging.info(
                    f"  One-to-All (L{layer:2d}): pitch={ota_pitch:6.2f}, entropy={ota_entropy:.3f}, groove={ota_groove:.3f}, scale={ota_scale:.3f}"
                )

        # Some-to-Some for each layer group
        if layer_groups:
            for layer_group in layer_groups:
                group_str = str(layer_group)
                sts_pitch = results["some_to_some"][group_str][alpha]["pitch"]["mean"]
                sts_entropy = results["some_to_some"][group_str][alpha]["quality"][
                    "pitch_class_entropy"
                ]
                sts_groove = results["some_to_some"][group_str][alpha]["quality"][
                    "groove_consistency"
                ]
                sts_scale = results["some_to_some"][group_str][alpha]["quality"][
                    "scale_consistency"
                ]
                logging.info(
                    f"  Some-to-Some {group_str:12s}: pitch={sts_pitch:6.2f}, entropy={sts_entropy:.3f}, groove={sts_groove:.3f}, scale={sts_scale:.3f}"
                )

    logging.info("\nComparison complete!")


if __name__ == "__main__":
    main()
