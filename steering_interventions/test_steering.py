"""Test script to verify steering interventions work correctly.

This script:
1. Loads the model and steering vectors
2. Generates 3 samples: baseline (alpha=0), high velocity (alpha=+2), low velocity (alpha=-2)
3. Extracts velocities from generated MIDI
4. Compares statistics to verify steering effect
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors


# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> dict:
    """Evaluate objective quality metrics.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with quality metrics
    """
    try:
        import muspy

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


def calculate_degradation(metrics: dict, baseline: dict) -> dict:
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


def extract_velocities_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract velocity values from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        List of velocity values
    """
    try:
        music = representation.decode(tokens, encoding)

        velocities = []
        for track in music.tracks:
            for note in track.notes:
                velocities.append(note.velocity)

        return velocities
    except Exception as e:
        logging.error(f"Error extracting velocities: {e}")
        return []


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


def extract_durations_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract duration values from generated tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        List of duration values (in ticks)
    """
    try:
        music = representation.decode(tokens, encoding)

        durations = []
        for track in music.tracks:
            for note in track.notes:
                durations.append(note.duration)

        return durations
    except Exception as e:
        logging.error(f"Error extracting durations: {e}")
        return []


def test_steering(
    model,
    steering_vectors,
    encoding,
    device,
    alphas=None,
    target_layers=None,
    seq_len=512,
    n_samples=3,
):
    """Test steering with different alpha values.

    Args:
        model: The model
        steering_vectors: Steering vectors dict
        encoding: Encoding dictionary
        device: Device to use
        alphas: List of alpha values to test
        target_layers: List of layer indices to apply steering (None = all)
        seq_len: Generation length
        n_samples: Number of samples per alpha

    Returns:
        Dictionary with results
    """
    if alphas is None:
        alphas = [-2.0, 0.0, 2.0]

    results = {}

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    generator = SteeredGenerator(model, steering_vectors, encoding)

    for alpha in alphas:
        logging.info(f"Testing alpha={alpha}")

        alpha_velocities = []
        alpha_pitches = []
        alpha_durations = []

        for i in range(n_samples):
            # Create start tokens
            start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start_tokens[:, 0, 0] = sos

            # Generate with steering
            generated = generator.generate(
                start_tokens,
                seq_len,
                alpha=alpha,
                target_layers=target_layers,
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Combine start and generated
            full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]

            # Extract velocities, pitches, and durations
            velocities = extract_velocities_from_tokens(full_seq, encoding)
            pitches = extract_pitches_from_tokens(full_seq, encoding)
            durations = extract_durations_from_tokens(full_seq, encoding)

            # Evaluate quality metrics
            quality_metrics = evaluate_quality_metrics(full_seq, encoding)

            if velocities:
                alpha_velocities.extend(velocities)
            if pitches:
                alpha_pitches.extend(pitches)
            if durations:
                alpha_durations.extend(durations)

            if pitches:
                logging.info(
                    f"  Sample {i}: {len(pitches)} notes, "
                    f"mean velocity={np.mean(velocities):.1f}, "
                    f"mean pitch={np.mean(pitches):.1f}, "
                    f"pitch_range=[{np.min(pitches)}, {np.max(pitches)}], "
                    f"mean duration={np.mean(durations):.2f} ticks"
                )
            else:
                logging.warning(
                    f"  Sample {i}: No notes extracted! Generation may have failed."
                )

        if alpha_pitches:
            # Calculate average quality metrics across samples
            avg_quality = {
                "pitch_class_entropy": np.mean(
                    [
                        quality_metrics.get("pitch_class_entropy", np.nan)
                        for _ in range(n_samples)
                    ]
                ),
                "scale_consistency": np.mean(
                    [
                        quality_metrics.get("scale_consistency", np.nan)
                        for _ in range(n_samples)
                    ]
                ),
                "groove_consistency": np.mean(
                    [
                        quality_metrics.get("groove_consistency", np.nan)
                        for _ in range(n_samples)
                    ]
                ),
            }
            degradation = calculate_degradation(avg_quality, GROUND_TRUTH_METRICS)

            results[alpha] = {
                "velocities": alpha_velocities,
                "pitches": alpha_pitches,
                "durations": alpha_durations,
                "velocity_mean": np.mean(alpha_velocities) if alpha_velocities else 0.0,
                "velocity_std": np.std(alpha_velocities) if alpha_velocities else 0.0,
                "pitch_mean": np.mean(alpha_pitches),
                "pitch_std": np.std(alpha_pitches),
                "pitch_min": np.min(alpha_pitches),
                "pitch_max": np.max(alpha_pitches),
                "duration_mean": np.mean(alpha_durations) if alpha_durations else 0.0,
                "duration_std": np.std(alpha_durations) if alpha_durations else 0.0,
                "duration_min": np.min(alpha_durations) if alpha_durations else 0,
                "duration_max": np.max(alpha_durations) if alpha_durations else 0,
                "n_notes": len(alpha_pitches),
                "quality_metrics": avg_quality,
                "degradation": degradation,
            }
        else:
            results[alpha] = {
                "velocities": [],
                "pitches": [],
                "durations": [],
                "velocity_mean": 0.0,
                "velocity_std": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0,
                "pitch_max": 0,
                "duration_mean": 0.0,
                "duration_std": 0.0,
                "duration_min": 0,
                "duration_max": 0,
                "n_notes": 0,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Test steering interventions")
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to use"
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
        "--alphas",
        type=str,
        default="-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to apply steering (default: all layers). Example: '3,4,5' for middle layers only",
    )
    parser.add_argument(
        "--n_samples", type=int, default=3, help="Number of samples per alpha"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Generation length")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (0, 1, etc.)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = (
            config.OUTPUT_DIR
            / "steering_vectors"
            / f"{args.concept}_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Load model
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

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    # Parse target layers if specified
    target_layers = None
    if args.target_layers is not None:
        target_layers = [int(l.strip()) for l in args.target_layers.split(",")]
        logging.info(f"Will apply steering to layers: {target_layers}")
    else:
        logging.info("Will apply steering to all layers")

    # Run test
    logging.info("=" * 60)
    logging.info("Testing steering interventions")
    logging.info("=" * 60)

    results = test_steering(
        model,
        steering_vectors,
        encoding,
        device,
        alphas=alphas,
        target_layers=target_layers,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
    )

    # Print results
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS SUMMARY")
    logging.info("=" * 60)

    for alpha in sorted(results.keys()):
        stats = results[alpha]
        deg = stats.get("degradation", {})
        logging.info(
            f"Alpha {alpha:+5.1f}: "
            f"velocity_mean={stats['velocity_mean']:6.2f}, "
            f"pitch_mean={stats['pitch_mean']:6.2f}, "
            f"pitch_std={stats['pitch_std']:5.2f}, "
            f"pitch_range=[{stats['pitch_min']:3d}, {stats['pitch_max']:3d}], "
            f"duration_mean={stats['duration_mean']:6.2f} ticks, "
            f"duration_std={stats['duration_std']:5.2f}, "
            f"n={stats['n_notes']:4d} notes, "
            f"degradation={deg.get('total_degradation', 0):.2f}"
        )

    # Verify steering effect
    if 0.0 in results and len(results) >= 3:
        # Find the most negative and most positive alphas
        sorted_alphas = sorted(results.keys())
        min_alpha = sorted_alphas[0]
        max_alpha = sorted_alphas[-1]

        # Collect all metrics across alphas
        all_velocities = {
            alpha: results[alpha]["velocity_mean"] for alpha in sorted_alphas
        }
        all_pitches = {alpha: results[alpha]["pitch_mean"] for alpha in sorted_alphas}
        all_durations = {
            alpha: results[alpha]["duration_mean"] for alpha in sorted_alphas
        }

        baseline_vel = results[0.0]["velocity_mean"]
        low_vel = results[min_alpha]["velocity_mean"]
        high_vel = results[max_alpha]["velocity_mean"]

        baseline_pitch = results[0.0]["pitch_mean"]
        low_pitch = results[min_alpha]["pitch_mean"]
        high_pitch = results[max_alpha]["pitch_mean"]

        baseline_duration = results[0.0]["duration_mean"]
        low_duration = results[min_alpha]["duration_mean"]
        high_duration = results[max_alpha]["duration_mean"]

        # Calculate statistics across all alphas
        velocity_values = [v for v in all_velocities.values() if v > 0]
        pitch_values = [v for v in all_pitches.values() if v > 0]
        duration_values = [v for v in all_durations.values() if v > 0]

        velocity_range = (
            max(velocity_values) - min(velocity_values) if velocity_values else 0
        )
        pitch_range = max(pitch_values) - min(pitch_values) if pitch_values else 0
        duration_range = (
            max(duration_values) - min(duration_values) if duration_values else 0
        )

        # Calculate correlation and regression statistics
        alphas_array = np.array(sorted_alphas)
        pitch_array = np.array([all_pitches[a] for a in sorted_alphas])
        duration_array = np.array([all_durations[a] for a in sorted_alphas])
        velocity_array = np.array([all_velocities[a] for a in sorted_alphas])

        # Pearson correlation (linear relationship)
        pitch_corr, pitch_corr_pvalue = scipy_stats.pearsonr(alphas_array, pitch_array)
        duration_corr, duration_corr_pvalue = scipy_stats.pearsonr(
            alphas_array, duration_array
        )
        velocity_corr, velocity_corr_pvalue = scipy_stats.pearsonr(
            alphas_array, velocity_array
        )

        # Spearman correlation (monotonic relationship)
        pitch_spearman, pitch_spearman_pvalue = scipy_stats.spearmanr(
            alphas_array, pitch_array
        )
        duration_spearman, duration_spearman_pvalue = scipy_stats.spearmanr(
            alphas_array, duration_array
        )

        # R² for linear fit
        pitch_slope, pitch_intercept, pitch_r_value, _, _ = scipy_stats.linregress(
            alphas_array, pitch_array
        )
        duration_slope, duration_intercept, duration_r_value, _, _ = (
            scipy_stats.linregress(alphas_array, duration_array)
        )

        # Effect size (Cohen's d) for extreme alphas vs baseline
        def cohens_d(mean1, std1, mean2, std2, n1, n2):
            pooled_std = np.sqrt(
                ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
            )
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        pitch_effect_size_low = cohens_d(
            results[min_alpha]["pitch_mean"],
            results[min_alpha]["pitch_std"],
            baseline_pitch,
            results[0.0]["pitch_std"],
            results[min_alpha]["n_notes"],
            results[0.0]["n_notes"],
        )
        pitch_effect_size_high = cohens_d(
            results[max_alpha]["pitch_mean"],
            results[max_alpha]["pitch_std"],
            baseline_pitch,
            results[0.0]["pitch_std"],
            results[max_alpha]["n_notes"],
            results[0.0]["n_notes"],
        )

        duration_effect_size_low = cohens_d(
            results[min_alpha]["duration_mean"],
            results[min_alpha]["duration_std"],
            baseline_duration,
            results[0.0]["duration_std"],
            results[min_alpha]["n_notes"],
            results[0.0]["n_notes"],
        )
        duration_effect_size_high = cohens_d(
            results[max_alpha]["duration_mean"],
            results[max_alpha]["duration_std"],
            baseline_duration,
            results[0.0]["duration_std"],
            results[max_alpha]["n_notes"],
            results[0.0]["n_notes"],
        )

        logging.info("\n" + "=" * 60)
        logging.info("STEERING EFFECT VERIFICATION")
        logging.info("=" * 60)

        logging.info("\nOVERALL STATISTICS:")
        logging.info(f"  Number of alpha values tested: {len(sorted_alphas)}")
        logging.info(f"  Alpha range: [{min_alpha}, {max_alpha}]")
        logging.info(
            f"  Total notes generated: {sum(results[a]['n_notes'] for a in sorted_alphas)}"
        )

        logging.info("\nVELOCITY:")
        logging.info(f"  Baseline (alpha=0.0):       {baseline_vel:.2f}")
        logging.info(f"  Min (alpha={min_alpha:+.1f}):        {low_vel:.2f}")
        logging.info(f"  Max (alpha={max_alpha:+.1f}):        {high_vel:.2f}")
        logging.info(f"  Range across all alphas:    {velocity_range:.2f}")
        logging.info(
            f"  Min vs Baseline:            {low_vel - baseline_vel:+.2f} ({((low_vel - baseline_vel) / baseline_vel * 100) if baseline_vel > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Max vs Baseline:            {high_vel - baseline_vel:+.2f} ({((high_vel - baseline_vel) / baseline_vel * 100) if baseline_vel > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Pearson correlation (r):    {velocity_corr:+.4f} (p={velocity_corr_pvalue:.4f})"
        )

        logging.info("\nPITCH:")
        logging.info(f"  Baseline (alpha=0.0):       {baseline_pitch:.2f}")
        logging.info(f"  Min (alpha={min_alpha:+.1f}):        {low_pitch:.2f}")
        logging.info(f"  Max (alpha={max_alpha:+.1f}):        {high_pitch:.2f}")
        logging.info(f"  Range across all alphas:    {pitch_range:.2f} semitones")
        logging.info(
            f"  Min vs Baseline:            {low_pitch - baseline_pitch:+.2f} ({((low_pitch - baseline_pitch) / baseline_pitch * 100) if baseline_pitch > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Max vs Baseline:            {high_pitch - baseline_pitch:+.2f} ({((high_pitch - baseline_pitch) / baseline_pitch * 100) if baseline_pitch > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Pearson correlation (r):    {pitch_corr:+.4f} (p={pitch_corr_pvalue:.4f})"
        )
        logging.info(
            f"  Spearman correlation (ρ):   {pitch_spearman:+.4f} (p={pitch_spearman_pvalue:.4f})"
        )
        logging.info(f"  Linear fit R²:              {pitch_r_value**2:.4f}")
        logging.info(
            f"  Linear slope:               {pitch_slope:+.4f} semitones per α"
        )
        logging.info(
            f"  Effect size (Cohen's d):    min={pitch_effect_size_low:+.3f}, max={pitch_effect_size_high:+.3f}"
        )

        logging.info("\nDURATION (ticks):")
        logging.info(f"  Baseline (alpha=0.0):       {baseline_duration:.2f}")
        logging.info(f"  Min (alpha={min_alpha:+.1f}):        {low_duration:.2f}")
        logging.info(f"  Max (alpha={max_alpha:+.1f}):        {high_duration:.2f}")
        logging.info(f"  Range across all alphas:    {duration_range:.2f} ticks")
        logging.info(
            f"  Min vs Baseline:            {low_duration - baseline_duration:+.2f} ({((low_duration - baseline_duration) / baseline_duration * 100) if baseline_duration > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Max vs Baseline:            {high_duration - baseline_duration:+.2f} ({((high_duration - baseline_duration) / baseline_duration * 100) if baseline_duration > 0 else 0:+.1f}%)"
        )
        logging.info(
            f"  Pearson correlation (r):    {duration_corr:+.4f} (p={duration_corr_pvalue:.4f})"
        )
        logging.info(
            f"  Spearman correlation (ρ):   {duration_spearman:+.4f} (p={duration_spearman_pvalue:.4f})"
        )
        logging.info(f"  Linear fit R²:              {duration_r_value**2:.4f}")
        logging.info(f"  Linear slope:               {duration_slope:+.4f} ticks per α")
        logging.info(
            f"  Effect size (Cohen's d):    min={duration_effect_size_low:+.3f}, max={duration_effect_size_high:+.3f}"
        )

        # Show progression across all alphas
        logging.info("\nPROGRESSION ACROSS ALPHAS:")
        for alpha in sorted_alphas:
            logging.info(
                f"  α={alpha:+5.1f}: "
                f"pitch={results[alpha]['pitch_mean']:6.2f}, "
                f"duration={results[alpha]['duration_mean']:6.2f}"
            )

        # Evaluation
        logging.info("\nSTEERING EFFECTIVENESS:")

        # Interpretation helper
        def interpret_correlation(r, pval):
            if pval > 0.05:
                return "NOT SIGNIFICANT"
            elif abs(r) >= 0.9:
                return "VERY STRONG"
            elif abs(r) >= 0.7:
                return "STRONG"
            elif abs(r) >= 0.5:
                return "MODERATE"
            elif abs(r) >= 0.3:
                return "WEAK"
            else:
                return "VERY WEAK"

        def interpret_effect_size(d):
            abs_d = abs(d)
            if abs_d >= 0.8:
                return "LARGE"
            elif abs_d >= 0.5:
                return "MEDIUM"
            elif abs_d >= 0.2:
                return "SMALL"
            else:
                return "NEGLIGIBLE"

        # Check pitch steering
        pitch_correct = high_pitch > baseline_pitch and low_pitch < baseline_pitch
        pitch_monotonic = all(
            all_pitches[sorted_alphas[i]] <= all_pitches[sorted_alphas[i + 1]]
            for i in range(len(sorted_alphas) - 1)
        )
        pitch_strength = interpret_correlation(pitch_corr, pitch_corr_pvalue)
        pitch_effect = interpret_effect_size(
            max(abs(pitch_effect_size_low), abs(pitch_effect_size_high))
        )

        logging.info(f"\n  Pitch Steering:")
        logging.info(
            f"    Direction: {'✓ Correct' if pitch_correct else '✗ Incorrect/None'}"
        )
        logging.info(
            f"    Monotonicity: {'✓ Monotonic' if pitch_monotonic else '✗ Non-monotonic'}"
        )
        logging.info(
            f"    Correlation: {pitch_strength} (r={pitch_corr:+.3f}, p={pitch_corr_pvalue:.4f})"
        )
        logging.info(
            f"    Linearity: R²={pitch_r_value**2:.3f} ({'Good fit' if pitch_r_value**2 > 0.8 else 'Moderate fit' if pitch_r_value**2 > 0.5 else 'Poor fit'})"
        )
        logging.info(
            f"    Effect Size: {pitch_effect} (d={max(abs(pitch_effect_size_low), abs(pitch_effect_size_high)):.3f})"
        )

        if pitch_correct and pitch_monotonic and abs(pitch_corr) > 0.7:
            logging.info(f"    ✓✓✓ EXCELLENT: Strong monotonic steering effect")
        elif pitch_correct and abs(pitch_corr) > 0.5:
            logging.info(f"    ✓✓ GOOD: Clear steering effect with some variability")
        elif pitch_correct:
            logging.info(f"    ✓ WEAK: Correct direction but inconsistent")
        else:
            logging.warning(f"    ✗ FAILED: No effective steering detected")

        # Check duration steering
        duration_correct = (
            high_duration > baseline_duration and low_duration < baseline_duration
        )
        duration_monotonic = all(
            all_durations[sorted_alphas[i]] <= all_durations[sorted_alphas[i + 1]]
            for i in range(len(sorted_alphas) - 1)
        )
        duration_strength = interpret_correlation(duration_corr, duration_corr_pvalue)
        duration_effect = interpret_effect_size(
            max(abs(duration_effect_size_low), abs(duration_effect_size_high))
        )

        logging.info(f"\n  Duration Steering:")
        logging.info(
            f"    Direction: {'✓ Correct' if duration_correct else '✗ Incorrect/None'}"
        )
        logging.info(
            f"    Monotonicity: {'✓ Monotonic' if duration_monotonic else '✗ Non-monotonic'}"
        )
        logging.info(
            f"    Correlation: {duration_strength} (r={duration_corr:+.3f}, p={duration_corr_pvalue:.4f})"
        )
        logging.info(
            f"    Linearity: R²={duration_r_value**2:.3f} ({'Good fit' if duration_r_value**2 > 0.8 else 'Moderate fit' if duration_r_value**2 > 0.5 else 'Poor fit'})"
        )
        logging.info(
            f"    Effect Size: {duration_effect} (d={max(abs(duration_effect_size_low), abs(duration_effect_size_high)):.3f})"
        )

        if duration_correct and duration_monotonic and abs(duration_corr) > 0.7:
            logging.info(f"    ✓✓✓ EXCELLENT: Strong monotonic steering effect")
        elif duration_correct and abs(duration_corr) > 0.5:
            logging.info(f"    ✓✓ GOOD: Clear steering effect with some variability")
        elif duration_correct:
            logging.info(f"    ✓ WEAK: Correct direction but inconsistent")
        else:
            logging.warning(f"    ✗ FAILED: No effective steering detected")
    else:
        logging.info("\nNot all alphas tested, skipping verification")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
