#!/usr/bin/env python3
"""Test only Scenario 2: High Pitch Major → Low Minor with NEGATIVE alphas (bug fix)"""

import sys
import pathlib

# Import the main script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_multi_conditioned import *

def main():
    """Run only scenario 2."""
    parser = argparse.ArgumentParser(
        description="Test Scenario 2 only: High Pitch Major → Low Minor"
    )
    parser.add_argument(
        "--model_checkpoint", type=pathlib.Path, default=None, help="Model checkpoint"
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "outputs/steering_vectors/average_pitch_steering_vectors.pt"
        ),
        help="Pitch steering vectors",
    )
    parser.add_argument(
        "--modality_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        ),
        help="Modality steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/scenario2_fixed"
        ),
        help="Output directory",
    )
    parser.add_argument("--n_songs", type=int, default=5, help="Songs per category")
    parser.add_argument(
        "--conditioning_beats", type=int, default=8, help="Beats for conditioning"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=512, help="Tokens to generate"
    )
    parser.add_argument(
        "--alphas_pitch",
        type=str,
        default="-2.5,-2.0,-1.5,0.0",
        help="Pitch alphas (NEGATIVE for high→low)",
    )
    parser.add_argument(
        "--alphas_modality",
        type=str,
        default="-2.5,-2.0,-1.5,0.0",
        help="Modality alphas (NEGATIVE for major→minor)",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "scenario2.log"),
            logging.StreamHandler(),
        ],
    )

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA not available, using CPU")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load model
    logging.info("Loading model...")
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
    logging.info("Creating vector composer...")
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(
        str(args.pitch_vectors), str(args.modality_vectors)
    )

    # Parse alphas
    alphas_pitch = [float(a.strip()) for a in args.alphas_pitch.split(",")]
    alphas_modality = [float(a.strip()) for a in args.alphas_modality.split(",")]

    logging.info(f"Testing SCENARIO 2: High Pitch Major → Low Minor (NEGATIVE alphas)")
    logging.info(f"Pitch alphas: {alphas_pitch}")
    logging.info(f"Modality alphas: {alphas_modality}")

    # Find extreme songs - only need high_pitch_major
    logging.info("\n" + "=" * 80)
    logging.info("FINDING HIGH PITCH MAJOR SONGS")
    logging.info("=" * 80)

    cache_file = args.output_dir / "extreme_songs_cache.json"

    extreme_songs = find_extreme_songs(
        config.NOTES_DIR,
        encoding,
        args.n_songs,
        args.conditioning_beats,
        cache_file=cache_file,
    )

    if len(extreme_songs["high_pitch_major"]) == 0:
        logging.error("No high pitch major songs found!")
        return

    # Run ONLY scenario 2
    logging.info("\n" + "=" * 80)
    logging.info("TESTING SCENARIO 2: High Pitch Major → Low Minor")
    logging.info("=" * 80)

    start_time = time.time()

    results = conditioned_generate_and_evaluate(
        model=model,
        composer=composer,
        strategy="gram_schmidt",
        song_list=extreme_songs["high_pitch_major"],
        scenario="high_pitch_major_to_low_minor",
        alpha_pitch_list=alphas_pitch,
        alpha_modality_list=alphas_modality,
        encoding=encoding,
        device=device,
        conditioning_beats=args.conditioning_beats,
        continuation_len=args.continuation_len,
        output_dir=args.output_dir,
    )

    elapsed_time = time.time() - start_time

    # Analyze
    logging.info("\nAnalyzing results...")
    
    # Calculate statistics
    valid = [r for r in results if r["generated_n_notes"] >= 10]
    
    if valid:
        pitch_success = np.mean([r["pitch_steering_success"] for r in valid])
        mode_success = np.mean([r["mode_steering_success"] for r in valid])
        overall_success = np.mean([r["overall_success"] for r in valid])
        mean_pitch_change = np.mean([r["pitch_change"] for r in valid])
        mean_degradation = np.mean([r["degradation"]["total_degradation"] for r in valid])
        mode_change_rate = np.mean([r["mode_changed"] for r in valid])

        # Save results
        results_file = args.output_dir / "scenario2_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "config": {
                        "scenario": "high_pitch_major_to_low_minor",
                        "n_songs": args.n_songs,
                        "conditioning_beats": args.conditioning_beats,
                        "continuation_len": args.continuation_len,
                        "alphas_pitch": alphas_pitch,
                        "alphas_modality": alphas_modality,
                        "strategy": "gram_schmidt",
                        "total_generations": len(results),
                        "elapsed_time_seconds": elapsed_time,
                    },
                    "results": results,
                    "summary": {
                        "n_samples": len(valid),
                        "pitch_success_rate": float(pitch_success),
                        "mode_success_rate": float(mode_success),
                        "overall_success_rate": float(overall_success),
                        "mean_pitch_change": float(mean_pitch_change),
                        "mean_degradation": float(mean_degradation),
                        "mode_change_rate": float(mode_change_rate),
                    },
                },
                f,
                indent=2,
            )

        logging.info(f"Saved results to: {results_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SCENARIO 2 RESULTS: High Pitch Major → Low Minor")
        print("=" * 80)
        print(f"\nTotal generations: {len(results)}")
        print(f"Valid generations: {len(valid)}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"\nPitch success: {pitch_success*100:.1f}%")
        print(f"Mode success: {mode_success*100:.1f}%")
        print(f"Overall success: {overall_success*100:.1f}%")
        print(f"Mean pitch change: {mean_pitch_change:+.1f} semitones")
        print(f"Mode change rate: {mode_change_rate*100:.1f}%")
        print(f"Mean degradation: {mean_degradation:.2f}")
        
        # Find best example
        best = max(valid, key=lambda x: (x["overall_success"], -x["degradation"]["total_degradation"]))
        print(f"\n### Best Example ###")
        print(f"Song: {best['song_name']}")
        print(f"α_pitch: {best['alpha_pitch']:+.1f}, α_modality: {best['alpha_modality']:+.1f}")
        print(f"Pitch: {best['conditioning_pitch']:.1f} → {best['generated_pitch_mean']:.1f} (Δ{best['pitch_change']:+.1f})")
        print(f"Mode: {best['conditioning_mode']} → {best['generated_mode']} (conf={best['generated_confidence']:.2f})")
        print(f"Success: pitch={best['pitch_steering_success']}, mode={best['mode_steering_success']}, overall={best['overall_success']}")
        print(f"Degradation: {best['degradation']['total_degradation']:.2f}")
        
        print("\n" + "=" * 80)
        print("✅ SCENARIO 2 COMPLETE!")
        print("=" * 80)

if __name__ == "__main__":
    main()
