#!/usr/bin/env python3
"""Step 2: Evaluate Audio with Music Flamingo.

This script:
1. Loads manifest.json from step 1
2. Queries Music Flamingo for each audio file with concept-specific prompts
3. Runs each query 3 times for self-consistency
4. Applies majority voting to determine final rating
5. Saves results.json with all responses and final ratings

Supports resuming from checkpoint if interrupted.

Usage:
    python steering_interventions/flamingo_eval/2_evaluate_music_flamingo.py \\
        --config steering_interventions/flamingo_eval/config.yaml \\
        --resume  # Resume from checkpoint
"""

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2  # seconds
MAX_DELAY = 60  # seconds

# Import will be done dynamically based on api_method
# from gradio_client import Client, handle_file  # For Gradio API
# from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor  # For Transformers


def load_config(config_path: pathlib.Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    """Load evaluation checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"completed_ids": [], "results": []}


def save_checkpoint(checkpoint_path: pathlib.Path, checkpoint_data: dict):
    """Save evaluation checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def query_music_flamingo(
    audio_path: pathlib.Path,
    system_prompt: str,
    user_prompt: str,
    model,  # Gradio Client or Transformers model
    music_flamingo_config: dict,
) -> str:
    """Query Music Flamingo model with audio and prompt (with retry logic).

    Args:
        audio_path: Path to WAV file
        system_prompt: System message (may be ignored depending on API method)
        user_prompt: User prompt template
        model: Gradio Client instance or Transformers model
        music_flamingo_config: Music Flamingo config dict (from config["music_flamingo"])

    Returns:
        Model response text

    Raises:
        Exception: If all retries fail
    """
    api_method = music_flamingo_config["api_method"]

    for attempt in range(MAX_RETRIES):
        try:
            if api_method == "gradio":
                # Query via Gradio API
                result = model.predict(
                    audio_path=model.handle_file(str(audio_path)),
                    youtube_url="",
                    prompt_text=user_prompt,
                    api_name="/infer",
                )

                if result:
                    return result
                else:
                    raise ValueError("Empty response from API")

            elif api_method == "transformers":
                # Query via HuggingFace Transformers (local inference)
                processor = model["processor"]
                transformer_model = model["model"]

                logging.info(f"Processing audio: {audio_path.name}")

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "audio", "path": str(audio_path)},
                        ],
                    }
                ]

                logging.info("Applying chat template...")
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                ).to(transformer_model.device, dtype=transformer_model.dtype)

                logging.info("Generating response...")

                # Generation parameters
                generate_kwargs = {
                    "max_new_tokens": music_flamingo_config.get("max_tokens", 256),
                    "do_sample": True,
                    "temperature": music_flamingo_config.get("temperature", 0.1),
                    "top_p": music_flamingo_config.get("top_p", 0.9),
                }

                outputs = transformer_model.generate(**inputs, **generate_kwargs)

                logging.info("Decoding output...")
                decoded_outputs = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                result = decoded_outputs[0] if decoded_outputs else ""

                if result:
                    logging.info(f"Response received ({len(result)} chars)")
                    return result
                else:
                    raise ValueError("Empty response from model")

            else:
                raise ValueError(
                    f"Unknown api_method: {api_method}. Use 'gradio' or 'transformers'"
                )

        except Exception as e:
            delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)

            if attempt < MAX_RETRIES - 1:
                logging.warning(
                    f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"API failed after {MAX_RETRIES} attempts for {audio_path.name}: {e}"
                )
                raise

    raise Exception(f"Failed to query Music Flamingo after {MAX_RETRIES} attempts")


def parse_response(
    response: str, parse_regex: str, concept: Optional[str] = None
) -> Optional[str]:
    """Parse rating from model response using regex.

    Args:
        response: Model response text
        parse_regex: Regex pattern to extract rating
        concept: Optional concept name for dual prompts

    Returns:
        Extracted rating (as string) or None if not found
    """
    match = re.search(parse_regex, response, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def majority_vote(ratings: List[str]) -> Tuple[str, int, bool]:
    """Apply majority voting to list of ratings.

    Args:
        ratings: List of rating strings (e.g., ["3", "4", "3"])

    Returns:
        (majority_rating, vote_count, is_unanimous)
    """
    if not ratings:
        return (None, 0, False)

    counter = Counter(ratings)
    majority_rating, vote_count = counter.most_common(1)[0]
    is_unanimous = vote_count == len(ratings)

    return (majority_rating, vote_count, is_unanimous)


def map_pitch_direction(code: str) -> str:
    """Map numeric pitch direction code to text.

    Args:
        code: "1", "2", or "3"

    Returns:
        "UP", "DOWN", or "CONSTANT"
    """
    mapping = {"1": "UP", "2": "DOWN", "3": "CONSTANT"}
    return mapping.get(code, code)


def map_duration_direction(code: str) -> str:
    """Map numeric duration direction code to text.

    Args:
        code: "1", "2", or "3"

    Returns:
        "LONGER", "SHORTER", or "CONSTANT"
    """
    mapping = {"1": "LONGER", "2": "SHORTER", "3": "CONSTANT"}
    return mapping.get(code, code)


def evaluate_sample_category_1(
    sample: dict, config: dict, model, output_root: pathlib.Path
) -> dict:
    """Evaluate unconditional single steering sample.

    Returns:
        Result dictionary with ratings and metadata
    """
    concept = sample["concept"]
    prompt_config = config["prompts"][f"category_1_{concept}"]

    # Audio path (wav_path is already complete relative path)
    audio_path = pathlib.Path(sample["wav_path"])

    # Run multiple times for self-consistency
    num_runs = config["music_flamingo"]["num_runs"]
    responses = []
    ratings = []

    for run_idx in range(num_runs):
        try:
            response = query_music_flamingo(
                audio_path,
                prompt_config["system"],
                prompt_config["user_template"],
                model,
                config["music_flamingo"],
            )
            responses.append(response)

            logging.info(f"Run {run_idx+1} response: {response}")

            # Parse rating
            rating = parse_response(response, prompt_config["parse_regex"])
            logging.info(f"Run {run_idx+1} parsed rating: {rating}")

            if rating:
                ratings.append(rating)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse rating. Regex: {prompt_config['parse_regex']}"
                )

        except Exception as e:
            logging.error(f"Error in run {run_idx+1}: {e}")
            responses.append(f"ERROR: {e}")

    # Majority vote
    final_rating, vote_count, is_unanimous = majority_vote(ratings)

    logging.info(f"Category 1 - All ratings: {ratings}")
    logging.info(
        f"Category 1 - Final rating: {final_rating}, Vote count: {vote_count}/{num_runs}"
    )

    result = {
        **sample,
        "responses": responses,
        "parsed_ratings": ratings,
        "final_rating": int(final_rating) if final_rating else None,
        "vote_count": vote_count,
        "total_runs": num_runs,
        "is_unanimous": is_unanimous,
        "has_ambiguity": len(set(ratings)) > 1 if ratings else True,
    }

    return result


def evaluate_sample_category_2(
    sample: dict, config: dict, model, output_root: pathlib.Path
) -> dict:
    """Evaluate unconditional dual steering sample.

    Returns:
        Result dictionary with ratings for both concepts
    """
    prompt_config = config["prompts"]["category_2_dual"]

    # Audio path (wav_path is already complete relative path)
    audio_path = pathlib.Path(sample["wav_path"])

    # Run multiple times
    num_runs = config["music_flamingo"]["num_runs"]
    responses = []
    pitch_ratings = []
    duration_ratings = []

    for run_idx in range(num_runs):
        try:
            response = query_music_flamingo(
                audio_path,
                prompt_config["system"],
                prompt_config["user_template"],
                model,
                config["music_flamingo"],
            )
            responses.append(response)

            logging.info(f"Run {run_idx+1} response: {response}")

            # Parse both ratings
            pitch_rating = parse_response(response, prompt_config["parse_pitch_regex"])
            duration_rating = parse_response(
                response, prompt_config["parse_duration_regex"]
            )

            logging.info(
                f"Run {run_idx+1} parsed pitch: {pitch_rating}, duration: {duration_rating}"
            )

            if pitch_rating:
                pitch_ratings.append(pitch_rating)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse pitch. Regex: {prompt_config['parse_pitch_regex']}"
                )

            if duration_rating:
                duration_ratings.append(duration_rating)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse duration. Regex: {prompt_config['parse_duration_regex']}"
                )

        except Exception as e:
            logging.error(f"Error in run {run_idx+1}: {e}")
            responses.append(f"ERROR: {e}")

    # Majority vote for each concept
    final_pitch, pitch_votes, pitch_unanimous = majority_vote(pitch_ratings)
    final_duration, duration_votes, duration_unanimous = majority_vote(duration_ratings)

    logging.info(
        f"Category 2 - Pitch ratings: {pitch_ratings}, Final: {final_pitch} ({pitch_votes}/{num_runs})"
    )
    logging.info(
        f"Category 2 - Duration ratings: {duration_ratings}, Final: {final_duration} ({duration_votes}/{num_runs})"
    )

    result = {
        **sample,
        "responses": responses,
        "parsed_pitch_ratings": pitch_ratings,
        "parsed_duration_ratings": duration_ratings,
        "final_pitch_rating": int(final_pitch) if final_pitch else None,
        "final_duration_rating": int(final_duration) if final_duration else None,
        "pitch_vote_count": pitch_votes,
        "duration_vote_count": duration_votes,
        "total_runs": num_runs,
        "pitch_unanimous": pitch_unanimous,
        "duration_unanimous": duration_unanimous,
        "has_ambiguity": (
            (len(set(pitch_ratings)) > 1 or len(set(duration_ratings)) > 1)
            if (pitch_ratings and duration_ratings)
            else True
        ),
    }

    return result


def evaluate_sample_category_3(
    sample: dict, config: dict, model, output_root: pathlib.Path
) -> dict:
    """Evaluate conditional single steering sample.

    Returns:
        Result dictionary with change direction
    """
    concept = sample["concept"]
    prompt_config = config["prompts"][f"category_3_{concept}"]

    # Audio path (wav_path is already complete relative path)
    audio_path = pathlib.Path(sample["wav_path"])

    # Run multiple times
    num_runs = config["music_flamingo"]["num_runs"]
    responses = []
    directions = []

    for run_idx in range(num_runs):
        try:
            response = query_music_flamingo(
                audio_path,
                prompt_config["system"],
                prompt_config["user_template"],
                model,
                config["music_flamingo"],
            )
            responses.append(response)

            logging.info(f"Run {run_idx+1} response: {response}")

            # Parse direction
            direction = parse_response(response, prompt_config["parse_regex"])
            logging.info(f"Run {run_idx+1} parsed direction: {direction}")

            if direction:
                directions.append(direction)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse direction. Regex: {prompt_config['parse_regex']}"
                )

        except Exception as e:
            logging.error(f"Error in run {run_idx+1}: {e}")
            responses.append(f"ERROR: {e}")

    # Majority vote
    final_direction, vote_count, is_unanimous = majority_vote(directions)

    # Map numeric code to text for pitch direction
    if final_direction and concept == "pitch":
        final_direction_text = map_pitch_direction(final_direction)
    elif final_direction and concept == "duration":
        final_direction_text = map_duration_direction(final_direction)
    else:
        final_direction_text = final_direction

    logging.info(f"Category 3 - All directions: {directions}")
    logging.info(
        f"Category 3 - Final direction: {final_direction_text} (code: {final_direction}), Vote count: {vote_count}/{num_runs}"
    )

    result = {
        **sample,
        "responses": responses,
        "parsed_directions": directions,
        "final_direction": final_direction_text,
        "final_direction_code": final_direction,
        "vote_count": vote_count,
        "total_runs": num_runs,
        "is_unanimous": is_unanimous,
        "has_ambiguity": len(set(directions)) > 1 if directions else True,
    }

    return result


def evaluate_sample_category_4(
    sample: dict, config: dict, model, output_root: pathlib.Path
) -> dict:
    """Evaluate conditional dual steering sample.

    Returns:
        Result dictionary with change directions for both concepts
    """
    prompt_config = config["prompts"]["category_4_dual"]

    # Audio path (wav_path is already complete relative path)
    audio_path = pathlib.Path(sample["wav_path"])

    # Run multiple times
    num_runs = config["music_flamingo"]["num_runs"]
    responses = []
    pitch_directions = []
    duration_directions = []

    for run_idx in range(num_runs):
        try:
            response = query_music_flamingo(
                audio_path,
                prompt_config["system"],
                prompt_config["user_template"],
                model,
                config["music_flamingo"],
            )
            responses.append(response)

            logging.info(f"Run {run_idx+1} response: {response}")

            # Parse both directions
            pitch_dir = parse_response(response, prompt_config["parse_pitch_regex"])
            duration_dir = parse_response(
                response, prompt_config["parse_duration_regex"]
            )

            logging.info(
                f"Run {run_idx+1} parsed pitch direction: {pitch_dir}, duration direction: {duration_dir}"
            )

            if pitch_dir:
                pitch_directions.append(pitch_dir)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse pitch direction. Regex: {prompt_config['parse_pitch_regex']}"
                )

            if duration_dir:
                duration_directions.append(duration_dir)
            else:
                logging.warning(
                    f"Run {run_idx+1} - Failed to parse duration direction. Regex: {prompt_config['parse_duration_regex']}"
                )

        except Exception as e:
            logging.error(f"Error in run {run_idx+1}: {e}")
            responses.append(f"ERROR: {e}")

    # Majority votes
    final_pitch_dir, pitch_votes, pitch_unanimous = majority_vote(pitch_directions)
    final_duration_dir, duration_votes, duration_unanimous = majority_vote(
        duration_directions
    )

    # Map numeric codes to text
    final_pitch_dir_text = (
        map_pitch_direction(final_pitch_dir) if final_pitch_dir else None
    )
    final_duration_dir_text = (
        map_duration_direction(final_duration_dir) if final_duration_dir else None
    )

    logging.info(
        f"Category 4 - Pitch directions: {pitch_directions}, Final: {final_pitch_dir_text} (code: {final_pitch_dir}) ({pitch_votes}/{num_runs})"
    )
    logging.info(
        f"Category 4 - Duration directions: {duration_directions}, Final: {final_duration_dir_text} (code: {final_duration_dir}) ({duration_votes}/{num_runs})"
    )

    result = {
        **sample,
        "responses": responses,
        "parsed_pitch_directions": pitch_directions,
        "parsed_duration_directions": duration_directions,
        "final_pitch_direction": final_pitch_dir_text,
        "final_duration_direction": final_duration_dir_text,
        "final_pitch_direction_code": final_pitch_dir,
        "final_duration_direction_code": final_duration_dir,
        "pitch_vote_count": pitch_votes,
        "duration_vote_count": duration_votes,
        "total_runs": num_runs,
        "pitch_unanimous": pitch_unanimous,
        "duration_unanimous": duration_unanimous,
        "has_ambiguity": (
            (len(set(pitch_directions)) > 1 or len(set(duration_directions)) > 1)
            if (pitch_directions and duration_directions)
            else True
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio with Music Flamingo")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/flamingo_eval/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (don't actually query model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N samples (for testing)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    log_level = getattr(logging, config["logging"]["level"])
    handlers = [logging.StreamHandler()]

    if config["logging"]["save_to_file"]:
        log_file = pathlib.Path(config["logging"]["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.info("=" * 70)
    logging.info("Music Flamingo Evaluation - Model Querying")
    logging.info("=" * 70)
    logging.info(f"Config: {args.config}")
    logging.info(f"Resume: {args.resume}")
    logging.info(f"Dry run: {args.dry_run}")

    # Setup paths
    output_root = pathlib.Path(config["paths"]["output_root"])
    manifest_path = output_root / "manifest.json"
    results_path = output_root / "results.json"
    checkpoint_path = pathlib.Path(config["resume"]["checkpoint_file"])

    # Load manifest
    logging.info(f"Loading manifest from {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    logging.info(f"Total samples in manifest: {len(manifest)}")

    # Load checkpoint if resuming
    checkpoint_data = {"completed_ids": [], "results": []}
    if args.resume and config["resume"]["enabled"]:
        checkpoint_data = load_checkpoint(checkpoint_path)
        logging.info(
            f"Loaded checkpoint: {len(checkpoint_data['completed_ids'])} samples already completed"
        )

    # Initialize Music Flamingo model
    if not args.dry_run:
        api_method = config["music_flamingo"]["api_method"]
        logging.info(f"Initializing Music Flamingo ({api_method} method)...")

        try:
            if api_method == "gradio":
                # Import Gradio Client
                from gradio_client import Client, handle_file

                hf_token = config["music_flamingo"].get("hf_token", "").strip()

                if hf_token:
                    logging.info("Using HuggingFace token for authentication")
                    model = Client("nvidia/music-flamingo", hf_token=hf_token)
                else:
                    logging.info("Using anonymous access (limited quota)")
                    model = Client("nvidia/music-flamingo")

                # Store handle_file as method for query function
                model.handle_file = handle_file

                logging.info("✅ Music Flamingo Gradio client initialized")

            elif api_method == "transformers":
                # Import Transformers
                import torch
                from transformers import (
                    AudioFlamingo3ForConditionalGeneration,
                    AutoProcessor,
                )

                model_id = "nvidia/music-flamingo-hf"
                device = config["music_flamingo"].get("device", "cuda:0")
                dtype = torch.float16

                logging.info(f"Loading model from {model_id}...")
                logging.info("⚠️ This will download ~10GB of model weights")

                processor = AutoProcessor.from_pretrained(model_id)
                transformer_model = (
                    AudioFlamingo3ForConditionalGeneration.from_pretrained(
                        model_id, device_map="auto", torch_dtype=dtype
                    )
                )

                # Package both processor and model
                model = {"processor": processor, "model": transformer_model}

                logging.info(f"✅ Music Flamingo Transformers model loaded on {device}")

            else:
                raise ValueError(
                    f"Unknown api_method: {api_method}. Use 'gradio' or 'transformers'"
                )

        except Exception as e:
            logging.error(f"Failed to initialize Music Flamingo: {e}")
            sys.exit(1)
    else:
        model = None
        logging.info("Dry run mode - skipping model initialization")

    # Process each sample
    all_results = checkpoint_data["results"].copy()
    completed_ids = set(checkpoint_data["completed_ids"])

    checkpoint_interval = config["resume"]["checkpoint_interval"]
    samples_since_checkpoint = 0

    # Apply limit if specified
    samples_to_process = manifest
    if args.limit is not None:
        remaining = [s for s in manifest if s["id"] not in completed_ids]
        samples_to_process = remaining[: args.limit]
        logging.info(f"LIMIT MODE: Processing only {len(samples_to_process)} samples")

    for sample in tqdm(samples_to_process, desc="Evaluating samples"):
        sample_id = sample["id"]

        # Skip if already completed
        if sample_id in completed_ids:
            continue

        category = sample["category"]

        try:
            # Evaluate based on category
            if category == "unconditional_single":
                if args.dry_run:
                    result = {**sample, "dry_run": True}
                else:
                    result = evaluate_sample_category_1(
                        sample, config, model, output_root
                    )

            elif category == "unconditional_dual":
                if args.dry_run:
                    result = {**sample, "dry_run": True}
                else:
                    result = evaluate_sample_category_2(
                        sample, config, model, output_root
                    )

            elif category == "conditional_single":
                if args.dry_run:
                    result = {**sample, "dry_run": True}
                else:
                    result = evaluate_sample_category_3(
                        sample, config, model, output_root
                    )

            elif category == "conditional_dual":
                if args.dry_run:
                    result = {**sample, "dry_run": True}
                else:
                    result = evaluate_sample_category_4(
                        sample, config, model, output_root
                    )

            else:
                logging.warning(f"Unknown category: {category}")
                continue

            all_results.append(result)
            completed_ids.add(sample_id)
            samples_since_checkpoint += 1

            # Save checkpoint periodically
            if samples_since_checkpoint >= checkpoint_interval:
                if config["resume"]["enabled"]:
                    checkpoint_data = {
                        "completed_ids": list(completed_ids),
                        "results": all_results,
                    }
                    save_checkpoint(checkpoint_path, checkpoint_data)
                    logging.info(f"Checkpoint saved ({len(completed_ids)} samples)")
                samples_since_checkpoint = 0

        except Exception as e:
            logging.error(f"Error evaluating {sample_id}: {e}")
            continue

    # Save final results
    logging.info(f"Saving results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save final checkpoint
    if config["resume"]["enabled"]:
        checkpoint_data = {"completed_ids": list(completed_ids), "results": all_results}
        save_checkpoint(checkpoint_path, checkpoint_data)

    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Total samples evaluated: {len(all_results)}")

    # Breakdown by category
    by_category = {}
    for result in all_results:
        category = result["category"]
        by_category[category] = by_category.get(category, 0) + 1

    for category, count in sorted(by_category.items()):
        logging.info(f"  {category}: {count} samples")

    logging.info(f"\nResults saved to: {results_path}")
    logging.info("=" * 70)
    logging.info("Evaluation complete!")
    logging.info("Next step: Run 3_analyze_results.py")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
