#!/usr/bin/env python3
"""
Evaluate Contrasting Interventions using LLM MIDI Comparison

This script compares intervention MIDI files against baselines using OpenAI's API
to determine if musical features are successfully added or removed.

Usage:
    export OPENAI_API_KEY="your-key-here"
    
    # Single feature evaluation
    python evaluate_interventions_llm.py \
        --input-dir contrasting_intervention_results_5_1950/layer5 \
        --output-file evaluation_results_layer5.json
    
    # Batch evaluation with multiple directories
    python evaluate_interventions_llm.py \
        --input-dirs contrasting_intervention_results_1_325/layer1 \
                     contrasting_intervention_results_3_182/layer3 \
                     contrasting_intervention_results_5_1950/layer5 \
        --output-file evaluation_results_all.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import time

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    import music21
except ImportError:
    print("Error: music21 package not installed. Run: pip install music21")
    sys.exit(1)


# Feature descriptions for each layer/feature combination
FEATURE_DESCRIPTIONS = {
    "1_325": {
        "name": "rhythmic_displacement_syncopation",
        "description": "Rhythmic displacement patterns where melodic motifs are shifted by sixteenth notes, creating syncopation effects and off-beat emphasis with rhythmic tension",
    },
    "1_256": {
        "name": "dynamic_contrast_accents",
        "description": "Dramatic dynamic contrast patterns with sudden accents followed by immediate soft dynamics, creating forte-piano alternations and expressive ebb-and-flow",
    },
    "1_1323": {
        "name": "wide_pitch_range_texture",
        "description": "Wide pitch range utilization across multiple instruments, emphasizing broad harmonic spectrum and textural depth through multi-register coverage",
    },
    "3_182": {
        "name": "steady_pulse_march_rhythm",
        "description": "Steady pulse patterns with metronomic consistency and march-like rhythms, creating regular ordered rhythmic structures with consistent motifs",
    },
    "3_855": {
        "name": "dynamic_contrast_tension",
        "description": "Dynamic contrast patterns with sudden shifts between high and low velocity ranges, creating dramatic tension and release within short passages",
    },
    "3_997": {
        "name": "antiphonal_call_response",
        "description": "Antiphonal textures with call-and-response interactions between instrumental groups, creating spatial and dialogic effects where musical ideas echo between instruments",
    },
    "5_471": {
        "name": "unison_doubling_octaves",
        "description": "Unison doubling patterns where multiple instruments play the same melodic line in octaves, creating reinforced unified sound with enhanced textural weight",
    },
    "5_904": {
        "name": "rhythmic_augmentation",
        "description": "Rhythmic augmentation patterns where motifs are systematically lengthened in duration, creating expansion and temporal stretching effects",
    },
    "5_1950": {
        "name": "dramatic_dynamic_swells",
        "description": "Dramatic dynamic swell patterns with abrupt shifts between soft and loud dynamics, creating expressive swells and dramatic textural effects",
    },
}


class InterventionEvaluator:
    """Evaluate interventions using LLM-based MIDI comparison."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_events: int = 500,
    ):
        """Initialize evaluator with OpenAI API key."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_events = max_events

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def midi_to_text(self, midi_path: Path) -> str:
        """Convert MIDI file to full textual representation for LLM analysis."""
        try:
            score = music21.converter.parse(str(midi_path))

            text_parts = []

            # Header info
            time_sigs = score.flat.getTimeSignatures()
            if time_sigs:
                text_parts.append(f"Time Signature: {time_sigs[0].ratioString}")

            tempos = score.flat.getElementsByClass("MetronomeMark")
            if tempos:
                text_parts.append(f"Tempo: {tempos[0].number} BPM")

            text_parts.append(f"\nMUSICAL EVENTS (in chronological order):\n")

            # Collect all events from all parts with timing
            all_events = []
            for part_idx, part in enumerate(score.parts):
                for element in part.flatten():
                    if hasattr(element, "offset"):
                        offset = element.offset

                        if isinstance(element, music21.note.Note):
                            all_events.append(
                                {
                                    "offset": offset,
                                    "part": part_idx + 1,
                                    "type": "Note",
                                    "pitch": element.pitch.nameWithOctave,
                                    "midi": element.pitch.midi,
                                    "duration": element.quarterLength,
                                    "velocity": (
                                        element.volume.velocity
                                        if element.volume.velocity
                                        else 64
                                    ),
                                }
                            )
                        elif isinstance(element, music21.chord.Chord):
                            pitches = [p.nameWithOctave for p in element.pitches]
                            midis = [p.midi for p in element.pitches]
                            all_events.append(
                                {
                                    "offset": offset,
                                    "part": part_idx + 1,
                                    "type": "Chord",
                                    "pitches": pitches,
                                    "midis": midis,
                                    "duration": element.quarterLength,
                                    "velocity": (
                                        element.volume.velocity
                                        if element.volume.velocity
                                        else 64
                                    ),
                                }
                            )
                        elif isinstance(element, music21.note.Rest):
                            all_events.append(
                                {
                                    "offset": offset,
                                    "part": part_idx + 1,
                                    "type": "Rest",
                                    "duration": element.quarterLength,
                                }
                            )

            # Sort by offset (chronological order)
            all_events.sort(key=lambda x: (x["offset"], x["part"]))

            # Limit to max_events to avoid token limits
            if len(all_events) > self.max_events:
                all_events = all_events[: self.max_events]
                text_parts.append(
                    f"[Showing first {self.max_events} events of {len(all_events)} total]\n"
                )

            # Format events
            for i, event in enumerate(all_events):
                if event["type"] == "Note":
                    text_parts.append(
                        f"[{event['offset']:.2f}] Part {event['part']}: "
                        f"Note {event['pitch']} (MIDI {event['midi']}), "
                        f"duration={event['duration']:.3f}, velocity={event['velocity']}"
                    )
                elif event["type"] == "Chord":
                    pitches_str = ", ".join(event["pitches"])
                    text_parts.append(
                        f"[{event['offset']:.2f}] Part {event['part']}: "
                        f"Chord [{pitches_str}], "
                        f"duration={event['duration']:.3f}, velocity={event['velocity']}"
                    )
                elif event["type"] == "Rest":
                    text_parts.append(
                        f"[{event['offset']:.2f}] Part {event['part']}: "
                        f"Rest, duration={event['duration']:.3f}"
                    )

            return "\n".join(text_parts)

        except Exception as e:
            self.logger.error(f"Failed to parse MIDI {midi_path}: {e}")
            return f"[Error parsing MIDI file: {e}]"

    def create_comparison_prompt(
        self, midi_a_text: str, midi_b_text: str, feature_description: str
    ) -> str:
        """Create unbiased comparison prompt for LLM."""

        prompt = f"""You are a music analysis expert. Compare these two MIDI files and determine which one exhibits MORE of the following musical characteristic:

MUSICAL CHARACTERISTIC TO EVALUATE:
"{feature_description}"

MIDI FILE A:
{midi_a_text}

MIDI FILE B:
{midi_b_text}

TASK:
Analyze both MIDI files and determine which one (A or B) shows MORE presence of the described musical characteristic.

Provide your answer in the following JSON format:
{{
    "choice": "A" or "B",
    "confidence": <number from 1-10>,
    "reasoning": "<brief explanation of your choice>",
    "difference_magnitude": <number from 1-5, where 1=subtle difference, 5=dramatic difference>
}}

Be objective and focus on the specific musical characteristic described. Do not make assumptions about which file is "better" overall."""

        return prompt

    def compare_midi_pair(
        self, midi_a: Path, midi_b: Path, feature_description: str, max_retries: int = 3
    ) -> Dict:
        """Compare two MIDI files using LLM."""

        self.logger.info(f"Comparing: {midi_a.name} vs {midi_b.name}")

        # Convert MIDIs to text
        midi_a_text = self.midi_to_text(midi_a)
        midi_b_text = self.midi_to_text(midi_b)

        # Create prompt
        prompt = self.create_comparison_prompt(
            midi_a_text, midi_b_text, feature_description
        )

        # Call LLM with retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a music analysis expert specializing in comparing MIDI files.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent analysis
                    response_format={"type": "json_object"},
                )

                result = json.loads(response.choices[0].message.content)

                # Add metadata
                result["midi_a"] = midi_a.name
                result["midi_b"] = midi_b.name
                result["timestamp"] = datetime.now().isoformat()

                return result

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed after {max_retries} attempts")
                    return {
                        "error": str(e),
                        "midi_a": midi_a.name,
                        "midi_b": midi_b.name,
                    }

    def evaluate_intervention_set(self, intervention_dir: Path) -> Dict:
        """Evaluate all interventions in a directory (feature/song combination)."""

        self.logger.info(f"Evaluating intervention set: {intervention_dir}")

        # Find baseline MIDI
        baseline_files = list(intervention_dir.glob("*_baseline.mid"))
        if not baseline_files:
            self.logger.error(f"No baseline file found in {intervention_dir}")
            return {"error": "No baseline file found"}

        baseline_midi = baseline_files[0]
        song_name = baseline_midi.stem.replace("_baseline", "")

        # Find all intervention MIDIs
        intervention_midis = [
            f
            for f in intervention_dir.glob(f"{song_name}_*.mid")
            if "baseline" not in f.stem and "ablation" not in f.stem
        ]

        # Find ablation if exists
        ablation_files = list(intervention_dir.glob("*_ablation.mid"))
        ablation_midi = ablation_files[0] if ablation_files else None

        results = {
            "song_name": song_name,
            "baseline": baseline_midi.name,
            "comparisons": [],
        }

        # Compare each intervention with baseline
        for intervention_midi in intervention_midis:
            # Extract strength from filename
            stem = intervention_midi.stem
            if "_add_" in stem:
                strength_str = stem.split("_add_")[-1]
                strength = float(strength_str)
            else:
                continue

            # Randomize order to avoid bias (A/B positions)
            import random

            if random.random() < 0.5:
                midi_a, midi_b = baseline_midi, intervention_midi
                is_flipped = False
            else:
                midi_a, midi_b = intervention_midi, baseline_midi
                is_flipped = True

            comparison = self.compare_midi_pair(
                midi_a, midi_b, self.get_feature_description(intervention_dir)
            )
            comparison["strength"] = strength
            comparison["intervention_file"] = intervention_midi.name
            comparison["is_flipped"] = is_flipped

            # Interpret result based on flip
            if "choice" in comparison:
                # If not flipped: A=baseline, B=intervention
                # If flipped: A=intervention, B=baseline
                if not is_flipped:
                    comparison["feature_more_present_in"] = (
                        "intervention" if comparison["choice"] == "B" else "baseline"
                    )
                else:
                    comparison["feature_more_present_in"] = (
                        "intervention" if comparison["choice"] == "A" else "baseline"
                    )

            results["comparisons"].append(comparison)

        # Compare ablation with baseline if exists
        if ablation_midi:
            import random

            if random.random() < 0.5:
                midi_a, midi_b = baseline_midi, ablation_midi
                is_flipped = False
            else:
                midi_a, midi_b = ablation_midi, baseline_midi
                is_flipped = True

            comparison = self.compare_midi_pair(
                midi_a, midi_b, self.get_feature_description(intervention_dir)
            )
            comparison["strength"] = "ablation"
            comparison["intervention_file"] = ablation_midi.name
            comparison["is_flipped"] = is_flipped

            if "choice" in comparison:
                if not is_flipped:
                    comparison["feature_more_present_in"] = (
                        "ablation" if comparison["choice"] == "B" else "baseline"
                    )
                else:
                    comparison["feature_more_present_in"] = (
                        "ablation" if comparison["choice"] == "A" else "baseline"
                    )

            results["ablation_comparison"] = comparison

        return results

    def get_feature_description(self, intervention_dir: Path) -> str:
        """Extract feature description from directory path."""
        # Parse layer and feature from path
        # Expected format: .../layerX/featureY/...
        parts = intervention_dir.parts

        layer = None
        feature = None

        for part in parts:
            if part.startswith("layer"):
                layer = part.replace("layer", "")
            elif part.startswith("feature"):
                # Extract feature number from "featureXXX_songYYY" format
                feature = part.split("_")[0].replace("feature", "")

        if layer and feature:
            key = f"{layer}_{feature}"
            if key in FEATURE_DESCRIPTIONS:
                return FEATURE_DESCRIPTIONS[key]["description"]

        return "Musical characteristic (description not found)"

    def evaluate_directory(self, input_dir: Path) -> Dict:
        """Evaluate all intervention sets in a directory."""

        self.logger.info(f"Scanning directory: {input_dir}")

        # Find all feature directories
        feature_dirs = []
        for feature_dir in input_dir.glob("feature*"):
            if feature_dir.is_dir():
                # Find song-specific intervention directories
                for intervention_dir in feature_dir.glob("feature*_song*"):
                    if intervention_dir.is_dir():
                        feature_dirs.append(intervention_dir)

        self.logger.info(f"Found {len(feature_dirs)} intervention sets to evaluate")

        results = {
            "input_dir": str(input_dir),
            "timestamp": datetime.now().isoformat(),
            "total_sets": len(feature_dirs),
            "evaluations": [],
        }

        for i, intervention_dir in enumerate(feature_dirs, 1):
            self.logger.info(
                f"[{i}/{len(feature_dirs)}] Evaluating: {intervention_dir.name}"
            )
            evaluation = self.evaluate_intervention_set(intervention_dir)
            evaluation["intervention_dir"] = str(intervention_dir)
            results["evaluations"].append(evaluation)

            # Small delay to avoid rate limiting
            time.sleep(1)

        return results

    def calculate_accuracy(self, results: Dict) -> Dict:
        """Calculate accuracy metrics from evaluation results."""

        total_comparisons = 0
        correct_additions = 0
        correct_ablations = 0
        total_additions = 0
        total_ablations = 0

        # Handle both single directory and multiple directories
        evaluations_list = []
        if "directories" in results:
            # Multiple directories case
            for directory_result in results["directories"]:
                evaluations_list.extend(directory_result.get("evaluations", []))
        else:
            # Single directory case
            evaluations_list = results.get("evaluations", [])

        for evaluation in evaluations_list:
            # Check addition interventions
            for comparison in evaluation.get("comparisons", []):
                if "feature_more_present_in" not in comparison:
                    continue

                total_comparisons += 1
                total_additions += 1

                strength = comparison.get("strength", 0)
                feature_present_in = comparison["feature_more_present_in"]

                # Positive additions should increase feature presence
                # Negative additions should decrease feature presence
                if strength > 0:
                    if feature_present_in == "intervention":
                        correct_additions += 1
                elif strength < 0:
                    if feature_present_in == "baseline":
                        correct_additions += 1

            # Check ablation
            ablation_comp = evaluation.get("ablation_comparison")
            if ablation_comp and "feature_more_present_in" in ablation_comp:
                total_comparisons += 1
                total_ablations += 1

                # Ablation should have LESS feature presence than baseline
                if ablation_comp["feature_more_present_in"] == "baseline":
                    correct_ablations += 1

        accuracy_metrics = {
            "total_comparisons": total_comparisons,
            "total_additions": total_additions,
            "total_ablations": total_ablations,
            "correct_additions": correct_additions,
            "correct_ablations": correct_ablations,
            "addition_accuracy": (
                correct_additions / total_additions if total_additions > 0 else 0
            ),
            "ablation_accuracy": (
                correct_ablations / total_ablations if total_ablations > 0 else 0
            ),
            "overall_accuracy": (
                (correct_additions + correct_ablations) / total_comparisons
                if total_comparisons > 0
                else 0
            ),
        }

        return accuracy_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contrasting interventions using LLM-based MIDI comparison"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing intervention results (e.g., contrasting_intervention_results_5_1950/layer5)",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        help="Multiple input directories for batch evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSON file for evaluation results",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o, or use gpt-4o-mini for lower cost)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=500,
        help="Maximum number of MIDI events to send per file (default: 500)",
    )

    args = parser.parse_args()

    # Determine input directories
    input_dirs = []
    if args.input_dir:
        input_dirs.append(args.input_dir)
    if args.input_dirs:
        input_dirs.extend(args.input_dirs)

    if not input_dirs:
        print("Error: Must specify --input-dir or --input-dirs")
        sys.exit(1)

    # Validate directories exist
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)

    # Create evaluator
    try:
        evaluator = InterventionEvaluator(
            api_key=args.api_key, model=args.model, max_events=args.max_events
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Evaluate each directory
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "directories": [],
    }

    for input_dir in input_dirs:
        print(f"\n{'='*80}")
        print(f"Evaluating: {input_dir}")
        print(f"{'='*80}\n")

        results = evaluator.evaluate_directory(input_dir)
        all_results["directories"].append(results)

    # Calculate overall accuracy
    all_results["accuracy_metrics"] = evaluator.calculate_accuracy(all_results)

    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_file}")
    print(f"\nAccuracy Metrics:")
    print(
        f"  - Addition accuracy: {all_results['accuracy_metrics']['addition_accuracy']:.1%}"
    )
    print(
        f"  - Ablation accuracy: {all_results['accuracy_metrics']['ablation_accuracy']:.1%}"
    )
    print(
        f"  - Overall accuracy: {all_results['accuracy_metrics']['overall_accuracy']:.1%}"
    )
    print(
        f"  - Total comparisons: {all_results['accuracy_metrics']['total_comparisons']}"
    )


if __name__ == "__main__":
    main()
