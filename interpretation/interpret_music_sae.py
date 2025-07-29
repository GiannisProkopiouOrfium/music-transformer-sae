import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pathlib
import json
import h5py
import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae.analyze_sae import analyze_sae_features


def load_musical_json_data(json_file_path: str) -> dict:
    """Load musical data from JSON file."""
    try:
        with open(json_file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file_path}: {e}")
        return {}


def extract_musical_patterns(musical_data: dict) -> dict:
    """Extract specific, measurable musical patterns for SAE interpretation."""
    patterns = {
        "instrument_sequences": [],
        "pitch_changes": [],
        "duration_patterns": [],
        "velocity_patterns": [],
        "rhythmic_patterns": [],
        "harmonic_patterns": [],
    }

    tracks = musical_data.get("tracks", [])

    # Extract instrument sequences
    if len(tracks) > 1:
        for i in range(len(tracks) - 1):
            curr_instrument = get_instrument_name(tracks[i].get("program", 0))
            next_instrument = get_instrument_name(tracks[i + 1].get("program", 0))
            if curr_instrument != next_instrument:
                patterns["instrument_sequences"].append(
                    f"{curr_instrument} followed by {next_instrument}"
                )

    # Analyze each track for patterns
    for track in tracks:
        notes = track.get("notes", [])
        if len(notes) < 2:
            continue

        instrument_name = get_instrument_name(track.get("program", 0))

        # Extract pitch change patterns
        for i in range(len(notes) - 1):
            pitch_diff = notes[i + 1]["pitch"] - notes[i]["pitch"]
            if abs(pitch_diff) >= 7:  # Large leap
                direction = "ascending" if pitch_diff > 0 else "descending"
                patterns["pitch_changes"].append(
                    f"sudden {direction} leap of {abs(pitch_diff)} semitones in {instrument_name}"
                )
            elif abs(pitch_diff) <= 2 and pitch_diff != 0:  # Stepwise motion
                direction = "ascending" if pitch_diff > 0 else "descending"
                patterns["pitch_changes"].append(
                    f"stepwise {direction} motion in {instrument_name}"
                )

        # Extract duration patterns
        durations = [note["duration"] for note in notes]
        long_notes = [
            d for d in durations if d >= 24
        ]  # Assuming resolution 24 = quarter note
        short_notes = [d for d in durations if d <= 6]  # Very short notes

        if long_notes:
            avg_long = sum(long_notes) / len(long_notes)
            patterns["duration_patterns"].append(
                f"sustained notes (avg duration {avg_long:.1f} ticks) in {instrument_name}"
            )

        if short_notes:
            patterns["duration_patterns"].append(
                f"staccato notes ({len(short_notes)} short notes) in {instrument_name}"
            )

        # Extract velocity patterns
        velocities = [note["velocity"] for note in notes]
        if velocities:
            max_vel = max(velocities)
            min_vel = min(velocities)
            vel_range = max_vel - min_vel

            if vel_range > 40:  # Significant dynamic range
                patterns["velocity_patterns"].append(
                    f"dynamic contrasts (range {vel_range}) in {instrument_name}"
                )
            elif max_vel > 100:
                patterns["velocity_patterns"].append(
                    f"forte passages (velocity > 100) in {instrument_name}"
                )
            elif max_vel < 60:
                patterns["velocity_patterns"].append(
                    f"piano passages (velocity < 60) in {instrument_name}"
                )

        # Extract rhythmic patterns
        if len(notes) > 2:
            onset_intervals = [
                notes[i + 1]["time"] - notes[i]["time"] for i in range(len(notes) - 1)
            ]
            unique_intervals = len(set(onset_intervals))
            total_intervals = len(onset_intervals)

            if unique_intervals / total_intervals < 0.5:  # Regular rhythm
                most_common_interval = max(
                    set(onset_intervals), key=onset_intervals.count
                )
                patterns["rhythmic_patterns"].append(
                    f"regular rhythm (interval {most_common_interval} ticks) in {instrument_name}"
                )
            else:  # Irregular rhythm
                patterns["rhythmic_patterns"].append(
                    f"irregular rhythm ({unique_intervals}/{total_intervals} unique intervals) in {instrument_name}"
                )

    # Extract harmonic patterns (chord analysis)
    resolution = musical_data.get("resolution", 24)
    time_slices = {}

    # Group notes by time windows to find simultaneous notes (chords)
    for track in tracks:
        for note in track.get("notes", []):
            time_window = note["time"] // (resolution // 4)  # Group by 16th notes
            if time_window not in time_slices:
                time_slices[time_window] = []
            time_slices[time_window].append(note["pitch"] % 12)  # Pitch class

    # Find chords (3+ simultaneous pitch classes)
    chord_count = 0
    for pitches in time_slices.values():
        unique_pitches = list(set(pitches))
        if len(unique_pitches) >= 3:
            chord_count += 1

    if chord_count > 0:
        patterns["harmonic_patterns"].append(
            f"chordal texture ({chord_count} chord events)"
        )

    return patterns


def calculate_pattern_interpretability_score(
    pattern_description: str, feature_activations: list, musical_contexts: list
) -> float:
    """Calculate interpretability score (0-1) based on pattern consistency across activations."""

    # Count how many contexts exhibit this pattern
    pattern_count = 0
    total_contexts = len(musical_contexts)

    if total_contexts == 0:
        return 0.0

    for context in musical_contexts:
        patterns = extract_musical_patterns(context)
        all_patterns = []
        for pattern_list in patterns.values():
            all_patterns.extend(pattern_list)

        # Check if any extracted pattern matches the description (fuzzy matching)
        pattern_words = set(pattern_description.lower().split())
        for extracted_pattern in all_patterns:
            extracted_words = set(extracted_pattern.lower().split())
            overlap = len(pattern_words.intersection(extracted_words))
            if overlap >= 2:  # At least 2 words match
                pattern_count += 1
                break

    # Base score on consistency
    consistency_score = pattern_count / total_contexts

    # Adjust score based on activation strength variance
    if feature_activations:
        activations = [act.get("activation_value", 0) for act in feature_activations]
        if len(activations) > 1:
            mean_activation = sum(activations) / len(activations)
            variance = sum((a - mean_activation) ** 2 for a in activations) / len(
                activations
            )
            # Lower variance = more consistent = higher interpretability
            variance_penalty = (
                min(variance / mean_activation, 0.5) if mean_activation > 0 else 0
            )
            consistency_score = max(0, consistency_score - variance_penalty)

    return consistency_score


def generate_pattern_based_interpretation(
    feature_id: int, feature_stats: dict, musical_contexts: list
) -> dict:
    """Generate pattern-based interpretation similar to the text domain example."""

    # Extract patterns from all contexts
    all_patterns = {
        "instrument_sequences": [],
        "pitch_changes": [],
        "duration_patterns": [],
        "velocity_patterns": [],
        "rhythmic_patterns": [],
        "harmonic_patterns": [],
    }

    for context in musical_contexts:
        patterns = extract_musical_patterns(context)
        for pattern_type, pattern_list in patterns.items():
            all_patterns[pattern_type].extend(pattern_list)

    # Find most common patterns
    pattern_counts = {}
    for pattern_type, pattern_list in all_patterns.items():
        for pattern in pattern_list:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Sort by frequency and select top patterns
    top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    if not top_patterns:
        return {
            "description": f"feature {feature_id}: no clear musical pattern detected",
            "interpretability_score": 0.0,
            "pattern_type": "unknown",
            "supporting_contexts": 0,
            "total_contexts": len(musical_contexts),
            "alternative_patterns": [],
        }

    # Use the most frequent pattern as the main description
    main_pattern, frequency = top_patterns[0]

    # Calculate interpretability score
    interpretability_score = calculate_pattern_interpretability_score(
        main_pattern, feature_stats.get("top_activations", []), musical_contexts
    )

    # Determine pattern type
    pattern_type = "unknown"
    for ptype, patterns in all_patterns.items():
        if main_pattern in patterns:
            pattern_type = ptype.replace("_", " ")
            break

    return {
        "description": f"feature {feature_id}: {main_pattern}",
        "interpretability_score": round(interpretability_score, 2),
        "pattern_type": pattern_type,
        "supporting_contexts": frequency,
        "total_contexts": len(musical_contexts),
        "alternative_patterns": [pattern for pattern, _ in top_patterns[1:]],
    }


def analyze_musical_features(musical_data: dict) -> dict:
    analysis = {
        "title": musical_data.get("metadata", {}).get("title", "Unknown"),
        "tempo": None,
        "key_signature": None,
        "time_signature": None,
        "tracks_analysis": [],
        "harmonic_analysis": {},
        "rhythmic_analysis": {},
        "melodic_analysis": {},
        "texture_analysis": {},
    }

    # Extract tempo information
    tempos = musical_data.get("tempos", [])
    if tempos:
        analysis["tempo"] = f"{tempos[0].get('qpm', 'Unknown')} BPM"

    # Extract key signature
    key_sigs = musical_data.get("key_signatures", [])
    if key_sigs:
        root = key_sigs[0].get("root", "Unknown")
        mode = key_sigs[0].get("mode", "Unknown")
        analysis["key_signature"] = f"Root: {root}, Mode: {mode}"

    # Extract time signature
    time_sigs = musical_data.get("time_signatures", [])
    if time_sigs:
        num = time_sigs[0].get("numerator", "Unknown")
        den = time_sigs[0].get("denominator", "Unknown")
        analysis["time_signature"] = f"{num}/{den}"

    # Analyze each track in detail
    tracks = musical_data.get("tracks", [])
    all_pitches = []
    all_durations = []
    all_velocities = []
    all_intervals = []

    for track_idx, track in enumerate(tracks):
        notes = track.get("notes", [])
        if not notes:
            continue

        track_analysis = {
            "track_id": track_idx,
            "program": track.get("program", "Unknown"),
            "instrument_name": get_instrument_name(track.get("program", 0)),
            "note_count": len(notes),
            "pitch_range": {},
            "velocity_profile": {},
            "duration_profile": {},
            "rhythmic_features": {},
            "melodic_features": {},
        }

        # Extract note features
        pitches = [note.get("pitch", 0) for note in notes]
        velocities = [note.get("velocity", 0) for note in notes]
        durations = [note.get("duration", 0) for note in notes]
        times = [note.get("time", 0) for note in notes]

        all_pitches.extend(pitches)
        all_durations.extend(durations)
        all_velocities.extend(velocities)

        if pitches:
            # Pitch analysis
            track_analysis["pitch_range"] = {
                "lowest": min(pitches),
                "highest": max(pitches),
                "span": max(pitches) - min(pitches),
                "most_common": (
                    max(set(pitches), key=pitches.count) if pitches else None
                ),
            }

            # Calculate intervals for melodic analysis
            intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
            all_intervals.extend(intervals)

            track_analysis["melodic_features"] = {
                "avg_interval": sum(intervals) / len(intervals) if intervals else 0,
                "large_leaps": sum(1 for i in intervals if abs(i) > 7),
                "stepwise_motion": sum(1 for i in intervals if abs(i) <= 2),
                "direction_changes": sum(
                    1
                    for i in range(len(intervals) - 1)
                    if intervals[i] * intervals[i + 1] < 0
                ),
            }

        if velocities:
            # Velocity analysis
            track_analysis["velocity_profile"] = {
                "average": sum(velocities) / len(velocities),
                "dynamic_range": max(velocities) - min(velocities),
                "loud_notes": sum(1 for v in velocities if v > 100),
                "soft_notes": sum(1 for v in velocities if v < 60),
            }

        if durations:
            # Duration analysis
            track_analysis["duration_profile"] = {
                "average": sum(durations) / len(durations),
                "shortest": min(durations),
                "longest": max(durations),
                "variation": max(durations) - min(durations),
            }

        if times and len(times) > 1:
            # Rhythmic analysis
            onset_intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
            track_analysis["rhythmic_features"] = {
                "avg_onset_interval": sum(onset_intervals) / len(onset_intervals),
                "rhythmic_regularity": (
                    len(set(onset_intervals)) / len(onset_intervals)
                    if onset_intervals
                    else 1
                ),
                "syncopation_hints": sum(
                    1 for interval in onset_intervals if interval % 12 != 0
                ),  # Assuming 24 resolution
            }

        analysis["tracks_analysis"].append(track_analysis)

    # Overall musical analysis
    if all_pitches:
        # Harmonic analysis
        pitch_classes = [p % 12 for p in all_pitches]
        pitch_class_counts = {pc: pitch_classes.count(pc) for pc in set(pitch_classes)}
        most_common_pcs = sorted(
            pitch_class_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        analysis["harmonic_analysis"] = {
            "pitch_class_distribution": dict(most_common_pcs),
            "tonal_center_hint": most_common_pcs[0][0] if most_common_pcs else None,
            "chromatic_diversity": len(set(pitch_classes)),
            "overall_range": max(all_pitches) - min(all_pitches),
        }

    if all_intervals:
        # Melodic analysis
        analysis["melodic_analysis"] = {
            "avg_interval_size": sum(abs(i) for i in all_intervals)
            / len(all_intervals),
            "predominantly_stepwise": sum(1 for i in all_intervals if abs(i) <= 2)
            / len(all_intervals),
            "leap_frequency": sum(1 for i in all_intervals if abs(i) > 4)
            / len(all_intervals),
        }

    if all_velocities:
        # Texture analysis
        analysis["texture_analysis"] = {
            "dynamic_complexity": (max(all_velocities) - min(all_velocities)) / 127,
            "average_velocity": sum(all_velocities) / len(all_velocities),
            "velocity_variation": (
                len(set(all_velocities)) / len(all_velocities) if all_velocities else 0
            ),
        }

    return analysis


def get_instrument_name(program_number: int) -> str:
    """Convert MIDI program number to instrument name."""
    # General MIDI instrument names (simplified)
    instruments = {
        0: "Acoustic Grand Piano",
        1: "Bright Acoustic Piano",
        2: "Electric Grand Piano",
        3: "Honky-tonk Piano",
        4: "Electric Piano 1",
        5: "Electric Piano 2",
        24: "Acoustic Guitar (nylon)",
        25: "Acoustic Guitar (steel)",
        26: "Electric Guitar (jazz)",
        32: "Acoustic Bass",
        33: "Electric Bass (finger)",
        34: "Electric Bass (pick)",
        40: "Violin",
        41: "Viola",
        42: "Cello",
        43: "Contrabass",
        44: "Tremolo Strings",
        56: "Trumpet",
        57: "Trombone",
        58: "Tuba",
        59: "Muted Trumpet",
        60: "French Horn",
        64: "Soprano Sax",
        65: "Alto Sax",
        66: "Tenor Sax",
        67: "Baritone Sax",
        68: "Oboe",
        72: "Piccolo",
        73: "Flute",
        74: "Recorder",
        75: "Pan Flute",
        76: "Blown Bottle",
    }
    return instruments.get(program_number, f"Program {program_number}")


def create_openai_interpretation_prompt(
    feature_id: int,
    feature_stats: dict,
    top_activations: list,
    musical_contexts: list,
    layer: int = 3,
) -> str:
    """
    Create a comprehensive prompt for OpenAI to interpret SAE features
    combining numerical statistics with detailed musical context.
    """

    prompt = f"""You are a musicologist and machine learning expert analyzing features learned by a Sparse Autoencoder (SAE) applied to a Music Transformer at layer {layer}.

## Feature {feature_id} Analysis

### Statistical Profile:
- Activation frequency: {feature_stats.get('activation_frequency', 0):.1%}
- Mean activation strength: {feature_stats.get('mean_activation', 0):.3f} 
- Maximum activation: {feature_stats.get('max_activation', 0):.3f}
- Total occurrences: {feature_stats.get('total_activations', 0)}

### Top Activation Contexts with Detailed Musical Analysis:
"""

    # Add detailed musical contexts
    for i, (activation_data, musical_context) in enumerate(
        zip(top_activations[:5], musical_contexts[:5])
    ):
        # Analyze musical features in detail
        musical_analysis = analyze_musical_features(musical_context)

        prompt += f"""
#### Context {i+1} (Activation: {activation_data.get('activation_value', 0):.3f}):
**Source:** {musical_analysis.get('title', 'Unknown')} ({musical_context.get('source_file', 'Unknown')})

**Musical Structure:**
- Tempo: {musical_analysis.get('tempo', 'Unknown')}
- Key: {musical_analysis.get('key_signature', 'Unknown')}
- Time Signature: {musical_analysis.get('time_signature', 'Unknown')}

**Track Details:**"""

        for track_analysis in musical_analysis.get("tracks_analysis", [])[
            :3
        ]:  # First 3 tracks
            prompt += f"""
  - Track {track_analysis.get('track_id', 'Unknown')}: {track_analysis.get('instrument_name', 'Unknown')} 
    • Notes: {track_analysis.get('note_count', 0)}
    • Pitch range: {track_analysis.get('pitch_range', {}).get('lowest', 'N/A')}-{track_analysis.get('pitch_range', {}).get('highest', 'N/A')} (span: {track_analysis.get('pitch_range', {}).get('span', 'N/A')})
    • Velocity: avg={track_analysis.get('velocity_profile', {}).get('average', 0):.0f}, range={track_analysis.get('velocity_profile', {}).get('dynamic_range', 0)}
    • Rhythm: avg_onset={track_analysis.get('rhythmic_features', {}).get('avg_onset_interval', 0):.1f}, regularity={track_analysis.get('rhythmic_features', {}).get('rhythmic_regularity', 0):.2f}
    • Melody: avg_interval={track_analysis.get('melodic_features', {}).get('avg_interval', 0):.1f}, stepwise={track_analysis.get('melodic_features', {}).get('stepwise_motion', 0)}, leaps={track_analysis.get('melodic_features', {}).get('large_leaps', 0)}"""

        prompt += f"""

**Overall Musical Characteristics:**
- Harmonic: Tonal center hint: {musical_analysis.get('harmonic_analysis', {}).get('tonal_center_hint', 'N/A')}, Chromatic diversity: {musical_analysis.get('harmonic_analysis', {}).get('chromatic_diversity', 'N/A')}/12
- Melodic: Avg interval size: {musical_analysis.get('melodic_analysis', {}).get('avg_interval_size', 0):.1f}, Stepwise motion: {musical_analysis.get('melodic_analysis', {}).get('predominantly_stepwise', 0):.1%}
- Texture: Dynamic complexity: {musical_analysis.get('texture_analysis', {}).get('dynamic_complexity', 0):.2f}, Velocity variation: {musical_analysis.get('texture_analysis', {}).get('velocity_variation', 0):.2f}

**Raw Activation Vector Sample:** {str(activation_data.get('original_input', []))[:100]}...
"""

    prompt += """

## Analysis Request:

Based on this detailed musical and statistical data, provide a specific interpretation:

1. **FEATURE INTERPRETATION** (2-3 sentences): What specific musical concept, pattern, or structure does this feature detect? Consider:
   - Specific harmonic patterns (chord progressions, intervals, voice leading)
   - Rhythmic patterns (syncopation, meter, note durations)
   - Melodic patterns (scales, motifs, contour)
   - Textural elements (instrument combinations, dynamics)
   - Structural elements (phrase boundaries, cadences)

2. **MUSICAL EVIDENCE** (bullet points): Cite specific numerical evidence from the musical analysis that supports your interpretation.

3. **PREDICTIVE SCORE** (0-100): How confident are you that this interpretation would predict other activations? Base this on:
   - Consistency of musical patterns across examples
   - Specificity and measurability of the pattern
   - Musical theory plausibility

4. **ALTERNATIVE HYPOTHESES** (1-2 alternatives): What other specific musical patterns might this feature represent?

Format your response as JSON:
```json
{
  "interpretation": "Specific interpretation based on musical evidence",
  "evidence": ["Specific musical evidence 1", "Specific musical evidence 2", "Specific musical evidence 3"],
  "confidence_score": 85,
  "alternative_hypotheses": ["Specific alternative 1", "Specific alternative 2"],
  "musical_category": "harmony|rhythm|melody|texture|form|style|dynamics|timbre"
}
```"""

    return prompt


def call_openai_for_interpretation(
    prompt: str, api_key: str, model: str = "gpt-4o-mini"
) -> dict:
    """Call OpenAI API to get feature interpretation."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert musicologist and machine learning researcher specializing in interpretability of music AI systems.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000,
        )

        # Extract the JSON response
        content = response.choices[0].message.content

        # Try to parse JSON from the response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content.strip()

        return json.loads(json_str)

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "interpretation": f"Error in interpretation for feature",
            "evidence": ["API call failed"],
            "confidence_score": 0,
            "alternative_hypotheses": ["Manual analysis needed"],
            "musical_category": "error",
        }


def run_pattern_based_interpretation_pipeline(
    model_path: str,
    activations_path: str,
    data_dir: str,
    output_dir: str = "interpretation/pattern_analysis",
    max_features: int = 20,
    device: str = None,
    subsample: int = 50000,
):
    """
    Pattern-based SAE feature interpretation pipeline.
    Generates structured interpretations similar to text domain examples.
    """

    print("🎼 PATTERN-BASED SAE INTERPRETATION PIPELINE 🎼")
    print("=" * 60)

    # Determine device for GPU acceleration
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device for interpretation: {device}")

    # Step 1: Extract feature-input mappings
    print("Step 1: Extracting feature activations...")
    _, feature_summaries = save_feature_activations_for_interpretation(
        model_path=model_path,
        activations_path=activations_path,
        output_dir=f"{output_dir}/feature_data",
        max_samples_per_feature=10,
        min_activation_threshold=0.1,
        device=device,  # Pass device for GPU acceleration
        subsample=subsample,  # Pass subsample parameter
    )

    # Step 2: Load original musical data
    print("Step 2: Loading musical contexts...")
    data_path = pathlib.Path(data_dir)

    # Sort features by activation frequency for analysis, but also prioritize diversity
    sorted_features = sorted(
        feature_summaries.items(),
        key=lambda x: x[1]["activation_frequency"],
        reverse=True,
    )

    # Filter to get more diverse features by checking source file diversity
    diverse_features = []
    seen_source_files = set()
    grouped_by_source = {}  # Group features by their primary source file

    # First, group features by their most common source file
    for feature_id, feature_stats in sorted_features:
        # Find the most common source file for this feature
        source_file_counts = {}
        for activation in feature_stats.get("top_activations", [])[:10]:
            if "source_file" in activation:
                source_file = activation["source_file"]
                source_file_counts[source_file] = (
                    source_file_counts.get(source_file, 0) + 1
                )

        if source_file_counts:
            primary_source = max(source_file_counts.items(), key=lambda x: x[1])[0]
            if primary_source not in grouped_by_source:
                grouped_by_source[primary_source] = []
            grouped_by_source[primary_source].append((feature_id, feature_stats))

    print(
        f"Found features from {len(grouped_by_source)} different primary source files:"
    )
    for source_file, features in list(grouped_by_source.items())[:5]:
        print(f"  {source_file}: {len(features)} features")

    # Select one representative feature from each source file, plus some extras
    for source_file, features in grouped_by_source.items():
        if len(diverse_features) < max_features:
            # Take the highest activation frequency feature from this source
            best_feature = max(features, key=lambda x: x[1]["activation_frequency"])
            diverse_features.append(best_feature)
            seen_source_files.add(source_file)

    # If we still don't have enough, add more features from high-frequency sources
    if len(diverse_features) < max_features:
        for source_file, features in grouped_by_source.items():
            for feature_id, feature_stats in features:
                if (feature_id, feature_stats) not in diverse_features and len(
                    diverse_features
                ) < max_features:
                    diverse_features.append((feature_id, feature_stats))

    selected_features = diverse_features[:max_features]
    print(
        f"Selected {len(selected_features)} diverse features from {len(set(seen_source_files))} unique sources"
    )

    # Step 3: Generate pattern-based interpretations
    interpretations = []
    processed_feature_contexts = {}  # Track contexts to avoid similar features

    for feature_id, feature_stats in selected_features:
        print(f"\n🎵 Analyzing Feature {feature_id}...")
        print(
            f"   📊 Stats: {feature_stats['total_activations']} total activations, "
            f"freq={feature_stats['activation_frequency']:.1%}, "
            f"max_act={feature_stats['max_activation']:.3f}"
        )

        # Load musical contexts for this feature's top activations
        musical_contexts = []
        unique_source_files = set()  # Track unique files to ensure diversity

        # Debug: Check what activation data contains
        print(f"   🔍 Debug - Top activation sample:")
        if feature_stats["top_activations"]:
            sample_activation = feature_stats["top_activations"][0]
            print(f"      Keys: {list(sample_activation.keys())}")
            if "source_file" in sample_activation:
                print(f"      Source file: {sample_activation['source_file']}")
            else:
                print(f"      No source_file found in activation data")

        # Before processing, check if this feature is too similar to already processed ones
        feature_activation_files = set()
        feature_tokens = []

        # Collect this feature's activation files and tokens
        for activation in feature_stats["top_activations"][:10]:
            if "source_file" in activation:
                feature_activation_files.add(activation["source_file"])
            if "token_idx" in activation:
                feature_tokens.append(activation["token_idx"])

        # Check similarity with previously processed features
        skip_feature = False
        for prev_feature_id, prev_data in processed_feature_contexts.items():
            prev_files = prev_data["files"]
            prev_tokens = prev_data["tokens"]

            # Calculate file overlap
            file_overlap = len(feature_activation_files & prev_files) / max(
                len(feature_activation_files | prev_files), 1
            )

            # Calculate token position similarity (if same files)
            token_similarity = 0
            if feature_activation_files & prev_files:  # If there are common files
                common_tokens = set(feature_tokens) & set(prev_tokens)
                token_similarity = len(common_tokens) / max(
                    len(set(feature_tokens) | set(prev_tokens)), 1
                )

            # Skip if too similar
            if file_overlap > 0.8 and token_similarity > 0.5:
                print(
                    f"   🔄 Skipping Feature {feature_id} (too similar to Feature {prev_feature_id})"
                )
                print(
                    f"       File overlap: {file_overlap:.2f}, Token similarity: {token_similarity:.2f}"
                )
                skip_feature = True
                break

        if skip_feature:
            continue

        # Store this feature's context for future similarity checks
        processed_feature_contexts[feature_id] = {
            "files": feature_activation_files,
            "tokens": feature_tokens,
        }

        # Try to find corresponding JSON files with diversity
        for activation_data in feature_stats["top_activations"][
            :20
        ]:  # Look at more activations to find diversity
            # Try to find the original JSON file
            if "source_file" in activation_data:
                source_file = activation_data["source_file"]

                # Skip if we already have this file (unless we have < 2 contexts)
                if source_file in unique_source_files and len(musical_contexts) >= 2:
                    continue

                # Try multiple possible paths for the JSON file
                possible_paths = [
                    data_path / source_file,  # Direct path
                    data_path / f"{source_file}.json",  # Add .json extension
                    data_path
                    / "json"
                    / f"{source_file}.json",  # In json subdirectory with extension
                    data_path / "json" / source_file,  # In json subdirectory
                ]

                json_file_path = None
                for path in possible_paths:
                    if path.exists():
                        json_file_path = path
                        break

                if json_file_path:
                    musical_data = load_musical_json_data(str(json_file_path))
                    if musical_data.get("tracks"):  # Only add if it has valid tracks
                        # Add specific context about this activation
                        musical_data["source_file"] = source_file
                        musical_data["activation_token_position"] = activation_data.get(
                            "token_position", 0
                        )
                        musical_data["activation_sequence_length"] = (
                            activation_data.get("sequence_length", 0)
                        )
                        musical_data["activation_value"] = activation_data.get(
                            "activation_value", 0
                        )

                        musical_contexts.append(musical_data)
                        unique_source_files.add(source_file)
                        print(
                            f"      ✅ Loaded: {source_file} from {json_file_path.name} (token {activation_data.get('token_position', '?')})"
                        )

                        # Stop if we have enough diverse contexts
                        if len(musical_contexts) >= 5:
                            break
                else:
                    if (
                        len(musical_contexts) < 2
                    ):  # Only show errors if we don't have enough contexts
                        print(
                            f"      ❌ File not found: {source_file} (tried {len(possible_paths)} paths)"
                        )
            else:
                if len(musical_contexts) < 2:
                    print(
                        f"      ⚠️  No source_file in activation {activation_data.get('input_index', 'N/A')}"
                    )

        print(
            f"   📈 Musical contexts loaded: {len(musical_contexts)} ({len(unique_source_files)} unique files)"
        )

        # Fallback: If no source files, try to load some sample JSON files to test pattern extraction
        if not musical_contexts:
            print("   🔄 Fallback: Loading sample JSON files for pattern testing...")
            sample_json_files = list(data_path.glob("**/*.json"))[
                :5
            ]  # Get first 5 JSON files
            for json_file in sample_json_files:
                musical_data = load_musical_json_data(str(json_file))
                if musical_data.get("tracks"):  # Only add if it has valid tracks
                    musical_data["source_file"] = str(json_file.relative_to(data_path))
                    musical_contexts.append(musical_data)
                    print(f"      📁 Fallback loaded: {json_file.name}")

        # Show a sample of what we extracted if we have contexts
        if musical_contexts:
            sample_context = musical_contexts[0]
            tracks = sample_context.get("tracks", [])
            total_notes = sum(len(track.get("notes", [])) for track in tracks)
            print(
                f"   🎼 Sample context: {len(tracks)} tracks, {total_notes} total notes"
            )
            if "activation_token_position" in sample_context:
                print(
                    f"       Activation at token position {sample_context['activation_token_position']}/{sample_context['activation_sequence_length']}"
                )

            # Test pattern extraction on the sample
            test_patterns = extract_musical_patterns(sample_context)
            total_patterns = sum(len(patterns) for patterns in test_patterns.values())
            print(
                f"   🎯 Patterns extracted from sample: {total_patterns} total patterns"
            )
            if total_patterns > 0:
                for pattern_type, patterns in test_patterns.items():
                    if patterns:
                        print(f"      - {pattern_type}: {len(patterns)} patterns")
                        if len(patterns) > 0:
                            print(f"        Example: {patterns[0][:80]}...")

        # Generate pattern-based interpretation
        interpretation = generate_pattern_based_interpretation(
            feature_id=feature_id,
            feature_stats=feature_stats,
            musical_contexts=musical_contexts,
        )

        interpretations.append(interpretation)

        print(f"   Pattern: {interpretation['description']}")
        print(f"   Interpretability: {interpretation['interpretability_score']}")
        print(
            f"   Support: {interpretation['supporting_contexts']}/{len(musical_contexts)} contexts"
        )

    # Step 4: Create structured output like text domain example
    create_structured_interpretation_table(interpretations, output_dir)

    print(f"\n✅ PATTERN-BASED INTERPRETATION COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")

    return interpretations


def create_structured_interpretation_table(interpretations: list, output_dir: str):
    """Create structured output table similar to the text domain example."""

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create markdown table
    table_file = output_path / "feature_interpretations_table.md"

    with open(table_file, "w") as f:
        f.write("# 🎼 Music SAE Feature Interpretations\n\n")
        f.write("| Feature | Description | Interpretability Score |\n")
        f.write("|---------|-------------|----------------------|\n")

        for interp in interpretations:
            feature_desc = interp["description"]
            score = interp["interpretability_score"]
            f.write(f"| {feature_desc} | {score} |\n")

    # Create CSV for analysis
    csv_file = output_path / "feature_interpretations.csv"

    with open(csv_file, "w") as f:
        f.write(
            "feature_id,description,interpretability_score,pattern_type,supporting_contexts,total_contexts\n"
        )

        for interp in interpretations:
            # Extract feature ID from description
            feature_id = interp["description"].split(":")[0].replace("feature ", "")
            desc = (
                interp["description"].split(": ", 1)[1]
                if ": " in interp["description"]
                else interp["description"]
            )
            f.write(
                f"{feature_id},\"{desc}\",{interp['interpretability_score']},{interp['pattern_type']},{interp['supporting_contexts']},{interp['total_contexts']}\n"
            )

    # Create detailed JSON report
    json_file = output_path / "detailed_interpretations.json"

    with open(json_file, "w") as f:
        json.dump(
            {
                "interpretations": interpretations,
                "summary": {
                    "total_features": len(interpretations),
                    "avg_interpretability": (
                        sum(i["interpretability_score"] for i in interpretations)
                        / len(interpretations)
                        if interpretations
                        else 0
                    ),
                    "pattern_types": list(
                        set(i["pattern_type"] for i in interpretations)
                    ),
                },
            },
            f,
            indent=2,
        )

    # Print console table
    print(f"\n📊 FEATURE INTERPRETATION TABLE:")
    print("=" * 80)
    print(f"{'Feature':<15} {'Description':<50} {'Score':<8}")
    print("-" * 80)

    for interp in interpretations:
        feature_desc = interp["description"]
        score = interp["interpretability_score"]
        # Truncate description if too long
        if len(feature_desc) > 50:
            feature_desc = feature_desc[:47] + "..."
        print(f"{feature_desc:<65} {score:<8}")

    print(f"\n📁 Saved structured outputs:")
    print(f"   - Markdown table: {table_file}")
    print(f"   - CSV data: {csv_file}")
    print(f"   - Detailed JSON: {json_file}")


def run_openai_sae_interpretation_pipeline(
    model_path: str,
    activations_path: str,
    data_dir: str,
    openai_api_key: str,
    output_dir: str = "interpretation/openai_analysis",
    max_features: int = 20,
    model_name: str = "gpt-4o-mini",
):
    """
    Complete pipeline for OpenAI-based SAE feature interpretation.
    Combines activation analysis with original musical context.
    """

    print("🤖 OPENAI SAE INTERPRETATION PIPELINE 🤖")
    print("=" * 60)

    # Step 1: Extract feature-input mappings
    print("Step 1: Extracting feature activations...")
    feature_data_dir, feature_summaries = save_feature_activations_for_interpretation(
        model_path=model_path,
        activations_path=activations_path,
        output_dir=f"{output_dir}/feature_data",
        max_samples_per_feature=10,
        min_activation_threshold=0.1,
    )

    # Step 2: Load original musical data
    print("Step 2: Loading musical contexts...")
    data_path = pathlib.Path(data_dir)

    # Sort features by activation frequency for analysis
    sorted_features = sorted(
        feature_summaries.items(),
        key=lambda x: x[1]["activation_frequency"],
        reverse=True,
    )[:max_features]

    print(f"Analyzing top {len(sorted_features)} features...")

    # Step 3: Run OpenAI interpretation for each feature
    interpretations = {}

    for feature_id, feature_stats in sorted_features:
        print(f"\n🎵 Analyzing Feature {feature_id}...")

        # Load musical contexts for this feature's top activations
        musical_contexts = []
        for activation_data in feature_stats["top_activations"][:5]:
            # Try to find the original JSON file
            if "source_file" in activation_data:
                source_file = activation_data["source_file"]
                json_file_path = data_path / source_file
                if json_file_path.exists():
                    musical_data = load_musical_json_data(str(json_file_path))
                    musical_data["source_file"] = source_file

                    # Add track summary for context
                    if "tracks" in musical_data:
                        track_info = []
                        for i, track in enumerate(
                            musical_data.get("tracks", [])[:3]
                        ):  # First 3 tracks
                            notes = track.get("notes", [])
                            track_info.append(f"Track {i}: {len(notes)} notes")
                        musical_data["track_info"] = ", ".join(track_info)

                    musical_contexts.append(musical_data)
                else:
                    # Fallback: create basic context from activation data
                    musical_contexts.append(
                        {
                            "source_file": source_file,
                            "tracks": [],
                            "resolution": "unknown",
                            "track_info": "Source file not found",
                        }
                    )
            else:
                musical_contexts.append(
                    {
                        "source_file": "unknown",
                        "tracks": [],
                        "resolution": "unknown",
                        "track_info": "No source file information",
                    }
                )

        # Create prompt
        prompt = create_openai_interpretation_prompt(
            feature_id=feature_id,
            feature_stats=feature_stats,
            top_activations=feature_stats["top_activations"],
            musical_contexts=musical_contexts,
        )

        # Get OpenAI interpretation
        interpretation = call_openai_for_interpretation(
            prompt=prompt, api_key=openai_api_key, model=model_name
        )

        # Add metadata
        interpretation["feature_id"] = feature_id
        interpretation["feature_stats"] = feature_stats
        interpretation["musical_contexts"] = musical_contexts[
            :3
        ]  # Keep sample contexts

        interpretations[feature_id] = interpretation

        print(f"   Interpretation: {interpretation.get('interpretation', 'Failed')}")
        print(f"   Confidence: {interpretation.get('confidence_score', 0)}/100")
        print(f"   Category: {interpretation.get('musical_category', 'unknown')}")

    # Step 4: Save comprehensive analysis
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual interpretations
    for feature_id, interpretation in interpretations.items():
        interpretation_file = (
            output_path / f"feature_{feature_id:04d}_interpretation.json"
        )
        with open(interpretation_file, "w") as f:
            json.dump(interpretation, f, indent=2)

    # Save summary report
    summary_report = {
        "pipeline_info": {
            "model_path": model_path,
            "activations_path": activations_path,
            "data_dir": data_dir,
            "openai_model": model_name,
            "features_analyzed": len(interpretations),
            "timestamp": str(pathlib.Path().resolve()),
        },
        "interpretations_summary": {
            feature_id: {
                "interpretation": interp.get("interpretation", ""),
                "confidence": interp.get("confidence_score", 0),
                "category": interp.get("musical_category", "unknown"),
            }
            for feature_id, interp in interpretations.items()
        },
        "category_distribution": {},
    }

    # Calculate category distribution
    categories = [
        interp.get("musical_category", "unknown") for interp in interpretations.values()
    ]
    for category in set(categories):
        summary_report["category_distribution"][category] = categories.count(category)

    summary_file = output_path / "interpretation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Create human-readable report
    create_human_readable_report(
        interpretations, output_path / "interpretation_report.md"
    )

    print("\n✅ OPENAI INTERPRETATION COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Summary: {summary_file}")
    print(f"📄 Report: {output_path / 'interpretation_report.md'}")

    return interpretations, summary_report


def create_human_readable_report(interpretations: dict, output_file: pathlib.Path):
    """Create a human-readable markdown report of the interpretations."""

    with open(output_file, "w") as f:
        f.write("# 🎵 Music SAE Feature Interpretation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"Total features analyzed: {len(interpretations)}\n\n")

        # Category breakdown
        categories = [
            interp.get("musical_category", "unknown")
            for interp in interpretations.values()
        ]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}

        f.write("### Feature Categories:\n")
        for category, count in sorted(category_counts.items()):
            f.write(f"- **{category.title()}**: {count} features\n")
        f.write("\n")

        # High confidence interpretations
        high_conf = [
            (fid, interp)
            for fid, interp in interpretations.items()
            if interp.get("confidence_score", 0) >= 70
        ]

        f.write(
            f"### High Confidence Interpretations ({len(high_conf)} features with 70%+ confidence):\n\n"
        )

        for feature_id, interpretation in sorted(
            high_conf, key=lambda x: x[1].get("confidence_score", 0), reverse=True
        ):
            f.write(
                f"#### Feature {feature_id} ({interpretation.get('confidence_score', 0)}% confidence)\n"
            )
            f.write(
                f"**Category**: {interpretation.get('musical_category', 'unknown').title()}\n\n"
            )
            f.write(
                f"**Interpretation**: {interpretation.get('interpretation', 'No interpretation')}\n\n"
            )

            evidence = interpretation.get("evidence", [])
            if evidence:
                f.write("**Supporting Evidence**:\n")
                for point in evidence:
                    f.write(f"- {point}\n")
                f.write("\n")

            alternatives = interpretation.get("alternative_hypotheses", [])
            if alternatives:
                f.write("**Alternative Hypotheses**:\n")
                for alt in alternatives:
                    f.write(f"- {alt}\n")
                f.write("\n")

            f.write("---\n\n")

        # All interpretations
        f.write("## Complete Analysis\n\n")

        for feature_id, interpretation in sorted(interpretations.items()):
            f.write(f"### Feature {feature_id}\n")
            f.write(f"- **Confidence**: {interpretation.get('confidence_score', 0)}%\n")
            f.write(
                f"- **Category**: {interpretation.get('musical_category', 'unknown')}\n"
            )
            f.write(
                f"- **Interpretation**: {interpretation.get('interpretation', 'No interpretation')}\n\n"
            )

    print(f"📄 Human-readable report saved to: {output_file}")


def save_feature_activations_for_interpretation(
    model_path: str,
    activations_path: str,
    output_dir: str = "interpretation/feature_data",
    max_samples_per_feature: int = 20,
    min_activation_threshold: float = 0.1,
    device: str = None,
    subsample: int = 50000,
):
    """
    Save input-activation mappings for each SAE feature for LLM interpretation.

    This function:
    1. Loads the trained SAE model
    2. Processes activations to find which inputs strongly activate each feature
    3. Saves mappings between original musical inputs and SAE feature activations
    4. Prepares data for LLM-based auto-interpretation
    """
    print("🎵 EXTRACTING FEATURE-INPUT MAPPINGS FOR INTERPRETATION 🎵")
    print("=" * 70)

    # Determine device - prioritize GPU for faster processing
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load trained SAE model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Recreate model architecture
    from sae.train_sae import SparseAutoencoder

    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        sparsity_coeff=checkpoint["config"]["sparsity_coeff"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)  # Move model to GPU
    model.eval()

    print(
        f"✅ Loaded SAE model: {checkpoint['input_dim']} -> {checkpoint['config']['hidden_dim']}"
    )

    # Load activations data
    from sae.sae_data import create_sae_dataloader

    layer_key = checkpoint.get("layer_key", "layer_3")

    dataloader, data_info = create_sae_dataloader(
        activations_path,
        layer_key=layer_key,
        batch_size=(
            4096 if device == "cuda" else 1024
        ),  # Larger GPU batch for faster processing
        shuffle=False,
        normalize=True,
        subsample=subsample,  # Use configurable subsample size
        num_workers=4,  # System recommended max workers
    )

    print(f"✅ Loaded activations from layer: {layer_key}")
    print(f"   Data shape: {data_info['shape']}")

    # Load original musical metadata if available
    musical_metadata = load_musical_metadata(activations_path, layer_key)

    # Process all data to find feature activations
    feature_activations = defaultdict(
        list
    )  # feature_id -> list of (input_idx, activation_value, metadata)

    print("🔍 Processing activations to find feature triggers...")

    # Enable memory efficient processing for GPU
    torch.backends.cudnn.benchmark = True if device == "cuda" else False

    # Clear GPU cache if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
        print(
            f"🧠 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total"
        )

    # Use automatic mixed precision for faster GPU processing
    use_amp = device == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        input_idx = 0
        total_batches = len(dataloader)
        print(
            f"Processing {total_batches} batches with batch_size={dataloader.batch_size}"
        )

        for batch_idx, batch in enumerate(dataloader):
            # Immediate progress feedback to show script is not stuck
            if batch_idx == 0:
                print(
                    f"   ✅ Started processing first batch (size: {batch.size(0)})..."
                )

            # Move data to device for GPU acceleration
            batch = batch.to(
                device, non_blocking=True
            )  # Non-blocking for better performance

            # Forward pass through SAE with mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, hidden = model(batch)
            else:
                _, hidden = model(batch)

            # Show progress immediately after first batch
            if batch_idx == 0:
                print(f"   ✅ Completed SAE forward pass for first batch")

            # Move results back to CPU for processing to save GPU memory
            hidden = hidden.cpu()
            batch = batch.cpu()

            # Vectorized processing - find all activated features across the batch
            batch_activated_mask = (
                hidden > min_activation_threshold
            )  # Shape: [batch_size, hidden_dim]

            # Much more efficient batch processing - avoid nested loops
            if torch.any(
                batch_activated_mask
            ):  # Only process if there are any activations
                # Get all activated positions at once
                batch_indices, feature_indices = torch.where(batch_activated_mask)

                # Process all activations in this batch efficiently
                for i in range(len(batch_indices)):
                    sample_idx_in_batch = batch_indices[i].item()
                    feature_id = feature_indices[i].item()
                    activation_value = hidden[sample_idx_in_batch, feature_id].item()

                    current_input_idx = input_idx + sample_idx_in_batch

                    # Minimal metadata for faster processing
                    metadata = {
                        "input_index": current_input_idx,
                        "activation_value": activation_value,
                    }

                    # Only add expensive operations if really needed
                    if musical_metadata and current_input_idx in musical_metadata:
                        metadata.update(musical_metadata[current_input_idx])

                    feature_activations[feature_id].append(metadata)

            # Update input index for the entire batch
            input_idx += batch.size(0)

            if batch_idx % 1 == 0:  # Report every single batch to show progress
                progress_pct = (batch_idx + 1) / total_batches * 100
                features_found = len(feature_activations)
                print(
                    f"   Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) - Found {features_found} active features"
                )

                # Show GPU utilization if available
                if device == "cuda":  # Show GPU stats every batch
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(
                        f"   GPU Memory: {gpu_memory_used:.1f}GB used, {gpu_memory_cached:.1f}GB cached"
                    )

    print(f"✅ Found activations for {len(feature_activations)} features")

    # Save feature activation data for interpretation
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort and save top activations for each feature
    feature_summaries = {}

    for feature_id, activations in feature_activations.items():
        # Sort by activation value (strongest first)
        activations.sort(key=lambda x: x["activation_value"], reverse=True)

        # Keep top activations
        top_activations = activations[:max_samples_per_feature]

        feature_summaries[feature_id] = {
            "feature_id": feature_id,
            "total_activations": len(activations),
            "mean_activation": np.mean([a["activation_value"] for a in activations]),
            "max_activation": max([a["activation_value"] for a in activations]),
            "top_activations": top_activations,
            "activation_frequency": (
                len(activations) / input_idx if input_idx > 0 else 0
            ),
        }

    # Save individual feature files for LLM processing
    for feature_id, summary in feature_summaries.items():
        feature_file = output_path / f"feature_{feature_id:04d}.json"
        with open(feature_file, "w") as f:
            json.dump(summary, f, indent=2)

    # Save overall summary
    overall_summary = {
        "model_info": {
            "model_path": model_path,
            "activations_path": activations_path,
            "layer_key": layer_key,
            "input_dim": checkpoint["input_dim"],
            "hidden_dim": checkpoint["config"]["hidden_dim"],
            "sparsity_coeff": checkpoint["config"]["sparsity_coeff"],
            "final_sparsity": checkpoint.get("final_sparsity", "unknown"),
        },
        "extraction_info": {
            "total_samples_processed": input_idx,
            "min_activation_threshold": min_activation_threshold,
            "max_samples_per_feature": max_samples_per_feature,
            "features_with_activations": len(feature_summaries),
        },
        "feature_statistics": {
            "most_frequent_feature": (
                max(
                    feature_summaries.keys(),
                    key=lambda x: feature_summaries[x]["activation_frequency"],
                )
                if feature_summaries
                else None
            ),
            "highest_activation_feature": (
                max(
                    feature_summaries.keys(),
                    key=lambda x: feature_summaries[x]["max_activation"],
                )
                if feature_summaries
                else None
            ),
        },
    }

    summary_file = output_path / "interpretation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"✅ Saved feature interpretation data to: {output_path}")
    print(f"   - {len(feature_summaries)} feature files")
    print(f"   - Overall summary: {summary_file}")
    print()
    print("🤖 Ready for LLM auto-interpretation!")
    print(f"   Use the files in {output_path} for feature interpretation")

    return output_path, feature_summaries


def load_musical_metadata(activations_path: str, layer_key: str = "layer_3"):
    """Load musical metadata from the activations file if available."""
    try:
        with h5py.File(activations_path, "r") as f:
            # Try to load metadata for the specific layer
            if "metadata" in f and layer_key in f["metadata"]:
                metadata_group = f["metadata"][layer_key]

                # Load the metadata arrays
                source_files = metadata_group["source_files"][:]
                sequence_lengths = metadata_group["sequence_lengths"][:]
                batch_starts = metadata_group["batch_starts"][:]

                # Convert bytes to strings if needed
                if isinstance(source_files[0], bytes):
                    source_files = [f.decode("utf-8") for f in source_files]

                print(
                    f"✅ Loaded metadata: {len(source_files)} files, {sum(sequence_lengths)} tokens"
                )

                # Create a mapping from token index to source file
                token_to_source = {}
                token_index = 0

                for i, (source_file, seq_len) in enumerate(
                    zip(source_files, sequence_lengths)
                ):
                    for token_pos in range(int(seq_len)):
                        token_to_source[token_index] = {
                            "source_file": source_file,
                            "sample_index": int(i),
                            "token_position": int(token_pos),
                            "sequence_length": int(seq_len),
                        }
                        token_index += 1

                return token_to_source

    except Exception as e:
        print(f"Note: Could not load musical metadata: {e}")

    return None


def interpret_music_features(results, activations_path, top_k=10):
    """Interpret what the SAE features might represent in musical terms."""

    model = results["model"]
    feature_frequencies = results["feature_frequencies"]
    hidden_acts = results["hidden_activations"]
    interesting_features = results["interesting_features"]

    print("🎵 INTERPRETING MUSIC SAE FEATURES 🎵")
    print("=" * 60)

    # Load original data to understand what triggers each feature
    from sae.sae_data import create_sae_dataloader

    dataloader, data_info = create_sae_dataloader(
        activations_path, batch_size=64, shuffle=False, subsample=1000
    )

    # Get a representative sample
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        _, sample_hidden = model(sample_batch)

    print(f"Analyzing {len(interesting_features)} interesting features...")

    # Analysis 1: Find features with extreme activations
    print("\n🔥 TOP FEATURES BY ACTIVATION PATTERNS:")
    print("-" * 40)

    # Most selective features (activate rarely but strongly)
    feature_selectivity = torch.std(hidden_acts, dim=0) / (
        torch.mean(hidden_acts, dim=0) + 1e-8
    )
    most_selective = torch.topk(feature_selectivity, top_k).indices

    print("Most Selective Features (high variance/mean ratio):")
    for i, feat_idx in enumerate(most_selective):
        freq = feature_frequencies[feat_idx].item()
        selectivity = feature_selectivity[feat_idx].item()
        mean_act = torch.mean(hidden_acts[:, feat_idx]).item()
        max_act = torch.max(hidden_acts[:, feat_idx]).item()
        print(
            f"  Feature {feat_idx:4d}: {freq:.1%} active, selectivity={selectivity:.2f}, max_act={max_act:.2f}"
        )

    # Analysis 2: Feature clustering by activation patterns
    print(f"\n🎼 FEATURE ACTIVATION CLUSTERS:")
    print("-" * 40)

    # Sample a subset of features for clustering analysis
    n_sample_features = min(100, len(interesting_features))
    sample_features = interesting_features[
        torch.randperm(len(interesting_features))[:n_sample_features]
    ]

    # Compute pairwise correlations
    feature_correlations = torch.corrcoef(hidden_acts[:, sample_features].T)

    # Find highly correlated feature pairs (potential feature families)
    correlation_threshold = 0.3
    high_corr_pairs = []

    for i in range(feature_correlations.shape[0]):
        for j in range(i + 1, feature_correlations.shape[1]):
            corr = feature_correlations[i, j].item()
            if abs(corr) > correlation_threshold:
                feat_i = sample_features[i].item()
                feat_j = sample_features[j].item()
                high_corr_pairs.append((feat_i, feat_j, corr))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(
        f"Found {len(high_corr_pairs)} highly correlated feature pairs (|corr| > {correlation_threshold}):"
    )
    for feat_i, feat_j, corr in high_corr_pairs[:10]:
        print(f"  Features {feat_i:4d} ↔ {feat_j:4d}: correlation = {corr:+.3f}")

    # Analysis 3: Feature activation statistics
    print(f"\n📊 FEATURE STATISTICS:")
    print("-" * 40)

    # Distribution of activation frequencies
    freq_bins = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("Feature activation frequency distribution:")

    prev_threshold = 0.0
    for threshold in freq_bins:
        count = torch.sum(
            (feature_frequencies > prev_threshold) & (feature_frequencies <= threshold)
        ).item()
        print(f"  {prev_threshold:.1%} - {threshold:.1%}: {count:4d} features")
        prev_threshold = threshold

    count = torch.sum(feature_frequencies > prev_threshold).item()
    print(f"  >{prev_threshold:.1%}: {count:4d} features")

    # Analysis 4: Decoder weight analysis (what each feature reconstructs)
    print(f"\n🎹 DECODER PATTERN ANALYSIS:")
    print("-" * 40)

    decoder_weights = model.decoder.weight.data  # Shape: [512, 2048]

    # Find features with strongest reconstruction patterns
    decoder_norms = torch.norm(decoder_weights, dim=0)  # Norm for each feature
    strongest_features = torch.topk(decoder_norms, top_k).indices

    print("Features with strongest decoder patterns:")
    for i, feat_idx in enumerate(strongest_features):
        norm = decoder_norms[feat_idx].item()
        freq = feature_frequencies[feat_idx].item()

        # Analyze what this feature reconstructs
        decoder_pattern = decoder_weights[:, feat_idx]
        max_weight_idx = torch.argmax(torch.abs(decoder_pattern)).item()
        max_weight = decoder_pattern[max_weight_idx].item()

        print(
            f"  Feature {feat_idx:4d}: norm={norm:.3f}, freq={freq:.1%}, "
            f"peak_weight@{max_weight_idx}={max_weight:+.3f}"
        )

    # Analysis 5: Save feature activation heatmap
    print(f"\n💾 SAVING VISUALIZATIONS:")
    print("-" * 40)

    # Create a detailed heatmap of feature activations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Feature frequency distribution
    axes[0, 0].hist(feature_frequencies.numpy(), bins=50, alpha=0.7, color="skyblue")
    axes[0, 0].set_xlabel("Activation Frequency")
    axes[0, 0].set_ylabel("Number of Features")
    axes[0, 0].set_title("Feature Activation Frequency Distribution")
    axes[0, 0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="50% line")
    axes[0, 0].legend()

    # Plot 2: Selectivity vs Frequency scatter
    axes[0, 1].scatter(
        feature_frequencies.numpy(),
        feature_selectivity.numpy(),
        alpha=0.6,
        s=20,
        color="coral",
    )
    axes[0, 1].set_xlabel("Activation Frequency")
    axes[0, 1].set_ylabel("Selectivity (std/mean)")
    axes[0, 1].set_title("Feature Selectivity vs Activation Frequency")

    # Plot 3: Decoder weight norms
    axes[1, 0].hist(decoder_norms.numpy(), bins=50, alpha=0.7, color="lightgreen")
    axes[1, 0].set_xlabel("Decoder Weight Norm")
    axes[1, 0].set_ylabel("Number of Features")
    axes[1, 0].set_title("Decoder Pattern Strength Distribution")

    # Plot 4: Feature activation heatmap (sample)
    sample_features_for_heatmap = interesting_features[
        :50
    ]  # Top 50 interesting features
    sample_activations = hidden_acts[
        :100, sample_features_for_heatmap
    ]  # First 100 samples

    im = axes[1, 1].imshow(sample_activations.T.numpy(), aspect="auto", cmap="viridis")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Feature Index")
    axes[1, 1].set_title("Feature Activation Heatmap\n(50 features × 100 samples)")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("detailed_sae_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅ Saved detailed analysis to 'detailed_sae_analysis.png'")

    return {
        "most_selective_features": most_selective,
        "feature_correlations": high_corr_pairs,
        "strongest_decoder_features": strongest_features,
        "feature_selectivity": feature_selectivity,
        "decoder_norms": decoder_norms,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Music SAE Feature Interpretation with OpenAI"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "openai", "pattern"],
        default="pattern",
        help="Analysis mode: basic (local analysis), openai (LLM interpretation), or pattern (pattern-based interpretation)",
    )
    parser.add_argument(
        "--model-path",
        default="exp/sod/ape/sae_models/sae_layer_2048d.pt",
        help="Path to trained SAE model",
    )
    parser.add_argument(
        "--activations-path",
        default="exp/sod/ape/activations/activations_layers_3.h5",
        help="Path to activation data",
    )
    parser.add_argument(
        "--data-dir",
        default="data/sod/processed/json",
        help="Directory containing original JSON musical data",
    )
    parser.add_argument(
        "--openai-api-key", help="OpenAI API key for LLM interpretation"
    )
    parser.add_argument(
        "--output-dir",
        default="interpretation/analysis_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20,
        help="Maximum number of features to analyze with OpenAI",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-3.5-turbo"],
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to use: cpu, cuda, or auto (default: auto-detect)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=50000,
        help="Number of samples to process for interpretation (default: 50000, use 1000000 for full analysis)",
    )

    args = parser.parse_args()

    # Determine device based on argument or auto-detect
    if args.device == "auto" or args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🚀 Using device: {device}")

    # Validate CUDA availability if requested
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if args.mode == "pattern":
        print("🎼 Running pattern-based interpretation pipeline...")
        interpretations = run_pattern_based_interpretation_pipeline(
            model_path=args.model_path,
            activations_path=args.activations_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_features=args.max_features,
            device=device,  # Pass device for GPU acceleration
            subsample=args.subsample,  # Pass subsample parameter
        )

        print("\n🎉 Pattern-based interpretation complete!")
        print(f"Features analyzed: {len(interpretations)}")

        # Show summary of high-interpretability features
        high_interp = [
            interp
            for interp in interpretations
            if interp["interpretability_score"] >= 0.5
        ]

        if high_interp:
            print(
                f"\n🎯 High-interpretability features ({len(high_interp)} features with score ≥ 0.5):"
            )
            for interpretation in sorted(
                high_interp, key=lambda x: x["interpretability_score"], reverse=True
            )[:5]:
                desc = interpretation["description"]
                score = interpretation["interpretability_score"]
                print(f"  {desc} (score: {score})")

    elif args.mode == "openai":
        if not args.openai_api_key:
            print(
                "❌ OpenAI API key required for OpenAI mode. Use --openai-api-key or set OPENAI_API_KEY environment variable"
            )
            exit(1)

        print("🤖 Running OpenAI-powered interpretation pipeline...")
        interpretations, summary = run_openai_sae_interpretation_pipeline(
            model_path=args.model_path,
            activations_path=args.activations_path,
            data_dir=args.data_dir,
            openai_api_key=args.openai_api_key,
            output_dir=args.output_dir,
            max_features=args.max_features,
            model_name=args.model_name,
        )

        print("\n🎉 OpenAI interpretation complete!")
        print(f"Features analyzed: {len(interpretations)}")

        # Show summary of high-confidence interpretations
        high_conf = [
            (fid, interp)
            for fid, interp in interpretations.items()
            if interp.get("confidence_score", 0) >= 70
        ]

        if high_conf:
            print(f"\n🎯 High-confidence interpretations ({len(high_conf)} features):")
            for feature_id, interpretation in sorted(
                high_conf, key=lambda x: x[1].get("confidence_score", 0), reverse=True
            )[:5]:
                print(
                    f"  Feature {feature_id}: {interpretation.get('interpretation', '')[:80]}... "
                    f"({interpretation.get('confidence_score', 0)}%)"
                )

    else:
        print("🎵 Running basic feature analysis...")
        # Load previous results for basic analysis
        results = analyze_sae_features(
            args.model_path,
            args.activations_path,
        )

        # Detailed interpretation
        interpretation = interpret_music_features(
            results, args.activations_path, top_k=15
        )

        print("✅ Basic analysis complete!")
