#!/usr/bin/env python3
"""
Enhanced SAE Feature Diversity Implementation

This script implements the key improvements for achieving diverse musical pattern detection.
Run this to get significantly more diverse feature interpretations.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter, defaultdict
import torch


def select_maximally_diverse_features(feature_summaries, max_features=15):
    """
    Select features that are maximally different from each other using clustering.
    This replaces the current frequency-based selection.
    """
    print(
        f"🎯 Selecting {max_features} maximally diverse features from {len(feature_summaries)}..."
    )

    # Create multi-dimensional feature vectors
    feature_vectors = []
    feature_ids = []

    for fid, stats in feature_summaries.items():
        # Get source file diversity
        source_files = set(
            act.get("source_file", "unknown")
            for act in stats.get("top_activations", [])
        )

        # Get activation position spread
        positions = [
            act.get("token_position", 0) for act in stats.get("top_activations", [])
        ]
        position_spread = np.std(positions) if positions else 0

        # Get activation strength characteristics
        activations = [
            act.get("activation_value", 0) for act in stats.get("top_activations", [])
        ]
        activation_variance = np.std(activations) if activations else 0

        # Create feature vector
        vector = [
            stats.get("activation_frequency", 0),
            stats.get("max_activation", 0),
            stats.get("mean_activation", 0),
            len(source_files),  # Source diversity
            position_spread,  # Position diversity
            activation_variance,  # Activation consistency
        ]

        feature_vectors.append(vector)
        feature_ids.append(fid)

    if len(feature_vectors) <= max_features:
        return feature_ids

    # Use k-means clustering to find diverse groups
    feature_vectors = np.array(feature_vectors)

    # Normalize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)

    # Cluster into max_features groups
    kmeans = KMeans(n_clusters=max_features, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_vectors_scaled)

    # Select the most representative feature from each cluster
    diverse_features = []
    for cluster_id in range(max_features):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        if cluster_indices:
            # Select feature with highest activation frequency within cluster
            best_idx = max(
                cluster_indices,
                key=lambda i: feature_summaries[feature_ids[i]]["activation_frequency"],
            )
            diverse_features.append(feature_ids[best_idx])

    print(f"✅ Selected {len(diverse_features)} diverse features using clustering")
    return diverse_features


def extract_musical_patterns(musical_data: dict) -> dict:
    """Extract specific, measurable musical patterns for SAE interpretation."""
    patterns = {
        "instrument_sequences": [],
        "pitch_changes": [],
        "duration_patterns": [],
        "velocity_patterns": [],
        "rhythmic_patterns": [],
        "harmonic_patterns": [],
        "textural_patterns": [],  # New category
        "dynamic_patterns": [],  # New category
        "structural_patterns": [],  # New category
    }

    tracks = musical_data.get("tracks", [])
    resolution = musical_data.get("resolution", 24)

    # ENHANCED RHYTHMIC PATTERN DETECTION
    def detect_rhythmic_patterns(notes, instrument_name):
        if len(notes) < 4:
            return []

        rhythmic_patterns = []
        onset_intervals = [
            notes[i + 1]["time"] - notes[i]["time"] for i in range(len(notes) - 1)
        ]
        durations = [note["duration"] for note in notes]

        # Syncopation detection (off-beat emphasis)
        beat_length = resolution  # Quarter note
        syncopated_count = 0
        for i, interval in enumerate(onset_intervals):
            if interval % beat_length != 0 and durations[i] >= beat_length // 2:
                syncopated_count += 1

        if syncopated_count > len(onset_intervals) * 0.3:
            rhythmic_patterns.append(
                f"syncopated patterns ({syncopated_count}/{len(onset_intervals)} off-beat) in {instrument_name}"
            )

        # Steady beat detection
        most_common_interval = (
            max(set(onset_intervals), key=onset_intervals.count)
            if onset_intervals
            else 0
        )
        steady_ratio = (
            onset_intervals.count(most_common_interval) / len(onset_intervals)
            if onset_intervals
            else 0
        )
        if steady_ratio > 0.7:
            rhythmic_patterns.append(
                f"steady beat (interval {most_common_interval} ticks, {steady_ratio:.1%} consistency) in {instrument_name}"
            )

        # Dotted rhythm detection (3:2 ratio patterns)
        dotted_count = 0
        for i in range(len(durations) - 1):
            ratio = durations[i] / durations[i + 1] if durations[i + 1] > 0 else 0
            if 1.4 < ratio < 1.6:  # Approximately 3:2 ratio
                dotted_count += 1

        if dotted_count > len(durations) * 0.2:
            rhythmic_patterns.append(
                f"dotted rhythms ({dotted_count} dotted patterns) in {instrument_name}"
            )

        # Triplet detection (irregular subdivisions)
        triplet_intervals = [
            interval
            for interval in onset_intervals
            if interval % (beat_length // 3) == 0
        ]
        if len(triplet_intervals) > len(onset_intervals) * 0.3:
            rhythmic_patterns.append(
                f"triplet rhythms ({len(triplet_intervals)} triplet subdivisions) in {instrument_name}"
            )

        return rhythmic_patterns

    # ENHANCED HARMONIC PATTERN DETECTION
    def detect_harmonic_patterns(all_tracks):
        harmonic_patterns = []

        # Collect simultaneous notes for chord analysis
        time_slices = {}
        for track in all_tracks:
            for note in track.get("notes", []):
                time_window = note["time"] // (resolution // 4)
                if time_window not in time_slices:
                    time_slices[time_window] = []
                time_slices[time_window].append(note["pitch"] % 12)

        # Chord progression analysis
        chord_sequence = []
        for time_window in sorted(time_slices.keys()):
            pitches = list(set(time_slices[time_window]))
            if len(pitches) >= 3:  # Chord
                pitches.sort()
                chord_sequence.append(tuple(pitches))

        if len(chord_sequence) >= 3:
            # Look for common progressions (simplified)
            progression_changes = len(set(chord_sequence))
            if progression_changes >= 3:
                harmonic_patterns.append(
                    f"chord progressions ({progression_changes} distinct chords in sequence)"
                )

        # Dissonance detection (tritones, major 7ths, minor 2nds)
        dissonant_intervals = []
        for pitches in time_slices.values():
            if len(pitches) >= 2:
                for i, p1 in enumerate(pitches):
                    for p2 in pitches[i + 1 :]:
                        interval = abs(p1 - p2) % 12
                        if interval in [1, 6, 10, 11]:  # Dissonant intervals
                            dissonant_intervals.append(interval)

        if len(dissonant_intervals) > 5:
            harmonic_patterns.append(
                f"dissonant intervals ({len(dissonant_intervals)} dissonances detected)"
            )

        # Cadence detection (simplified - look for V-I motion)
        for i in range(len(chord_sequence) - 1):
            curr_chord = chord_sequence[i]
            next_chord = chord_sequence[i + 1]
            # Simple dominant-tonic detection
            if len(curr_chord) >= 3 and len(next_chord) >= 3:
                # Look for fifth relationship in bass notes
                bass_motion = (next_chord[0] - curr_chord[0]) % 12
                if bass_motion == 5 or bass_motion == 7:  # Perfect 4th or 5th
                    harmonic_patterns.append(
                        "cadential motion (dominant-tonic bass movement)"
                    )
                    break

        return harmonic_patterns

    # TEXTURAL PATTERN DETECTION
    def detect_textural_patterns(all_tracks):
        textural_patterns = []

        # Polyphonic density analysis
        max_simultaneous = 0
        total_time_slices = 0
        dense_sections = 0

        time_coverage = {}
        for track in all_tracks:
            for note in track.get("notes", []):
                start_time = note["time"]
                end_time = start_time + note["duration"]
                for t in range(start_time, end_time, resolution // 4):
                    if t not in time_coverage:
                        time_coverage[t] = 0
                    time_coverage[t] += 1

        if time_coverage:
            max_simultaneous = max(time_coverage.values())
            dense_sections = sum(
                1 for count in time_coverage.values() if count >= len(all_tracks) * 0.8
            )
            total_time_slices = len(time_coverage)

            if max_simultaneous >= 4:
                textural_patterns.append(
                    f"dense polyphony (up to {max_simultaneous} simultaneous voices)"
                )

            if dense_sections > total_time_slices * 0.3:
                textural_patterns.append(
                    f"thick texture ({dense_sections}/{total_time_slices} dense sections)"
                )

        # Solo passage detection
        solo_sections = sum(1 for count in time_coverage.values() if count == 1)
        if solo_sections > total_time_slices * 0.2:
            textural_patterns.append(
                f"solo passages ({solo_sections}/{total_time_slices} single-voice sections)"
            )

        # Unison detection (same pitches across tracks)
        unison_count = 0
        for time_window in time_coverage:
            pitches_at_time = []
            for track in all_tracks:
                for note in track.get("notes", []):
                    if note["time"] <= time_window < note["time"] + note["duration"]:
                        pitches_at_time.append(note["pitch"])

            if len(set(pitches_at_time)) == 1 and len(pitches_at_time) > 1:
                unison_count += 1

        if unison_count > total_time_slices * 0.1:
            textural_patterns.append(f"unison sections ({unison_count} unison moments)")

        return textural_patterns

    # DYNAMIC PATTERN DETECTION
    def detect_dynamic_patterns(notes, instrument_name):
        if len(notes) < 5:
            return []

        dynamic_patterns = []
        velocities = [note["velocity"] for note in notes]

        # Crescendo detection (gradual increase)
        crescendo_sequences = 0
        for i in range(len(velocities) - 3):
            if all(velocities[j] < velocities[j + 1] for j in range(i, i + 3)):
                crescendo_sequences += 1

        if crescendo_sequences > 0:
            dynamic_patterns.append(
                f"crescendo patterns ({crescendo_sequences} ascending sequences) in {instrument_name}"
            )

        # Accent detection (sudden loud notes)
        avg_velocity = sum(velocities) / len(velocities)
        accents = [v for v in velocities if v > avg_velocity + 20]
        if len(accents) > len(velocities) * 0.2:
            dynamic_patterns.append(
                f"forte accents ({len(accents)} accent notes) in {instrument_name}"
            )

        # Pianissimo detection
        soft_notes = [v for v in velocities if v < 40]
        if len(soft_notes) > len(velocities) * 0.3:
            dynamic_patterns.append(
                f"pianissimo passages ({len(soft_notes)} very soft notes) in {instrument_name}"
            )

        # Dynamic contrast
        velocity_range = max(velocities) - min(velocities)
        if velocity_range > 60:
            dynamic_patterns.append(
                f"wide dynamic range (span: {velocity_range}) in {instrument_name}"
            )

        return dynamic_patterns

    # STRUCTURAL PATTERN DETECTION
    def detect_structural_patterns(notes, instrument_name):
        if len(notes) < 8:
            return []

        structural_patterns = []

        # Phrase boundary detection (gaps in time)
        times = [note["time"] for note in notes]
        gaps = []
        for i in range(len(times) - 1):
            gap = times[i + 1] - (times[i] + notes[i]["duration"])
            if gap > resolution:  # Gap longer than quarter note
                gaps.append(gap)

        if len(gaps) > 2:
            avg_gap = sum(gaps) / len(gaps)
            structural_patterns.append(
                f"phrase boundaries ({len(gaps)} gaps, avg {avg_gap:.0f} ticks) in {instrument_name}"
            )

        # Repetitive motif detection (pitch sequences)
        pitches = [note["pitch"] for note in notes]
        motif_length = 4
        motifs = {}
        for i in range(len(pitches) - motif_length + 1):
            motif = tuple(pitches[i : i + motif_length])
            motifs[motif] = motifs.get(motif, 0) + 1

        repeated_motifs = [
            (motif, count) for motif, count in motifs.items() if count >= 2
        ]
        if repeated_motifs:
            max_repetitions = max(count for _, count in repeated_motifs)
            structural_patterns.append(
                f"repetitive motifs ({len(repeated_motifs)} patterns, max {max_repetitions} repeats) in {instrument_name}"
            )

        return structural_patterns

    # USE ENHANCED PATTERN DETECTION FUNCTIONS
    if tracks:
        # Extract harmonic patterns from all tracks
        harmonic_patterns = detect_harmonic_patterns(tracks)
        patterns["harmonic_patterns"].extend(harmonic_patterns)

        # Extract textural patterns from all tracks
        textural_patterns = detect_textural_patterns(tracks)
        patterns["textural_patterns"].extend(textural_patterns)

        # Analyze each track for individual patterns
        for track in tracks:
            notes = track.get("notes", [])
            if len(notes) < 2:
                continue

            instrument_name = get_instrument_name(track.get("program", 0))

            # Enhanced rhythmic pattern detection
            rhythmic_patterns = detect_rhythmic_patterns(notes, instrument_name)
            patterns["rhythmic_patterns"].extend(rhythmic_patterns)

            # Enhanced dynamic pattern detection
            dynamic_patterns = detect_dynamic_patterns(notes, instrument_name)
            patterns["dynamic_patterns"].extend(dynamic_patterns)

            # Enhanced structural pattern detection
            structural_patterns = detect_structural_patterns(notes, instrument_name)
            patterns["structural_patterns"].extend(structural_patterns)

            # IMPROVED PITCH ANALYSIS (more specific patterns)
            for i in range(len(notes) - 1):
                pitch_diff = notes[i + 1]["pitch"] - notes[i]["pitch"]

                # Specific interval detection
                if abs(pitch_diff) == 12:  # Octave
                    direction = "ascending" if pitch_diff > 0 else "descending"
                    patterns["pitch_changes"].append(
                        f"octave leaps {direction} in {instrument_name}"
                    )
                elif abs(pitch_diff) == 7:  # Perfect fifth
                    direction = "ascending" if pitch_diff > 0 else "descending"
                    patterns["pitch_changes"].append(
                        f"perfect fifth leaps {direction} in {instrument_name}"
                    )
                elif abs(pitch_diff) >= 7:  # Large leap
                    direction = "ascending" if pitch_diff > 0 else "descending"
                    patterns["pitch_changes"].append(
                        f"wide leaps ({abs(pitch_diff)} semitones {direction}) in {instrument_name}"
                    )

            # IMPROVED DURATION ANALYSIS
            durations = [note["duration"] for note in notes]
            if durations:
                quarter_note = resolution  # Assuming resolution is quarter note

                # Specific rhythmic values
                whole_notes = [d for d in durations if d >= quarter_note * 3]
                sixteenth_notes = [d for d in durations if d <= quarter_note // 3]

                if whole_notes:
                    patterns["duration_patterns"].append(
                        f"sustained notes ({len(whole_notes)} long notes) in {instrument_name}"
                    )
                if len(sixteenth_notes) > len(durations) * 0.3:
                    patterns["duration_patterns"].append(
                        f"rapid passages ({len(sixteenth_notes)} fast notes) in {instrument_name}"
                    )

    return patterns


def create_contrastive_openai_prompt(
    feature_id, feature_stats, musical_contexts, existing_interpretations=None
):
    """
    Create an enhanced OpenAI prompt that avoids repetitive interpretations.
    """

    existing_patterns = []
    if existing_interpretations:
        existing_patterns = [
            interp.get("interpretation", "")
            for interp in existing_interpretations.values()
        ]

    prompt = f"""You are a musicologist analyzing Feature {feature_id} in a Music Transformer's SAE.

        CRITICAL CONSTRAINT: This feature must be DIFFERENT from these already-identified patterns:
        {chr(10).join(f"- {pattern}" for pattern in existing_patterns)}

        ## Feature {feature_id} Analysis

        ### Statistical Profile:
        - Activation frequency: {feature_stats.get('activation_frequency', 0):.1%}
        - Mean activation strength: {feature_stats.get('mean_activation', 0):.3f} 
        - Maximum activation: {feature_stats.get('max_activation', 0):.3f}
        - Total occurrences: {feature_stats.get('total_activations', 0)}

        ### Musical Contexts with Extracted Patterns:
    """

    # Add enhanced musical context analysis with extracted patterns
    for i, context in enumerate(musical_contexts[:3]):
        tracks = context.get("tracks", [])
        total_notes = sum(len(track.get("notes", [])) for track in tracks)

        prompt += f"""
            Context {i+1}:
            - Source: {context.get('source_file', 'Unknown')}
            - Tracks: {len(tracks)}, Total notes: {total_notes}
            - Activation position: {context.get('activation_token_position', 'Unknown')}
            - Activation strength: {context.get('activation_value', 0):.3f}
        """

        if tracks:
            # Extract detailed musical patterns from this context
            extracted_patterns = extract_musical_patterns(context)

            # Add basic track information
            for j, track in enumerate(tracks[:2]):  # First 2 tracks
                notes = track.get("notes", [])
                if notes:
                    instrument = get_instrument_name(track.get("program", 0))
                    pitches = [n["pitch"] for n in notes]
                    velocities = [n["velocity"] for n in notes]
                    durations = [n["duration"] for n in notes]

                    prompt += f"""  Track {j+1} ({instrument}): {len(notes)} notes
                        - Pitch range: {min(pitches)}-{max(pitches)} (span: {max(pitches)-min(pitches)})
                        - Velocity range: {min(velocities)}-{max(velocities)}
                        - Duration variety: {len(set(durations))} different values
                    """

            # Add extracted pattern analysis
            prompt += """
            🎼 EXTRACTED MUSICAL PATTERNS:"""

            for pattern_type, patterns in extracted_patterns.items():
                if patterns:
                    pattern_type_name = pattern_type.replace("_", " ").title()
                    prompt += f"""
            {pattern_type_name}:"""
                    for pattern in patterns[:3]:  # Top 3 patterns per type
                        prompt += f"""
            - {pattern}"""

    prompt += """

        ## ENHANCED ANALYSIS REQUIREMENTS:

        Focus on finding ONE UNIQUE musical pattern that this feature detects. Look specifically for:

        ### RHYTHMIC SPECIALIZATION:
        - Syncopated patterns (off-beat emphasis, cross-rhythms)  
        - Steady pulse patterns (metronomic consistency, march-like)
        - Dotted rhythm patterns (uneven long-short groupings)
        - Triplet feel patterns (3-against-2 subdivisions)
        - Polyrhythmic textures (multiple simultaneous rhythms)

        ### HARMONIC SPECIALIZATION:
        - Specific chord progressions (ii-V-I, vi-IV-I-V, etc.)
        - Dissonance patterns (tritones, 7th chords, cluster chords)
        - Cadential gestures (authentic, plagal, deceptive cadences)
        - Voice leading patterns (contrary motion, parallel fourths)
        - Modal characteristics (Dorian, Lydian, pentatonic scales)

        ### TEXTURAL SPECIALIZATION:
        - Dense polyphonic writing (4+ independent voices)
        - Solo instrumental passages (featured melodic lines)
        - Unison doubling (octave reinforcement, tutti effects)
        - Antiphonal textures (call-and-response between groups)
        - Accompaniment patterns (Alberti bass, arpeggiated support)

        ### DYNAMIC SPECIALIZATION:
        - Crescendo gestures (gradual builds, climactic approaches)
        - Sudden accents (sforzando attacks, rhythmic emphasis)  
        - Soft dynamics (pianissimo sections, intimate passages)
        - Dynamic contrasts (terraced levels, echo effects)
        - Expressive swells (phrase-level dynamics, musical breathing)

        ### STRUCTURAL SPECIALIZATION:
        - Opening gestures (introductory material, thematic presentation)
        - Closing patterns (cadential extensions, final affirmations)
        - Transitional passages (modulations, bridge sections)
        - Developmental techniques (fragmentation, augmentation, sequence)
        - Repetitive cycles (ostinato patterns, ground bass, rondo themes)

        ## RESPONSE FORMAT:

        Provide exactly ONE specific interpretation that:
        1. Is MUSICALLY DISTINCT from all existing patterns above
        2. Focuses on a specific musical technique or gesture
        3. Is supported by concrete evidence from the contexts
        4. Would be recognizable to a trained musician

        ```json
        {{
        "interpretation": "Specific, unique musical pattern (avoid generic descriptions)",
        "evidence": ["Evidence 1 from musical analysis", "Evidence 2 from statistical data", "Evidence 3 from context"],
        "confidence_score": 85,
        "alternative_hypotheses": ["Alternative 1", "Alternative 2"],
        "musical_category": "rhythmic_specific|harmonic_specific|textural_specific|dynamic_specific|structural_specific"
        }}
        ```
    """

    return prompt


def get_instrument_name(program_number: int) -> str:
    """Convert MIDI program number to instrument name."""
    instruments = {
        0: "Acoustic Grand Piano",
        1: "Bright Acoustic Piano",
        2: "Electric Grand Piano",
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
    }
    return instruments.get(program_number, f"Program {program_number}")


def apply_enhanced_threshold_strategy(feature_summaries, threshold=1.5):
    """
    Apply intelligent thresholding to get more selective features.
    """
    print(f"🎯 Applying enhanced threshold strategy (min_activation >= {threshold})...")

    # Filter features by activation threshold
    filtered_features = {}

    for fid, stats in feature_summaries.items():
        max_activation = stats.get("max_activation", 0)

        # Apply threshold
        if max_activation >= threshold:
            # Additional quality filters
            activation_freq = stats.get("activation_frequency", 0)
            total_activations = stats.get("total_activations", 0)

            # Must have reasonable frequency and multiple activations
            if (
                activation_freq >= 0.001 and total_activations >= 5
            ):  # At least 0.1% frequency and 5 occurrences
                filtered_features[fid] = stats

    print(
        f"✅ Filtered to {len(filtered_features)} high-quality features (from {len(feature_summaries)})"
    )
    return filtered_features


def run_enhanced_diversity_pipeline(
    model_path,
    activations_path,
    data_dir,
    openai_api_key,
    output_dir="interpretation/enhanced_diversity",
    max_features=15,
    min_activation_threshold=1.5,
    openai_model="gpt-4o",
):
    """
    Run the enhanced diversity pipeline with all improvements.
    """
    print("🚀 ENHANCED SAE DIVERSITY PIPELINE 🚀")
    print("=" * 60)

    # Import the original functions
    from interpretation.interpret_music_sae import (
        save_feature_activations_for_interpretation,
        load_musical_json_data,
        call_openai_for_interpretation,
    )
    import pathlib

    # Step 1: Extract feature activations with higher threshold
    print("Step 1: Extracting selective feature activations...")
    _, feature_summaries = save_feature_activations_for_interpretation(
        model_path=model_path,
        activations_path=activations_path,
        output_dir=f"{output_dir}/feature_data",
        max_samples_per_feature=10,
        min_activation_threshold=min_activation_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        subsample=50000,
    )

    # Step 2: Apply enhanced filtering
    filtered_features = apply_enhanced_threshold_strategy(
        feature_summaries, min_activation_threshold
    )

    # Step 3: Select maximally diverse features
    diverse_feature_ids = select_maximally_diverse_features(
        filtered_features, max_features
    )

    print(f"Selected {len(diverse_feature_ids)} diverse features for analysis...")

    # Step 4: Load musical contexts
    data_path = pathlib.Path(data_dir)

    # Step 5: Run contrastive OpenAI analysis
    interpretations = {}

    for i, feature_id in enumerate(diverse_feature_ids):
        print(
            f"\n🎵 Analyzing Feature {feature_id} ({i+1}/{len(diverse_feature_ids)})..."
        )

        feature_stats = filtered_features[feature_id]

        # Load musical contexts
        musical_contexts = []
        for activation_data in feature_stats["top_activations"][:5]:
            if "source_file" in activation_data:
                source_file = activation_data["source_file"]

                # Try multiple paths
                possible_paths = [
                    data_path / source_file,
                    data_path / f"{source_file}.json",
                    data_path / "json" / f"{source_file}.json",
                    data_path / "json" / source_file,
                ]

                for path in possible_paths:
                    if path.exists():
                        musical_data = load_musical_json_data(str(path))
                        if musical_data.get("tracks"):
                            musical_data["source_file"] = source_file
                            musical_data["activation_token_position"] = (
                                activation_data.get("token_position", 0)
                            )
                            musical_data["activation_value"] = activation_data.get(
                                "activation_value", 0
                            )
                            musical_contexts.append(musical_data)
                        break

        # Create contrastive prompt
        prompt = create_contrastive_openai_prompt(
            feature_id=feature_id,
            feature_stats=feature_stats,
            musical_contexts=musical_contexts,
            existing_interpretations=interpretations,  # Pass existing interpretations for contrast
        )

        # Get OpenAI interpretation
        interpretation = call_openai_for_interpretation(
            prompt=prompt, api_key=openai_api_key, model=openai_model
        )

        interpretation["feature_id"] = feature_id
        interpretation["feature_stats"] = feature_stats
        interpretations[feature_id] = interpretation

        print(f"   ✅ {interpretation.get('interpretation', 'Failed')}")
        print(f"   📊 Confidence: {interpretation.get('confidence_score', 0)}/100")
        print(f"   🎼 Category: {interpretation.get('musical_category', 'unknown')}")

    # Step 6: Save results with diversity analysis
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate diversity metrics
    categories = [
        interp.get("musical_category", "unknown") for interp in interpretations.values()
    ]
    category_distribution = Counter(categories)

    # Check for interpretation uniqueness
    interpretation_texts = [
        interp.get("interpretation", "") for interp in interpretations.values()
    ]
    unique_words = set()
    for text in interpretation_texts:
        unique_words.update(text.lower().split())

    diversity_report = {
        "pipeline_info": {
            "model_path": model_path,
            "total_features_found": len(feature_summaries),
            "filtered_features": len(filtered_features),
            "selected_diverse_features": len(diverse_feature_ids),
            "min_activation_threshold": min_activation_threshold,
            "openai_model": openai_model,
        },
        "diversity_metrics": {
            "category_distribution": dict(category_distribution),
            "unique_categories": len(category_distribution),
            "max_category_dominance": (
                max(category_distribution.values()) / len(interpretations)
                if interpretations
                else 0
            ),
            "vocabulary_diversity": len(unique_words),
            "avg_confidence": (
                np.mean(
                    [
                        interp.get("confidence_score", 0)
                        for interp in interpretations.values()
                    ]
                )
                if interpretations
                else 0
            ),
        },
        "interpretations": interpretations,
    }

    # Save comprehensive report
    with open(output_path / "enhanced_diversity_report.json", "w") as f:
        json.dump(diversity_report, f, indent=2)

    # Create summary table
    with open(output_path / "diversity_summary.md", "w") as f:
        f.write("# 🎵 Enhanced SAE Feature Diversity Report\n\n")
        f.write("## Diversity Metrics\n\n")
        f.write(f"- **Total features analyzed**: {len(interpretations)}\n")
        f.write(f"- **Unique categories**: {len(category_distribution)}\n")
        f.write(
            f"- **Average confidence**: {diversity_report['diversity_metrics']['avg_confidence']:.1f}%\n"
        )
        f.write(f"- **Vocabulary diversity**: {len(unique_words)} unique words\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in category_distribution.most_common():
            percentage = count / len(interpretations) * 100
            f.write(
                f"- **{category.replace('_', ' ').title()}**: {count} features ({percentage:.1f}%)\n"
            )

        f.write("\n## Feature Interpretations\n\n")
        f.write("| Feature | Category | Confidence | Interpretation |\n")
        f.write("|---------|----------|------------|----------------|\n")

        for fid, interp in sorted(interpretations.items()):
            category = (
                interp.get("musical_category", "unknown").replace("_", " ").title()
            )
            confidence = interp.get("confidence_score", 0)
            interpretation = (
                interp.get("interpretation", "No interpretation")[:80] + "..."
            )
            f.write(f"| {fid} | {category} | {confidence}% | {interpretation} |\n")

    print("\n🎯 ENHANCED DIVERSITY ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print("📊 Diversity metrics:")
    print(f"   - Categories: {len(category_distribution)}")
    print(
        f"   - Average confidence: {diversity_report['diversity_metrics']['avg_confidence']:.1f}%"
    )
    print(
        f"   - Max category dominance: {diversity_report['diversity_metrics']['max_category_dominance']:.1%}"
    )

    return interpretations, diversity_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced SAE Feature Diversity Analysis"
    )
    parser.add_argument("--model-path", required=True, help="Path to SAE model")
    parser.add_argument(
        "--activations-path", required=True, help="Path to activations file"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to musical data directory"
    )
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key")
    parser.add_argument(
        "--output-dir",
        default="interpretation/enhanced_diversity",
        help="Output directory",
    )
    parser.add_argument(
        "--max-features", type=int, default=15, help="Maximum features to analyze"
    )
    parser.add_argument(
        "--min-activation-threshold",
        type=float,
        default=1.5,
        help="Minimum activation threshold",
    )
    parser.add_argument("--openai-model", default="gpt-4o", help="OpenAI model to use")

    args = parser.parse_args()

    run_enhanced_diversity_pipeline(
        model_path=args.model_path,
        activations_path=args.activations_path,
        data_dir=args.data_dir,
        openai_api_key=args.openai_api_key,
        output_dir=args.output_dir,
        max_features=args.max_features,
        min_activation_threshold=args.min_activation_threshold,
        openai_model=args.openai_model,
    )
