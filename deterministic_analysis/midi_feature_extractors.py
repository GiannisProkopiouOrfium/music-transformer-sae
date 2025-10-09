#!/usr/bin/env python3
"""
MIDI Feature Extractors for Deterministic Musical Analysis

This module provides comprehensive MIDI-level analysis tools for feature interventions.
Extracts quantifiable musical characteristics from symbolic representations.

Features analyzed:
- Note-level patterns (density, velocity, duration)
- Pitch characteristics (range, contour, intervals)
- Rhythmic patterns (syncopation, groove, timing)
- Harmonic content (chord progressions, key signatures)
- Structural properties (phrases, repetition, form)
"""

import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import json
import sys
from collections import defaultdict, Counter
import logging

# Add mmt to path for imports
sys.path.append(str(Path(__file__).parent.parent / "mmt"))
import muspy
from mmt import representation


class MIDIFeatureExtractor:
    """Comprehensive MIDI feature extraction for musical analysis."""

    def __init__(self, encoding_path: str = "data/sod/processed/notes/encoding.json"):
        """Initialize with encoding information."""
        self.encoding = representation.load_encoding(encoding_path)
        self.logger = logging.getLogger(__name__)

    def extract_all_features(
        self, sequence, condition_info: Dict = None
    ) -> Dict[str, Any]:
        """
        Extract all MIDI features from a sequence.

        Args:
            sequence: Musical sequence array or MusPy Music object
            condition_info: Metadata about the condition (feature, strength, etc.)

        Returns:
            Comprehensive feature dictionary
        """
        try:
            # Handle both raw sequences and already-decoded MusPy Music objects
            if isinstance(sequence, muspy.Music):
                music = sequence
            else:
                # Convert to MusPy Music object
                music = representation.decode(sequence, self.encoding)
            
            if music.resolution:
                music.trim(music.resolution * 64)  # Trim to reasonable length

            if not music.tracks:
                return {"error": "No tracks found in music"}

            # Extract all feature categories
            features = {
                "basic_info": self._extract_basic_info(music, sequence),
                "note_patterns": self._extract_note_patterns(music),
                "pitch_analysis": self._extract_pitch_analysis(music),
                "rhythmic_analysis": self._extract_rhythmic_analysis(music),
                "harmonic_analysis": self._extract_harmonic_analysis(music),
                "structural_analysis": self._extract_structural_analysis(music),
                "velocity_dynamics": self._extract_velocity_dynamics(music),
                "temporal_analysis": self._extract_temporal_analysis(music),
                "musical_complexity": self._extract_complexity_metrics(music),
                "condition_metadata": condition_info or {},
            }

            return features

        except Exception as e:
            import traceback

            self.logger.error(f"Feature extraction failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e), "condition_metadata": condition_info or {}}

    def _extract_basic_info(
        self, music: muspy.Music, sequence: np.ndarray
    ) -> Dict[str, Any]:
        """Extract basic musical information."""
        return {
            "sequence_length": len(sequence),
            "total_tracks": len(music.tracks),
            "total_notes": sum(len(track.notes) for track in music.tracks),
            "duration_beats": (
                music.get_end_time() / music.resolution if music.resolution > 0 else 0
            ),
            "resolution": music.resolution,
            "tempo_changes": len(music.tempo_changes) if music.tempo_changes else 0,
            "key_signatures": len(music.key_signatures) if music.key_signatures else 0,
            "time_signatures": (
                len(music.time_signatures) if music.time_signatures else 0
            ),
        }

    def _extract_note_patterns(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract note-level patterns and statistics."""
        all_notes = []
        for track in music.tracks:
            all_notes.extend(track.notes)

        if not all_notes:
            return {"error": "No notes found"}

        # Note densities and patterns - Handle both Note objects and direct access
        try:
            durations = []
            velocities = []

            for note in all_notes:
                # Handle different note object types
                if hasattr(note, "duration"):
                    duration = note.duration
                    velocity = getattr(note, "velocity", None)
                else:
                    # Fallback for other formats
                    duration = getattr(note, "dur", 0)
                    velocity = getattr(note, "vel", None)

                # Ensure numeric values
                if duration is not None:
                    durations.append(float(duration))
                if velocity is not None:
                    velocities.append(int(velocity))

        except Exception as e:
            self.logger.error(f"Error extracting note properties: {e}")
            return {"error": f"Failed to extract note properties: {e}"}

        return {
            "notes_per_beat": (
                len(all_notes) / (music.get_end_time() / music.resolution)
                if music.get_end_time() > 0
                else 0
            ),
            "note_density": (
                len(all_notes) / (music.get_end_time() / music.resolution)
                if music.get_end_time() > 0
                else 0
            ),
            "average_note_duration": float(np.mean(durations)) if durations else 0,
            "note_duration_std": float(np.std(durations)) if durations else 0,
            "shortest_note": int(min(durations)) if durations else 0,
            "longest_note": int(max(durations)) if durations else 0,
            "average_velocity": float(np.mean(velocities)) if velocities else 0,
            "velocity_range": (
                int(max(velocities) - min(velocities)) if velocities else 0
            ),
            "unique_note_durations": len(set(durations)) if durations else 0,
            "note_onset_intervals": self._calculate_onset_intervals(all_notes),
        }

    def _extract_pitch_analysis(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract pitch-related characteristics."""
        all_pitches = []

        try:
            for track in music.tracks:
                for note in track.notes:
                    # Handle different note object types
                    if hasattr(note, "pitch"):
                        pitch = note.pitch
                    else:
                        pitch = getattr(note, "note", None)

                    if pitch is not None:
                        all_pitches.append(int(pitch))

            if not all_pitches:
                return {"error": "No pitches found"}

        except Exception as e:
            self.logger.error(f"Error extracting pitches: {e}")
            return {"error": f"Failed to extract pitches: {e}"}

        pitches = np.array(all_pitches)

        # Pitch statistics
        pitch_stats = {
            "pitch_range": int(np.max(pitches) - np.min(pitches)),
            "mean_pitch": float(np.mean(pitches)),
            "pitch_std": float(np.std(pitches)),
            "min_pitch": int(np.min(pitches)),
            "max_pitch": int(np.max(pitches)),
            "unique_pitches": len(set(all_pitches)),
            "pitch_class_distribution": self._calculate_pitch_class_distribution(
                all_pitches
            ),
        }

        # Melodic intervals and contour
        intervals = self._calculate_melodic_intervals(music)
        pitch_stats.update(
            {
                "melodic_intervals": intervals,
                "average_interval_size": (
                    float(np.mean(np.abs(intervals))) if intervals else 0
                ),
                "largest_leap": int(np.max(np.abs(intervals))) if intervals else 0,
                "step_motion_ratio": (
                    sum(1 for i in intervals if abs(i) <= 2) / len(intervals)
                    if intervals
                    else 0
                ),
            }
        )

        return pitch_stats

    def _extract_rhythmic_analysis(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract rhythmic patterns and timing characteristics."""
        all_notes = []
        for track in music.tracks:
            all_notes.extend(track.notes)

        if not all_notes:
            return {"error": "No notes for rhythmic analysis"}

        # Sort by onset time
        all_notes.sort(key=lambda n: n.time)

        # Inter-onset intervals
        onsets = [note.time for note in all_notes]
        iois = np.diff(onsets) if len(onsets) > 1 else []

        rhythmic_features = {
            "average_ioi": float(np.mean(iois)) if len(iois) > 0 else 0,
            "ioi_std": float(np.std(iois)) if len(iois) > 0 else 0,
            "rhythmic_regularity": self._calculate_rhythmic_regularity(iois),
            "syncopation_score": self._calculate_syncopation(music),
            "beat_strength_distribution": self._analyze_beat_strengths(music),
            "rhythmic_complexity": self._calculate_rhythmic_complexity(music),
        }

        return rhythmic_features

    def _extract_harmonic_analysis(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract harmonic content and progressions."""
        try:
            # Basic harmonic metrics using MusPy
            harmonic_features = {
                "scale_consistency": muspy.scale_consistency(music),
                "pitch_class_entropy": muspy.pitch_class_entropy(music),
            }

            # Chord analysis
            chord_analysis = self._analyze_chord_progressions(music)
            harmonic_features.update(chord_analysis)

            return harmonic_features

        except Exception as e:
            return {"error": f"Harmonic analysis failed: {str(e)}"}

    def _extract_structural_analysis(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract structural and formal characteristics."""
        try:
            # Phrase detection and analysis
            phrases = self._detect_phrases(music)

            structural_features = {
                "phrase_count": len(phrases),
                "average_phrase_length": (
                    float(np.mean([p["length"] for p in phrases])) if phrases else 0
                ),
                "phrase_length_std": (
                    float(np.std([p["length"] for p in phrases])) if phrases else 0
                ),
                "repetition_analysis": self._analyze_repetition_patterns(music),
                "structural_coherence": self._calculate_structural_coherence(phrases),
            }

            return structural_features

        except Exception as e:
            return {"error": f"Structural analysis failed: {str(e)}"}

    def _extract_velocity_dynamics(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract velocity and dynamic characteristics."""
        all_velocities = []
        for track in music.tracks:
            all_velocities.extend(
                [note.velocity for note in track.notes if note.velocity is not None]
            )

        if not all_velocities:
            return {"error": "No velocity data found"}

        velocities = np.array(all_velocities)

        return {
            "dynamic_range": int(np.max(velocities) - np.min(velocities)),
            "average_dynamics": float(np.mean(velocities)),
            "dynamic_variance": float(np.var(velocities)),
            "forte_notes_ratio": sum(1 for v in velocities if v > 100)
            / len(velocities),
            "piano_notes_ratio": sum(1 for v in velocities if v < 64) / len(velocities),
            "dynamic_contour": self._analyze_dynamic_contour(music),
        }

    def _extract_temporal_analysis(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract temporal flow and timing characteristics."""
        try:
            groove_consistency = muspy.groove_consistency(music, 4 * music.resolution)

            return {
                "groove_consistency": groove_consistency,
                "tempo_stability": self._calculate_tempo_stability(music),
                "timing_precision": self._analyze_timing_precision(music),
                "rhythmic_swing": self._detect_swing_characteristics(music),
            }

        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}

    def _extract_complexity_metrics(self, music: muspy.Music) -> Dict[str, Any]:
        """Extract overall musical complexity measures."""
        try:
            return {
                "polyphonic_complexity": self._calculate_polyphonic_complexity(music),
                "harmonic_complexity": self._calculate_harmonic_complexity(music),
                "rhythmic_diversity": self._calculate_rhythmic_diversity(music),
                "melodic_complexity": self._calculate_melodic_complexity(music),
                "overall_complexity_score": self._calculate_overall_complexity(music),
            }

        except Exception as e:
            return {"error": f"Complexity analysis failed: {str(e)}"}

    # Helper methods for specific calculations

    def _calculate_onset_intervals(self, notes: List) -> Dict[str, float]:
        """Calculate statistics about note onset intervals."""
        if len(notes) < 2:
            return {"error": "Insufficient notes for interval calculation"}

        sorted_notes = sorted(notes, key=lambda n: n.time)
        intervals = [
            sorted_notes[i + 1].time - sorted_notes[i].time
            for i in range(len(sorted_notes) - 1)
        ]

        return {
            "mean_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "min_interval": float(np.min(intervals)),
            "max_interval": float(np.max(intervals)),
        }

    def _calculate_pitch_class_distribution(
        self, pitches: List[int]
    ) -> Dict[str, float]:
        """Calculate pitch class distribution."""
        pitch_classes = [p % 12 for p in pitches]
        pc_counts = Counter(pitch_classes)
        total = len(pitches)

        return {f"pc_{pc}": count / total for pc, count in pc_counts.items()}

    def _calculate_melodic_intervals(self, music: muspy.Music) -> List[int]:
        """Calculate melodic intervals for the main melodic line."""
        # Find the track with the most notes (likely the melody)
        if not music.tracks:
            return []

        main_track = max(music.tracks, key=lambda t: len(t.notes))
        if len(main_track.notes) < 2:
            return []

        sorted_notes = sorted(main_track.notes, key=lambda n: n.time)
        intervals = [
            sorted_notes[i + 1].pitch - sorted_notes[i].pitch
            for i in range(len(sorted_notes) - 1)
        ]

        return intervals

    def _calculate_rhythmic_regularity(self, iois: np.ndarray) -> float:
        """Calculate rhythmic regularity score."""
        if len(iois) == 0:
            return 0.0

        # Regularity based on coefficient of variation
        mean_ioi = np.mean(iois)
        if mean_ioi == 0:
            return 0.0

        cv = np.std(iois) / mean_ioi
        return float(1.0 / (1.0 + cv))  # Higher score = more regular

    def _calculate_syncopation(self, music: muspy.Music) -> float:
        """Calculate syncopation score."""
        # Simplified syncopation based on off-beat emphasis
        if not music.tracks or not music.resolution:
            return 0.0

        all_notes = []
        for track in music.tracks:
            all_notes.extend(track.notes)

        if not all_notes:
            return 0.0

        # Count notes on strong vs weak beats
        beat_resolution = music.resolution
        strong_beat_notes = 0
        weak_beat_notes = 0

        for note in all_notes:
            beat_position = (note.time // beat_resolution) % 4
            if beat_position in [0, 2]:  # Strong beats
                strong_beat_notes += 1
            else:  # Weak beats
                weak_beat_notes += 1

        total_notes = strong_beat_notes + weak_beat_notes
        if total_notes == 0:
            return 0.0

        return weak_beat_notes / total_notes

    def _analyze_beat_strengths(self, music: muspy.Music) -> Dict[str, float]:
        """Analyze distribution of notes across beat positions."""
        if not music.tracks or not music.resolution:
            return {"error": "No tracks or resolution info"}

        beat_counts = defaultdict(int)
        total_notes = 0

        for track in music.tracks:
            for note in track.notes:
                beat_pos = (note.time // music.resolution) % 4
                beat_counts[beat_pos] += 1
                total_notes += 1

        if total_notes == 0:
            return {"error": "No notes found"}

        return {
            f"beat_{pos}": count / total_notes for pos, count in beat_counts.items()
        }

    def _calculate_rhythmic_complexity(self, music: muspy.Music) -> float:
        """Calculate overall rhythmic complexity."""
        if not music.tracks:
            return 0.0

        # Count unique rhythmic patterns
        rhythm_patterns = set()

        for track in music.tracks:
            if len(track.notes) < 2:
                continue

            sorted_notes = sorted(track.notes, key=lambda n: n.time)
            durations = tuple(
                note.duration for note in sorted_notes[:10]
            )  # First 10 notes
            rhythm_patterns.add(durations)

        return len(rhythm_patterns)

    def _analyze_chord_progressions(self, music: muspy.Music) -> Dict[str, Any]:
        """Analyze harmonic progressions."""
        # Simplified chord analysis
        try:
            # Count simultaneous note groups (potential chords)
            chord_count = 0
            time_groups = defaultdict(list)

            for track in music.tracks:
                for note in track.notes:
                    time_groups[note.time].append(note.pitch)

            chord_sizes = []
            for time_point, pitches in time_groups.items():
                unique_pitches = len(set(pitches))
                if unique_pitches > 1:
                    chord_count += 1
                    chord_sizes.append(unique_pitches)

            return {
                "chord_count": chord_count,
                "average_chord_size": float(np.mean(chord_sizes)) if chord_sizes else 0,
                "max_chord_size": int(max(chord_sizes)) if chord_sizes else 0,
                "harmonic_rhythm": (
                    chord_count / (music.get_end_time() / music.resolution)
                    if music.get_end_time() > 0
                    else 0
                ),
            }

        except Exception as e:
            return {"error": f"Chord analysis failed: {str(e)}"}

    def _detect_phrases(self, music: muspy.Music) -> List[Dict[str, Any]]:
        """Detect musical phrases based on rests and note patterns."""
        phrases = []

        for track in music.tracks:
            if len(track.notes) < 2:
                continue

            sorted_notes = sorted(track.notes, key=lambda n: n.time)
            phrase_starts = [0]

            # Detect phrase boundaries (gaps longer than average)
            gaps = [
                sorted_notes[i + 1].time
                - (sorted_notes[i].time + sorted_notes[i].duration)
                for i in range(len(sorted_notes) - 1)
            ]

            if gaps:
                avg_gap = np.mean(gaps)
                threshold = avg_gap * 2  # Phrases separated by longer gaps

                for i, gap in enumerate(gaps):
                    if gap > threshold:
                        phrase_starts.append(i + 1)

            phrase_starts.append(len(sorted_notes))

            # Create phrase objects
            for i in range(len(phrase_starts) - 1):
                start_idx = phrase_starts[i]
                end_idx = phrase_starts[i + 1]
                phrase_notes = sorted_notes[start_idx:end_idx]

                if phrase_notes:
                    phrases.append(
                        {
                            "start_time": phrase_notes[0].time,
                            "end_time": phrase_notes[-1].time
                            + phrase_notes[-1].duration,
                            "length": len(phrase_notes),
                            "track": id(track),
                        }
                    )

        return phrases

    def _analyze_repetition_patterns(self, music: muspy.Music) -> Dict[str, Any]:
        """Analyze repetition and variation patterns."""
        if not music.tracks:
            return {"error": "No tracks for repetition analysis"}

        # Simplified repetition analysis based on pitch sequences
        sequences = []
        for track in music.tracks:
            if len(track.notes) >= 4:
                sorted_notes = sorted(track.notes, key=lambda n: n.time)
                pitch_sequence = tuple(
                    note.pitch for note in sorted_notes[:20]
                )  # First 20 notes
                sequences.append(pitch_sequence)

        # Count unique vs repeated patterns
        sequence_counts = Counter(sequences)
        unique_sequences = len(sequence_counts)
        total_sequences = len(sequences)
        repeated_sequences = sum(1 for count in sequence_counts.values() if count > 1)

        return {
            "unique_patterns": unique_sequences,
            "total_sequences": total_sequences,
            "repetition_ratio": (
                repeated_sequences / total_sequences if total_sequences > 0 else 0
            ),
        }

    def _calculate_structural_coherence(self, phrases: List[Dict]) -> float:
        """Calculate structural coherence based on phrase regularity."""
        if len(phrases) < 2:
            return 0.0

        phrase_lengths = [p["length"] for p in phrases]
        mean_length = np.mean(phrase_lengths)

        if mean_length == 0:
            return 0.0

        # Coherence based on phrase length consistency
        cv = np.std(phrase_lengths) / mean_length
        return float(1.0 / (1.0 + cv))

    def _analyze_dynamic_contour(self, music: muspy.Music) -> Dict[str, Any]:
        """Analyze dynamic contour and changes."""
        all_velocities = []
        all_times = []

        for track in music.tracks:
            for note in track.notes:
                if note.velocity is not None:
                    all_velocities.append(note.velocity)
                    all_times.append(note.time)

        if len(all_velocities) < 2:
            return {"error": "Insufficient velocity data"}

        # Sort by time
        sorted_pairs = sorted(zip(all_times, all_velocities))
        velocities = [v for _, v in sorted_pairs]

        # Calculate contour characteristics
        changes = np.diff(velocities)

        return {
            "dynamic_changes": len(changes),
            "increasing_dynamics": sum(1 for c in changes if c > 5),
            "decreasing_dynamics": sum(1 for c in changes if c < -5),
            "dynamic_stability": sum(1 for c in changes if abs(c) <= 5),
            "average_change": float(np.mean(np.abs(changes))),
        }

    def _calculate_tempo_stability(self, music: muspy.Music) -> float:
        """Calculate tempo stability measure."""
        if not music.tempo_changes or len(music.tempo_changes) <= 1:
            return 1.0  # Perfectly stable if no tempo changes

        tempos = [tc.tempo for tc in music.tempo_changes]
        if len(tempos) < 2:
            return 1.0

        tempo_std = np.std(tempos)
        tempo_mean = np.mean(tempos)

        if tempo_mean == 0:
            return 0.0

        return float(1.0 / (1.0 + tempo_std / tempo_mean))

    def _analyze_timing_precision(self, music: muspy.Music) -> float:
        """Analyze timing precision and quantization."""
        if not music.tracks or not music.resolution:
            return 0.0

        # Check how many notes fall exactly on grid positions
        grid_aligned = 0
        total_notes = 0

        sixteenth_note = music.resolution // 4  # 16th note resolution

        for track in music.tracks:
            for note in track.notes:
                total_notes += 1
                if note.time % sixteenth_note == 0:
                    grid_aligned += 1

        return grid_aligned / total_notes if total_notes > 0 else 0.0

    def _detect_swing_characteristics(self, music: muspy.Music) -> Dict[str, float]:
        """Detect swing/shuffle characteristics."""
        # Simplified swing detection
        if not music.tracks or not music.resolution:
            return {"swing_ratio": 0.0, "swing_detected": False}

        eighth_note = music.resolution // 2
        swing_count = 0
        straight_count = 0

        for track in music.tracks:
            sorted_notes = sorted(track.notes, key=lambda n: n.time)

            for i in range(len(sorted_notes) - 1):
                interval = sorted_notes[i + 1].time - sorted_notes[i].time

                # Check if interval suggests swing timing
                if abs(interval - eighth_note * 1.5) < abs(interval - eighth_note):
                    swing_count += 1
                else:
                    straight_count += 1

        total = swing_count + straight_count
        swing_ratio = swing_count / total if total > 0 else 0.0

        return {"swing_ratio": swing_ratio, "swing_detected": swing_ratio > 0.6}

    def _calculate_polyphonic_complexity(self, music: muspy.Music) -> float:
        """Calculate polyphonic complexity."""
        if not music.tracks:
            return 0.0

        # Average number of simultaneous notes
        time_points = set()
        for track in music.tracks:
            for note in track.notes:
                time_points.add(note.time)
                time_points.add(note.time + note.duration)

        if not time_points:
            return 0.0

        simultaneous_counts = []
        for time_point in time_points:
            count = 0
            for track in music.tracks:
                for note in track.notes:
                    if note.time <= time_point < note.time + note.duration:
                        count += 1
            simultaneous_counts.append(count)

        return float(np.mean(simultaneous_counts)) if simultaneous_counts else 0.0

    def _calculate_harmonic_complexity(self, music: muspy.Music) -> float:
        """Calculate harmonic complexity."""
        try:
            # Based on pitch class entropy
            return muspy.pitch_class_entropy(music)
        except:
            return 0.0

    def _calculate_rhythmic_diversity(self, music: muspy.Music) -> float:
        """Calculate rhythmic diversity."""
        if not music.tracks:
            return 0.0

        # Count unique inter-onset intervals
        all_iois = []
        for track in music.tracks:
            sorted_notes = sorted(track.notes, key=lambda n: n.time)
            iois = [
                sorted_notes[i + 1].time - sorted_notes[i].time
                for i in range(len(sorted_notes) - 1)
            ]
            all_iois.extend(iois)

        if not all_iois:
            return 0.0

        # Quantize IOIs to reduce noise
        quantized_iois = [
            round(ioi / (music.resolution // 4)) * (music.resolution // 4)
            for ioi in all_iois
        ]
        unique_iois = len(set(quantized_iois))

        return unique_iois / len(quantized_iois) if quantized_iois else 0.0

    def _calculate_melodic_complexity(self, music: muspy.Music) -> float:
        """Calculate melodic complexity."""
        if not music.tracks:
            return 0.0

        # Find main melodic track
        main_track = max(music.tracks, key=lambda t: len(t.notes))
        if len(main_track.notes) < 3:
            return 0.0

        sorted_notes = sorted(main_track.notes, key=lambda n: n.time)
        pitches = [note.pitch for note in sorted_notes]

        # Count unique intervals and directions
        intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
        unique_intervals = len(set(intervals))

        # Direction changes
        directions = [
            1 if interval > 0 else -1 if interval < 0 else 0 for interval in intervals
        ]
        direction_changes = sum(
            1
            for i in range(len(directions) - 1)
            if directions[i] != directions[i + 1]
            and directions[i] != 0
            and directions[i + 1] != 0
        )

        complexity = (
            (unique_intervals + direction_changes) / len(intervals)
            if intervals
            else 0.0
        )
        return min(complexity, 1.0)  # Cap at 1.0

    def _calculate_overall_complexity(self, music: muspy.Music) -> float:
        """Calculate overall musical complexity score."""
        try:
            # Weighted combination of different complexity measures
            poly_complexity = self._calculate_polyphonic_complexity(music)
            harmonic_complexity = self._calculate_harmonic_complexity(music)
            rhythmic_diversity = self._calculate_rhythmic_diversity(music)
            melodic_complexity = self._calculate_melodic_complexity(music)

            # Weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # Adjust weights as needed
            complexities = [
                poly_complexity,
                harmonic_complexity,
                rhythmic_diversity,
                melodic_complexity,
            ]

            overall = sum(w * c for w, c in zip(weights, complexities))
            return float(overall)

        except Exception:
            return 0.0


def load_and_analyze_sequence(
    sequence_path: str, encoding_path: str = None
) -> Dict[str, Any]:
    """
    Load a sequence from file and perform complete MIDI analysis.

    Args:
        sequence_path: Path to tensor/npy file containing sequence
        encoding_path: Path to encoding file

    Returns:
        Complete analysis results
    """
    # Load sequence
    if sequence_path.endswith(".pt"):
        import torch

        data = torch.load(sequence_path, map_location="cpu")
        if "generated" in data:
            sequence = data["generated"].numpy()
        elif "sequence" in data:
            sequence = data["sequence"].numpy()
        else:
            sequence = data.numpy() if hasattr(data, "numpy") else data
    elif sequence_path.endswith(".npy"):
        sequence = np.load(sequence_path)
    else:
        raise ValueError(f"Unsupported file format: {sequence_path}")

    # Handle batch dimension
    if len(sequence.shape) == 3 and sequence.shape[0] == 1:
        sequence = sequence[0]

    # Extract condition info from filename if available
    condition_info = {"source_file": sequence_path}

    # Initialize extractor and analyze
    if encoding_path is None:
        encoding_path = "data/sod/processed/notes/encoding.json"

    extractor = MIDIFeatureExtractor(encoding_path)
    features = extractor.extract_all_features(sequence, condition_info)

    return features


if __name__ == "__main__":
    # Test the extractor
    import argparse

    parser = argparse.ArgumentParser(description="Test MIDI feature extraction")
    parser.add_argument("--sequence-path", required=True, help="Path to sequence file")
    parser.add_argument(
        "--encoding-path",
        default="data/sod/processed/notes/encoding.json",
        help="Path to encoding",
    )
    parser.add_argument("--output-path", help="Path to save analysis results")

    args = parser.parse_args()

    # Analyze sequence
    results = load_and_analyze_sequence(args.sequence_path, args.encoding_path)

    # Print results
    print("MIDI Feature Analysis Results:")
    print("=" * 50)
    for category, features in results.items():
        if isinstance(features, dict) and "error" not in features:
            print(f"\n{category.upper()}:")
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    # Save results if requested
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_path}")
