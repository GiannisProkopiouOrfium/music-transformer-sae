#!/usr/bin/env python3
"""
Music Quality Assessment Module

Provides comprehensive music quality assessment capabilities for deterministic analysis.
Evaluates musical coherence, structural integrity, and aesthetic quality of generated music.

This module implements multiple quality assessment approaches:
- Rule-based musical constraints and conventions
- Statistical analysis of musical properties
- Comparative quality assessment (before/after intervention)
- Music theory-based evaluation criteria
- Aesthetic quality indicators
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class QualityDimension(Enum):
    """Different dimensions of musical quality."""

    HARMONIC_COHERENCE = "harmonic_coherence"
    RHYTHMIC_CONSISTENCY = "rhythmic_consistency"
    MELODIC_FLOW = "melodic_flow"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    AESTHETIC_APPEAL = "aesthetic_appeal"
    TECHNICAL_CORRECTNESS = "technical_correctness"


@dataclass
class QualityThreshold:
    """Quality threshold configuration for different musical aspects."""

    dimension: QualityDimension
    excellent_min: float
    good_min: float
    acceptable_min: float
    weight: float = 1.0


class MusicQualityAssessor:
    """Comprehensive music quality assessment system."""

    def __init__(self):
        """Initialize quality assessor with thresholds and weights."""

        self.logger = logging.getLogger(__name__)

        # Define quality thresholds for different dimensions
        self.quality_thresholds = {
            QualityDimension.HARMONIC_COHERENCE: QualityThreshold(
                dimension=QualityDimension.HARMONIC_COHERENCE,
                excellent_min=0.8,
                good_min=0.65,
                acceptable_min=0.5,
                weight=0.25,
            ),
            QualityDimension.RHYTHMIC_CONSISTENCY: QualityThreshold(
                dimension=QualityDimension.RHYTHMIC_CONSISTENCY,
                excellent_min=0.75,
                good_min=0.6,
                acceptable_min=0.45,
                weight=0.2,
            ),
            QualityDimension.MELODIC_FLOW: QualityThreshold(
                dimension=QualityDimension.MELODIC_FLOW,
                excellent_min=0.7,
                good_min=0.55,
                acceptable_min=0.4,
                weight=0.2,
            ),
            QualityDimension.STRUCTURAL_INTEGRITY: QualityThreshold(
                dimension=QualityDimension.STRUCTURAL_INTEGRITY,
                excellent_min=0.75,
                good_min=0.6,
                acceptable_min=0.45,
                weight=0.15,
            ),
            QualityDimension.AESTHETIC_APPEAL: QualityThreshold(
                dimension=QualityDimension.AESTHETIC_APPEAL,
                excellent_min=0.7,
                good_min=0.55,
                acceptable_min=0.4,
                weight=0.1,
            ),
            QualityDimension.TECHNICAL_CORRECTNESS: QualityThreshold(
                dimension=QualityDimension.TECHNICAL_CORRECTNESS,
                excellent_min=0.9,
                good_min=0.8,
                acceptable_min=0.7,
                weight=0.1,
            ),
        }

        # Musical constraints and rules
        self.musical_constraints = self._initialize_musical_constraints()

    def _initialize_musical_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize musical constraints and rules for quality assessment."""

        return {
            "pitch_constraints": {
                "reasonable_range": {"min_pitch": 21, "max_pitch": 108},  # A0 to C8
                "interval_limits": {
                    "max_leap": 24,
                    "preferred_max": 12,
                },  # 2 octaves max, 1 preferred
                "voice_leading": {"max_voice_crossing": 0.1},  # 10% max voice crossing
            },
            "rhythm_constraints": {
                "timing_precision": {"min_precision": 0.8, "preferred_precision": 0.9},
                "beat_consistency": {
                    "min_consistency": 0.7,
                    "preferred_consistency": 0.85,
                },
                "tempo_stability": {"max_deviation": 0.2},  # 20% tempo deviation max
            },
            "harmony_constraints": {
                "scale_consistency": {
                    "min_consistency": 0.6,
                    "preferred_consistency": 0.8,
                },
                "chord_progression": {"min_coherence": 0.5, "preferred_coherence": 0.7},
                "dissonance_handling": {
                    "max_unresolved": 0.3
                },  # 30% max unresolved dissonance
            },
            "structure_constraints": {
                "phrase_coherence": {"min_coherence": 0.4, "preferred_coherence": 0.6},
                "form_clarity": {"min_clarity": 0.5, "preferred_clarity": 0.7},
                "repetition_balance": {
                    "min_ratio": 0.2,
                    "max_ratio": 0.8,
                },  # 20-80% repetition
            },
        }

    def assess_comprehensive_quality(
        self, musical_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment across all dimensions.

        Args:
            musical_features: Extracted musical features dictionary

        Returns:
            Comprehensive quality assessment results
        """

        quality_assessment = {
            "overall_quality_score": 0.0,
            "dimension_scores": {},
            "constraint_violations": [],
            "quality_indicators": {},
            "improvement_suggestions": [],
            "quality_level": "unknown",
        }

        # Assess each quality dimension
        for dimension, threshold in self.quality_thresholds.items():
            dimension_score = self._assess_quality_dimension(
                dimension, musical_features
            )
            quality_assessment["dimension_scores"][dimension.value] = {
                "score": dimension_score,
                "level": self._get_quality_level(dimension_score, threshold),
                "weight": threshold.weight,
            }

        # Calculate overall quality score
        weighted_scores = []
        total_weight = 0

        for dimension_data in quality_assessment["dimension_scores"].values():
            weighted_scores.append(dimension_data["score"] * dimension_data["weight"])
            total_weight += dimension_data["weight"]

        if total_weight > 0:
            quality_assessment["overall_quality_score"] = (
                sum(weighted_scores) / total_weight
            )

        # Check constraint violations
        quality_assessment["constraint_violations"] = self._check_constraint_violations(
            musical_features
        )

        # Generate quality indicators
        quality_assessment["quality_indicators"] = self._generate_quality_indicators(
            musical_features
        )

        # Generate improvement suggestions
        quality_assessment["improvement_suggestions"] = (
            self._generate_improvement_suggestions(
                quality_assessment["dimension_scores"],
                quality_assessment["constraint_violations"],
            )
        )

        # Determine overall quality level
        quality_assessment["quality_level"] = self._determine_overall_quality_level(
            quality_assessment["overall_quality_score"]
        )

        return quality_assessment

    def _assess_quality_dimension(
        self, dimension: QualityDimension, features: Dict[str, Any]
    ) -> float:
        """
        Assess quality for a specific dimension.

        Args:
            dimension: Quality dimension to assess
            features: Musical features dictionary

        Returns:
            Quality score for this dimension (0.0 to 1.0)
        """

        if dimension == QualityDimension.HARMONIC_COHERENCE:
            return self._assess_harmonic_coherence(features)

        elif dimension == QualityDimension.RHYTHMIC_CONSISTENCY:
            return self._assess_rhythmic_consistency(features)

        elif dimension == QualityDimension.MELODIC_FLOW:
            return self._assess_melodic_flow(features)

        elif dimension == QualityDimension.STRUCTURAL_INTEGRITY:
            return self._assess_structural_integrity(features)

        elif dimension == QualityDimension.AESTHETIC_APPEAL:
            return self._assess_aesthetic_appeal(features)

        elif dimension == QualityDimension.TECHNICAL_CORRECTNESS:
            return self._assess_technical_correctness(features)

        return 0.0

    def _assess_harmonic_coherence(self, features: Dict[str, Any]) -> float:
        """Assess harmonic coherence and consistency."""

        score_components = []

        # Scale consistency
        scale_consistency = self._get_nested_value(
            features, "harmonic_analysis.scale_consistency", 0.0
        )
        score_components.append(min(scale_consistency, 1.0))

        # Pitch class entropy (lower entropy = more tonal coherence)
        entropy = self._get_nested_value(
            features, "harmonic_analysis.pitch_class_entropy", 3.0
        )
        # Normalize entropy score (typical range 0-4, invert so lower is better)
        entropy_score = max(0, 1 - (entropy / 4.0))
        score_components.append(entropy_score)

        # Harmonic complexity (moderate complexity is good)
        complexity = self._get_nested_value(
            features, "harmonic_analysis.harmonic_complexity", 0.0
        )
        # Optimal complexity around 0.5-0.7
        if 0.5 <= complexity <= 0.7:
            complexity_score = 1.0
        elif complexity < 0.5:
            complexity_score = complexity / 0.5
        else:
            complexity_score = max(0, 1 - (complexity - 0.7) / 0.3)
        score_components.append(complexity_score)

        # Chord count (reasonable number of chords)
        chord_count = self._get_nested_value(
            features, "harmonic_analysis.chord_count", 0
        )
        total_notes = self._get_nested_value(features, "basic_info.total_notes", 1)
        chord_density = chord_count / max(total_notes, 1)

        # Reasonable chord density: 0.1 to 0.5
        if 0.1 <= chord_density <= 0.5:
            chord_score = 1.0
        elif chord_density < 0.1:
            chord_score = chord_density / 0.1
        else:
            chord_score = max(0, 1 - (chord_density - 0.5) / 0.5)
        score_components.append(chord_score)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _assess_rhythmic_consistency(self, features: Dict[str, Any]) -> float:
        """Assess rhythmic consistency and regularity."""

        score_components = []

        # Timing precision
        timing_precision = self._get_nested_value(
            features, "temporal_analysis.timing_precision", 0.0
        )
        score_components.append(timing_precision)

        # Rhythmic regularity
        rhythmic_regularity = self._get_nested_value(
            features, "rhythmic_analysis.rhythmic_regularity", 0.0
        )
        score_components.append(rhythmic_regularity)

        # Groove consistency
        groove_consistency = self._get_nested_value(
            features, "temporal_analysis.groove_consistency", 0.0
        )
        score_components.append(groove_consistency)

        # Beat strength distribution (should be reasonable)
        beat_strength = self._get_nested_value(
            features, "rhythmic_analysis.beat_strength_distribution", []
        )
        if beat_strength and len(beat_strength) > 0:
            # Beat strength should have clear hierarchy
            beat_variance = (
                max(beat_strength) - min(beat_strength) if len(beat_strength) > 1 else 0
            )
            beat_score = min(beat_variance / 0.5, 1.0)  # Normalize to 0-1
            score_components.append(beat_score)

        # Inter-onset interval consistency
        ioi_std = self._get_nested_value(features, "temporal_analysis.ioi_std", 0.0)
        # Lower standard deviation is better (more consistent)
        ioi_score = max(0, 1 - (ioi_std / 1.0))  # Assuming reasonable range 0-1
        score_components.append(ioi_score)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _assess_melodic_flow(self, features: Dict[str, Any]) -> float:
        """Assess melodic flow and coherence."""

        score_components = []

        # Step motion ratio (smooth melodic motion)
        step_motion = self._get_nested_value(
            features, "pitch_analysis.step_motion_ratio", 0.0
        )
        score_components.append(step_motion)

        # Average interval size (moderate intervals are good)
        avg_interval = self._get_nested_value(
            features, "pitch_analysis.average_interval_size", 0.0
        )
        # Optimal interval size: 1-3 semitones
        if 1 <= avg_interval <= 3:
            interval_score = 1.0
        elif avg_interval < 1:
            interval_score = avg_interval
        else:
            interval_score = max(
                0, 1 - (avg_interval - 3) / 9
            )  # Degrade up to 12 semitones
        score_components.append(interval_score)

        # Melodic complexity (moderate complexity is good)
        melodic_complexity = self._get_nested_value(
            features, "pitch_analysis.melodic_complexity", 0.0
        )
        # Optimal complexity around 0.4-0.7
        if 0.4 <= melodic_complexity <= 0.7:
            complexity_score = 1.0
        elif melodic_complexity < 0.4:
            complexity_score = melodic_complexity / 0.4
        else:
            complexity_score = max(0, 1 - (melodic_complexity - 0.7) / 0.3)
        score_components.append(complexity_score)

        # Pitch range (reasonable range)
        pitch_range = self._get_nested_value(features, "pitch_analysis.pitch_range", 0)
        # Optimal range: 12-36 semitones (1-3 octaves)
        if 12 <= pitch_range <= 36:
            range_score = 1.0
        elif pitch_range < 12:
            range_score = pitch_range / 12
        else:
            range_score = max(0, 1 - (pitch_range - 36) / 48)  # Degrade up to 7 octaves
        score_components.append(range_score)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _assess_structural_integrity(self, features: Dict[str, Any]) -> float:
        """Assess structural integrity and coherence."""

        score_components = []

        # Structural coherence
        structural_coherence = self._get_nested_value(
            features, "structural_analysis.structural_coherence", 0.0
        )
        score_components.append(structural_coherence)

        # Phrase count (reasonable number of phrases)
        phrase_count = self._get_nested_value(
            features, "structural_analysis.phrase_count", 0
        )
        duration = self._get_nested_value(features, "basic_info.duration", 1)
        phrases_per_minute = phrase_count / max(duration / 60, 1)

        # Reasonable phrase density: 2-8 phrases per minute
        if 2 <= phrases_per_minute <= 8:
            phrase_score = 1.0
        elif phrases_per_minute < 2:
            phrase_score = phrases_per_minute / 2
        else:
            phrase_score = max(0, 1 - (phrases_per_minute - 8) / 8)
        score_components.append(phrase_score)

        # Repetition ratio (balanced repetition)
        repetition_ratio = self._get_nested_value(
            features, "structural_analysis.repetition_ratio", 0.0
        )
        # Optimal repetition: 0.3-0.7
        if 0.3 <= repetition_ratio <= 0.7:
            repetition_score = 1.0
        elif repetition_ratio < 0.3:
            repetition_score = repetition_ratio / 0.3
        else:
            repetition_score = max(0, 1 - (repetition_ratio - 0.7) / 0.3)
        score_components.append(repetition_score)

        # Average phrase length consistency
        avg_phrase_length = self._get_nested_value(
            features, "structural_analysis.average_phrase_length", 0
        )
        phrase_length_std = self._get_nested_value(
            features, "structural_analysis.phrase_length_std", 0
        )

        if avg_phrase_length > 0:
            phrase_consistency = max(0, 1 - (phrase_length_std / avg_phrase_length))
            score_components.append(phrase_consistency)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _assess_aesthetic_appeal(self, features: Dict[str, Any]) -> float:
        """Assess aesthetic appeal based on musical preferences."""

        score_components = []

        # Dynamic range (expressive dynamics)
        velocity_range = self._get_nested_value(
            features, "velocity_dynamics.velocity_range", 0
        )
        # Good dynamic range: 40-80 MIDI velocity units
        if 40 <= velocity_range <= 80:
            dynamic_score = 1.0
        elif velocity_range < 40:
            dynamic_score = velocity_range / 40
        else:
            dynamic_score = max(0, 1 - (velocity_range - 80) / 47)  # Max 127
        score_components.append(dynamic_score)

        # Rhythmic diversity (not too repetitive, not too chaotic)
        rhythmic_diversity = self._get_nested_value(
            features, "rhythmic_analysis.rhythmic_diversity", 0.0
        )
        # Optimal diversity: 0.4-0.8
        if 0.4 <= rhythmic_diversity <= 0.8:
            diversity_score = 1.0
        elif rhythmic_diversity < 0.4:
            diversity_score = rhythmic_diversity / 0.4
        else:
            diversity_score = max(0, 1 - (rhythmic_diversity - 0.8) / 0.2)
        score_components.append(diversity_score)

        # Polyphonic complexity (reasonable texture)
        polyphonic_complexity = self._get_nested_value(
            features, "musical_complexity.polyphonic_complexity", 0.0
        )
        # Optimal complexity: 0.3-0.7
        if 0.3 <= polyphonic_complexity <= 0.7:
            texture_score = 1.0
        elif polyphonic_complexity < 0.3:
            texture_score = polyphonic_complexity / 0.3
        else:
            texture_score = max(0, 1 - (polyphonic_complexity - 0.7) / 0.3)
        score_components.append(texture_score)

        # Note density (not too sparse, not too dense)
        notes_per_beat = self._get_nested_value(
            features, "note_patterns.notes_per_beat", 0.0
        )
        # Optimal density: 1-4 notes per beat
        if 1 <= notes_per_beat <= 4:
            density_score = 1.0
        elif notes_per_beat < 1:
            density_score = notes_per_beat
        else:
            density_score = max(0, 1 - (notes_per_beat - 4) / 6)  # Degrade up to 10
        score_components.append(density_score)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _assess_technical_correctness(self, features: Dict[str, Any]) -> float:
        """Assess technical correctness of the music."""

        score_components = []

        # Check basic validity
        total_notes = self._get_nested_value(features, "basic_info.total_notes", 0)
        if total_notes > 0:
            score_components.append(1.0)
        else:
            score_components.append(0.0)

        # Check pitch range validity
        min_pitch = self._get_nested_value(features, "pitch_analysis.min_pitch", 60)
        max_pitch = self._get_nested_value(features, "pitch_analysis.max_pitch", 60)

        if 21 <= min_pitch <= 108 and 21 <= max_pitch <= 108 and min_pitch <= max_pitch:
            pitch_validity = 1.0
        else:
            pitch_validity = 0.5
        score_components.append(pitch_validity)

        # Check duration validity
        duration = self._get_nested_value(features, "basic_info.duration", 0)
        if duration > 0:
            duration_validity = 1.0
        else:
            duration_validity = 0.0
        score_components.append(duration_validity)

        # Check note duration validity
        avg_note_duration = self._get_nested_value(
            features, "note_patterns.average_note_duration", 0
        )
        if avg_note_duration > 0:
            note_duration_validity = 1.0
        else:
            note_duration_validity = 0.5
        score_components.append(note_duration_validity)

        # Check tempo consistency (if available)
        tempo_changes = self._get_nested_value(
            features, "temporal_analysis.tempo_changes", 0
        )
        duration_beats = duration * 2  # Rough estimate

        if duration_beats > 0:
            tempo_change_rate = tempo_changes / duration_beats
            # Reasonable tempo change rate: < 0.1 changes per beat
            if tempo_change_rate <= 0.1:
                tempo_validity = 1.0
            else:
                tempo_validity = max(0, 1 - (tempo_change_rate - 0.1) / 0.5)
            score_components.append(tempo_validity)

        return (
            sum(score_components) / len(score_components) if score_components else 0.0
        )

    def _check_constraint_violations(
        self, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for violations of musical constraints."""

        violations = []

        # Check pitch constraints
        min_pitch = self._get_nested_value(features, "pitch_analysis.min_pitch", 60)
        max_pitch = self._get_nested_value(features, "pitch_analysis.max_pitch", 60)

        constraints = self.musical_constraints["pitch_constraints"]
        if min_pitch < constraints["reasonable_range"]["min_pitch"]:
            violations.append(
                {
                    "type": "pitch_range",
                    "severity": "warning",
                    "description": f"Minimum pitch {min_pitch} below reasonable range",
                }
            )

        if max_pitch > constraints["reasonable_range"]["max_pitch"]:
            violations.append(
                {
                    "type": "pitch_range",
                    "severity": "warning",
                    "description": f"Maximum pitch {max_pitch} above reasonable range",
                }
            )

        # Check interval constraints
        largest_leap = self._get_nested_value(
            features, "pitch_analysis.largest_leap", 0
        )
        if largest_leap > constraints["interval_limits"]["max_leap"]:
            violations.append(
                {
                    "type": "interval_leap",
                    "severity": "error",
                    "description": f"Interval leap {largest_leap} exceeds maximum allowed",
                }
            )
        elif largest_leap > constraints["interval_limits"]["preferred_max"]:
            violations.append(
                {
                    "type": "interval_leap",
                    "severity": "warning",
                    "description": f"Interval leap {largest_leap} exceeds preferred maximum",
                }
            )

        # Check rhythm constraints
        timing_precision = self._get_nested_value(
            features, "temporal_analysis.timing_precision", 1.0
        )
        rhythm_constraints = self.musical_constraints["rhythm_constraints"]

        if timing_precision < rhythm_constraints["timing_precision"]["min_precision"]:
            violations.append(
                {
                    "type": "timing_precision",
                    "severity": "warning",
                    "description": f"Timing precision {timing_precision:.2f} below minimum threshold",
                }
            )

        # Check harmony constraints
        scale_consistency = self._get_nested_value(
            features, "harmonic_analysis.scale_consistency", 1.0
        )
        harmony_constraints = self.musical_constraints["harmony_constraints"]

        if (
            scale_consistency
            < harmony_constraints["scale_consistency"]["min_consistency"]
        ):
            violations.append(
                {
                    "type": "harmonic_consistency",
                    "severity": "warning",
                    "description": f"Scale consistency {scale_consistency:.2f} below minimum threshold",
                }
            )

        return violations

    def _generate_quality_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality indicators and metrics."""

        indicators = {
            "complexity_balance": self._assess_complexity_balance(features),
            "musical_interest": self._assess_musical_interest(features),
            "coherence_measures": self._assess_coherence_measures(features),
            "expressiveness": self._assess_expressiveness(features),
        }

        return indicators

    def _assess_complexity_balance(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Assess balance of complexity across different musical dimensions."""

        rhythmic_complexity = self._get_nested_value(
            features, "rhythmic_analysis.rhythmic_complexity", 0.0
        )
        harmonic_complexity = self._get_nested_value(
            features, "harmonic_analysis.harmonic_complexity", 0.0
        )
        melodic_complexity = self._get_nested_value(
            features, "pitch_analysis.melodic_complexity", 0.0
        )
        overall_complexity = self._get_nested_value(
            features, "musical_complexity.overall_complexity_score", 0.0
        )

        # Calculate balance (lower standard deviation = better balance)
        complexities = [rhythmic_complexity, harmonic_complexity, melodic_complexity]
        if len(complexities) > 1:
            import statistics

            complexity_std = statistics.stdev(complexities)
            balance_score = max(0, 1 - complexity_std)
        else:
            balance_score = 1.0

        return {
            "rhythmic_complexity": rhythmic_complexity,
            "harmonic_complexity": harmonic_complexity,
            "melodic_complexity": melodic_complexity,
            "overall_complexity": overall_complexity,
            "balance_score": balance_score,
        }

    def _assess_musical_interest(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Assess musical interest and engagement factors."""

        rhythmic_diversity = self._get_nested_value(
            features, "rhythmic_analysis.rhythmic_diversity", 0.0
        )
        unique_pitches = self._get_nested_value(
            features, "pitch_analysis.unique_pitches", 0
        )
        total_notes = self._get_nested_value(features, "basic_info.total_notes", 1)

        pitch_diversity = unique_pitches / max(total_notes, 1)
        velocity_range = self._get_nested_value(
            features, "velocity_dynamics.velocity_range", 0
        )

        # Interest score based on diversity measures
        interest_factors = [rhythmic_diversity, pitch_diversity, velocity_range / 127]
        interest_score = sum(interest_factors) / len(interest_factors)

        return {
            "rhythmic_diversity": rhythmic_diversity,
            "pitch_diversity": pitch_diversity,
            "dynamic_range": velocity_range / 127,
            "overall_interest": interest_score,
        }

    def _assess_coherence_measures(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Assess various coherence measures."""

        structural_coherence = self._get_nested_value(
            features, "structural_analysis.structural_coherence", 0.0
        )
        scale_consistency = self._get_nested_value(
            features, "harmonic_analysis.scale_consistency", 0.0
        )
        rhythmic_regularity = self._get_nested_value(
            features, "rhythmic_analysis.rhythmic_regularity", 0.0
        )
        groove_consistency = self._get_nested_value(
            features, "temporal_analysis.groove_consistency", 0.0
        )

        overall_coherence = (
            structural_coherence
            + scale_consistency
            + rhythmic_regularity
            + groove_consistency
        ) / 4

        return {
            "structural_coherence": structural_coherence,
            "harmonic_coherence": scale_consistency,
            "rhythmic_coherence": rhythmic_regularity,
            "temporal_coherence": groove_consistency,
            "overall_coherence": overall_coherence,
        }

    def _assess_expressiveness(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Assess musical expressiveness."""

        velocity_range = self._get_nested_value(
            features, "velocity_dynamics.velocity_range", 0
        )
        dynamic_variance = self._get_nested_value(
            features, "velocity_dynamics.dynamic_variance", 0.0
        )
        timing_precision = self._get_nested_value(
            features, "temporal_analysis.timing_precision", 1.0
        )

        # Expressiveness often involves controlled imprecision
        timing_expressiveness = (
            1 - timing_precision if timing_precision > 0.8 else timing_precision
        )
        dynamic_expressiveness = min(
            velocity_range / 80, 1.0
        )  # Normalize to reasonable range

        overall_expressiveness = (
            timing_expressiveness + dynamic_expressiveness + dynamic_variance
        ) / 3

        return {
            "dynamic_expressiveness": dynamic_expressiveness,
            "timing_expressiveness": timing_expressiveness,
            "dynamic_variance": dynamic_variance,
            "overall_expressiveness": overall_expressiveness,
        }

    def _generate_improvement_suggestions(
        self, dimension_scores: Dict[str, Dict], violations: List[Dict]
    ) -> List[str]:
        """Generate improvement suggestions based on assessment results."""

        suggestions = []

        # Check each dimension for improvement opportunities
        for dimension, data in dimension_scores.items():
            score = data["score"]
            level = data["level"]

            if level in ["poor", "unacceptable"] or score < 0.5:
                if dimension == "harmonic_coherence":
                    suggestions.append(
                        "Improve harmonic coherence by maintaining consistent scale/key center"
                    )
                elif dimension == "rhythmic_consistency":
                    suggestions.append(
                        "Enhance rhythmic consistency with more regular beat patterns"
                    )
                elif dimension == "melodic_flow":
                    suggestions.append(
                        "Improve melodic flow with smoother interval progressions"
                    )
                elif dimension == "structural_integrity":
                    suggestions.append(
                        "Strengthen structural integrity with clearer phrase boundaries"
                    )
                elif dimension == "aesthetic_appeal":
                    suggestions.append(
                        "Enhance aesthetic appeal with better dynamic range and texture"
                    )
                elif dimension == "technical_correctness":
                    suggestions.append(
                        "Address technical issues with note ranges and durations"
                    )

        # Add suggestions based on constraint violations
        for violation in violations:
            if violation["severity"] == "error":
                suggestions.append(
                    f"Fix {violation['type']}: {violation['description']}"
                )
            elif violation["severity"] == "warning" and len(suggestions) < 5:
                suggestions.append(
                    f"Consider addressing {violation['type']}: {violation['description']}"
                )

        return suggestions[:5]  # Limit to top 5 suggestions

    def _get_quality_level(self, score: float, threshold: QualityThreshold) -> str:
        """Get quality level string based on score and thresholds."""

        if score >= threshold.excellent_min:
            return "excellent"
        elif score >= threshold.good_min:
            return "good"
        elif score >= threshold.acceptable_min:
            return "acceptable"
        else:
            return "poor"

    def _determine_overall_quality_level(self, overall_score: float) -> str:
        """Determine overall quality level from overall score."""

        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.65:
            return "good"
        elif overall_score >= 0.5:
            return "acceptable"
        elif overall_score >= 0.3:
            return "poor"
        else:
            return "unacceptable"

    def _get_nested_value(self, data: Dict, key_path: str, default: Any = None) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split(".")
        value = data

        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def compare_quality(
        self, baseline_features: Dict[str, Any], intervention_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare quality between baseline and intervention conditions.

        Args:
            baseline_features: Features from baseline condition
            intervention_features: Features from intervention condition

        Returns:
            Quality comparison results
        """

        baseline_quality = self.assess_comprehensive_quality(baseline_features)
        intervention_quality = self.assess_comprehensive_quality(intervention_features)

        comparison = {
            "baseline_quality": baseline_quality,
            "intervention_quality": intervention_quality,
            "quality_change": {
                "overall_score_change": (
                    intervention_quality["overall_quality_score"]
                    - baseline_quality["overall_quality_score"]
                ),
                "dimension_changes": {},
                "quality_preserved": True,
                "improvement_areas": [],
                "degradation_areas": [],
            },
        }

        # Compare each dimension
        for dimension in baseline_quality["dimension_scores"]:
            baseline_score = baseline_quality["dimension_scores"][dimension]["score"]
            intervention_score = intervention_quality["dimension_scores"][dimension][
                "score"
            ]
            change = intervention_score - baseline_score

            comparison["quality_change"]["dimension_changes"][dimension] = {
                "baseline_score": baseline_score,
                "intervention_score": intervention_score,
                "change": change,
                "change_percent": (change / max(baseline_score, 0.01)) * 100,
            }

            # Track improvements and degradations
            if change > 0.05:  # 5% improvement threshold
                comparison["quality_change"]["improvement_areas"].append(dimension)
            elif change < -0.05:  # 5% degradation threshold
                comparison["quality_change"]["degradation_areas"].append(dimension)

        # Overall quality preservation assessment
        overall_change = comparison["quality_change"]["overall_score_change"]
        if overall_change < -0.1:  # 10% overall degradation
            comparison["quality_change"]["quality_preserved"] = False

        return comparison


if __name__ == "__main__":
    # Test the quality assessor
    assessor = MusicQualityAssessor()

    print("Music Quality Assessor initialized")
    print(f"Quality dimensions: {[d.value for d in QualityDimension]}")
    print(f"Musical constraints defined: {list(assessor.musical_constraints.keys())}")
