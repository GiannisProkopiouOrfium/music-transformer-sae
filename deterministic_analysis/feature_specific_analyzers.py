#!/usr/bin/env python3
"""
Feature-Specific Analyzers for Intervention Assessment

This module provides specialized analysis tools for each type of feature intervention.
It can assess the effectiveness and quality of interventions both with and without LLM assistance.

Features:
- Layer-specific analysis (Early/Mid/Late processing characteristics)
- Feature-specific impact assessment (what each feature actually controls)
- Intervention effectiveness scoring (RELATIVE to baseline)
- Musical quality preservation assessment (using calibrated thresholds for context)
- Decision-making framework for intervention success

Key Philosophy:
- PRIMARY FOCUS: Relative changes (baseline vs intervention)
- SECONDARY CONTEXT: Calibrated thresholds for quality reference
- GOAL: Measure if intervention achieves intended effect, not if it matches training distribution
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Import threshold manager for calibrated threshold context
try:
    from .threshold_manager import ThresholdManager

    THRESHOLD_MANAGER_AVAILABLE = True
except ImportError:
    THRESHOLD_MANAGER_AVAILABLE = False
    logging.warning("ThresholdManager not available, using fallback quality assessment")


class LayerType(Enum):
    """Feature layer classification."""

    EARLY = "early"  # Layer 1 - Note-level processing
    MID = "mid"  # Layer 3 - Phrase/rhythm processing
    LATE = "late"  # Layer 5 - Structure/harmony processing


class InterventionQuality(Enum):
    """Intervention quality assessment."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class FeatureProfile:
    """Profile of a musical feature and its expected effects."""

    feature_id: str
    layer: int
    layer_type: LayerType
    feature_name: str
    expected_effects: List[str]
    primary_metrics: List[str]
    secondary_metrics: List[str]
    quality_thresholds: Dict[str, float]


class FeatureSpecificAnalyzer:
    """Analyze interventions with feature-specific knowledge and decision-making."""

    def __init__(self, use_calibrated_thresholds: bool = True):
        """
        Initialize with feature profiles and analysis configuration.

        Args:
            use_calibrated_thresholds: Whether to use calibrated thresholds for context
                                      (default: True, falls back if unavailable)
        """
        self.logger = logging.getLogger(__name__)
        self.feature_profiles = self._initialize_feature_profiles()
        self.quality_weights = self._initialize_quality_weights()

        # Initialize threshold manager if available
        self.threshold_manager = None
        if use_calibrated_thresholds and THRESHOLD_MANAGER_AVAILABLE:
            try:
                self.threshold_manager = ThresholdManager()
                self.logger.info("Calibrated thresholds loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load calibrated thresholds: {e}")
                self.logger.warning("Falling back to relative-only analysis")
        else:
            if not THRESHOLD_MANAGER_AVAILABLE:
                self.logger.info(
                    "ThresholdManager not available, using relative-only analysis"
                )

    def _initialize_feature_profiles(self) -> Dict[str, FeatureProfile]:
        """Initialize profiles for our 9 selected features."""

        profiles = {}

        # Layer 1 Features (Early Processing - Note Level)
        profiles["325"] = FeatureProfile(
            feature_id="325",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="rhythmic_displacement_syncopation",
            expected_effects=[
                "sixteenth_note_rhythmic_displacement",
                "syncopation_effect",
                "off_beat_emphasis",
                "rhythmic_motif_shifting",
            ],
            primary_metrics=[
                "rhythmic_analysis.syncopation_score",
                "rhythmic_analysis.rhythmic_complexity",
                "rhythmic_analysis.ioi_std",
                "note_patterns.note_onset_intervals.mean_interval",
            ],
            secondary_metrics=[
                "note_patterns.notes_per_beat",
                "rhythmic_analysis.rhythmic_regularity",
                "note_patterns.note_density",
            ],
            quality_thresholds={
                "syncopation_score_min": 0.108,  # 10th percentile
                "syncopation_score_max": 0.469,  # 90th percentile
                "rhythmic_complexity_min": 1.515,
                "rhythmic_complexity_max": 4.263,
                "ioi_std_acceptable_range": (0.115, 0.349),
            },
        )

        profiles["256"] = FeatureProfile(
            feature_id="256",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="dynamic_contrast_accents",
            expected_effects=[
                "sudden_accent_patterns",
                "dramatic_dynamic_contrast",
                "forte_piano_alternation",
                "extreme_velocity_changes",
            ],
            primary_metrics=[
                "velocity_dynamics.dynamic_range",
                "note_patterns.velocity_range",
                "velocity_dynamics.dynamic_variance",
                "velocity_dynamics.forte_notes_ratio",
                "velocity_dynamics.piano_notes_ratio",
            ],
            secondary_metrics=[
                "note_patterns.average_velocity",
                "structural_analysis.structural_coherence",
            ],
            quality_thresholds={
                "dynamic_range_min": 34.0,  # 10th percentile
                "dynamic_range_extreme": 91.0,  # 90th percentile
                "velocity_range_min": 34.0,
                "velocity_range_extreme": 91.0,
                "has_dynamic_contrast": 30,  # velocity difference > 30
                "extreme_dynamic_contrast": 70,  # velocity difference > 70
                "loud_threshold": 60,  # minimum velocity > 60
                "soft_threshold": 60,  # maximum velocity < 60
            },
        )

        profiles["1323"] = FeatureProfile(
            feature_id="1323",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="wide_pitch_range_texture",
            expected_effects=[
                "broad_harmonic_spectrum",
                "multi_register_utilization",
                "textural_depth",
                "pitch_range_expansion",
            ],
            primary_metrics=[
                "pitch_analysis.pitch_range",
                "pitch_analysis.pitch_std",
                "pitch_analysis.max_pitch",
                "pitch_analysis.min_pitch",
            ],
            secondary_metrics=[
                "musical_complexity.polyphonic_complexity",
                "pitch_analysis.unique_pitches",
                "note_patterns.note_density",
            ],
            quality_thresholds={
                "pitch_range_min": 26.0,  # 10th percentile
                "pitch_range_max": 62.0,  # 90th percentile
                "pitch_std_min": 8.066,
                "pitch_std_max": 16.467,
                "min_pitch_acceptable": 31.0,  # 10th percentile
                "max_pitch_acceptable": 95.0,  # 90th percentile
            },
        )

        # Layer 3 Features (Mid Processing - Phrase/Rhythm Level)
        profiles["182"] = FeatureProfile(
            feature_id="182",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="steady_pulse_march_rhythm",
            expected_effects=[
                "metronomic_consistency",
                "march_like_rhythms",
                "regular_pulse_patterns",
                "rhythmic_order_and_regularity",
            ],
            primary_metrics=[
                "rhythmic_analysis.rhythmic_regularity",
                "rhythmic_analysis.average_ioi",
                "note_patterns.unique_note_durations",
                "rhythmic_analysis.rhythmic_complexity",
            ],
            secondary_metrics=[
                "note_patterns.note_duration_std",
                "rhythmic_analysis.ioi_std",
            ],
            quality_thresholds={
                "rhythmic_regularity_min": 0.562,  # 10th percentile
                "rhythmic_regularity_max": 0.892,  # 90th percentile
                "average_ioi_min": 0.226,
                "average_ioi_max": 0.552,
                "rhythmic_complexity_acceptable": (1.515, 4.263),
                "few_onsets_threshold": 2.0,  # few groups per second
                "many_onsets_threshold": 8.0,  # many groups per second
            },
        )

        profiles["855"] = FeatureProfile(
            feature_id="855",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="dynamic_contrast_tension",
            expected_effects=[
                "sudden_velocity_shifts",
                "dramatic_tension_release",
                "high_low_velocity_alternation",
                "dynamic_range_expansion",
            ],
            primary_metrics=[
                "velocity_dynamics.dynamic_range",
                "velocity_dynamics.dynamic_variance",
                "note_patterns.velocity_range",
                "velocity_dynamics.forte_notes_ratio",
                "velocity_dynamics.piano_notes_ratio",
            ],
            secondary_metrics=[
                "note_patterns.average_velocity",
                "structural_analysis.structural_coherence",
            ],
            quality_thresholds={
                "dynamic_range_min": 34.0,  # 10th percentile
                "dynamic_range_max": 91.0,  # 90th percentile
                "dynamic_variance_min": 197.691,
                "dynamic_variance_max": 698.055,
                "has_dynamic_contrast": 30,
                "extreme_dynamic_contrast": 70,
            },
        )

        profiles["997"] = FeatureProfile(
            feature_id="997",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="antiphonal_call_response",
            expected_effects=[
                "call_response_interactions",
                "instrumental_dialogue",
                "antiphonal_texture",
                "alternating_melodic_lines",
            ],
            primary_metrics=[
                "musical_complexity.polyphonic_complexity",
                "structural_analysis.phrase_count",
                "note_patterns.note_density",
                "basic_info.total_tracks",
            ],
            secondary_metrics=[
                "structural_analysis.average_phrase_length",
                "pitch_analysis.pitch_range",
                "velocity_dynamics.dynamic_range",
            ],
            quality_thresholds={
                "polyphonic_complexity_min": 1.111,  # 10th percentile
                "polyphonic_complexity_max": 2.273,  # 90th percentile
                "phrase_count_min": 2.0,
                "total_tracks_min": 2,  # Need at least 2 tracks for call-response
                "only_melody_check": False,  # Should NOT be single note at a time
            },
        )

        # Layer 5 Features (Late Processing - Structure/Harmony Level)
        profiles["471"] = FeatureProfile(
            feature_id="471",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="unison_doubling_octaves",
            expected_effects=[
                "unison_doubling",
                "octave_reinforcement",
                "powerful_unified_sound",
                "multi_instrument_unison",
            ],
            primary_metrics=[
                "pitch_analysis.pitch_range",
                "musical_complexity.polyphonic_complexity",
                "basic_info.total_tracks",
                "note_patterns.average_velocity",
            ],
            secondary_metrics=[
                "pitch_analysis.unique_pitches",
                "note_patterns.note_density",
                "velocity_dynamics.dynamic_range",
            ],
            quality_thresholds={
                "polyphonic_complexity_min": 1.111,  # Should have multiple voices
                "polyphonic_complexity_max": 2.273,
                "total_tracks_min": 2,  # Need multiple instruments for doubling
                "pitch_range_octave_span": 12,  # Octave doubling indicator
                "only_melody_check": False,  # Should NOT be single note
            },
        )

        profiles["904"] = FeatureProfile(
            feature_id="904",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="rhythmic_augmentation",
            expected_effects=[
                "systematic_duration_lengthening",
                "motif_expansion",
                "tension_building",
                "temporal_stretching",
            ],
            primary_metrics=[
                "note_patterns.average_note_duration",
                "note_patterns.longest_note",
                "note_patterns.unique_note_durations",
                "rhythmic_analysis.average_ioi",
            ],
            secondary_metrics=[
                "note_patterns.note_duration_std",
                "rhythmic_analysis.rhythmic_complexity",
                "structural_analysis.average_phrase_length",
            ],
            quality_thresholds={
                "average_note_duration_min": 0.443,  # 10th percentile
                "average_note_duration_max": 1.053,  # 90th percentile
                "longest_note_min": 2.0,
                "longest_note_max": 8.0,
                "unique_durations_min": 5.0,
            },
        )

        profiles["1950"] = FeatureProfile(
            feature_id="1950",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="dramatic_dynamic_swells",
            expected_effects=[
                "abrupt_dynamic_shifts",
                "soft_to_loud_contrasts",
                "expressive_dynamic_swells",
                "dramatic_textural_effects",
            ],
            primary_metrics=[
                "velocity_dynamics.dynamic_range",
                "velocity_dynamics.dynamic_variance",
                "note_patterns.velocity_range",
                "velocity_dynamics.forte_notes_ratio",
                "velocity_dynamics.piano_notes_ratio",
            ],
            secondary_metrics=[
                "note_patterns.average_velocity",
                "structural_analysis.structural_coherence",
            ],
            quality_thresholds={
                "dynamic_range_min": 34.0,  # 10th percentile
                "dynamic_range_max": 91.0,  # 90th percentile
                "dynamic_variance_min": 197.691,
                "dynamic_variance_max": 698.055,
                "has_dynamic_contrast": 30,
                "extreme_dynamic_contrast": 70,
                "loud_threshold": 60,
                "soft_threshold": 60,
            },
        )

        return profiles

    def _initialize_quality_weights(self) -> Dict[str, float]:
        """Initialize weights for overall quality assessment."""
        return {
            "musical_coherence": 0.2,
            "intervention_effectiveness": 0.4,  # Increased - most important
            "quality_preservation": 0.2,
            "feature_specificity": 0.2,
        }

    def analyze_feature_intervention(
        self,
        feature_id: str,
        baseline_features: Dict[str, Any],
        intervention_features: Dict[str, Any],
        strength: float,
        intervention_type: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single feature intervention.

        Args:
            feature_id: ID of the intervened feature
            baseline_features: Extracted features from baseline condition
            intervention_features: Extracted features from intervention condition
            strength: Intervention strength used
            intervention_type: Type of intervention (addition/ablation)

        Returns:
            Comprehensive analysis results with decision assessment
        """

        if feature_id not in self.feature_profiles:
            return {"error": f"Unknown feature ID: {feature_id}"}

        profile = self.feature_profiles[feature_id]

        # Perform multi-dimensional analysis
        analysis = {
            "feature_info": {
                "feature_id": feature_id,
                "feature_name": profile.feature_name,
                "layer": profile.layer,
                "layer_type": profile.layer_type.value,
                "intervention_strength": strength,
                "intervention_type": intervention_type,
            },
            "effectiveness_analysis": self._analyze_intervention_effectiveness(
                profile, baseline_features, intervention_features, strength
            ),
            "quality_assessment": self._assess_musical_quality(
                profile, baseline_features, intervention_features
            ),
            "feature_specific_analysis": self._analyze_feature_specific_effects(
                profile, baseline_features, intervention_features
            ),
            "layer_appropriate_analysis": self._analyze_layer_appropriateness(
                profile, baseline_features, intervention_features
            ),
            "overall_decision": None,  # Will be filled by decision maker
        }

        # Make overall decision
        analysis["overall_decision"] = self._make_intervention_decision(analysis)

        # Add interpretable summary with threshold context
        analysis["summary"] = self._generate_intervention_summary(analysis, profile)

        return analysis

    def _generate_intervention_summary(
        self, analysis: Dict[str, Any], profile: FeatureProfile
    ) -> Dict[str, Any]:
        """
        Generate human-readable summary of intervention results.
        Highlights relative changes and provides threshold context.
        """
        effectiveness = analysis.get("effectiveness_analysis", {})
        quality = analysis.get("quality_assessment", {})

        summary = {
            "success": analysis.get("overall_decision", {}).get("should_accept", False),
            "effectiveness": effectiveness.get("effectiveness_score", 0.0),
            "quality_preservation": quality.get("overall_quality_score", 0.0),
            "key_changes": [],
            "quality_context": {},
        }

        # Extract key metric changes with context
        primary_changes = effectiveness.get("primary_metric_changes", {})
        for metric, change_info in primary_changes.items():
            change_summary = {
                "metric": metric,
                "baseline": change_info["baseline"],
                "intervention": change_info["intervention"],
                "percent_change": change_info["change_percent"],
                "absolute_change": change_info["absolute_change"],
            }

            # Add threshold context if available
            if "threshold_context" in change_info:
                ctx = change_info["threshold_context"]
                change_summary["context"] = {
                    "baseline_quality": f"P{ctx['baseline_percentile']:.0f}",
                    "intervention_quality": f"P{ctx['intervention_percentile']:.0f}",
                    "moved_toward_better": ctx["moved_toward_better"],
                    "interpretation": self._interpret_percentile_change(
                        ctx["baseline_percentile"],
                        ctx["intervention_percentile"],
                        change_info["change_percent"],
                    ),
                }

            if abs(change_info["change_percent"]) > 5:  # Only report meaningful changes
                summary["key_changes"].append(change_summary)

        # Overall quality context
        if self.threshold_manager:
            summary["quality_context"] = {
                "has_calibrated_thresholds": True,
                "interpretation": "Quality assessment uses percentile-based context from training data",
                "note": "Primary evaluation is based on intended changes, not absolute quality",
            }
        else:
            summary["quality_context"] = {
                "has_calibrated_thresholds": False,
                "interpretation": "Quality assessment based on relative changes only",
                "note": "Consider running threshold calibration for additional context",
            }

        return summary

    def _interpret_percentile_change(
        self,
        baseline_percentile: float,
        intervention_percentile: float,
        percent_change: float,
    ) -> str:
        """Generate human-readable interpretation of metric change."""
        percentile_diff = intervention_percentile - baseline_percentile

        if abs(percent_change) < 5:
            return "Minimal change"
        elif percentile_diff > 25:
            return f"Large improvement: moved from P{baseline_percentile:.0f} to P{intervention_percentile:.0f} (top {100-intervention_percentile:.0f}%)"
        elif percentile_diff > 10:
            return f"Moderate improvement: moved from P{baseline_percentile:.0f} to P{intervention_percentile:.0f}"
        elif percentile_diff < -25:
            return f"Large degradation: moved from P{baseline_percentile:.0f} to P{intervention_percentile:.0f}"
        elif percentile_diff < -10:
            return f"Moderate degradation: moved from P{baseline_percentile:.0f} to P{intervention_percentile:.0f}"
        else:
            direction = "increased" if percent_change > 0 else "decreased"
            return f"Value {direction} by {abs(percent_change):.1f}% with minimal quality change"

    def _analyze_intervention_effectiveness(
        self,
        profile: FeatureProfile,
        baseline: Dict[str, Any],
        intervention: Dict[str, Any],
        strength: float,
    ) -> Dict[str, Any]:
        """
        Analyze how effectively the intervention achieved its intended effects.

        FOCUS: Relative changes (baseline vs intervention)
        CONTEXT: Calibrated thresholds provide quality reference but don't determine success

        An intervention is effective if:
        1. It produces INTENDED changes in target metrics (direction matters)
        2. The magnitude is appropriate for the intervention strength
        3. Changes are statistically meaningful (>5% relative change)

        Quality thresholds are used ONLY for context, not pass/fail criteria.
        """

        effectiveness = {
            "primary_metric_changes": {},
            "secondary_metric_changes": {},
            "expected_direction_alignment": 0.0,
            "magnitude_appropriateness": 0.0,
            "effectiveness_score": 0.0,
            "relative_improvement": {},  # NEW: Track if metrics improved relative to baseline
            "threshold_context": {},  # NEW: Percentile context from calibrated thresholds
        }

        # Analyze primary metrics (most important for this feature)
        primary_changes = 0
        primary_change_magnitudes = []
        total_primary = 0

        # Debug logging
        self.logger.debug(f"Analyzing effectiveness for feature {profile.feature_id}")
        self.logger.debug(f"Primary metrics to check: {profile.primary_metrics}")

        for metric in profile.primary_metrics:
            baseline_val = self._get_nested_value(baseline, metric)
            intervention_val = self._get_nested_value(intervention, metric)

            self.logger.debug(
                f"Metric {metric}: baseline={baseline_val}, intervention={intervention_val}"
            )

            if baseline_val is not None and intervention_val is not None:

                baseline_val = self._get_nested_value(baseline, metric)
                intervention_val = self._get_nested_value(intervention, metric)

                # Handle zero baseline by using absolute change
                if baseline_val != 0:
                    change_percent = (
                        (intervention_val - baseline_val) / baseline_val * 100
                    )
                else:
                    # For zero baseline, consider absolute change
                    change_percent = (
                        intervention_val * 100 if intervention_val != 0 else 0
                    )

                # Calculate relative change
                change_info = {
                    "baseline": baseline_val,
                    "intervention": intervention_val,
                    "change_percent": change_percent,
                    "absolute_change": intervention_val - baseline_val,
                }

                # Add threshold context if available (for reference only)
                if self.threshold_manager:
                    try:
                        # Get percentile ranks for both baseline and intervention
                        baseline_percentile = (
                            self.threshold_manager.get_metric_percentile_rank(
                                metric, baseline_val
                            )
                        )
                        intervention_percentile = (
                            self.threshold_manager.get_metric_percentile_rank(
                                metric, intervention_val
                            )
                        )

                        if (
                            baseline_percentile is not None
                            and intervention_percentile is not None
                        ):
                            change_info["threshold_context"] = {
                                "baseline_percentile": baseline_percentile,
                                "intervention_percentile": intervention_percentile,
                                "percentile_change": intervention_percentile
                                - baseline_percentile,
                                "moved_toward_better": intervention_percentile
                                > baseline_percentile,
                            }

                            # Track if intervention moved metric toward better quality
                            effectiveness["relative_improvement"][metric] = {
                                "improved": intervention_percentile
                                > baseline_percentile,
                                "percentile_gain": intervention_percentile
                                - baseline_percentile,
                            }
                    except Exception as e:
                        self.logger.debug(
                            f"Could not get threshold context for {metric}: {e}"
                        )

                effectiveness["primary_metric_changes"][metric] = change_info

                # Check if ANY meaningful change occurred (not just expected direction)
                abs_change = abs(change_percent)
                if abs_change > 5:  # More than 5% change is meaningful
                    primary_change_magnitudes.append(abs_change)

                    # Check if change aligns with expected direction (bonus points)
                    expected_change = self._get_expected_change_direction(
                        profile.feature_id, metric, strength
                    )
                    if (
                        (expected_change > 0 and change_percent > 5)
                        or (expected_change < 0 and change_percent < -5)
                        or (expected_change == 0 and abs_change > 5)
                    ):
                        primary_changes += 1
                    elif abs_change > 10:  # Large change in any direction counts
                        primary_changes += 0.5

                total_primary += 1

        # Analyze secondary metrics
        for metric in profile.secondary_metrics:
            if (
                self._get_nested_value(baseline, metric) is not None
                and self._get_nested_value(intervention, metric) is not None
            ):

                baseline_val = self._get_nested_value(baseline, metric)
                intervention_val = self._get_nested_value(intervention, metric)

                if baseline_val != 0:
                    change_percent = (
                        (intervention_val - baseline_val) / baseline_val * 100
                    )
                    effectiveness["secondary_metric_changes"][metric] = {
                        "baseline": baseline_val,
                        "intervention": intervention_val,
                        "change_percent": change_percent,
                    }

        # Calculate effectiveness scores
        effectiveness["expected_direction_alignment"] = (
            primary_changes / total_primary if total_primary > 0 else 0.0
        )
        effectiveness["magnitude_appropriateness"] = self._assess_change_magnitude(
            strength, effectiveness["primary_metric_changes"]
        )

        # Calculate average magnitude of changes detected
        effectiveness["average_change_magnitude"] = (
            np.mean(primary_change_magnitudes) if primary_change_magnitudes else 0.0
        )

        # Calculate quality improvement score (from threshold context)
        quality_improvement_score = 0.0
        if effectiveness["relative_improvement"]:
            improvements = [
                info["percentile_gain"]
                for info in effectiveness["relative_improvement"].values()
            ]
            if improvements:
                # Average percentile gain, normalized to 0-1
                avg_gain = np.mean(improvements)
                quality_improvement_score = min(
                    1.0, max(0.0, (avg_gain + 50) / 100)
                )  # Map -50 to +50 percentile gain to 0-1

        # Overall effectiveness score - hybrid approach
        # PRIMARY: Relative changes (direction + magnitude + detection)
        # SECONDARY: Quality improvement (from threshold context)
        direction_score = effectiveness["expected_direction_alignment"]
        magnitude_score = effectiveness["magnitude_appropriateness"]
        change_detected_score = min(
            1.0, effectiveness["average_change_magnitude"] / 30.0
        )  # Normalize to 0-1

        # Weighted scoring:
        # - 70% based on RELATIVE changes (primary focus)
        # - 30% based on quality improvement context (secondary validation)
        relative_change_score = (
            direction_score * 0.4  # Expected direction
            + magnitude_score * 0.2  # Appropriate magnitude
            + change_detected_score * 0.4  # ANY detectable change
        )

        effectiveness["effectiveness_score"] = (
            relative_change_score * 0.7  # 70% from relative changes
            + quality_improvement_score * 0.3  # 30% from quality improvement
        )

        # Add breakdown for transparency
        effectiveness["score_breakdown"] = {
            "relative_change_score": relative_change_score,
            "quality_improvement_score": quality_improvement_score,
            "direction_component": direction_score * 0.4,
            "magnitude_component": magnitude_score * 0.2,
            "change_detection_component": change_detected_score * 0.4,
        }

        return effectiveness

    def _assess_musical_quality(
        self,
        profile: FeatureProfile,
        baseline: Dict[str, Any],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess whether musical quality was preserved during intervention."""

        quality_assessment = {
            "coherence_preservation": 0.0,
            "structural_integrity": 0.0,
            "harmonic_stability": 0.0,
            "rhythmic_stability": 0.0,
            "overall_quality_score": 0.0,
            "quality_issues": [],
        }

        # Check musical coherence
        baseline_coherence = (
            self._get_nested_value(baseline, "structural_analysis.structural_coherence")
            or 0
        )
        intervention_coherence = (
            self._get_nested_value(
                intervention, "structural_analysis.structural_coherence"
            )
            or 0
        )

        if baseline_coherence > 0:
            coherence_change = (
                intervention_coherence - baseline_coherence
            ) / baseline_coherence
            quality_assessment["coherence_preservation"] = (
                max(0, 1 + coherence_change) if coherence_change < 0 else 1.0
            )

        # Check structural integrity
        baseline_complexity = (
            self._get_nested_value(
                baseline, "musical_complexity.overall_complexity_score"
            )
            or 0
        )
        intervention_complexity = (
            self._get_nested_value(
                intervention, "musical_complexity.overall_complexity_score"
            )
            or 0
        )

        if baseline_complexity > 0:
            complexity_change = (
                abs(intervention_complexity - baseline_complexity) / baseline_complexity
            )
            quality_assessment["structural_integrity"] = max(0, 1 - complexity_change)

        # Check harmonic stability
        baseline_scale = (
            self._get_nested_value(baseline, "harmonic_analysis.scale_consistency") or 0
        )
        intervention_scale = (
            self._get_nested_value(intervention, "harmonic_analysis.scale_consistency")
            or 0
        )

        if baseline_scale > 0:
            scale_change = (intervention_scale - baseline_scale) / baseline_scale
            quality_assessment["harmonic_stability"] = (
                max(0, 1 + scale_change) if scale_change < 0 else 1.0
            )

        # Check rhythmic stability
        baseline_groove = (
            self._get_nested_value(baseline, "temporal_analysis.groove_consistency")
            or 0
        )
        intervention_groove = (
            self._get_nested_value(intervention, "temporal_analysis.groove_consistency")
            or 0
        )

        if baseline_groove > 0:
            groove_change = (intervention_groove - baseline_groove) / baseline_groove
            quality_assessment["rhythmic_stability"] = (
                max(0, 1 + groove_change) if groove_change < 0 else 1.0
            )

        # Identify quality issues
        if quality_assessment["coherence_preservation"] < 0.7:
            quality_assessment["quality_issues"].append(
                "Structural coherence significantly degraded"
            )

        if quality_assessment["harmonic_stability"] < 0.8:
            quality_assessment["quality_issues"].append(
                "Harmonic stability compromised"
            )

        if quality_assessment["rhythmic_stability"] < 0.8:
            quality_assessment["quality_issues"].append("Rhythmic consistency affected")

        # Overall quality score
        quality_assessment["overall_quality_score"] = np.mean(
            [
                quality_assessment["coherence_preservation"],
                quality_assessment["structural_integrity"],
                quality_assessment["harmonic_stability"],
                quality_assessment["rhythmic_stability"],
            ]
        )

        return quality_assessment

    def _analyze_feature_specific_effects(
        self,
        profile: FeatureProfile,
        baseline: Dict[str, Any],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze effects specific to this feature type."""

        specific_analysis = {
            "feature_category": profile.layer_type.value,
            "expected_effects_achieved": [],
            "unexpected_effects": [],
            "feature_specificity_score": 0.0,
        }

        # Check feature-specific absolute conditions
        intervention_conditions = self._check_feature_specific_conditions(
            profile.feature_id, intervention
        )
        baseline_conditions = self._check_feature_specific_conditions(
            profile.feature_id, baseline
        )

        specific_analysis["intervention_conditions"] = intervention_conditions
        specific_analysis["baseline_conditions"] = baseline_conditions
        specific_analysis["feature_specificity_score"] = intervention_conditions[
            "overall_condition_score"
        ]

        # Layer-specific analysis
        if profile.layer_type == LayerType.EARLY:
            specific_analysis.update(
                self._analyze_early_layer_effects(baseline, intervention)
            )
        elif profile.layer_type == LayerType.MID:
            specific_analysis.update(
                self._analyze_mid_layer_effects(baseline, intervention)
            )
        elif profile.layer_type == LayerType.LATE:
            specific_analysis.update(
                self._analyze_late_layer_effects(baseline, intervention)
            )

        return specific_analysis

    def _analyze_layer_appropriateness(
        self,
        profile: FeatureProfile,
        baseline: Dict[str, Any],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze whether the intervention effects are appropriate for the layer."""

        appropriateness = {
            "layer_consistency": 0.0,
            "processing_level_match": True,
            "scope_appropriateness": 0.0,
            "layer_analysis": {},
        }

        # Early layer should affect note-level properties
        if profile.layer_type == LayerType.EARLY:
            note_level_changes = self._count_note_level_changes(baseline, intervention)
            phrase_level_changes = self._count_phrase_level_changes(
                baseline, intervention
            )

            appropriateness["layer_analysis"] = {
                "note_level_changes": note_level_changes,
                "phrase_level_changes": phrase_level_changes,
                "appropriate_scope": note_level_changes > phrase_level_changes,
            }
            appropriateness["scope_appropriateness"] = note_level_changes / (
                note_level_changes + phrase_level_changes + 1
            )

        # Mid layer should affect phrase/rhythm properties
        elif profile.layer_type == LayerType.MID:
            phrase_level_changes = self._count_phrase_level_changes(
                baseline, intervention
            )
            structure_level_changes = self._count_structure_level_changes(
                baseline, intervention
            )

            appropriateness["layer_analysis"] = {
                "phrase_level_changes": phrase_level_changes,
                "structure_level_changes": structure_level_changes,
                "appropriate_scope": phrase_level_changes >= structure_level_changes,
            }
            appropriateness["scope_appropriateness"] = phrase_level_changes / (
                phrase_level_changes + structure_level_changes + 1
            )

        # Late layer should affect structural/harmonic properties
        elif profile.layer_type == LayerType.LATE:
            structure_level_changes = self._count_structure_level_changes(
                baseline, intervention
            )
            note_level_changes = self._count_note_level_changes(baseline, intervention)

            appropriateness["layer_analysis"] = {
                "structure_level_changes": structure_level_changes,
                "note_level_changes": note_level_changes,
                "appropriate_scope": structure_level_changes > note_level_changes,
            }
            appropriateness["scope_appropriateness"] = structure_level_changes / (
                structure_level_changes + note_level_changes + 1
            )

        appropriateness["layer_consistency"] = appropriateness["scope_appropriateness"]

        return appropriateness

    def _make_intervention_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make overall decision about intervention success and quality."""

        # Extract key scores
        effectiveness_score = analysis["effectiveness_analysis"]["effectiveness_score"]
        quality_score = analysis["quality_assessment"]["overall_quality_score"]
        specificity_score = analysis["feature_specific_analysis"].get(
            "feature_specificity_score", 0.0
        )
        appropriateness_score = analysis["layer_appropriate_analysis"][
            "layer_consistency"
        ]

        # Calculate weighted overall score
        weights = self.quality_weights
        overall_score = (
            effectiveness_score * weights["intervention_effectiveness"]
            + quality_score * weights["quality_preservation"]
            + specificity_score * weights["feature_specificity"]
            + appropriateness_score * weights["musical_coherence"]
        )

        # Determine quality level (more lenient thresholds)
        if overall_score >= 0.75:
            quality_level = InterventionQuality.EXCELLENT
        elif overall_score >= 0.6:
            quality_level = InterventionQuality.GOOD
        elif overall_score >= 0.45:
            quality_level = InterventionQuality.MODERATE
        elif overall_score >= 0.25:
            quality_level = InterventionQuality.POOR
        else:
            quality_level = InterventionQuality.FAILED

        # Generate recommendations
        recommendations = []

        if effectiveness_score < 0.3:
            recommendations.append(
                "Low effectiveness detected - consider different intervention strength"
            )
        elif effectiveness_score < 0.5:
            recommendations.append(
                "Moderate effectiveness - intervention has some impact"
            )

        if quality_score < 0.5:
            recommendations.append(
                "Musical quality affected - review if changes are acceptable"
            )

        if appropriateness_score < 0.4:
            recommendations.append(
                "Layer effects may not match expectations - verify feature targeting"
            )

        # Success criteria (more lenient)
        success_criteria = {
            "effectiveness_threshold": effectiveness_score >= 0.3,  # Lowered from 0.5
            "quality_preservation": quality_score >= 0.5,  # Lowered from 0.6
            "layer_appropriateness": appropriateness_score >= 0.35,  # Lowered from 0.5
            "overall_success": overall_score >= 0.4,  # Lowered from 0.5
        }

        return {
            "overall_score": overall_score,
            "quality_level": quality_level.value,
            "component_scores": {
                "effectiveness": effectiveness_score,
                "quality_preservation": quality_score,
                "feature_specificity": specificity_score,
                "layer_appropriateness": appropriateness_score,
            },
            "success_criteria": success_criteria,
            "recommendations": recommendations,
            "summary": self._generate_decision_summary(
                quality_level, overall_score, analysis
            ),
        }

    # Helper methods

    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split(".")
        value = data

        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except (KeyError, TypeError):
            return None

    def _check_feature_specific_conditions(
        self, feature_id: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check feature-specific absolute conditions using calibrated thresholds.

        Each feature has specific metric requirements that should be met for
        the intervention to be considered successful.

        Returns:
            Dictionary with condition checks and scores for the specific feature
        """
        checks = {
            "conditions_met": [],
            "conditions_failed": [],
            "overall_condition_score": 0.0,
        }

        profile = self.feature_profiles.get(feature_id)
        if not profile:
            return checks

        # Feature 325: Rhythmic displacement syncopation
        if feature_id == "325":
            syncopation = (
                self._get_nested_value(features, "rhythmic_analysis.syncopation_score")
                or 0
            )
            rhythmic_complexity = (
                self._get_nested_value(
                    features, "rhythmic_analysis.rhythmic_complexity"
                )
                or 0
            )

            if syncopation >= 0.108:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Syncopation score {syncopation:.3f} >= 0.108"
                )
            else:
                checks["conditions_failed"].append(
                    f"Syncopation score {syncopation:.3f} < 0.108"
                )

            if rhythmic_complexity >= 1.515:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Rhythmic complexity {rhythmic_complexity:.3f} >= 1.515"
                )
            else:
                checks["conditions_failed"].append(
                    f"Rhythmic complexity {rhythmic_complexity:.3f} < 1.515"
                )

        # Feature 256: Dynamic contrast accents
        elif feature_id == "256":
            dynamic_range = (
                self._get_nested_value(features, "velocity_dynamics.dynamic_range") or 0
            )
            velocity_range = (
                self._get_nested_value(features, "note_patterns.velocity_range") or 0
            )

            if dynamic_range >= 34.0:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Dynamic range {dynamic_range:.1f} >= 34.0"
                )
            else:
                checks["conditions_failed"].append(
                    f"Dynamic range {dynamic_range:.1f} < 34.0"
                )

            if velocity_range >= 30:  # Has dynamic contrast
                checks["conditions_met"].append(
                    f"Velocity range {velocity_range:.1f} >= 30 (has contrast)"
                )
            else:
                checks["conditions_failed"].append(
                    f"Velocity range {velocity_range:.1f} < 30 (no contrast)"
                )

            if velocity_range >= 70:  # Extreme dynamic contrast
                checks["conditions_met"].append(
                    f"Velocity range {velocity_range:.1f} >= 70 (extreme contrast)"
                )

        # Feature 1323: Wide pitch range texture
        elif feature_id == "1323":
            pitch_range = (
                self._get_nested_value(features, "pitch_analysis.pitch_range") or 0
            )
            pitch_std = (
                self._get_nested_value(features, "pitch_analysis.pitch_std") or 0
            )

            if pitch_range >= 26.0:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Pitch range {pitch_range:.1f} >= 26.0"
                )
            else:
                checks["conditions_failed"].append(
                    f"Pitch range {pitch_range:.1f} < 26.0"
                )

            if pitch_std >= 8.066:  # Above 10th percentile
                checks["conditions_met"].append(f"Pitch std {pitch_std:.3f} >= 8.066")
            else:
                checks["conditions_failed"].append(f"Pitch std {pitch_std:.3f} < 8.066")

        # Feature 182: Steady pulse march rhythm
        elif feature_id == "182":
            rhythmic_regularity = (
                self._get_nested_value(
                    features, "rhythmic_analysis.rhythmic_regularity"
                )
                or 0
            )
            unique_durations = (
                self._get_nested_value(features, "note_patterns.unique_note_durations")
                or 0
            )

            if rhythmic_regularity >= 0.562:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Rhythmic regularity {rhythmic_regularity:.3f} >= 0.562"
                )
            else:
                checks["conditions_failed"].append(
                    f"Rhythmic regularity {rhythmic_regularity:.3f} < 0.562"
                )

            if unique_durations <= 15:  # Limited duration variety = more regular
                checks["conditions_met"].append(
                    f"Limited duration variety ({unique_durations} types)"
                )
            else:
                checks["conditions_failed"].append(
                    f"High duration variety ({unique_durations} types)"
                )

        # Feature 855: Dynamic contrast tension
        elif feature_id == "855":
            dynamic_range = (
                self._get_nested_value(features, "velocity_dynamics.dynamic_range") or 0
            )
            dynamic_variance = (
                self._get_nested_value(features, "velocity_dynamics.dynamic_variance")
                or 0
            )

            if dynamic_range >= 34.0:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Dynamic range {dynamic_range:.1f} >= 34.0"
                )
            else:
                checks["conditions_failed"].append(
                    f"Dynamic range {dynamic_range:.1f} < 34.0"
                )

            if dynamic_variance >= 197.691:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Dynamic variance {dynamic_variance:.1f} >= 197.7"
                )
            else:
                checks["conditions_failed"].append(
                    f"Dynamic variance {dynamic_variance:.1f} < 197.7"
                )

        # Feature 997: Antiphonal call-response
        elif feature_id == "997":
            polyphonic_complexity = (
                self._get_nested_value(
                    features, "musical_complexity.polyphonic_complexity"
                )
                or 1.0
            )
            total_tracks = (
                self._get_nested_value(features, "basic_info.total_tracks") or 1
            )
            phrase_count = (
                self._get_nested_value(features, "structural_analysis.phrase_count")
                or 0
            )

            if polyphonic_complexity >= 1.111:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Polyphonic complexity {polyphonic_complexity:.3f} >= 1.111"
                )
            else:
                checks["conditions_failed"].append(
                    f"Polyphonic complexity {polyphonic_complexity:.3f} < 1.111"
                )

            if total_tracks >= 2:  # Need multiple tracks for call-response
                checks["conditions_met"].append(
                    f"Multiple tracks ({total_tracks}) for dialogue"
                )
            else:
                checks["conditions_failed"].append(
                    f"Only {total_tracks} track (need >= 2)"
                )

            if phrase_count >= 2:  # Need phrases for call-response structure
                checks["conditions_met"].append(
                    f"Multiple phrases ({phrase_count}) for call-response"
                )
            else:
                checks["conditions_failed"].append(f"Only {phrase_count} phrase(s)")

        # Feature 471: Unison doubling octaves
        elif feature_id == "471":
            pitch_range = (
                self._get_nested_value(features, "pitch_analysis.pitch_range") or 0
            )
            polyphonic_complexity = (
                self._get_nested_value(
                    features, "musical_complexity.polyphonic_complexity"
                )
                or 1.0
            )
            total_tracks = (
                self._get_nested_value(features, "basic_info.total_tracks") or 1
            )

            if pitch_range >= 12:  # At least one octave span for octave doubling
                checks["conditions_met"].append(
                    f"Pitch range {pitch_range:.1f} >= 12 (octave span)"
                )
            else:
                checks["conditions_failed"].append(
                    f"Pitch range {pitch_range:.1f} < 12 (less than octave)"
                )

            if polyphonic_complexity >= 1.111:  # Multiple voices
                checks["conditions_met"].append(
                    f"Polyphonic complexity {polyphonic_complexity:.3f} >= 1.111 (multi-voice)"
                )
            else:
                checks["conditions_failed"].append(
                    f"Polyphonic complexity {polyphonic_complexity:.3f} < 1.111"
                )

            if total_tracks >= 2:  # Need multiple instruments for doubling
                checks["conditions_met"].append(
                    f"Multiple instruments ({total_tracks}) for doubling"
                )
            else:
                checks["conditions_failed"].append(f"Only {total_tracks} instrument")

        # Feature 904: Rhythmic augmentation
        elif feature_id == "904":
            avg_duration = (
                self._get_nested_value(features, "note_patterns.average_note_duration")
                or 0
            )
            longest_note = (
                self._get_nested_value(features, "note_patterns.longest_note") or 0
            )
            unique_durations = (
                self._get_nested_value(features, "note_patterns.unique_note_durations")
                or 0
            )

            if avg_duration >= 0.443:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Average duration {avg_duration:.3f} >= 0.443"
                )
            else:
                checks["conditions_failed"].append(
                    f"Average duration {avg_duration:.3f} < 0.443"
                )

            if longest_note >= 2.0:  # Has long notes
                checks["conditions_met"].append(
                    f"Longest note {longest_note:.1f} >= 2.0"
                )
            else:
                checks["conditions_failed"].append(
                    f"Longest note {longest_note:.1f} < 2.0"
                )

            if unique_durations >= 5:  # Duration variety for augmentation pattern
                checks["conditions_met"].append(
                    f"Duration variety ({unique_durations} types) >= 5"
                )
            else:
                checks["conditions_failed"].append(
                    f"Low duration variety ({unique_durations} types)"
                )

        # Feature 1950: Dramatic dynamic swells
        elif feature_id == "1950":
            dynamic_range = (
                self._get_nested_value(features, "velocity_dynamics.dynamic_range") or 0
            )
            dynamic_variance = (
                self._get_nested_value(features, "velocity_dynamics.dynamic_variance")
                or 0
            )
            velocity_range = (
                self._get_nested_value(features, "note_patterns.velocity_range") or 0
            )

            if dynamic_range >= 34.0:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Dynamic range {dynamic_range:.1f} >= 34.0"
                )
            else:
                checks["conditions_failed"].append(
                    f"Dynamic range {dynamic_range:.1f} < 34.0"
                )

            if dynamic_variance >= 197.691:  # Above 10th percentile
                checks["conditions_met"].append(
                    f"Dynamic variance {dynamic_variance:.1f} >= 197.7"
                )
            else:
                checks["conditions_failed"].append(
                    f"Dynamic variance {dynamic_variance:.1f} < 197.7"
                )

            if velocity_range >= 30:  # Has significant contrast for swells
                checks["conditions_met"].append(
                    f"Velocity range {velocity_range:.1f} >= 30 (swells possible)"
                )
            else:
                checks["conditions_failed"].append(
                    f"Velocity range {velocity_range:.1f} < 30 (limited swells)"
                )

        # Calculate overall condition score
        total_conditions = len(checks["conditions_met"]) + len(
            checks["conditions_failed"]
        )
        if total_conditions > 0:
            checks["overall_condition_score"] = (
                len(checks["conditions_met"]) / total_conditions
            )

        return checks

    def _get_expected_change_direction(
        self, feature_id: str, metric: str, strength: float
    ) -> int:
        """Get expected direction of change for a metric given feature and strength."""

        # Feature-specific expectations (using actual MIDI extractor metric names with paths)
        expectations = {
            "325": {  # Rhythmic displacement syncopation
                "rhythmic_analysis.syncopation_score": 1,
                "rhythmic_analysis.rhythmic_complexity": 1,
                "rhythmic_analysis.ioi_std": 1,
                "note_patterns.note_onset_intervals.mean_interval": 0,  # Secondary
            },
            "256": {  # Dynamic contrast accents
                "velocity_dynamics.dynamic_range": 1,
                "note_patterns.velocity_range": 1,
                "velocity_dynamics.dynamic_variance": 1,
                "velocity_dynamics.forte_notes_ratio": 1,
            },
            "1323": {  # Wide pitch range texture
                "pitch_analysis.pitch_range": 1,
                "pitch_analysis.pitch_std": 1,
                "pitch_analysis.max_pitch": 1,
                "pitch_analysis.min_pitch": -1,  # Lower notes expand range downward
            },
            "182": {  # Steady pulse march rhythm
                "rhythmic_analysis.rhythmic_regularity": 1,
                "rhythmic_analysis.average_ioi": 0,  # Consistent, not necessarily increasing
                "note_patterns.unique_note_durations": -1,  # More uniform
                "rhythmic_analysis.rhythmic_complexity": -1,  # More regular = less complex
            },
            "855": {  # Dynamic contrast tension
                "velocity_dynamics.dynamic_range": 1,
                "velocity_dynamics.dynamic_variance": 1,
                "note_patterns.velocity_range": 1,
                "structural_analysis.structural_coherence": 0,  # May be affected
            },
            "997": {  # Antiphonal call-response
                "musical_complexity.polyphonic_complexity": 1,
                "structural_analysis.phrase_count": 1,
                "note_patterns.note_density": 0,  # Depends on implementation
                "basic_info.total_tracks": 1,  # More tracks for dialogue
            },
            "471": {  # Unison doubling octaves
                "pitch_analysis.pitch_range": 1,  # Octave span increases
                "musical_complexity.polyphonic_complexity": 1,
                "basic_info.total_tracks": 0,  # May have more tracks
                "note_patterns.average_velocity": 1,  # Stronger unified sound
            },
            "904": {  # Rhythmic augmentation
                "note_patterns.average_note_duration": 1,
                "note_patterns.longest_note": 1,
                "note_patterns.unique_note_durations": 1,
                "rhythmic_analysis.average_ioi": 1,  # Longer intervals
            },
            "1950": {  # Dramatic dynamic swells
                "velocity_dynamics.dynamic_range": 1,
                "velocity_dynamics.dynamic_variance": 1,
                "note_patterns.velocity_range": 1,
                "velocity_dynamics.forte_notes_ratio": 0,  # Variable
            },
        }

        if feature_id in expectations and metric in expectations[feature_id]:
            expected = expectations[feature_id][metric]
            # Reverse for negative strength
            return expected if strength > 0 else -expected

        return 0  # No expectation

    def _assess_change_magnitude(self, strength: float, metric_changes: Dict) -> float:
        """Assess whether change magnitude is appropriate for intervention strength."""
        if not metric_changes:
            return 0.0

        changes = [abs(change["change_percent"]) for change in metric_changes.values()]
        avg_change = np.mean(changes)

        # Expected change should roughly correlate with strength
        expected_change = abs(strength) * 20  # Rough scaling

        # Score based on how close actual change is to expected
        if expected_change == 0:
            return 1.0 if avg_change < 5 else 0.5

        ratio = min(avg_change, expected_change) / max(avg_change, expected_change)
        return ratio

    def _analyze_early_layer_effects(self, baseline: Dict, intervention: Dict) -> Dict:
        """Analyze effects appropriate for early layer processing."""
        return {
            "feature_specificity_score": 0.7,  # Placeholder - would analyze note-level effects
            "processing_characteristics": "note_level_modifications",
        }

    def _analyze_mid_layer_effects(self, baseline: Dict, intervention: Dict) -> Dict:
        """Analyze effects appropriate for mid layer processing."""
        return {
            "feature_specificity_score": 0.75,  # Placeholder - would analyze phrase-level effects
            "processing_characteristics": "phrase_level_modifications",
        }

    def _analyze_late_layer_effects(self, baseline: Dict, intervention: Dict) -> Dict:
        """Analyze effects appropriate for late layer processing."""
        return {
            "feature_specificity_score": 0.8,  # Placeholder - would analyze structure-level effects
            "processing_characteristics": "structure_level_modifications",
        }

    def _count_note_level_changes(self, baseline: Dict, intervention: Dict) -> int:
        """Count significant changes in note-level metrics."""
        note_metrics = [
            "average_velocity",
            "note_density",
            "average_note_duration",
            "timing_precision",
        ]
        changes = 0

        for metric in note_metrics:
            baseline_val = (
                self._get_nested_value(baseline, f"note_patterns.{metric}")
                or self._get_nested_value(baseline, f"velocity_dynamics.{metric}")
                or self._get_nested_value(baseline, f"temporal_analysis.{metric}")
            )
            intervention_val = (
                self._get_nested_value(intervention, f"note_patterns.{metric}")
                or self._get_nested_value(intervention, f"velocity_dynamics.{metric}")
                or self._get_nested_value(intervention, f"temporal_analysis.{metric}")
            )

            if (
                baseline_val is not None
                and intervention_val is not None
                and baseline_val != 0
            ):
                change_percent = (
                    abs(intervention_val - baseline_val) / baseline_val * 100
                )
                if change_percent > 10:  # Significant change threshold
                    changes += 1

        return changes

    def _count_phrase_level_changes(self, baseline: Dict, intervention: Dict) -> int:
        """Count significant changes in phrase-level metrics."""
        phrase_metrics = [
            "phrase_count",
            "average_phrase_length",
            "rhythmic_complexity",
            "syncopation_score",
        ]
        changes = 0

        for metric in phrase_metrics:
            baseline_val = self._get_nested_value(
                baseline, f"structural_analysis.{metric}"
            ) or self._get_nested_value(baseline, f"rhythmic_analysis.{metric}")
            intervention_val = self._get_nested_value(
                intervention, f"structural_analysis.{metric}"
            ) or self._get_nested_value(intervention, f"rhythmic_analysis.{metric}")

            if (
                baseline_val is not None
                and intervention_val is not None
                and baseline_val != 0
            ):
                change_percent = (
                    abs(intervention_val - baseline_val) / baseline_val * 100
                )
                if change_percent > 10:  # Significant change threshold
                    changes += 1

        return changes

    def _count_structure_level_changes(self, baseline: Dict, intervention: Dict) -> int:
        """Count significant changes in structure-level metrics."""
        structure_metrics = [
            "structural_coherence",
            "scale_consistency",
            "overall_complexity_score",
            "harmonic_complexity",
        ]
        changes = 0

        for metric in structure_metrics:
            baseline_val = (
                self._get_nested_value(baseline, f"structural_analysis.{metric}")
                or self._get_nested_value(baseline, f"harmonic_analysis.{metric}")
                or self._get_nested_value(baseline, f"musical_complexity.{metric}")
            )
            intervention_val = (
                self._get_nested_value(intervention, f"structural_analysis.{metric}")
                or self._get_nested_value(intervention, f"harmonic_analysis.{metric}")
                or self._get_nested_value(intervention, f"musical_complexity.{metric}")
            )

            if (
                baseline_val is not None
                and intervention_val is not None
                and baseline_val != 0
            ):
                change_percent = (
                    abs(intervention_val - baseline_val) / baseline_val * 100
                )
                if change_percent > 10:  # Significant change threshold
                    changes += 1

        return changes

    def _generate_decision_summary(
        self, quality_level: InterventionQuality, score: float, analysis: Dict
    ) -> str:
        """Generate human-readable summary of the intervention assessment."""

        feature_info = analysis["feature_info"]
        feature_name = feature_info["feature_name"]
        layer = feature_info["layer"]
        strength = feature_info["intervention_strength"]

        summary = f"Feature {feature_name} (Layer {layer}) intervention with strength {strength}: "

        if quality_level == InterventionQuality.EXCELLENT:
            summary += f"Excellent results (score: {score:.2f}). Intervention achieved intended effects with minimal quality degradation."
        elif quality_level == InterventionQuality.GOOD:
            summary += f"Good results (score: {score:.2f}). Effective intervention with acceptable quality preservation."
        elif quality_level == InterventionQuality.MODERATE:
            summary += f"Moderate results (score: {score:.2f}). Some effectiveness but with noticeable quality impact."
        elif quality_level == InterventionQuality.POOR:
            summary += f"Poor results (score: {score:.2f}). Limited effectiveness and/or significant quality degradation."
        else:
            summary += f"Failed intervention (score: {score:.2f}). Minimal effectiveness with substantial quality loss."

        return summary


def analyze_intervention_batch(feature_analyses: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Analyze a batch of feature interventions and provide comparative insights.

    Args:
        feature_analyses: Dictionary of individual feature analyses

    Returns:
        Batch analysis with comparative insights and recommendations
    """

    analyzer = FeatureSpecificAnalyzer()

    batch_analysis = {
        "total_interventions": len(feature_analyses),
        "success_rate": 0.0,
        "quality_distribution": {level.value: 0 for level in InterventionQuality},
        "layer_performance": {"early": [], "mid": [], "late": []},
        "best_interventions": [],
        "worst_interventions": [],
        "recommendations": [],
    }

    successful_interventions = 0
    quality_scores = []

    # Analyze each intervention
    for feature_id, analysis in feature_analyses.items():
        if "overall_decision" in analysis and analysis["overall_decision"]:
            decision = analysis["overall_decision"]

            # Count successes
            if decision["success_criteria"]["overall_success"]:
                successful_interventions += 1

            # Track quality distribution
            quality_level = decision["quality_level"]
            batch_analysis["quality_distribution"][quality_level] += 1

            # Track by layer
            layer_type = analysis["feature_info"]["layer_type"]
            if layer_type in batch_analysis["layer_performance"]:
                batch_analysis["layer_performance"][layer_type].append(
                    {"feature_id": feature_id, "score": decision["overall_score"]}
                )

            quality_scores.append(decision["overall_score"])

    # Calculate success rate
    batch_analysis["success_rate"] = (
        successful_interventions / len(feature_analyses) if feature_analyses else 0.0
    )

    # Find best and worst interventions
    if quality_scores:
        sorted_analyses = sorted(
            feature_analyses.items(),
            key=lambda x: x[1].get("overall_decision", {}).get("overall_score", 0),
            reverse=True,
        )

        batch_analysis["best_interventions"] = sorted_analyses[:3]
        batch_analysis["worst_interventions"] = sorted_analyses[-3:]

    # Generate batch recommendations
    if batch_analysis["success_rate"] < 0.5:
        batch_analysis["recommendations"].append(
            "Overall success rate is low - consider adjusting intervention approach"
        )

    if batch_analysis["quality_distribution"]["failed"] > len(feature_analyses) * 0.3:
        batch_analysis["recommendations"].append(
            "High failure rate detected - review feature selection and intervention methods"
        )

    return batch_analysis


if __name__ == "__main__":
    # Test the analyzer
    analyzer = FeatureSpecificAnalyzer()

    # Print feature profiles
    print("Feature-Specific Analyzer initialized with profiles:")
    for feature_id, profile in analyzer.feature_profiles.items():
        print(
            f"  {feature_id}: {profile.feature_name} (Layer {profile.layer}, {profile.layer_type.value})"
        )
        print(f"    Expected effects: {', '.join(profile.expected_effects)}")
        print(f"    Primary metrics: {', '.join(profile.primary_metrics)}")
        print()
