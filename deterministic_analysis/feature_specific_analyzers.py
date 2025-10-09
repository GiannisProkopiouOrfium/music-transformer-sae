#!/usr/bin/env python3
"""
Feature-Specific Analyzers for Intervention Assessment

This module provides specialized analysis tools for each type of feature intervention.
It can assess the effectiveness and quality of interventions both with and without LLM assistance.

Features:
- Layer-specific analysis (Early/Mid/Late processing characteristics)
- Feature-specific impact assessment (what each feature actually controls)
- Intervention effectiveness scoring
- Musical quality preservation assessment
- Decision-making framework for intervention success
"""

import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum


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

    def __init__(self):
        """Initialize with feature profiles and analysis configuration."""
        self.logger = logging.getLogger(__name__)
        self.feature_profiles = self._initialize_feature_profiles()
        self.quality_weights = self._initialize_quality_weights()

    def _initialize_feature_profiles(self) -> Dict[str, FeatureProfile]:
        """Initialize profiles for our 9 selected features."""

        profiles = {}

        # Layer 1 Features (Early Processing - Note Level)
        profiles["325"] = FeatureProfile(
            feature_id="325",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="dynamic_emphasis",
            expected_effects=["velocity_changes", "note_emphasis", "dynamic_contrast"],
            primary_metrics=[
                "velocity_dynamics.average_dynamics",
                "velocity_dynamics.dynamic_range",
                "velocity_dynamics.dynamic_variance",
            ],
            secondary_metrics=[
                "note_patterns.note_density",
                "velocity_dynamics.forte_notes_ratio",
            ],
            quality_thresholds={"min_velocity_range": 20, "max_velocity_std": 50},
        )

        profiles["256"] = FeatureProfile(
            feature_id="256",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="note_timing",
            expected_effects=[
                "timing_precision",
                "rhythmic_variations",
                "micro_timing",
            ],
            primary_metrics=[
                "temporal_analysis.timing_precision",
                "temporal_analysis.groove_consistency",
                "temporal_analysis.tempo_stability",
            ],
            secondary_metrics=[
                "temporal_analysis.rhythmic_swing",
                "note_patterns.note_density",
            ],
            quality_thresholds={
                "min_timing_precision": 0.3,
                "max_rhythmic_deviation": 0.8,
            },
        )

        profiles["1323"] = FeatureProfile(
            feature_id="1323",
            layer=1,
            layer_type=LayerType.EARLY,
            feature_name="note_density_control",
            expected_effects=["note_frequency", "texture_density", "activity_level"],
            primary_metrics=[
                "note_patterns.notes_per_beat",
                "note_patterns.note_density",
                "basic_info.total_notes",
            ],
            secondary_metrics=[
                "musical_complexity.polyphonic_complexity",
                "note_patterns.average_note_duration",
            ],
            quality_thresholds={"min_notes_per_beat": 0.5, "max_notes_per_beat": 10.0},
        )

        # Layer 3 Features (Mid Processing - Phrase/Rhythm Level)
        profiles["182"] = FeatureProfile(
            feature_id="182",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="rhythmic_pattern",
            expected_effects=[
                "rhythm_complexity",
                "beat_patterns",
                "groove_characteristics",
            ],
            primary_metrics=[
                "musical_complexity.rhythmic_complexity",
                "rhythmic_analysis.syncopation_score",
                "rhythmic_analysis.beat_strength_distribution",
            ],
            secondary_metrics=[
                "temporal_analysis.groove_consistency",
                "rhythmic_analysis.rhythmic_diversity",
            ],
            quality_thresholds={"min_rhythmic_complexity": 1.0, "max_syncopation": 0.8},
        )

        profiles["855"] = FeatureProfile(
            feature_id="855",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="phrase_structure",
            expected_effects=[
                "phrase_boundaries",
                "musical_phrases",
                "structural_units",
            ],
            primary_metrics=[
                "phrase_count",
                "average_phrase_length",
                "structural_coherence",
            ],
            secondary_metrics=["repetition_ratio", "phrase_length_std"],
            quality_thresholds={
                "min_phrase_count": 2,
                "max_phrase_length_variation": 0.7,
            },
        )

        profiles["997"] = FeatureProfile(
            feature_id="997",
            layer=3,
            layer_type=LayerType.MID,
            feature_name="melodic_contour",
            expected_effects=["melodic_shape", "interval_patterns", "pitch_movements"],
            primary_metrics=[
                "average_interval_size",
                "step_motion_ratio",
                "melodic_complexity",
            ],
            secondary_metrics=["pitch_range", "largest_leap"],
            quality_thresholds={"min_step_motion": 0.3, "max_interval_size": 8.0},
        )

        # Layer 5 Features (Late Processing - Structure/Harmony Level)
        profiles["471"] = FeatureProfile(
            feature_id="471",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="harmonic_progression",
            expected_effects=[
                "chord_progressions",
                "harmonic_rhythm",
                "tonal_structure",
            ],
            primary_metrics=[
                "scale_consistency",
                "pitch_class_entropy",
                "harmonic_complexity",
            ],
            secondary_metrics=["chord_count", "harmonic_rhythm"],
            quality_thresholds={
                "min_scale_consistency": 0.7,
                "min_harmonic_complexity": 1.5,
            },
        )

        profiles["904"] = FeatureProfile(
            feature_id="904",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="musical_structure",
            expected_effects=[
                "overall_form",
                "structural_coherence",
                "musical_architecture",
            ],
            primary_metrics=[
                "structural_coherence",
                "overall_complexity_score",
                "repetition_ratio",
            ],
            secondary_metrics=["phrase_count", "polyphonic_complexity"],
            quality_thresholds={"min_structural_coherence": 0.4, "max_complexity": 0.9},
        )

        profiles["1950"] = FeatureProfile(
            feature_id="1950",
            layer=5,
            layer_type=LayerType.LATE,
            feature_name="tonal_center",
            expected_effects=["key_stability", "tonal_relationships", "pitch_center"],
            primary_metrics=["scale_consistency", "pitch_class_entropy", "mean_pitch"],
            secondary_metrics=["pitch_range", "unique_pitches"],
            quality_thresholds={
                "min_scale_consistency": 0.8,
                "pitch_stability_range": 12,
            },
        )

        return profiles

    def _initialize_quality_weights(self) -> Dict[str, float]:
        """Initialize weights for overall quality assessment."""
        return {
            "musical_coherence": 0.3,
            "intervention_effectiveness": 0.25,
            "quality_preservation": 0.25,
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

        return analysis

    def _analyze_intervention_effectiveness(
        self,
        profile: FeatureProfile,
        baseline: Dict[str, Any],
        intervention: Dict[str, Any],
        strength: float,
    ) -> Dict[str, Any]:
        """Analyze how effectively the intervention achieved its intended effects."""

        effectiveness = {
            "primary_metric_changes": {},
            "secondary_metric_changes": {},
            "expected_direction_alignment": 0.0,
            "magnitude_appropriateness": 0.0,
            "effectiveness_score": 0.0,
        }

        # Analyze primary metrics (most important for this feature)
        primary_changes = 0
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

                if baseline_val != 0:
                    change_percent = (
                        (intervention_val - baseline_val) / baseline_val * 100
                    )
                    effectiveness["primary_metric_changes"][metric] = {
                        "baseline": baseline_val,
                        "intervention": intervention_val,
                        "change_percent": change_percent,
                        "absolute_change": intervention_val - baseline_val,
                    }

                    # Check if change aligns with expected direction
                    expected_change = self._get_expected_change_direction(
                        profile.feature_id, metric, strength
                    )
                    if (
                        (expected_change > 0 and change_percent > 0)
                        or (expected_change < 0 and change_percent < 0)
                        or (expected_change == 0 and abs(change_percent) < 5)
                    ):
                        primary_changes += 1

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

        # Overall effectiveness score
        effectiveness["effectiveness_score"] = (
            effectiveness["expected_direction_alignment"] * 0.7
            + effectiveness["magnitude_appropriateness"] * 0.3
        )

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

        # Determine quality level
        if overall_score >= 0.8:
            quality_level = InterventionQuality.EXCELLENT
        elif overall_score >= 0.65:
            quality_level = InterventionQuality.GOOD
        elif overall_score >= 0.5:
            quality_level = InterventionQuality.MODERATE
        elif overall_score >= 0.3:
            quality_level = InterventionQuality.POOR
        else:
            quality_level = InterventionQuality.FAILED

        # Generate recommendations
        recommendations = []

        if effectiveness_score < 0.5:
            recommendations.append(
                "Consider different intervention strength or approach"
            )

        if quality_score < 0.6:
            recommendations.append(
                "Musical quality significantly affected - review intervention method"
            )

        if appropriateness_score < 0.5:
            recommendations.append(
                "Intervention effects not appropriate for this layer - check feature targeting"
            )

        # Success criteria
        success_criteria = {
            "effectiveness_threshold": effectiveness_score >= 0.5,
            "quality_preservation": quality_score >= 0.6,
            "layer_appropriateness": appropriateness_score >= 0.5,
            "overall_success": overall_score >= 0.5,
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

    def _get_expected_change_direction(
        self, feature_id: str, metric: str, strength: float
    ) -> int:
        """Get expected direction of change for a metric given feature and strength."""

        # Feature-specific expectations (using actual MIDI extractor metric names with paths)
        expectations = {
            "325": {
                "velocity_dynamics.average_dynamics": 1,
                "velocity_dynamics.dynamic_range": 1,
            },  # Dynamic emphasis
            "256": {
                "temporal_analysis.timing_precision": -1,
                "temporal_analysis.groove_consistency": -1,
            },  # Timing variations
            "1323": {
                "note_patterns.notes_per_beat": 1,
                "note_patterns.note_density": 1,
            },  # Note density
            "182": {
                "musical_complexity.rhythmic_complexity": 1,
                "rhythmic_analysis.syncopation_score": 1,
            },  # Rhythmic patterns
            "855": {
                "phrase_length_avg": 1,
                "structural_coherence": 1,
            },  # Phrase structure
            "997": {
                "pitch_range": 1,
                "average_interval_size": 1,
            },  # Melodic contour
            "471": {
                "scale_consistency": 1,
                "harmonic_complexity": 1,
            },  # Harmonic progression
            "904": {
                "structural_coherence": 1,
                "overall_complexity_score": 1,
            },  # Structure
            "1950": {"scale_consistency": 1, "pitch_class_entropy": -1},  # Tonal center
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
