"""
Deterministic Analysis Package for Musical Feature Intervention Assessment

This package provides comprehensive deterministic analysis capabilities for evaluating
the effectiveness and quality of musical feature interventions. It includes:

- MIDI-level musical feature extraction
- Feature-specific intervention analysis
- Music quality assessment
- Batch processing for complete intervention pipelines
- Decision-making support for intervention success

Key modules:
- midi_feature_extractors: Extract comprehensive musical characteristics from MIDI
- feature_specific_analyzers: Analyze interventions with feature-specific knowledge
- music_quality_assessor: Assess musical quality and coherence
- batch_deterministic_analyzer: Orchestrate complete analysis pipelines

Usage:
    from deterministic_analysis import BatchDeterministicAnalyzer

    analyzer = BatchDeterministicAnalyzer()
    results = analyzer.run_batch_analysis("path/to/intervention/results")
"""

from .midi_feature_extractors import MIDIFeatureExtractor
from .feature_specific_analyzers import (
    FeatureSpecificAnalyzer,
    InterventionQuality,
    LayerType,
    analyze_intervention_batch,
)
from .music_quality_assessor import (
    MusicQualityAssessor,
    QualityDimension,
    QualityThreshold,
)
from .batch_deterministic_analyzer import BatchDeterministicAnalyzer

__version__ = "1.0.0"
__author__ = "MMT Project"

__all__ = [
    # Main analyzer classes
    "MIDIFeatureExtractor",
    "FeatureSpecificAnalyzer",
    "MusicQualityAssessor",
    "BatchDeterministicAnalyzer",
    # Enums and utilities
    "InterventionQuality",
    "LayerType",
    "QualityDimension",
    "QualityThreshold",
    # Functions
    "analyze_intervention_batch",
]

# Package metadata
FEATURES_ANALYZED = {
    # Layer 1 - Early Processing (Note Level)
    "325": {"layer": 1, "name": "dynamic_emphasis", "type": "early"},
    "256": {"layer": 1, "name": "note_timing", "type": "early"},
    "1323": {"layer": 1, "name": "note_density_control", "type": "early"},
    # Layer 3 - Mid Processing (Phrase/Rhythm Level)
    "182": {"layer": 3, "name": "rhythmic_pattern", "type": "mid"},
    "855": {"layer": 3, "name": "phrase_structure", "type": "mid"},
    "997": {"layer": 3, "name": "melodic_contour", "type": "mid"},
    # Layer 5 - Late Processing (Structure/Harmony Level)
    "471": {"layer": 5, "name": "harmonic_progression", "type": "late"},
    "904": {"layer": 5, "name": "musical_structure", "type": "late"},
    "1950": {"layer": 5, "name": "tonal_center", "type": "late"},
}

MUSICAL_ANALYSIS_CATEGORIES = [
    "basic_info",
    "note_patterns",
    "pitch_analysis",
    "rhythmic_analysis",
    "harmonic_analysis",
    "structural_analysis",
    "velocity_dynamics",
    "temporal_analysis",
    "musical_complexity",
]

QUALITY_DIMENSIONS = [
    "harmonic_coherence",
    "rhythmic_consistency",
    "melodic_flow",
    "structural_integrity",
    "aesthetic_appeal",
    "technical_correctness",
]
