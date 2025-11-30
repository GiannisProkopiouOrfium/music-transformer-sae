"""Configuration for steering interventions."""

import pathlib

# Paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sod" / "processed"
JSON_DIR = DATA_DIR / "json"
NOTES_DIR = DATA_DIR / "notes"
OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"

# Model paths
MODEL_DIR = PROJECT_ROOT / "exp" / "sod" / "ape"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Concept definitions
CONCEPTS = {
    "velocity": {
        "metric_name": "average_velocity",
        "high_threshold": 127.0,  # High velocity (louder music)
        "low_threshold": 76.1,  # Low velocity (quieter music)
        "json_field": "velocity",  # Field in note objects
    },
    "pitch_range": {
        "metric_name": "pitch_range",
        "high_threshold": 48,  # Wide range (4 octaves)
        "low_threshold": 29,  # Narrow range (1 octave)
        "json_field": "pitch",
    },
    "note_density": {
        "metric_name": "notes_per_beat",
        "high_threshold": 8.0,
        "low_threshold": 2.0,
        "json_field": None,  # Calculated from count
    },
    "average_pitch": {
        "metric_name": "average_pitch",
        "high_threshold": 67.6,  # High pitch notes (above middle C)
        "low_threshold": 60,  # Low pitch notes (below middle C)
        "json_field": "pitch",
    },
    "modality": {
        "metric_name": "modality",
        "metric_type": "categorical",  # Not threshold-based
        "categories": ["major", "minor"],
        "confidence_threshold": 0.7,  # Minimum confidence for classification
        "min_notes": 10,  # Minimum notes required for reliable detection
        "json_field": "pitch",  # Uses pitch for key detection
    },
    "average_duration": {
        "metric_name": "average_duration",
        "high_threshold": 48,  # Long notes (4 beats) - PLACEHOLDER, run find_thresholds.py
        "low_threshold": 12,  # Short notes (1 beat) - PLACEHOLDER, run find_thresholds.py
        "json_field": "duration",
    },
}

# Segmentation parameters
SEGMENT_N_BEATS = 16  # Length of segments to analyze
USE_SEGMENTATION = True  # Whether to segment songs or use whole songs

# Activation extraction parameters
BATCH_SIZE = 8
MAX_SEQ_LEN = 1024
MAX_BEAT = 256

# Steering parameters
ALPHA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
TARGET_LAYERS = None  # None means all layers, or specify list like [6, 7, 8, 9, 10, 11]

# Generation parameters
N_GENERATION_SAMPLES = 50
GENERATION_SEQ_LEN = 512
GENERATION_TEMPERATURE = 1.0
GENERATION_FILTER = "top_k"  # Changed from top_p to avoid repetitive generation
GENERATION_FILTER_THRESHOLD = 0.9
