"""Configuration for SAS dual-concept (pitch + duration) steering.

Centralises paths, alpha grids, conditioned-scenario definitions and
ground-truth quality metrics so every script in this package shares
consistent settings.
"""

import pathlib

# ── Paths (relative to project root) ────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
SPARSE_STEERING_DIR = PROJECT_ROOT / "sparse_steering"

OUTPUT_BASE = PROJECT_ROOT / "exp" / "sod" / "sparse_steering"
SAE_CHECKPOINT_DIR = OUTPUT_BASE / "sae_checkpoints"
SAS_VECTORS_DIR = OUTPUT_BASE / "sas_vectors"

# Model & data
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "exp" / "sod" / "ape" / "checkpoints" / "best_model.pt"
)
DEFAULT_TRAIN_ARGS = PROJECT_ROOT / "exp" / "sod" / "ape" / "train-args.json"
DEFAULT_ENCODING = (
    PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
)
DEFAULT_NOTES_DIR = PROJECT_ROOT / "data" / "sod" / "processed" / "notes"

# SAS vectors for each concept
PITCH_SAS_VECTORS = SAS_VECTORS_DIR / "average_pitch_sas_vectors.pt"
DURATION_SAS_VECTORS = SAS_VECTORS_DIR / "average_duration_sas_vectors.pt"

# Default output directories
OVERLAP_OUTPUT_DIR = OUTPUT_BASE / "dual_steering" / "overlap_analysis"
UNCONDITIONED_OUTPUT_DIR = OUTPUT_BASE / "dual_steering" / "unconditioned"
CONDITIONED_OUTPUT_DIR = OUTPUT_BASE / "dual_steering" / "conditioned"

# ── SAE / SAS architecture constants  ───────────────────────────────────────
HIDDEN_DIM = 512
SPARSE_DIM = 4096
NUM_LAYERS = 12

# ── Ground-truth quality metrics ────────────────────────────────────────────
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}

# ── Default steering layers (from single-concept optimisation) ──────────────
# Layer 10 was optimal for both pitch and duration individually.
DEFAULT_LAYERS_TO_STEER = [10]

# ── Alpha grids for unconditioned grid search ───────────────────────────────
# Symmetric, centred on 0, matching single-concept ranges
ALPHA_PITCH_GRID = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
ALPHA_DURATION_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

N_SAMPLES_PER_CONFIG = 5  # samples generated per (α_p, α_d) pair
SEQ_LEN = 512  # target tokens per sample

# ── Composition strategies to compare ───────────────────────────────────────
STRATEGIES = [
    "direct",
    "cross_concept_masking",
    "gram_schmidt_pitch",
    "gram_schmidt_duration",
    "cross_concept_sas",
    "expanded_k",
    "expanded_k_2x",
    "sequential",
    "topk_budget",
    "opposite_sign_masking",
    "opposite_sign_masking_ek2",
    "cross_concept_masking_ek2",
]

# ── Conditioned evaluation scenarios ────────────────────────────────────────
# Each scenario finds extreme songs and steers the opposite direction.
CONDITIONED_SCENARIOS = {
    "low_pitch_short_duration_to_high_long": {
        "song_category": "low_pitch_short_duration",
        "alpha_pitch_sign": +1,  # push pitch UP
        "alpha_duration_sign": +1,  # push duration UP (longer)
        "description": "Low pitch + short → steer high + long (fight both up)",
    },
    "high_pitch_long_duration_to_low_short": {
        "song_category": "high_pitch_long_duration",
        "alpha_pitch_sign": -1,  # push pitch DOWN
        "alpha_duration_sign": -1,  # push duration DOWN (shorter)
        "description": "High pitch + long → steer low + short (fight both down)",
    },
    "low_pitch_long_duration_to_high_short": {
        "song_category": "low_pitch_long_duration",
        "alpha_pitch_sign": +1,  # push pitch UP
        "alpha_duration_sign": -1,  # push duration DOWN (shorter)
        "description": "Low pitch + long → steer high + short (opposing fight)",
    },
    "high_pitch_short_duration_to_low_long": {
        "song_category": "high_pitch_short_duration",
        "alpha_pitch_sign": -1,  # push pitch DOWN
        "alpha_duration_sign": +1,  # push duration UP (longer)
        "description": "High pitch + short → steer low + long (opposing fight)",
    },
}

# Song classification thresholds (from DiffMean config)
PITCH_HIGH_THRESHOLD = 67.6
PITCH_LOW_THRESHOLD = 60.0
DURATION_HIGH_THRESHOLD = 14.5  # "long" notes in ticks
DURATION_LOW_THRESHOLD = 6.5  # "short" notes in ticks

N_SONGS_PER_SCENARIO = 5
CONDITIONING_BEATS = 16

# Alphas used during conditioned evaluation (only the direction appropriate
# for each scenario is kept)
CONDITIONED_ALPHA_PITCH = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
CONDITIONED_ALPHA_DURATION = [0.0, 0.5, 1.0, 1.5]
