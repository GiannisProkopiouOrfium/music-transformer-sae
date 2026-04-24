"""Configuration for PID Activation Steering experiments."""

import pathlib

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR = PROJECT_ROOT / "exp" / "sod"
MODEL_CHECKPOINT = EXP_DIR / "ape" / "checkpoints" / "best_model.pt"
TRAIN_ARGS_PATH = EXP_DIR / "ape" / "train-args.json"
ENCODING_PATH = PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"

# Steering vector inputs (from DiffMean pipeline)
STEERING_VECTORS_DIR = (
    PROJECT_ROOT / "steering_interventions" / "outputs" / "steering_vectors"
)
ACTIVATIONS_DIR = PROJECT_ROOT / "steering_interventions" / "outputs" / "activations"

# SAE models (for SAS temporal PID)
SAE_CHECKPOINT_DIR = EXP_DIR / "sparse_steering" / "sae_checkpoints"
SAS_VECTORS_DIR = EXP_DIR / "sparse_steering" / "sas_vectors"

# PID outputs
PID_OUTPUT_DIR = EXP_DIR / "pid_steering"
PID_VECTORS_DIR = PID_OUTPUT_DIR / "vectors"
PID_EXPERIMENTS_DIR = PID_OUTPUT_DIR / "experiments"
PID_AUDIO_DIR = PID_OUTPUT_DIR / "audio"
PID_PLOTS_DIR = PID_OUTPUT_DIR / "plots"

# ─── Model Architecture ──────────────────────────────────────────────────────
NUM_SUBLAYERS = 12  # 6 transformer blocks × 2 sublayers (attention + FF)
MODEL_DIM = 512  # Residual stream dimension
DEPTH = 6  # Number of transformer blocks

# ─── Contrastive Set Configuration ────────────────────────────────────────────
CONCEPTS = {
    "average_pitch": {
        "low_threshold": 60.0,  # ≤ 60 semitones (20th percentile)
        "high_threshold": 67.6,  # ≥ 67.6 semitones (80th percentile)
        "unit": "semitones",
    },
    "average_duration": {
        "low_threshold": 6.5,  # ≤ 6.5 ticks (20th percentile)
        "high_threshold": 14.5,  # ≥ 14.5 ticks (80th percentile)
        "unit": "ticks",
    },
}

# ─── Ground Truth Quality Metrics (from SOD corpus) ──────────────────────────
GROUND_TRUTH = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}

# ─── PID Default Gains ───────────────────────────────────────────────────────
# PID paper (Nguyen et al., ICLR 2026) found optimal Ki ∈ [0.05, 0.10],
# Kd ∈ [0.01, 0.05] for 26-42 layer LLMs.
# Grid search on MMT (12 sublayers) confirmed Ki=0.025 for duration,
# Ki=0.05 neighborhood for pitch. Using conservative unified gains.

SPATIAL_PID_DEFAULTS = {
    "Kp": 1.0,  # Proportional gain (1.0 = standard DiffMean strength)
    "Ki": 0.05,  # Integral gain (grid search validated, paper's lower optimal)
    "Kd": 0.01,  # Derivative gain (grid search confirmed across both concepts)
    "max_I": 5.0,  # Anti-windup clamp for integral accumulator
}

TEMPORAL_PID_DEFAULTS = {
    "Kp": 1.0,  # Proportional gain
    "Ki": 0.05,  # Integral gain
    "Kd": 0.01,  # Derivative gain
    "max_I": 10.0,  # Anti-windup clamp
    "lambda_min": 0.0,  # Minimum steering strength
    "lambda_max": 5.0,  # Maximum steering strength
}

# ─── Grid Search Ranges ──────────────────────────────────────────────────────
GRID_SEARCH = {
    "Kp": [0.5, 0.75, 1.0, 1.25, 1.5],
    "Ki": [0.0, 0.025, 0.05, 0.10, 0.15, 0.20],
    "Kd": [0.0, 0.01, 0.025, 0.05, 0.10],
    "max_I": [2.0, 5.0, 10.0],
}

# ─── Experiment Configuration ─────────────────────────────────────────────────
# Conditioned generation settings (aligned with SAS evaluator)
CONDITIONING_BEATS = 16  # Same as SAS conditioned evaluator
CONTINUATION_LEN = 512  # Tokens to generate after conditioning prefix

# Alpha values for conditioned evaluation (direction-aware: +α on low, -α on high)
PITCH_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5]
DURATION_ALPHAS = [0.5, 1.0, 1.5]

ALPHA_GRID = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]  # unconditioned sweep
N_GENERATIONS = 50  # Per alpha value for main experiments
N_GENERATIONS_QUICK = 10  # For grid search / ablation
MAX_SEQ_LEN = 512
TEMPERATURE = 1.0
FILTER_THRESHOLD = 0.9

# ─── Hook Ablation Configurations ────────────────────────────────────────────
HOOK_CONFIGS = {
    "all_12": list(range(12)),
    "attention_only": [0, 2, 4, 6, 8, 10],
    "feedforward_only": [1, 3, 5, 7, 9, 11],
    "deep_only": [8, 9, 10, 11],
    "mid_deep": [4, 5, 6, 7, 8, 9, 10, 11],
}

# ─── SAS Configuration (for temporal PID) ─────────────────────────────────────
SAS_LAYER = 10  # Optimal SAS intervention layer
SAS_TAU = 0.08  # Frequency filtering threshold
SPARSE_DIM = 4096  # SAE sparse dimension (8× expansion of 512)
