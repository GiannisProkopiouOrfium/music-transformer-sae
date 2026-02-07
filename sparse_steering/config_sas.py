"""Configuration for Sparse Activation Steering (SAS)."""

import pathlib

# Paths (relative to project root)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
STEERING_DIR = PROJECT_ROOT / "steering_interventions"
OUTPUT_DIR = PROJECT_ROOT / "exp" / "sod"
SPARSE_OUTPUT_DIR = OUTPUT_DIR / "sparse_steering"

# Directories
SAE_CHECKPOINT_DIR = SPARSE_OUTPUT_DIR / "sae_checkpoints"
SAE_TRAINING_DATA_DIR = SPARSE_OUTPUT_DIR / "sae_training_data"
SAS_VECTORS_DIR = SPARSE_OUTPUT_DIR / "sas_vectors"
SPARSE_EXPERIMENTS_DIR = SPARSE_OUTPUT_DIR / "experiments"

# SAE Architecture Hyperparameters
# Starting with smaller expansion for initial testing
EXPANSION_FACTOR = 8  # 512 * 8 = 4096 features (can scale to 16 or 32 later)
HIDDEN_DIM = 512  # Model's residual stream dimension
SPARSE_DIM = HIDDEN_DIM * EXPANSION_FACTOR  # 4096 for 8x expansion

# TopK Sparsity
K = 32  # Number of active features (adjust based on expansion factor)
# For 8x expansion (4096 features), K=32 means ~0.78% sparsity
# For 16x expansion (8192 features), K=64 means ~0.78% sparsity

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64  # For SAE training
EPOCHS = 50
L1_COEFFICIENT = 1e-3  # Weight for L1 sparsity loss

# Data Collection
N_TRAINING_SEGMENTS = 10000  # Number of random segments for SAE training
N_TEST_SEGMENTS = 100  # Quick test mode

# Validation Criteria
TARGET_MSE = 0.01  # Maximum acceptable reconstruction error
TARGET_SPARSITY = K  # Expected number of active features (L0 norm)
SPARSITY_TOLERANCE = 10  # Allow K ± tolerance active features

# Frequency Filtering (for SAS vector generation)
TAU_THRESHOLD = 0.3  # Frequency threshold for feature selection

# Model Information (from MMT training)
NUM_LAYERS = 12  # Number of transformer layers in MMT
MODEL_DIM = 512  # Residual stream dimension

# Quick Test Mode
QUICK_TEST = False  # Set to True for fast configuration validation
QUICK_TEST_SEGMENTS = 100  # Reduced segments for quick test
QUICK_TEST_EPOCHS = 5  # Reduced epochs for quick test


def get_quick_test_config():
    """Return configuration for quick testing mode."""
    return {
        "n_segments": QUICK_TEST_SEGMENTS,
        "epochs": QUICK_TEST_EPOCHS,
        "batch_size": BATCH_SIZE,
        "expansion_factor": EXPANSION_FACTOR,
        "k": K,
        "learning_rate": LEARNING_RATE,
        "l1_coefficient": L1_COEFFICIENT,
    }


def get_full_config():
    """Return configuration for full training."""
    return {
        "n_segments": N_TRAINING_SEGMENTS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "expansion_factor": EXPANSION_FACTOR,
        "k": K,
        "learning_rate": LEARNING_RATE,
        "l1_coefficient": L1_COEFFICIENT,
    }


def print_config(quick_test=False):
    """Print current configuration."""
    config = get_quick_test_config() if quick_test else get_full_config()

    print("=" * 60)
    print("SAS Configuration")
    print("=" * 60)
    print(f"Mode: {'QUICK TEST' if quick_test else 'FULL TRAINING'}")
    print(f"Training segments: {config['n_segments']:,}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(
        f"Expansion factor: {config['expansion_factor']}x ({HIDDEN_DIM} → {SPARSE_DIM})"
    )
    print(f"TopK sparsity: ADAPTIVE (K=32/64/128 by layer)")
    print(f"  Layers 0-3: K=32 (0.78% active)")
    print(f"  Layers 4-7: K=64 (1.56% active)")
    print(f"  Layers 8-11: K=128 (3.12% active)")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"L1 coefficient: {config['l1_coefficient']}")
    print(f"Target MSE: ADAPTIVE (<0.05/<0.5/<2.0 by layer)")
    print(f"Target L0: ADAPTIVE (32/64/128 ± {SPARSITY_TOLERANCE} by layer)")
    print("=" * 60)
