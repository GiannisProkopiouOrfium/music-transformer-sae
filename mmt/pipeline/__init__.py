"""
Music Transformer SAE Pipeline

This package provides a unified pipeline for Sparse Autoencoder analysis
of Music Transformers. It includes:

- extract: Extract activations from transformer layers
- train: Train sparse autoencoders on extracted activations
- analyze: Analyze learned features and correlate with musical concepts
- main: Unified pipeline runner
"""

# Import functions only when explicitly needed to avoid circular imports
__all__ = [
    "extract_activations_pipeline",
    "train_sae_pipeline",
    "analyze_sae_pipeline",
    "run_pipeline",
    "PipelineConfig",
]


def __getattr__(name):
    if name == "extract_activations_pipeline":
        from .extract import extract_activations_pipeline

        return extract_activations_pipeline
    elif name == "train_sae_pipeline":
        from .train import train_sae_pipeline

        return train_sae_pipeline
    elif name == "analyze_sae_pipeline":
        from .analyze import analyze_sae_pipeline

        return analyze_sae_pipeline
    elif name == "run_pipeline":
        from .main import run_pipeline

        return run_pipeline
    elif name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
