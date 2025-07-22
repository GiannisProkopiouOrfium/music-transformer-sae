"""
Pipeline Configuration Schema

Defines the configuration structure for the SAE pipeline.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pathlib


@dataclass
class ModelConfig:
    """Configuration for model loading and setup."""

    dataset: str = "sod"
    representation: str = "ape"
    model_steps: Optional[int] = None  # None = best model
    gpu: int = -1  # -1 = CPU, >=0 = GPU index
    max_seq_len: int = 1024
    max_beat: int = 256


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""

    layers: List[int] = field(default_factory=lambda: [3])  # Transformer layer indices
    n_samples: int = 500
    batch_size: int = 4
    max_seq_len: int = 1024
    use_generation: bool = False  # Whether to also extract from generated samples
    generation_samples: int = 100

    # Generation parameters (if use_generation=True)
    seq_len: int = 512
    temperature: float = 1.0
    filter_fn: str = "top_k"
    filter_threshold: float = 0.9


@dataclass
class SAEConfig:
    """Configuration for SAE training."""

    hidden_dim: int = 2048  # Latent dimension (expansion factor ~4x)
    sparsity_coeff: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 1024
    num_epochs: int = 5
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Advanced options
    k_sparse: Optional[int] = None  # For top-k sparsity (alternative to L1)
    weight_decay: float = 0.0
    scheduler: str = "none"  # "none", "cosine", "linear"
    
    # Data loading optimization
    num_workers: int = 0  # Number of DataLoader workers (0=single-threaded, 2-4 recommended for GPU)
    
    # Data subsampling for faster training
    subsample_training: Optional[int] = None  # Subsample training data for speed
    subsample_validation: Optional[int] = None  # Subsample validation data


@dataclass
class AnalysisConfig:
    """Configuration for feature analysis."""

    top_k_segments: int = 10  # Top segments per feature
    min_activation_freq: float = 0.01
    max_activation_freq: float = 0.5
    correlation_threshold: float = 0.3
    save_visualizations: bool = True

    # Musical analysis
    analyze_harmony: bool = True
    analyze_rhythm: bool = True
    analyze_melody: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Paths
    input_dir: Optional[pathlib.Path] = None
    output_dir: Optional[pathlib.Path] = None
    config_name: str = "default"

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Pipeline control
    run_extraction: bool = True
    run_training: bool = True
    run_analysis: bool = True

    # Logging
    log_level: str = "INFO"
    save_logs: bool = True

    def __post_init__(self):
        """Set default paths based on dataset and representation."""
        if self.input_dir is None:
            self.input_dir = pathlib.Path(f"data/{self.model.dataset}/processed/notes")

        if self.output_dir is None:
            self.output_dir = pathlib.Path(
                f"exp/{self.model.dataset}/{self.model.representation}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        # Extract nested configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        extraction_config = ExtractionConfig(**config_dict.get("extraction", {}))
        sae_config = SAEConfig(**config_dict.get("sae", {}))
        analysis_config = AnalysisConfig(**config_dict.get("analysis", {}))

        # Extract top-level config
        top_level = {
            k: v
            for k, v in config_dict.items()
            if k not in ["model", "extraction", "sae", "analysis"]
        }

        return cls(
            model=model_config,
            extraction=extraction_config,
            sae=sae_config,
            analysis=analysis_config,
            **top_level,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "extraction": self.extraction.__dict__,
            "sae": self.sae.__dict__,
            "analysis": self.analysis.__dict__,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "config_name": self.config_name,
            "run_extraction": self.run_extraction,
            "run_training": self.run_training,
            "run_analysis": self.run_analysis,
            "log_level": self.log_level,
            "save_logs": self.save_logs,
        }
