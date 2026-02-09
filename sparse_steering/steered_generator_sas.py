"""
Steered music generation using Sparse Activation Steering (SAS).

Implements Algorithm 2 from the SAS paper:
- Encodes activations to sparse space via SAE
- Adds scaled SAS steering vector: s = f(a) + λ·v_SAS
- Applies activation function and decodes back
- Adds correction term to preserve reconstruction quality

Prerequisites:
    - Trained MMT model at: exp/sod/ape/checkpoints/best_model.pt
    - Encoding at: data/sod/processed/notes/encoding.json
    - Trained SAE models at: exp/sod/sparse_steering/sae_models/
    - SAS vectors at: exp/sod/sparse_steering/sas_vectors/
    (Run compute_sas_vectors.py --tau 0.08 first)

Usage:
    # Generate with pitch steering
    python sparse_steering/steered_generator_sas.py \
        --concept average_pitch \
        --steering_strengths -2.0 -1.0 0.0 1.0 2.0 \
        --n_samples 5
    
    # Generate with duration steering at specific layers
    python sparse_steering/steered_generator_sas.py \
        --concept average_duration \
        --steering_strengths -1.5 0.0 1.5 \
        --layers 6 7 8 9 \
        --n_samples 3
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

# Add mmt directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import music_x_transformers
import representation
import utils

# Try to import config, but it's optional (only used for defaults)
try:
    import config

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from generated tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
        return pitches
    except Exception as e:
        logger.error(f"Error extracting pitches: {e}")
        return []


def extract_durations_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract duration values from generated tokens (in ticks)."""
    try:
        music = representation.decode(tokens, encoding)
        durations = []
        for track in music.tracks:
            for note in track.notes:
                durations.append(note.duration)
        return durations
    except Exception as e:
        logger.error(f"Error extracting durations: {e}")
        return []


class SASSteeringHook:
    """
    Forward hook that applies Sparse Activation Steering (SAS) during generation.

    Implements Algorithm 2: SAS Vectors in Inference
    """

    def __init__(
        self, sae_models, sas_vectors, concept, steering_strength, layers_to_steer=None
    ):
        """
        Args:
            sae_models: Dict mapping layer_idx -> SAE model
            sas_vectors: Dict mapping layer_idx -> SAS vector (numpy array)
            concept: Concept name (for logging)
            steering_strength: λ parameter (positive = amplify, negative = suppress)
            layers_to_steer: List of layer indices to apply steering (None = all)
        """
        self.sae_models = sae_models
        self.sas_vectors = sas_vectors
        self.concept = concept
        self.steering_strength = steering_strength
        self.layers_to_steer = layers_to_steer

        # Convert SAS vectors to torch tensors
        self.sas_vectors_torch = {
            layer_idx: torch.from_numpy(vec).float()
            for layer_idx, vec in sas_vectors.items()
        }

        self.device = None
        self._initialized = False

    def _initialize_device(self, activations):
        """Initialize device from first activation."""
        if not self._initialized:
            self.device = activations.device
            # Move SAS vectors to device
            for layer_idx in self.sas_vectors_torch:
                self.sas_vectors_torch[layer_idx] = self.sas_vectors_torch[
                    layer_idx
                ].to(self.device)
            self._initialized = True

    def __call__(self, module, input, output, layer_idx):
        """
        Forward hook implementing Algorithm 2.

        Args:
            module: The layer module
            input: Input to the layer
            output: Dense activations a_ℓ from the layer (may be tuple)
            layer_idx: Layer index

        Returns:
            Steered activations ã_ℓ (same format as output)
        """
        # Handle tuple outputs from Attention module
        # Attention modules return (output, attention_weights)
        is_tuple_output = isinstance(output, tuple)
        if is_tuple_output:
            actual_output = output[0]
            other_outputs = output[1:]  # Save for returning later
        else:
            actual_output = output
            other_outputs = None
        
        # Check if we should steer this layer
        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output

        if layer_idx not in self.sae_models or layer_idx not in self.sas_vectors_torch:
            return output

        # Initialize device on first call
        self._initialize_device(actual_output)

        # Get SAE and SAS vector for this layer
        sae = self.sae_models[layer_idx]
        v_sas = self.sas_vectors_torch[layer_idx]  # (D,) where D=4096

        # Algorithm 2: Sparse Activation Steering
        # Input: a_ℓ (dense activations), v_(b,ℓ) (SAS vector), λ (steering strength)

        a_l = actual_output  # (batch, seq, 512) dense activations
        batch_size, seq_len, d_model = a_l.shape

        # Flatten for processing
        a_flat = a_l.reshape(-1, d_model)  # (batch*seq, 512)

        # Step 1: Encode to sparse space
        # f(a_ℓ) = encoder(a_ℓ)
        f_a = sae.encode(a_flat)  # (batch*seq, 4096) sparse representation

        # Step 2: Compute correction term Δ
        # Δ := a_ℓ - decoder(f(a_ℓ))
        reconstructed = sae.decode(f_a)  # (batch*seq, 512)
        delta = a_flat - reconstructed  # (batch*seq, 512)

        # Step 3: Add sparse steering vector scaled by λ
        # s_ℓ := f(a_ℓ) + λ · v_(b,ℓ)
        s_l = f_a + self.steering_strength * v_sas.unsqueeze(0)  # (batch*seq, 4096)

        # Step 4: Apply activation function σ (ReLU + TopK to match SAE training)
        # and decode back to dense space
        # a'_ℓ := decoder(σ(s_ℓ))
        # CRITICAL: Must apply same sparsity as during encoding!
        # The SAE was trained with TopK, so decoder expects exactly K non-zero values
        s_l_relu = torch.relu(s_l)  # First apply ReLU (non-negativity)
        s_l_activated = sae.topk(s_l_relu)  # Then apply TopK (same K as encoding)
        a_prime = sae.decode(s_l_activated)  # (batch*seq, 512)

        # Step 5: Add correction term
        # ã_ℓ = a'_ℓ + Δ
        a_steered = a_prime + delta  # (batch*seq, 512)

        # Reshape back
        a_steered = a_steered.reshape(batch_size, seq_len, d_model)

        # Return in same format as input (tuple or tensor)
        if is_tuple_output:
            return (a_steered,) + other_outputs
        else:
            return a_steered


def load_model(checkpoint_path, train_args_path, encoding_path, device):
    """Load the trained MMT model."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load training args and encoding
    train_args = utils.load_json(train_args_path)
    encoding = representation.load_encoding(encoding_path)

    # Create model - match the parameter names from test_steering.py
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],  # Note: "layers" not "n_layers"
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    # Load checkpoint - note: might be just state_dict, not wrapped in {"model": ...}
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.info("Trying direct load...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    logger.info(f"✓ Model loaded ({train_args['layers']} layers)")
    return model, encoding, train_args


def get_adaptive_k(layer_idx: int) -> int:
    """Get adaptive K for layer."""
    if layer_idx == 0:
        return 32
    elif layer_idx < 4:
        return 64
    elif layer_idx < 8:
        return 96
    else:
        return 128


def load_sae_models(sae_dir, device):
    """Load trained SAE models for all layers."""
    logger.info(f"Loading SAE models from {sae_dir}")

    # Add sparse_steering directory to path for sae_model import
    sparse_steering_path = pathlib.Path(__file__).parent
    if str(sparse_steering_path) not in sys.path:
        sys.path.insert(0, str(sparse_steering_path))

    from sae_model import SparseAutoencoder

    HIDDEN_DIM = 512
    SPARSE_DIM = 4096

    sae_models = {}
    for layer_idx in range(12):  # Assuming 12 layers
        checkpoint_path = sae_dir / f"sae_layer_{layer_idx}_best.pt"
        if not checkpoint_path.exists():
            logger.warning(f"SAE checkpoint not found for layer {layer_idx}")
            continue

        # Get adaptive K for this layer
        k = get_adaptive_k(layer_idx)

        # Create SAE model
        sae = SparseAutoencoder(
            input_dim=HIDDEN_DIM,
            sparse_dim=SPARSE_DIM,
            k=k,
            tied_weights=True,
            normalize_input=True,
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = checkpoint["model_state_dict"]

        # Backward compatibility: add _normalization_fitted if missing
        if "_normalization_fitted" not in state_dict:
            if "input_mean" in state_dict:
                has_normalization = state_dict["input_mean"].abs().sum() > 0
                state_dict["_normalization_fitted"] = torch.tensor(
                    1 if has_normalization else 0
                )
            else:
                state_dict["_normalization_fitted"] = torch.tensor(0)

        sae.load_state_dict(state_dict)
        sae.eval()
        sae = sae.to(device)

        sae_models[layer_idx] = sae

    logger.info(f"✓ Loaded {len(sae_models)} SAE models")
    return sae_models


def load_sas_vectors(sas_vectors_path):
    """Load pre-computed SAS vectors."""
    logger.info(f"Loading SAS vectors from {sas_vectors_path}")

    data = torch.load(sas_vectors_path, map_location="cpu")

    # Data is already numpy arrays (saved as dict of numpy arrays)
    sas_vectors = {}
    for layer_idx, vec_data in data.items():
        # Handle both numpy arrays and tensors
        if isinstance(vec_data, np.ndarray):
            sas_vectors[layer_idx] = vec_data
        else:
            sas_vectors[layer_idx] = vec_data.numpy()

    logger.info(f"✓ Loaded SAS vectors for {len(sas_vectors)} layers")

    # Diagnostic: Show vector statistics
    logger.info("SAS Vector Statistics:")
    for layer_idx in sorted(sas_vectors.keys()):
        vec = sas_vectors[layer_idx]
        l0 = (vec != 0).sum()
        magnitude = np.linalg.norm(vec)
        n_positive = (vec > 0).sum()
        n_negative = (vec < 0).sum()
        logger.info(
            f"  Layer {layer_idx}: L0={l0:4d}, ||v||={magnitude:7.2f}, "
            f"pos={n_positive:4d}, neg={n_negative:4d}, "
            f"mean={vec.mean():+.4f}"
        )

    return sas_vectors


def register_steering_hooks(
    model, sae_models, sas_vectors, concept, steering_strength, layers_to_steer=None
):
    """
    Register forward hooks to apply SAS steering during generation.

    Args:
        model: The MMT model
        sae_models: Dictionary of SAE models
        sas_vectors: Dictionary of SAS vectors
        concept: Concept name
        steering_strength: λ parameter
        layers_to_steer: Specific layers to steer (None = all)

    Returns:
        List of hook handles (for cleanup)
    """
    hook = SASSteeringHook(
        sae_models=sae_models,
        sas_vectors=sas_vectors,
        concept=concept,
        steering_strength=steering_strength,
        layers_to_steer=layers_to_steer,
    )

    handles = []
    layers = model.decoder.net.attn_layers.layers

    for layer_idx, layer in enumerate(layers):
        # CRITICAL: Hook the same module we used during activation extraction!
        # The SAE was trained on attention module outputs (layer_module_list[1]),
        # NOT on full layer outputs. We must hook the same module for consistency.
        #
        # Layer structure: ModuleList[prenorm/ModuleList, Attention, Residual]
        # We want index 1 (the Attention module) to match activation extraction
        if isinstance(layer, nn.ModuleList) and len(layer) > 1:
            target_module = layer[1]  # The Attention module
        else:
            target_module = layer  # Fallback: hook the whole layer

        handle = target_module.register_forward_hook(
            lambda module, input, output, idx=layer_idx: hook(
                module, input, output, idx
            )
        )
        handles.append(handle)

    return handles


def remove_hooks(handles):
    """Remove all registered hooks."""
    for handle in handles:
        handle.remove()


def generate_with_steering(
    model,
    encoding,
    sae_models,
    sas_vectors,
    concept,
    steering_strength,
    n_samples=5,
    layers_to_steer=None,
    max_seq_len=512,
    output_dir=None,
    device="cuda",
):
    """
    Generate music samples with SAS steering.

    Args:
        model: MMT model
        encoding: Encoding dictionary
        sae_models: Dictionary of SAE models
        sas_vectors: Dictionary of SAS vectors
        concept: Concept name
        steering_strength: λ parameter
        n_samples: Number of samples to generate
        layers_to_steer: Specific layers to steer
        max_seq_len: Maximum sequence length
        output_dir: Directory to save generated sequences
        device: Device

    Returns:
        List of generated token sequences (numpy arrays)
    """
    logger.info(f"Generating {n_samples} samples with SAS steering")
    logger.info(f"  Concept: {concept}")
    logger.info(f"  Steering strength λ: {steering_strength}")
    logger.info(f"  Layers: {layers_to_steer if layers_to_steer else 'all'}")

    # Register hooks
    handles = register_steering_hooks(
        model, sae_models, sas_vectors, concept, steering_strength, layers_to_steer
    )

    # Setup output directory
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get SOS and EOS tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    try:
        generated_sequences = []

        for i in range(n_samples):
            logger.info(f"  Generating sample {i+1}/{n_samples}...")

            # Create start tokens (1, 1, 6) - matches test_steering.py
            start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start_tokens[:, 0, 0] = sos

            # Generate with steering
            with torch.no_grad():
                # Use config defaults if available, otherwise use sensible defaults
                temperature = 1.0
                filter_fn = "top_k"
                filter_thresh = 0.9

                if HAS_CONFIG:
                    temperature = getattr(config, "GENERATION_TEMPERATURE", 1.0)
                    filter_fn = getattr(config, "GENERATION_FILTER", "top_k")
                    filter_thresh = getattr(config, "GENERATION_FILTER_THRESHOLD", 0.9)

                generated = model.generate(
                    start_tokens,
                    max_seq_len,
                    eos_token=eos,
                    temperature=temperature,
                    filter_logits_fn=filter_fn,
                    filter_thres=filter_thresh,
                    monotonicity_dim=("type", "beat"),
                )

            # Combine start and generated
            full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
            generated_sequences.append(full_seq)

            # Save if output directory provided
            if output_dir is not None:
                strength_str = f"{'pos' if steering_strength >= 0 else 'neg'}{abs(steering_strength):.1f}"
                filename = f"sample_lambda_{strength_str}_num_{i}.npy"
                filepath = output_dir / filename
                np.save(filepath, full_seq)
                logger.info(f"    Saved: {filename}")

        # Extract metrics from all generated sequences
        all_pitches = []
        all_durations = []
        for seq in generated_sequences:
            pitches = extract_pitches_from_tokens(seq, encoding)
            durations = extract_durations_from_tokens(seq, encoding)
            all_pitches.extend(pitches)
            all_durations.extend(durations)

        # Compute aggregate statistics
        metrics = {
            "pitch_mean": float(np.mean(all_pitches)) if all_pitches else 0.0,
            "pitch_std": float(np.std(all_pitches)) if all_pitches else 0.0,
            "pitch_min": int(np.min(all_pitches)) if all_pitches else 0,
            "pitch_max": int(np.max(all_pitches)) if all_pitches else 0,
            "duration_mean": float(np.mean(all_durations)) if all_durations else 0.0,
            "duration_std": float(np.std(all_durations)) if all_durations else 0.0,
            "n_notes": len(all_pitches),
        }

        return generated_sequences, metrics

    finally:
        # Always remove hooks
        remove_hooks(handles)
        logger.info("✓ Steering hooks removed")


def analyze_steering_effect(all_results: dict, concept: str):
    """Analyze steering effectiveness across all lambda values.

    Args:
        all_results: Dict mapping lambda values to metrics
        concept: Concept being steered (average_pitch or average_duration)
    """
    if len(all_results) < 3:
        logger.warning("Not enough lambda values for statistical analysis")
        return

    sorted_lambdas = sorted(all_results.keys())
    lambdas_array = np.array(sorted_lambdas)

    # Extract metrics
    pitch_means = np.array([all_results[lam]["pitch_mean"] for lam in sorted_lambdas])
    duration_means = np.array(
        [all_results[lam]["duration_mean"] for lam in sorted_lambdas]
    )

    # Calculate correlations
    pitch_corr, pitch_pval = scipy_stats.pearsonr(lambdas_array, pitch_means)
    pitch_spearman, pitch_spearman_pval = scipy_stats.spearmanr(
        lambdas_array, pitch_means
    )
    duration_corr, duration_pval = scipy_stats.pearsonr(lambdas_array, duration_means)
    duration_spearman, duration_spearman_pval = scipy_stats.spearmanr(
        lambdas_array, duration_means
    )

    # Linear regression
    pitch_slope, pitch_intercept, pitch_r_value, _, _ = scipy_stats.linregress(
        lambdas_array, pitch_means
    )
    duration_slope, duration_intercept, duration_r_value, _, _ = scipy_stats.linregress(
        lambdas_array, duration_means
    )

    # Check monotonicity
    pitch_monotonic = all(
        pitch_means[i] <= pitch_means[i + 1] for i in range(len(pitch_means) - 1)
    )
    duration_monotonic = all(
        duration_means[i] <= duration_means[i + 1]
        for i in range(len(duration_means) - 1)
    )

    # Baseline stats (lambda = 0)
    if 0.0 in all_results:
        baseline = all_results[0.0]
        min_lambda = sorted_lambdas[0]
        max_lambda = sorted_lambdas[-1]

        baseline_pitch = baseline["pitch_mean"]
        min_pitch = all_results[min_lambda]["pitch_mean"]
        max_pitch = all_results[max_lambda]["pitch_mean"]

        baseline_duration = baseline["duration_mean"]
        min_duration = all_results[min_lambda]["duration_mean"]
        max_duration = all_results[max_lambda]["duration_mean"]

    # Helper functions
    def interpret_correlation(r, pval):
        if pval > 0.05:
            return "NOT SIGNIFICANT"
        elif abs(r) >= 0.9:
            return "VERY STRONG"
        elif abs(r) >= 0.7:
            return "STRONG"
        elif abs(r) >= 0.5:
            return "MODERATE"
        else:
            return "WEAK"

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("STEERING EFFECT ANALYSIS")
    logger.info("=" * 80)

    logger.info(f"\nConcept: {concept}")
    logger.info(f"Lambda values tested: {sorted_lambdas}")
    logger.info(
        f"Total samples: {sum(all_results[lam]['n_notes'] for lam in sorted_lambdas)} notes"
    )

    logger.info("\n" + "-" * 80)
    logger.info("PITCH ANALYSIS")
    logger.info("-" * 80)

    if 0.0 in all_results:
        logger.info(f"Baseline (λ=0.0):       {baseline_pitch:.2f}")
        logger.info(
            f"Min (λ={min_lambda:+.1f}):        {min_pitch:.2f} ({min_pitch - baseline_pitch:+.2f}, {((min_pitch - baseline_pitch) / baseline_pitch * 100):+.1f}%)"
        )
        logger.info(
            f"Max (λ={max_lambda:+.1f}):        {max_pitch:.2f} ({max_pitch - baseline_pitch:+.2f}, {((max_pitch - baseline_pitch) / baseline_pitch * 100):+.1f}%)"
        )
        logger.info(f"Range:                  {max_pitch - min_pitch:.2f} semitones")

    logger.info(f"\nCorrelation Analysis:")
    logger.info(
        f"  Pearson r:            {pitch_corr:+.4f} (p={pitch_pval:.4f}) - {interpret_correlation(pitch_corr, pitch_pval)}"
    )
    logger.info(
        f"  Spearman ρ:           {pitch_spearman:+.4f} (p={pitch_spearman_pval:.4f})"
    )
    logger.info(
        f"  Linear R²:            {pitch_r_value**2:.4f} ({'Good' if pitch_r_value**2 > 0.8 else 'Moderate' if pitch_r_value**2 > 0.5 else 'Poor'} fit)"
    )
    logger.info(f"  Slope:                {pitch_slope:+.4f} semitones per λ")
    logger.info(f"  Monotonic:            {'✓ Yes' if pitch_monotonic else '✗ No'}")

    logger.info("\n" + "-" * 80)
    logger.info("DURATION ANALYSIS")
    logger.info("-" * 80)

    if 0.0 in all_results:
        logger.info(f"Baseline (λ=0.0):       {baseline_duration:.2f} ticks")
        logger.info(
            f"Min (λ={min_lambda:+.1f}):        {min_duration:.2f} ({min_duration - baseline_duration:+.2f}, {((min_duration - baseline_duration) / baseline_duration * 100) if baseline_duration > 0 else 0:+.1f}%)"
        )
        logger.info(
            f"Max (λ={max_lambda:+.1f}):        {max_duration:.2f} ({max_duration - baseline_duration:+.2f}, {((max_duration - baseline_duration) / baseline_duration * 100) if baseline_duration > 0 else 0:+.1f}%)"
        )
        logger.info(f"Range:                  {max_duration - min_duration:.2f} ticks")

    logger.info(f"\nCorrelation Analysis:")
    logger.info(
        f"  Pearson r:            {duration_corr:+.4f} (p={duration_pval:.4f}) - {interpret_correlation(duration_corr, duration_pval)}"
    )
    logger.info(
        f"  Spearman ρ:           {duration_spearman:+.4f} (p={duration_spearman_pval:.4f})"
    )
    logger.info(
        f"  Linear R²:            {duration_r_value**2:.4f} ({'Good' if duration_r_value**2 > 0.8 else 'Moderate' if duration_r_value**2 > 0.5 else 'Poor'} fit)"
    )
    logger.info(f"  Slope:                {duration_slope:+.4f} ticks per λ")
    logger.info(f"  Monotonic:            {'✓ Yes' if duration_monotonic else '✗ No'}")

    logger.info("\n" + "-" * 80)
    logger.info("PROGRESSION ACROSS LAMBDA")
    logger.info("-" * 80)
    for lam in sorted_lambdas:
        r = all_results[lam]
        logger.info(
            f"λ={lam:+5.1f}: pitch={r['pitch_mean']:6.2f}, duration={r['duration_mean']:6.2f} ticks, n={r['n_notes']:4d} notes"
        )

    # Effectiveness evaluation
    logger.info("\n" + "-" * 80)
    logger.info("STEERING EFFECTIVENESS")
    logger.info("-" * 80)

    # Determine target metric based on concept
    if "pitch" in concept.lower():
        target_metric = "pitch"
        target_corr = pitch_corr
        target_pval = pitch_pval
        target_r2 = pitch_r_value**2
        target_monotonic = pitch_monotonic
    else:
        target_metric = "duration"
        target_corr = duration_corr
        target_pval = duration_pval
        target_r2 = duration_r_value**2
        target_monotonic = duration_monotonic

    logger.info(f"\nTarget metric: {target_metric.upper()}")

    if target_pval > 0.05:
        logger.warning(f"  ✗ FAILED: No significant correlation (p={target_pval:.4f})")
    elif abs(target_corr) > 0.7 and target_monotonic and target_r2 > 0.7:
        logger.info(f"  ✓✓✓ EXCELLENT: Strong monotonic steering effect")
        logger.info(
            f"      r={target_corr:+.3f}, R²={target_r2:.3f}, monotonic={target_monotonic}"
        )
    elif abs(target_corr) > 0.5 and target_r2 > 0.5:
        logger.info(f"  ✓✓ GOOD: Clear steering effect")
        logger.info(
            f"      r={target_corr:+.3f}, R²={target_r2:.3f}, monotonic={target_monotonic}"
        )
    elif abs(target_corr) > 0.3:
        logger.info(f"  ✓ WEAK: Some steering detected but inconsistent")
        logger.info(
            f"      r={target_corr:+.3f}, R²={target_r2:.3f}, monotonic={target_monotonic}"
        )
    else:
        logger.warning(f"  ✗ FAILED: Steering effect too weak or inconsistent")
        logger.warning(
            f"      r={target_corr:+.3f}, R²={target_r2:.3f}, monotonic={target_monotonic}"
        )

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with Sparse Activation Steering (SAS)"
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=["average_pitch", "average_duration"],
        help="Concept to steer",
    )
    parser.add_argument(
        "--steering_strengths",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Steering strengths λ to test (positive=amplify, negative=suppress, 0=baseline)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to steer (default: all layers)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples per steering strength",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--exp_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: exp_dir/sparse_steering/generations_sas)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")

    args = parser.parse_args()

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths - matching the structure from encode_concept_activations.py
    checkpoint_path = args.exp_dir / "ape" / "checkpoints" / "best_model.pt"
    train_args_path = args.exp_dir / "ape" / "train-args.json"
    encoding_path = pathlib.Path("data/sod/processed/notes") / "encoding.json"
    sae_dir = args.exp_dir / "sparse_steering" / "sae_checkpoints"
    sas_vectors_path = (
        args.exp_dir
        / "sparse_steering"
        / "sas_vectors"
        / f"{args.concept}_sas_vectors.pt"
    )

    if args.output_dir is None:
        args.output_dir = (
            args.exp_dir / "sparse_steering" / "generations_sas" / args.concept
        )

    # Validate paths
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found: {checkpoint_path}")
        logger.error(f"Expected at: {checkpoint_path.absolute()}")
        logger.error("Please check --exp_dir argument or ensure model is trained")
        sys.exit(1)
    if not encoding_path.exists():
        logger.error(f"Encoding not found: {encoding_path}")
        logger.error(f"Expected at: {encoding_path.absolute()}")
        sys.exit(1)
    if not sas_vectors_path.exists():
        logger.error(f"SAS vectors not found: {sas_vectors_path}")
        logger.error(f"Expected at: {sas_vectors_path.absolute()}")
        logger.error("Run: python sparse_steering/compute_sas_vectors.py --tau 0.08")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Sparse Activation Steering (SAS) - Music Generation")
    logger.info("=" * 80)
    logger.info(f"Concept: {args.concept}")
    logger.info(f"Steering strengths: {args.steering_strengths}")
    logger.info(f"Layers: {args.layers if args.layers else 'all'}")
    logger.info(f"Samples per strength: {args.n_samples}")

    # Load model
    model, encoding, train_args = load_model(
        checkpoint_path, train_args_path, encoding_path, device
    )

    # Load SAE models
    sae_models = load_sae_models(sae_dir, device)

    # Load SAS vectors
    sas_vectors = load_sas_vectors(sas_vectors_path)

    # Store results for analysis
    all_results = {}

    # Generate for each steering strength
    for strength in args.steering_strengths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Steering strength λ = {strength}")
        logger.info(f"{'='*80}")

        # Create output directory for this steering strength
        strength_output_dir = (
            args.output_dir / f"lambda_{'+' if strength >= 0 else ''}{strength}"
        )

        sequences, metrics = generate_with_steering(
            model=model,
            encoding=encoding,
            sae_models=sae_models,
            sas_vectors=sas_vectors,
            concept=args.concept,
            steering_strength=strength,
            n_samples=args.n_samples,
            layers_to_steer=args.layers,
            max_seq_len=args.max_seq_len,
            output_dir=strength_output_dir,
            device=device,
        )

        # Store metrics
        all_results[strength] = metrics
        logger.info(
            f"  λ={strength:+.1f}: pitch_mean={metrics['pitch_mean']:.2f}, duration_mean={metrics['duration_mean']:.2f}, n={metrics['n_notes']} notes"
        )

    # Save all metrics to JSON
    metrics_file = args.output_dir / "steering_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    logger.info(f"\n✓ Saved metrics to: {metrics_file}")

    # Analyze steering effectiveness
    analyze_steering_effect(all_results, args.concept)

    logger.info(f"\n{'='*80}")
    logger.info("✓ Generation complete!")
    logger.info(f"✓ Outputs saved to: {args.output_dir}")
    logger.info(f"{'='*80}")
    logger.info("\nNext steps:")
    logger.info("1. Analyze generated .npy files with deterministic metrics")
    logger.info("2. Compare SAS vs DiffMean steering effects")
    logger.info("3. Run ablation studies on different layers")


if __name__ == "__main__":
    main()
