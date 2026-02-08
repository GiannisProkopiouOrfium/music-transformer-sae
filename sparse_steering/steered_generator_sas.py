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
import logging
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

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
            output: Dense activations a_ℓ from the layer
            layer_idx: Layer index

        Returns:
            Steered activations ã_ℓ
        """
        # Check if we should steer this layer
        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output

        if layer_idx not in self.sae_models or layer_idx not in self.sas_vectors_torch:
            return output

        # Initialize device on first call
        self._initialize_device(output)

        # Get SAE and SAS vector for this layer
        sae = self.sae_models[layer_idx]
        v_sas = self.sas_vectors_torch[layer_idx]  # (D,) where D=4096

        # Algorithm 2: Sparse Activation Steering
        # Input: a_ℓ (dense activations), v_(b,ℓ) (SAS vector), λ (steering strength)

        a_l = output  # (batch, seq, 512) dense activations
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

        # Step 4: Apply activation function σ (ReLU for TopK SAE)
        # and decode back to dense space
        # a'_ℓ := decoder(σ(s_ℓ))
        s_l_activated = torch.relu(s_l)  # Ensure non-negativity
        a_prime = sae.decode(s_l_activated)  # (batch*seq, 512)

        # Step 5: Add correction term
        # ã_ℓ = a'_ℓ + Δ
        a_steered = a_prime + delta  # (batch*seq, 512)

        # Reshape back
        a_steered = a_steered.reshape(batch_size, seq_len, d_model)

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


def load_sae_models(sae_dir, device):
    """Load trained SAE models for all layers."""
    logger.info(f"Loading SAE models from {sae_dir}")

    # Add sae directory to path
    sae_path = pathlib.Path(__file__).parent.parent / "sae"
    if str(sae_path) not in sys.path:
        sys.path.insert(0, str(sae_path))

    from train_adaptive_sae import AdaptiveTopKSAE

    sae_models = {}
    for layer_idx in range(12):  # Assuming 12 layers
        checkpoint_path = sae_dir / f"layer{layer_idx}_sae.pt"
        if not checkpoint_path.exists():
            logger.warning(f"SAE checkpoint not found for layer {layer_idx}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        sae_config = checkpoint["config"]

        # Create SAE model
        sae = AdaptiveTopKSAE(
            d_model=sae_config["d_model"],
            d_hidden=sae_config["d_hidden"],
            target_l0_values=sae_config["target_l0_values"],
        ).to(device)

        sae.load_state_dict(checkpoint["model_state_dict"])
        sae.eval()

        sae_models[layer_idx] = sae

    logger.info(f"✓ Loaded {len(sae_models)} SAE models")
    return sae_models


def load_sas_vectors(sas_vectors_path):
    """Load pre-computed SAS vectors."""
    logger.info(f"Loading SAS vectors from {sas_vectors_path}")

    data = torch.load(sas_vectors_path, map_location="cpu")

    # Convert to numpy for easier manipulation
    sas_vectors = {}
    for layer_idx, vec_tensor in data.items():
        sas_vectors[layer_idx] = vec_tensor.numpy()

    logger.info(f"✓ Loaded SAS vectors for {len(sas_vectors)} layers")
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
        # Register hook on the layer's residual connection output
        # This captures the dense activations before the next layer
        handle = layer.register_forward_hook(
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

        return generated_sequences

    finally:
        # Always remove hooks
        remove_hooks(handles)
        logger.info("✓ Steering hooks removed")


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
    sae_dir = args.exp_dir / "sparse_steering" / "sae_models"
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

    # Generate for each steering strength
    for strength in args.steering_strengths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Steering strength λ = {strength}")
        logger.info(f"{'='*80}")

        # Create output directory for this steering strength
        strength_output_dir = (
            args.output_dir / f"lambda_{'+' if strength >= 0 else ''}{strength}"
        )

        sequences = generate_with_steering(
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
