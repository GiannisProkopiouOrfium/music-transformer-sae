"""
Steered music generation using Sparse Activation Steering (SAS).

Implements Algorithm 2 from the SAS paper:
- Encodes activations to sparse space via SAE
- Adds scaled SAS steering vector: s = f(a) + λ·v_SAS
- Applies activation function and decodes back
- Adds correction term to preserve reconstruction quality

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
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baseline import representation
from mmt import MusicXTransformer

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

    # Load encoding
    encoding = representation.load_encoding(encoding_path)

    # Load training args
    import json

    with open(train_args_path) as f:
        train_args = json.load(f)

    # Create model
    model = MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["n_layers"],
        heads=train_args["n_heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["emb_dropout"],
        attn_dropout=train_args["attn_dropout"],
        ff_dropout=train_args["ff_dropout"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    logger.info(f"✓ Model loaded ({train_args['n_layers']} layers)")
    return model, encoding, train_args


def load_sae_models(sae_dir, device):
    """Load trained SAE models for all layers."""
    logger.info(f"Loading SAE models from {sae_dir}")

    from sae.train_adaptive_sae import AdaptiveTopKSAE

    sae_models = {}
    for layer_idx in range(12):  # Assuming 12 layers
        checkpoint_path = sae_dir / f"layer{layer_idx}_sae.pt"
        if not checkpoint_path.exists():
            logger.warning(f"SAE checkpoint not found for layer {layer_idx}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        # Create SAE model
        sae = AdaptiveTopKSAE(
            d_model=config["d_model"],
            d_hidden=config["d_hidden"],
            target_l0_values=config["target_l0_values"],
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
    max_seq_len=1024,
    temperature=1.0,
    device="cuda",
):
    """
    Generate music samples with SAS steering.

    Args:
        model: MMT model
        encoding: Encoding object
        sae_models: Dictionary of SAE models
        sas_vectors: Dictionary of SAS vectors
        concept: Concept name
        steering_strength: λ parameter
        n_samples: Number of samples to generate
        layers_to_steer: Specific layers to steer
        max_seq_len: Maximum sequence length
        temperature: Sampling temperature
        device: Device

    Returns:
        List of generated note sequences
    """
    logger.info(f"Generating {n_samples} samples with SAS steering")
    logger.info(f"  Concept: {concept}")
    logger.info(f"  Steering strength λ: {steering_strength}")
    logger.info(f"  Layers: {layers_to_steer if layers_to_steer else 'all'}")

    # Register hooks
    handles = register_steering_hooks(
        model, sae_models, sas_vectors, concept, steering_strength, layers_to_steer
    )

    try:
        generated_sequences = []

        for i in range(n_samples):
            logger.info(f"  Generating sample {i+1}/{n_samples}...")

            # Start with BOS token
            start_seq = torch.tensor(
                [[encoding["type_code_map"]["start-of-song"]]], device=device
            )

            # Generate
            with torch.no_grad():
                output = model.generate(
                    start_seq,
                    max_seq_len,
                    temperature=temperature,
                    filter_thres=0.9,  # Top-p sampling
                )

            # Convert to note sequence
            notes = representation.decode_notes(output[0].cpu().numpy(), encoding)
            generated_sequences.append(notes)

        return generated_sequences

    finally:
        # Always remove hooks
        remove_hooks(handles)
        logger.info("✓ Steering hooks removed")


def save_generations(sequences, output_dir, concept, steering_strength):
    """Save generated sequences to MIDI files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import muspy

    for i, notes in enumerate(sequences):
        # Convert to MusPy Music object
        music = representation.notes_to_music(notes)

        # Save to MIDI
        strength_str = (
            f"{'pos' if steering_strength >= 0 else 'neg'}{abs(steering_strength):.1f}"
        )
        filename = f"{concept}_lambda_{strength_str}_sample{i+1}.mid"
        filepath = output_dir / filename

        muspy.write_midi(filepath, music)
        logger.info(f"  Saved: {filename}")

    logger.info(f"✓ Saved {len(sequences)} generations to {output_dir}")


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
        "--max_seq_len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("exp/sod"), help="Experiment directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: exp_dir/sparse_steering/generations_sas)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")

    args = parser.parse_args()

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    checkpoint_path = args.exp_dir / "checkpoints" / "best_model.pt"
    train_args_path = args.exp_dir / "train-args.json"
    encoding_path = args.exp_dir / "notes" / "encoding.json"
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
        sys.exit(1)
    if not sas_vectors_path.exists():
        logger.error(f"SAS vectors not found: {sas_vectors_path}")
        logger.error("Run compute_sas_vectors.py first")
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
            temperature=args.temperature,
            device=device,
        )

        # Save generations
        save_generations(sequences, args.output_dir, args.concept, strength)

    logger.info(f"\n{'='*80}")
    logger.info("✓ Generation complete!")
    logger.info(f"✓ Outputs saved to: {args.output_dir}")
    logger.info(f"{'='*80}")
    logger.info("\nNext steps:")
    logger.info("1. Listen to generated MIDI files")
    logger.info("2. Analyze steering effects with deterministic metrics")
    logger.info("3. Compare SAS vs DiffMean steering")


if __name__ == "__main__":
    main()
