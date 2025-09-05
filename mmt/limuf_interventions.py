"""
LiMuF-based interventions for controlling music generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class LiMuFInterventionWrapper(nn.Module):
    """
    Wrapper around MusicXTransformer that enables LiMuF interventions during generation.

    This implements inference-time interventions similar to the LiReFs paper,
    but for musical concepts discovered by SAEs.
    """

    def __init__(
        self, model, limufs: Dict[int, torch.Tensor], intervention_layer: int = 3
    ):
        super().__init__()
        self.model = model
        self.limufs = {k: v.to(model.device) for k, v in limufs.items()}
        self.intervention_layer = intervention_layer
        self.device = model.device

        # Store original hooks for restoration
        self._original_hooks = []
        self._intervention_active = False
        self._intervention_config = {}

        print(f"🎛️  LiMuF Intervention Wrapper initialized")
        print(f"   Available features: {sorted(self.limufs.keys())}")
        print(f"   Target intervention layer: {intervention_layer}")

    def set_intervention(
        self,
        feature_id: int,
        strength: float = 1.0,
        target_tokens: Optional[List[int]] = None,
    ):
        """
        Configure intervention for a specific musical feature.

        Args:
            feature_id: ID of the musical feature to intervene on
            strength: Intervention strength (positive to promote, negative to suppress)
            target_tokens: Specific token positions to intervene on (None = all tokens)
        """
        if feature_id not in self.limufs:
            available_features = sorted(self.limufs.keys())
            raise ValueError(
                f"Feature {feature_id} not found in loaded LiMuFs. Available: {available_features}"
            )

        self._intervention_config = {
            "feature_id": feature_id,
            "strength": strength,
            "target_tokens": target_tokens,
            "direction": self.limufs[feature_id],
        }

        print(f"🎛️  Intervention configured:")
        print(f"   Feature ID: {feature_id}")
        print(
            f"   Strength: {strength:+.2f} ({'promote' if strength > 0 else 'suppress'})"
        )
        if target_tokens:
            print(f"   Target tokens: {target_tokens}")
        else:
            print(f"   Target tokens: all positions")

    def _apply_intervention_hook(self, module, input, output):
        """Hook function to apply LiMuF intervention at specified layer."""
        if not self._intervention_active or not self._intervention_config:
            return output

        config = self._intervention_config
        direction = config["direction"]
        strength = config["strength"]

        # output shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = output.shape

        # Create intervention vector
        intervention_vector = strength * direction.to(output.device)

        # Apply to all tokens or specific positions
        if config["target_tokens"] is not None:
            # Apply only to specific token positions
            for token_pos in config["target_tokens"]:
                if token_pos < seq_len:
                    output[:, token_pos, :] += intervention_vector
        else:
            # Apply to all positions (broadcast across batch and sequence)
            output += intervention_vector.unsqueeze(0).unsqueeze(0)

        return output

    def enable_interventions(self):
        """Enable LiMuF interventions by registering hooks."""
        if self._intervention_active:
            print("⚠️  Interventions already active")
            return

        # Get the target layer for intervention
        target_layer = self._get_intervention_layer()

        # Register forward hook
        hook = target_layer.register_forward_hook(self._apply_intervention_hook)
        self._original_hooks.append(hook)
        self._intervention_active = True

        print(f"✅ LiMuF interventions enabled at layer {self.intervention_layer}")

    def disable_interventions(self):
        """Disable LiMuF interventions by removing hooks."""
        for hook in self._original_hooks:
            hook.remove()
        self._original_hooks = []
        self._intervention_active = False

        print("❌ LiMuF interventions disabled")

    def _get_intervention_layer(self):
        """Get the specific layer where interventions should be applied."""
        # Navigate to the correct layer in MusicXTransformer
        # This assumes the model structure from your codebase

        # For MusicXTransformer -> MusicAutoregressiveWrapper -> MusicTransformerWrapper
        decoder = self.model.decoder if hasattr(self.model, "decoder") else self.model
        net = decoder.net if hasattr(decoder, "net") else decoder

        # Get the transformer layers
        if hasattr(net, "attn_layers"):
            layers = net.attn_layers.layers
            if self.intervention_layer * 2 < len(
                layers
            ):  # Each transformer layer has 2 sublayers
                # Get the attention layer (before feed-forward)
                # layers is structured as: [attn, ff, attn, ff, ...]
                layer_idx = self.intervention_layer * 2
                return layers[layer_idx]  # Attention layer
            else:
                raise ValueError(
                    f"Layer {self.intervention_layer} not found. Model has {len(layers)//2} transformer layers"
                )

        raise ValueError(f"Could not find attn_layers in model structure")

    @torch.no_grad()
    def generate_with_intervention(
        self,
        start_tokens,
        seq_len: int,
        feature_id: int,
        strength: float = 1.0,
        target_tokens: Optional[List[int]] = None,
        **generation_kwargs,
    ):
        """
        Generate music with LiMuF intervention.

        Args:
            start_tokens: Initial tokens for generation
            seq_len: Length of sequence to generate
            feature_id: Musical feature to intervene on
            strength: Intervention strength (positive to promote, negative to suppress)
            target_tokens: Specific positions to intervene on
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Generated sequence with intervention applied
        """

        print(f"🎼 Generating with LiMuF intervention...")
        print(f"   Feature: {feature_id}, Strength: {strength:+.2f}, Length: {seq_len}")

        # Configure intervention
        self.set_intervention(feature_id, strength, target_tokens)

        # Enable interventions
        self.enable_interventions()

        try:
            # Generate with intervention
            generated = self.model.generate(start_tokens, seq_len, **generation_kwargs)

            print(f"✅ Generation complete with intervention")
            return generated

        finally:
            # Always disable interventions after generation
            self.disable_interventions()

    @torch.no_grad()
    def generate_comparison(
        self,
        start_tokens,
        seq_len: int,
        feature_id: int,
        strengths: List[float] = [-2.0, 0.0, 2.0],
        **generation_kwargs,
    ) -> Dict[float, torch.Tensor]:
        """
        Generate multiple sequences with different intervention strengths for comparison.

        Args:
            start_tokens: Initial tokens for generation
            seq_len: Length of sequence to generate
            feature_id: Musical feature to intervene on
            strengths: List of intervention strengths to test
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Dictionary mapping strength to generated sequence
        """
        results = {}

        print(f"🎼 Generating comparison with feature {feature_id}")
        print(f"   Testing strengths: {strengths}")

        for strength in strengths:
            print(f"\n   🎛️  Generating with strength {strength:+.1f}")

            if strength == 0.0:
                # Baseline generation (no intervention)
                generated = self.model.generate(
                    start_tokens, seq_len, **generation_kwargs
                )
            else:
                # Generate with intervention
                generated = self.generate_with_intervention(
                    start_tokens=start_tokens,
                    seq_len=seq_len,
                    feature_id=feature_id,
                    strength=strength,
                    **generation_kwargs,
                )

            results[strength] = generated

        print(f"✅ Comparison generation complete")
        return results

    def get_available_features(self) -> List[int]:
        """Get list of available LiMuF feature IDs."""
        return sorted(self.limufs.keys())

    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)


def load_model_with_limuf_intervention(
    model_path: str, limuf_path: str, device: str = "cuda", intervention_layer: int = 3
) -> LiMuFInterventionWrapper:
    """
    Load a trained music model and wrap it with LiMuF intervention capabilities.

    Args:
        model_path: Path to trained MusicXTransformer model
        limuf_path: Path to extracted LiMuFs directory
        device: Device to load model on
        intervention_layer: Layer to apply interventions at

    Returns:
        LiMuF intervention wrapper
    """
    # Import here to avoid circular imports
    from mmt.limuf_extractor import LiMuFExtractor

    print(f"🎼 Loading model with LiMuF intervention capabilities...")

    # Load LiMuFs
    print(f"📁 Loading LiMuFs from {limuf_path}")
    limufs, metadata = LiMuFExtractor.load_limufs(limuf_path)

    # Load the music model (you'll need to implement this based on your model loading)
    print(f"🎵 Loading music model from {model_path}")
    model = _load_music_model(model_path, device)

    # Create intervention wrapper
    intervention_model = LiMuFInterventionWrapper(
        model=model, limufs=limufs, intervention_layer=intervention_layer
    )

    print(f"✅ Model loaded with LiMuF intervention capabilities")
    return intervention_model


def _load_music_model(model_path: str, device: str = "cuda"):
    """
    Load the pre-trained Multi-track Music Transformer.

    This is a placeholder function - adapt to your actual model loading logic.
    """
    # Import your model loading utilities with full paths
    import sys
    from pathlib import Path

    # Add the mmt directory to sys.path if needed
    mmt_dir = Path(__file__).parent.parent
    if str(mmt_dir) not in sys.path:
        sys.path.insert(0, str(mmt_dir))

    # Now import with proper paths
    from mmt import music_x_transformers
    from mmt import representation
    from mmt import utils
    import pathlib

    model_path = pathlib.Path(model_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load configuration (adapt this to your actual setup)
    config_path = model_path.parent / "train-args.json"
    if config_path.exists():
        train_args = utils.load_json(config_path)
    else:
        # Use default config or extract from checkpoint
        train_args = checkpoint.get(
            "train_args",
            {
                "dim": 512,
                "layers": 6,
                "heads": 8,
                "max_seq_len": 1024,
                "max_beat": 256,
                "dropout": 0.1,
                "rel_pos_emb": True,
                "abs_pos_emb": True,
            },
        )  # Load encoding
    # encoding_path = model_path.parent.parent / "processed" / "notes" / "encoding.json"
    encoding_path = "data/sod/processed/notes/encoding.json"
    encoding = representation.load_encoding(encoding_path)
    if encoding is None:
        encoding = checkpoint.get("encoding")
        if encoding is None:
            raise ValueError(f"Cannot find encoding file at {encoding_path}")

    # Create model
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args.get("max_beat", 256),
        rotary_pos_emb=train_args.get("rel_pos_emb", True),
        use_abs_pos_emb=train_args.get("abs_pos_emb", True),
        emb_dropout=train_args.get("dropout", 0.1),
        attn_dropout=train_args.get("dropout", 0.1),
        ff_dropout=train_args.get("dropout", 0.1),
    ).to(device)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"✅ Music model loaded: {train_args['dim']}d, {train_args['layers']} layers")
    return model
