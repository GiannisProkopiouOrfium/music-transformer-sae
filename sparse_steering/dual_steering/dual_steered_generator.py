#!/usr/bin/env python3
"""Dual-concept SAS Steered Generator.

Extends the single-concept SASSteeringHook to inject a **combined** pitch +
duration steering vector into the model during generation.

Architecture (Algorithm 2 — dual-concept variant):
  1. Encode:   f(a_L) = SAE.encode(a_L)
  2. Correct:  Δ = a_L − SAE.decode(f(a_L))
  3. Steer:    s_L = f(a_L) + combined_vector_L     ← λ already baked in
  4. Activate: s'_L = TopK(ReLU(s_L))
  5. Decode:   ã_L = SAE.decode(s'_L) + Δ

The combined_vector is produced by SparseVectorComposer *before* generation
starts, so the hook is as lean as the single-concept hook.

Usage:
    from sparse_vector_composer import SparseVectorComposer
    from dual_steered_generator import DualSASSteeringHook, register_dual_hooks

    composer = SparseVectorComposer(pitch_vecs, duration_vecs)
    combined = composer.compose(lambda_pitch=0.75, lambda_duration=1.0,
                                strategy="cross_concept_masking")

    handles = register_dual_hooks(model, sae_models, combined, layers_to_steer=[10])
    # ... generate ...
    remove_hooks(handles)
"""

import logging
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Ensure sparse_steering is on the path for sae_model imports
_SPARSE_DIR = pathlib.Path(__file__).parent.parent
if str(_SPARSE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPARSE_DIR))


class DualSASSteeringHook:
    """Forward hook that applies a pre-composed dual-concept SAS vector.

    Identical to single-concept SASSteeringHook except:
      • Accepts a *combined* vector instead of a single-concept vector + λ.
      • The combined vector already contains both concepts scaled by their λ.
    """

    def __init__(
        self,
        sae_models: Dict[int, nn.Module],
        combined_vectors: Dict[int, np.ndarray],
        layers_to_steer: Optional[List[int]] = None,
    ):
        """
        Args:
            sae_models:       {layer_idx: trained SparseAutoencoder}
            combined_vectors: {layer_idx: (4096,) np.ndarray} already-scaled
            layers_to_steer:  restrict steering to these layers (None → all)
        """
        self.sae_models = sae_models
        self.layers_to_steer = (
            set(layers_to_steer) if layers_to_steer is not None else None
        )

        # Convert to torch tensors (moved to device lazily)
        self.combined_torch = {
            layer_idx: torch.from_numpy(vec).float()
            for layer_idx, vec in combined_vectors.items()
        }

        self.device = None
        self._initialised = False

    def _ensure_device(self, activations: torch.Tensor):
        """Move vectors to the correct device on first call."""
        if not self._initialised:
            self.device = activations.device
            for k in self.combined_torch:
                self.combined_torch[k] = self.combined_torch[k].to(self.device)
            self._initialised = True

    def __call__(self, module, input, output, layer_idx: int):
        """Forward hook implementing Algorithm 2 (dual-concept)."""
        # Unpack tuple output from Attention
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        others = output[1:] if is_tuple else None

        # Gate: skip layers not targeted
        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.sae_models or layer_idx not in self.combined_torch:
            return output

        self._ensure_device(actual)

        sae = self.sae_models[layer_idx]
        v_combined = self.combined_torch[layer_idx]  # (4096,)

        a_l = actual  # (B, T, 512)
        B, T, D = a_l.shape
        a_flat = a_l.reshape(-1, D)  # (B*T, 512)

        # 1. Encode → sparse
        f_a = sae.encode(a_flat)  # (B*T, 4096)

        # 2. Correction Δ
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed  # (B*T, 512)

        # 3. Add combined steering vector (λ already included)
        s_l = f_a + v_combined.unsqueeze(0)  # (B*T, 4096)

        # 4. Re-sparsify
        s_l = torch.relu(s_l)
        s_l = sae.topk(s_l)  # (B*T, 4096)

        # 5. Decode + correct
        a_prime = sae.decode(s_l)  # (B*T, 512)
        a_steered = a_prime + delta

        a_steered = a_steered.reshape(B, T, D)

        return (a_steered,) + others if is_tuple else a_steered


# ════════════════════════════════════════════════════════════════════════════
# Hook registration / removal helpers (mirror single-concept API)
# ════════════════════════════════════════════════════════════════════════════


class ExpandedKDualSASSteeringHook:
    """Forward hook that temporarily increases TopK budget for dual steering.

    During dual-concept steering, two concepts compete for the same K slots
    in TopK re-sparsification.  This hook temporarily increases K by a
    configurable factor (default 1.5×) so that both concepts can retain
    enough active features.

    The SAE decoder is a simple linear map and is reasonably robust to
    moderate K increases.
    """

    def __init__(
        self,
        sae_models: Dict[int, nn.Module],
        combined_vectors: Dict[int, np.ndarray],
        layers_to_steer: Optional[List[int]] = None,
        k_multiplier: float = 1.5,
    ):
        self.sae_models = sae_models
        self.layers_to_steer = (
            set(layers_to_steer) if layers_to_steer is not None else None
        )
        self.k_multiplier = k_multiplier

        self.combined_torch = {
            layer_idx: torch.from_numpy(vec).float()
            for layer_idx, vec in combined_vectors.items()
        }
        self.device = None
        self._initialised = False

    def _ensure_device(self, activations: torch.Tensor):
        if not self._initialised:
            self.device = activations.device
            for k in self.combined_torch:
                self.combined_torch[k] = self.combined_torch[k].to(self.device)
            self._initialised = True

    def __call__(self, module, input, output, layer_idx: int):
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        others = output[1:] if is_tuple else None

        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.sae_models or layer_idx not in self.combined_torch:
            return output

        self._ensure_device(actual)

        sae = self.sae_models[layer_idx]
        v_combined = self.combined_torch[layer_idx]

        a_l = actual
        B, T, D = a_l.shape
        a_flat = a_l.reshape(-1, D)

        # 1. Encode
        f_a = sae.encode(a_flat)

        # 2. Correction
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        # 3. Steer
        s_l = f_a + v_combined.unsqueeze(0)

        # 4. Re-sparsify with EXPANDED K
        s_l = torch.relu(s_l)
        original_k = sae.topk.k
        expanded_k = min(int(original_k * self.k_multiplier), s_l.shape[-1])
        sae.topk.k = expanded_k
        try:
            s_l = sae.topk(s_l)
        finally:
            sae.topk.k = original_k  # always restore

        # 5. Decode + correct
        a_prime = sae.decode(s_l)
        a_steered = (a_prime + delta).reshape(B, T, D)

        return (a_steered,) + others if is_tuple else a_steered


class SequentialDualSASSteeringHook:
    """Forward hook that applies pitch and duration as two sequential Algorithm 2 passes.

    Pass 1:  a  → encode → add λ_p·v_pitch → TopK → decode + Δ₁ → a₁
    Pass 2:  a₁ → encode → add λ_d·v_dur   → TopK → decode + Δ₂ → a₂

    Each concept gets the full K budget independently (no competition).
    Trade-off: 2× compute cost per layer, and potential order dependence.
    """

    def __init__(
        self,
        sae_models: Dict[int, nn.Module],
        pitch_vectors: Dict[int, np.ndarray],
        duration_vectors: Dict[int, np.ndarray],
        layers_to_steer: Optional[List[int]] = None,
    ):
        self.sae_models = sae_models
        self.layers_to_steer = (
            set(layers_to_steer) if layers_to_steer is not None else None
        )

        self.pitch_torch = {
            idx: torch.from_numpy(v).float() for idx, v in pitch_vectors.items()
        }
        self.duration_torch = {
            idx: torch.from_numpy(v).float() for idx, v in duration_vectors.items()
        }
        self.device = None
        self._initialised = False

    def _ensure_device(self, activations: torch.Tensor):
        if not self._initialised:
            self.device = activations.device
            for k in self.pitch_torch:
                self.pitch_torch[k] = self.pitch_torch[k].to(self.device)
            for k in self.duration_torch:
                self.duration_torch[k] = self.duration_torch[k].to(self.device)
            self._initialised = True

    def _single_pass(self, sae, a_flat, v_concept):
        """One full Algorithm 2 pass."""
        f_a = sae.encode(a_flat)
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        s_l = f_a + v_concept.unsqueeze(0)
        s_l = torch.relu(s_l)
        s_l = sae.topk(s_l)

        a_prime = sae.decode(s_l)
        return a_prime + delta

    def __call__(self, module, input, output, layer_idx: int):
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        others = output[1:] if is_tuple else None

        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.sae_models:
            return output

        self._ensure_device(actual)
        sae = self.sae_models[layer_idx]

        a_l = actual
        B, T, D = a_l.shape
        a_flat = a_l.reshape(-1, D)

        # Pass 1: pitch steering
        if layer_idx in self.pitch_torch:
            a_flat = self._single_pass(sae, a_flat, self.pitch_torch[layer_idx])

        # Pass 2: duration steering (on already-pitch-steered activations)
        if layer_idx in self.duration_torch:
            a_flat = self._single_pass(sae, a_flat, self.duration_torch[layer_idx])

        a_steered = a_flat.reshape(B, T, D)
        return (a_steered,) + others if is_tuple else a_steered


class TopKBudgetAllocationHook:
    """Forward hook that guarantees each concept a minimum TopK budget.

    Instead of applying global TopK(K) after adding the combined vector
    (where one concept can dominate), this hook:

    1. Partitions the 4096 features into concept groups based on the
       original SAS vectors:
         - pitch_only:   non-zero in v_pitch only
         - duration_only: non-zero in v_duration only
         - shared:       non-zero in both
         - neutral:      zero in both (the vast majority)

    2. After steering, applies TopK within each concept's features
       separately, then merges:
         - K_pitch slots reserved for pitch_only + shared features
         - K_dur slots reserved for duration_only + shared features
         - K_neutral slots for the remaining neutral features
         - Total budget = K_pitch + K_dur + K_neutral

    This ensures neither concept is starved.  The budget split defaults to:
         K_pitch = K_dur = K // 3,  K_neutral = K - 2*(K//3)
    """

    def __init__(
        self,
        sae_models: Dict[int, nn.Module],
        combined_vectors: Dict[int, np.ndarray],
        pitch_vectors_raw: Dict[int, np.ndarray],
        duration_vectors_raw: Dict[int, np.ndarray],
        layers_to_steer: Optional[List[int]] = None,
        budget_split: str = "equal",  # "equal" or "proportional"
    ):
        self.sae_models = sae_models
        self.layers_to_steer = (
            set(layers_to_steer) if layers_to_steer is not None else None
        )
        self.budget_split = budget_split

        self.combined_torch = {
            idx: torch.from_numpy(v).float() for idx, v in combined_vectors.items()
        }

        # Pre-compute feature group masks per layer
        self.pitch_mask = {}  # features in pitch only
        self.duration_mask = {}  # features in duration only
        self.shared_mask = {}  # features in both
        self.neutral_mask = {}  # features in neither

        for layer_idx in combined_vectors:
            vp = pitch_vectors_raw.get(layer_idx)
            vd = duration_vectors_raw.get(layer_idx)
            if vp is None or vd is None:
                continue
            nz_p = vp != 0
            nz_d = vd != 0
            shared = nz_p & nz_d
            p_only = nz_p & ~nz_d
            d_only = nz_d & ~nz_p
            neutral = ~nz_p & ~nz_d

            self.pitch_mask[layer_idx] = torch.from_numpy(p_only | shared).bool()
            self.duration_mask[layer_idx] = torch.from_numpy(d_only | shared).bool()
            self.neutral_mask[layer_idx] = torch.from_numpy(neutral).bool()

        self.device = None
        self._initialised = False

    def _ensure_device(self, activations: torch.Tensor):
        if not self._initialised:
            self.device = activations.device
            for k in self.combined_torch:
                self.combined_torch[k] = self.combined_torch[k].to(self.device)
            for k in self.pitch_mask:
                self.pitch_mask[k] = self.pitch_mask[k].to(self.device)
                self.duration_mask[k] = self.duration_mask[k].to(self.device)
                self.neutral_mask[k] = self.neutral_mask[k].to(self.device)
            self._initialised = True

    def _topk_within_mask(self, x: torch.Tensor, mask: torch.Tensor, k: int):
        """Apply TopK only on features where mask=True, zero the rest."""
        # x: (B*T, 4096), mask: (4096,) bool
        if k <= 0 or not mask.any():
            return torch.zeros_like(x)
        masked = x * mask.unsqueeze(0)  # zero out non-concept features
        n_active = int(mask.sum().item())
        actual_k = min(k, n_active)
        topk_vals, topk_idx = torch.topk(masked, actual_k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_idx, topk_vals)
        return result

    def __call__(self, module, input, output, layer_idx: int):
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        others = output[1:] if is_tuple else None

        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.sae_models or layer_idx not in self.combined_torch:
            return output
        if layer_idx not in self.pitch_mask:
            return output

        self._ensure_device(actual)

        sae = self.sae_models[layer_idx]
        v_combined = self.combined_torch[layer_idx]
        K = sae.topk.k  # original K budget

        a_l = actual
        B, T, D = a_l.shape
        a_flat = a_l.reshape(-1, D)

        # 1. Encode
        f_a = sae.encode(a_flat)

        # 2. Correction
        reconstructed = sae.decode(f_a)
        delta = a_flat - reconstructed

        # 3. Steer
        s_l = f_a + v_combined.unsqueeze(0)
        s_l = torch.relu(s_l)

        # 4. Budget-allocated TopK
        # Split K into: pitch_budget + duration_budget + neutral_budget = K
        K_concept = K // 3
        K_neutral = K - 2 * K_concept

        s_pitch = self._topk_within_mask(s_l, self.pitch_mask[layer_idx], K_concept)
        s_dur = self._topk_within_mask(s_l, self.duration_mask[layer_idx], K_concept)
        s_neutral = self._topk_within_mask(s_l, self.neutral_mask[layer_idx], K_neutral)

        # Merge (masks are non-overlapping for pitch_only/dur_only/neutral,
        # but shared features can appear in both pitch and duration masks.
        # Take the max activation to avoid double-counting.)
        s_merged = torch.max(torch.max(s_pitch, s_dur), s_neutral)

        # 5. Decode + correct
        a_prime = sae.decode(s_merged)
        a_steered = (a_prime + delta).reshape(B, T, D)

        return (a_steered,) + others if is_tuple else a_steered


class DenseSASSteeringHook:
    """Forward hook that steers in dense 512-dim space using SAE-projected SAS vectors.

    Instead of the full encode → steer → TopK → decode cycle (Algorithm 2),
    this projects the SAS vectors from 4096-dim sparse space to 512-dim dense
    space via the SAE decoder, then adds them directly to activations:

        v_dense = (W_dec @ v_SAS) * input_std
        h' = h + v_dense

    Benefits:
      • No reconstruction error (no encode/decode round-trip)
      • No TopK information loss (no feature competition)
      • SAS-quality directions (derived from interpretable sparse features)
      • Same computational cost as difference-in-means steering
    """

    def __init__(
        self,
        dense_vectors: Dict[int, np.ndarray],
        layers_to_steer: Optional[List[int]] = None,
    ):
        """
        Args:
            dense_vectors:   {layer_idx: (512,) np.ndarray} projected vectors
            layers_to_steer: restrict steering to these layers
        """
        self.layers_to_steer = (
            set(layers_to_steer) if layers_to_steer is not None else None
        )
        self.dense_torch = {
            idx: torch.from_numpy(v).float() for idx, v in dense_vectors.items()
        }
        self.device = None
        self._initialised = False

    def _ensure_device(self, activations: torch.Tensor):
        if not self._initialised:
            self.device = activations.device
            for k in self.dense_torch:
                self.dense_torch[k] = self.dense_torch[k].to(self.device)
            self._initialised = True

    def __call__(self, module, input, output, layer_idx: int):
        is_tuple = isinstance(output, tuple)
        actual = output[0] if is_tuple else output
        others = output[1:] if is_tuple else None

        if self.layers_to_steer is not None and layer_idx not in self.layers_to_steer:
            return output
        if layer_idx not in self.dense_torch:
            return output

        self._ensure_device(actual)
        v_dense = self.dense_torch[layer_idx]  # (512,)

        # Direct dense-space addition — no encode/decode/TopK cycle
        a_steered = actual + v_dense.unsqueeze(0).unsqueeze(0)  # (1, 1, 512) broadcast

        return (a_steered,) + others if is_tuple else a_steered


def project_sas_to_dense(
    combined_vectors: Dict[int, np.ndarray],
    sae_models: Dict[int, nn.Module],
) -> Dict[int, np.ndarray]:
    """Project SAS vectors from 4096-dim sparse to 512-dim dense via SAE decoder.

    The SAE decode path is:
        reconstruction_norm = sparse @ W_encoder + decoder_bias
        reconstruction = reconstruction_norm * input_std + input_mean

    So the *directional* effect of adding v_SAS in sparse space is:
        Δ_output = (v_SAS @ W_encoder) * input_std

    We compute this projection without the bias/mean terms (which are constant
    offsets, not directional).

    Args:
        combined_vectors: {layer_idx: (4096,)} combined SAS vector (λ already baked in)
        sae_models:       {layer_idx: trained SparseAutoencoder}

    Returns:
        {layer_idx: (512,) np.ndarray} projected dense vectors
    """
    dense_vectors = {}
    for layer_idx, v_sparse in combined_vectors.items():
        if layer_idx not in sae_models:
            continue
        sae = sae_models[layer_idx]
        v_t = torch.from_numpy(v_sparse).float()

        with torch.no_grad():
            # Move to same device as SAE weights
            device = next(sae.parameters()).device
            v_t = v_t.to(device)

            # Get effective decoder weight matrix: sparse(4096) → dense(512)
            if sae.tied_weights:
                w_dec = sae.encoder.weight  # (4096, 512)
            else:
                w_dec = sae.decoder.weight.t()  # (512, out) → t() → (out, 512)

            # Project: v_sparse (4096,) @ w_dec (4096, 512) → (512,)
            v_dense = v_t @ w_dec

            # Account for denormalization: decode output is
            # reconstruction_norm * input_std + input_mean
            # The directional component is scaled by input_std
            if sae.normalize_input:
                v_dense = v_dense * sae.input_std

        dense_vectors[layer_idx] = v_dense.cpu().numpy()
        norm_sparse = float(np.linalg.norm(v_sparse))
        norm_dense = float(np.linalg.norm(dense_vectors[layer_idx]))
        logger.info(
            f"L{layer_idx}: projected 4096→512  "
            f"||v_sparse||={norm_sparse:.3f} → ||v_dense||={norm_dense:.3f}"
        )

    return dense_vectors


def register_dual_hooks(
    model,
    sae_models: Dict[int, nn.Module],
    combined_vectors: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
) -> List:
    """Register forward hooks for dual-concept SAS steering.

    Args:
        model:            MMT model (MusicXTransformer)
        sae_models:       {layer: SparseAutoencoder}
        combined_vectors: {layer: (4096,)} from SparseVectorComposer.compose()
        layers_to_steer:  restrict to these layers (None → all)

    Returns:
        List of hook handles.  Call remove_hooks(handles) to clean up.
    """
    hook = DualSASSteeringHook(sae_models, combined_vectors, layers_to_steer)

    handles = []
    layers = model.decoder.net.attn_layers.layers

    for layer_idx, layer in enumerate(layers):
        # Must hook the same module used during SAE training:
        # layer[1] = the Attention module inside each ModuleList
        if isinstance(layer, nn.ModuleList) and len(layer) > 1:
            target_module = layer[1]
        else:
            target_module = layer

        handle = target_module.register_forward_hook(
            lambda module, inp, out, idx=layer_idx: hook(module, inp, out, idx)
        )
        handles.append(handle)

    logger.info(f"Registered {len(handles)} dual-steering hooks")
    return handles


def remove_hooks(handles: List):
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    logger.info(f"Removed {len(handles)} hooks")


def _register_hook_on_layers(model, hook_callable) -> List:
    """Shared helper: attach a hook_callable to every attention module."""
    handles = []
    layers = model.decoder.net.attn_layers.layers
    for layer_idx, layer in enumerate(layers):
        if isinstance(layer, nn.ModuleList) and len(layer) > 1:
            target_module = layer[1]
        else:
            target_module = layer
        handle = target_module.register_forward_hook(
            lambda module, inp, out, idx=layer_idx: hook_callable(module, inp, out, idx)
        )
        handles.append(handle)
    return handles


def register_expanded_k_hooks(
    model,
    sae_models: Dict[int, nn.Module],
    combined_vectors: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
    k_multiplier: float = 1.5,
) -> List:
    """Register hooks that expand TopK budget during dual steering.

    Args:
        model:           MMT model
        sae_models:      {layer: SparseAutoencoder}
        combined_vectors: {layer: (4096,)} composed vector (direct addition)
        layers_to_steer: layers to steer
        k_multiplier:    factor to multiply K by (default 1.5)

    Returns:
        List of hook handles
    """
    hook = ExpandedKDualSASSteeringHook(
        sae_models, combined_vectors, layers_to_steer, k_multiplier
    )
    handles = _register_hook_on_layers(model, hook)
    logger.info(
        f"Registered {len(handles)} expanded-K dual-steering hooks "
        f"(K×{k_multiplier:.1f})"
    )
    return handles


def register_sequential_hooks(
    model,
    sae_models: Dict[int, nn.Module],
    pitch_vectors: Dict[int, np.ndarray],
    duration_vectors: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
) -> List:
    """Register hooks that apply pitch and duration as two sequential passes.

    Args:
        model:            MMT model
        sae_models:       {layer: SparseAutoencoder}
        pitch_vectors:    {layer: (4096,)} scaled pitch vector
        duration_vectors: {layer: (4096,)} scaled duration vector
        layers_to_steer:  layers to steer

    Returns:
        List of hook handles
    """
    hook = SequentialDualSASSteeringHook(
        sae_models, pitch_vectors, duration_vectors, layers_to_steer
    )
    handles = _register_hook_on_layers(model, hook)
    logger.info(f"Registered {len(handles)} sequential dual-steering hooks")
    return handles


def register_budget_allocation_hooks(
    model,
    sae_models: Dict[int, nn.Module],
    combined_vectors: Dict[int, np.ndarray],
    pitch_vectors_raw: Dict[int, np.ndarray],
    duration_vectors_raw: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
) -> List:
    """Register hooks with TopK budget allocation per concept.

    Args:
        model:                MMT model
        sae_models:           {layer: SparseAutoencoder}
        combined_vectors:     {layer: (4096,)} composed vector
        pitch_vectors_raw:    {layer: (4096,)} raw pitch SAS vectors (for masks)
        duration_vectors_raw: {layer: (4096,)} raw duration SAS vectors (for masks)
        layers_to_steer:      layers to steer

    Returns:
        List of hook handles
    """
    hook = TopKBudgetAllocationHook(
        sae_models,
        combined_vectors,
        pitch_vectors_raw,
        duration_vectors_raw,
        layers_to_steer,
    )
    handles = _register_hook_on_layers(model, hook)
    logger.info(f"Registered {len(handles)} budget-allocation dual-steering hooks")
    return handles


def register_dense_sas_hooks(
    model,
    sae_models: Dict[int, nn.Module],
    combined_vectors: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
) -> List:
    """Register hooks for SAS-informed dense-space steering.

    Projects combined SAS vectors from 4096-dim sparse to 512-dim dense
    using the SAE decoder, then steers via direct addition in dense space.
    No encode/decode/TopK cycle — minimal quality degradation.

    Args:
        model:            MMT model
        sae_models:       {layer: SparseAutoencoder}
        combined_vectors: {layer: (4096,)} composed SAS vector (λ baked in)
        layers_to_steer:  layers to steer

    Returns:
        List of hook handles
    """
    # Project sparse → dense
    dense_vectors = project_sas_to_dense(combined_vectors, sae_models)

    hook = DenseSASSteeringHook(dense_vectors, layers_to_steer)
    handles = _register_hook_on_layers(model, hook)
    logger.info(f"Registered {len(handles)} dense-SAS steering hooks")
    return handles


# ════════════════════════════════════════════════════════════════════════════
# High-level generate function
# ════════════════════════════════════════════════════════════════════════════


def generate_with_dual_steering(
    model,
    encoding: dict,
    sae_models: Dict[int, nn.Module],
    combined_vectors: Dict[int, np.ndarray],
    layers_to_steer: Optional[List[int]] = None,
    n_samples: int = 5,
    max_seq_len: int = 512,
    primer: Optional[torch.Tensor] = None,
) -> List[np.ndarray]:
    """Generate MIDI token sequences with dual-concept SAS steering.

    Args:
        model:            Loaded MMT model in eval mode
        encoding:         Encoding dict (from representation.load_encoding)
        sae_models:       Trained SAE models per layer
        combined_vectors: Composed vectors from SparseVectorComposer
        layers_to_steer:  Layers to apply steering (default [10])
        n_samples:        Number of sequences to generate
        max_seq_len:      Target sequence length
        primer:           Optional conditioning prefix (B, T, 6) tensor

    Returns:
        List of np.ndarray token sequences, each (seq_len, 6)
    """
    if layers_to_steer is None:
        layers_to_steer = [10]

    eos = encoding["type_code_map"]["end-of-song"]
    results = []

    handles = register_dual_hooks(model, sae_models, combined_vectors, layers_to_steer)

    try:
        for i in range(n_samples):
            with torch.no_grad():
                if primer is not None:
                    start = primer.clone()
                else:
                    start = torch.zeros(
                        1,
                        1,
                        6,
                        dtype=torch.long,
                        device=next(model.parameters()).device,
                    )

                generated = model.generate(
                    start,
                    max_seq_len,
                    eos_token=eos,
                )

            tokens = generated[0].cpu().numpy()
            results.append(tokens)
            logger.debug(f"Sample {i + 1}/{n_samples}: {tokens.shape[0]} tokens")

    finally:
        remove_hooks(handles)

    return results
