#!/usr/bin/env python3
"""Sparse Vector Composer — combine pitch and duration SAS vectors.

Implements four composition strategies for dual-concept steering in
sparse (4096-dim) space:

Strategy 1 — Direct Addition
  s = f(a) + λ_p · v_pitch + λ_d · v_duration
  Simplest. Works well when SAE features are disjoint for the two concepts.

Strategy 2 — Cross-Concept Masking (unique to sparse space)
  Identify shared non-zero indices. Assign each shared feature to the
  concept with higher |v[i]|.  Zero it in the other vector.
  Then combine as direct addition.

Strategy 3 — Gram-Schmidt: Orthogonalise Duration w.r.t. Pitch
  v_dur⊥ = v_dur − proj(v_dur → v_pitch)
  Re-sparsify v_dur⊥ via TopK to stay on the SAE manifold.

Strategy 4 — Gram-Schmidt: Orthogonalise Pitch w.r.t. Duration
  Symmetric to Strategy 3 but preserves duration and modifies pitch.

Strategy 5 — Cross-Concept SAS (extend Algorithm 1 Step 5 across concepts)
  Zero out all features that are non-zero in BOTH v_pitch and v_duration.
  This extends the SAS paper's within-concept shared removal to the
  cross-concept case.  Result: perfectly disjoint supports.
  Then combine via direct addition.

Strategy 6 — Expanded K (handled at the hook level, not here)
  Uses direct addition for composition, but the *hook* temporarily
  increases K during TopK re-sparsification to give both concepts
  enough budget.

Strategy 7 — Sequential (handled at the hook level, not here)
  Two separate Algorithm 2 passes: first pitch, then duration.
  Each concept gets the full K budget independently.

Usage:
    from sparse_vector_composer import SparseVectorComposer

    composer = SparseVectorComposer(pitch_vectors, duration_vectors)
    combined = composer.compose(
        lambda_pitch=0.75,
        lambda_duration=1.0,
        strategy="cross_concept_masking",
    )
    # combined: Dict[int, np.ndarray]  layer → (4096,) composed vector
"""

import logging
from typing import Dict, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

CompositionStrategy = Literal[
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
    "sas_dense",
]

ALL_STRATEGIES = [
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
    "sas_dense",
]

# Strategies that need special hook handling (not just a combined vector)
HOOK_SPECIAL_STRATEGIES = {
    "expanded_k",
    "expanded_k_2x",
    "sequential",
    "topk_budget",
    "opposite_sign_masking_ek2",
    "cross_concept_masking_ek2",
    "sas_dense",
}


class SparseVectorComposer:
    """Combine pitch and duration SAS vectors using various strategies.

    All operations stay in numpy (4096-dim sparse space).  The result is
    a *per-layer* combined SAS vector plus a pair of per-concept lambda
    values that are baked-in to the combined vector.

    The caller (DualSteeringHook) then adds the combined vector in the
    same way as single-concept steering: s = f(a) + combined.
    """

    def __init__(
        self,
        pitch_vectors: Dict[int, np.ndarray],
        duration_vectors: Dict[int, np.ndarray],
    ):
        """
        Args:
            pitch_vectors:    {layer_idx: (4096,) np.ndarray}
            duration_vectors: {layer_idx: (4096,) np.ndarray}
        """
        self.pitch_vectors = pitch_vectors
        self.duration_vectors = duration_vectors

        # Verify shared layers
        self.layers = sorted(set(pitch_vectors.keys()) & set(duration_vectors.keys()))
        if not self.layers:
            raise ValueError("No common layers between pitch and duration vectors")

        # Pre-compute per-layer overlap stats (used by cross_concept_masking)
        self._overlap_cache: Dict[int, dict] = {}
        for layer_idx in self.layers:
            vp = pitch_vectors[layer_idx]
            vd = duration_vectors[layer_idx]
            nz_p = vp != 0
            nz_d = vd != 0
            shared = nz_p & nz_d
            self._overlap_cache[layer_idx] = {
                "shared_mask": shared,
                "n_pitch": int(nz_p.sum()),
                "n_duration": int(nz_d.sum()),
                "n_shared": int(shared.sum()),
            }

        # Log summary
        total_shared = sum(v["n_shared"] for v in self._overlap_cache.values())
        logger.info(
            f"SparseVectorComposer: {len(self.layers)} layers, "
            f"total shared features={total_shared}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────

    def compose(
        self,
        lambda_pitch: float,
        lambda_duration: float,
        strategy: CompositionStrategy = "direct",
    ) -> Dict[int, np.ndarray]:
        """Compose scaled pitch + duration SAS vectors.

        Returns a dict {layer_idx: combined_vector} where combined_vector
        already includes λ scaling.  The hook should add this with
        strength=1.0.
        """
        if strategy == "direct":
            return self._direct(lambda_pitch, lambda_duration)
        elif strategy == "cross_concept_masking":
            return self._cross_concept_masking(lambda_pitch, lambda_duration)
        elif strategy == "gram_schmidt_pitch":
            return self._gram_schmidt_pitch(lambda_pitch, lambda_duration)
        elif strategy == "gram_schmidt_duration":
            return self._gram_schmidt_duration(lambda_pitch, lambda_duration)
        elif strategy == "cross_concept_sas":
            return self._cross_concept_sas(lambda_pitch, lambda_duration)
        elif strategy == "opposite_sign_masking":
            return self._opposite_sign_masking(lambda_pitch, lambda_duration)
        elif strategy == "opposite_sign_masking_ek2":
            # Composition uses opposite-sign masking; hook uses expanded K
            return self._opposite_sign_masking(lambda_pitch, lambda_duration)
        elif strategy == "cross_concept_masking_ek2":
            # Full shared-feature masking composition; hook uses expanded K
            return self._cross_concept_masking(lambda_pitch, lambda_duration)
        elif strategy in ("expanded_k", "expanded_k_2x", "sequential", "topk_budget", "sas_dense"):
            # These strategies use direct composition; the special handling
            # happens at the hook level, not here.
            return self._direct(lambda_pitch, lambda_duration)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. " f"Choose from: {ALL_STRATEGIES}"
            )

    def compose_separate(
        self,
        lambda_pitch: float,
        lambda_duration: float,
    ) -> tuple:
        """Return scaled pitch and duration vectors separately.

        Used by sequential strategy which needs to apply them in two passes.

        Returns:
            (pitch_vecs, duration_vecs) — each Dict[int, np.ndarray]
        """
        pitch_scaled = {}
        dur_scaled = {}
        for layer in self.layers:
            pitch_scaled[layer] = lambda_pitch * self.pitch_vectors[layer]
            dur_scaled[layer] = lambda_duration * self.duration_vectors[layer]
        return pitch_scaled, dur_scaled

    def get_overlap_summary(self) -> dict:
        """Return pre-computed overlap statistics for diagnostic logging."""
        return {
            layer: {k: v for k, v in info.items() if k != "shared_mask"}
            for layer, info in self._overlap_cache.items()
        }

    # ────────────────────────────────────────────────────────────────────
    # Strategy implementations
    # ────────────────────────────────────────────────────────────────────

    def _direct(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Strategy 1: Direct addition.

        combined = λ_p · v_pitch + λ_d · v_duration

        No special handling of overlapping features.
        """
        combined = {}
        for layer in self.layers:
            combined[layer] = (
                lambda_pitch * self.pitch_vectors[layer]
                + lambda_duration * self.duration_vectors[layer]
            )
        return combined

    def _cross_concept_masking(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Strategy 2: Cross-concept feature masking.

        For each shared feature index i:
          - If |v_pitch[i]| >= |v_duration[i]|: zero it in duration vector
          - Otherwise: zero it in pitch vector

        Then combine the masked vectors via direct addition.
        """
        combined = {}
        for layer in self.layers:
            vp = self.pitch_vectors[layer].copy()
            vd = self.duration_vectors[layer].copy()
            shared = self._overlap_cache[layer]["shared_mask"]

            if shared.any():
                # Winner-take-all on shared features
                pitch_stronger = np.abs(vp[shared]) >= np.abs(vd[shared])
                # Zero out loser in shared positions
                # Where pitch is stronger → zero duration; where duration stronger → zero pitch
                shared_indices = np.where(shared)[0]
                for idx, p_wins in zip(shared_indices, pitch_stronger):
                    if p_wins:
                        vd[idx] = 0.0
                    else:
                        vp[idx] = 0.0

            combined[layer] = lambda_pitch * vp + lambda_duration * vd
        return combined

    def _gram_schmidt_duration(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Strategy 3: Gram-Schmidt — orthogonalise duration w.r.t. pitch.

        v_dur⊥ = v_dur − (v_dur · v_pitch / ||v_pitch||²) · v_pitch
        Then re-sparsify by zeroing out small values (soft thresholding).

        combined = λ_p · v_pitch + λ_d · v_dur⊥
        """
        combined = {}
        for layer in self.layers:
            vp = self.pitch_vectors[layer]
            vd = self.duration_vectors[layer]

            denom = np.dot(vp, vp)
            if denom > 1e-12:
                proj = (np.dot(vd, vp) / denom) * vp
                vd_orth = vd - proj
            else:
                vd_orth = vd.copy()

            # Re-sparsify: zero out entries that were zero in original and
            # are small in the orthogonalised vector (Gram-Schmidt can
            # introduce non-zeros at positions that were originally zero).
            # Keep entries that are either:
            #   (a) non-zero in the original duration vector, or
            #   (b) large enough to matter (> 1% of max magnitude)
            orig_nz = vd != 0
            threshold = 0.01 * np.max(np.abs(vd_orth)) if np.any(vd_orth != 0) else 0
            keep = orig_nz | (np.abs(vd_orth) > threshold)
            vd_orth_sparse = np.where(keep, vd_orth, 0.0)

            combined[layer] = lambda_pitch * vp + lambda_duration * vd_orth_sparse
        return combined

    def _cross_concept_sas(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Strategy 5: Cross-concept SAS — extend Alg 1 Step 5 across concepts.

        Zero out ALL features where BOTH v_pitch[c] ≠ 0 AND v_duration[c] ≠ 0.
        This mirrors the SAS paper's within-concept shared removal but applied
        between concepts.  Result: perfectly disjoint feature supports.

        combined = λ_p · v_pitch_clean + λ_d · v_duration_clean
        """
        combined = {}
        for layer in self.layers:
            vp = self.pitch_vectors[layer].copy()
            vd = self.duration_vectors[layer].copy()
            shared = self._overlap_cache[layer]["shared_mask"]

            # Zero shared features in BOTH vectors (SAS paper philosophy)
            vp[shared] = 0.0
            vd[shared] = 0.0

            combined[layer] = lambda_pitch * vp + lambda_duration * vd
        return combined

    def _opposite_sign_masking(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Opposite-sign shared feature masking.

        Only zero out shared features where v_pitch[i] and v_duration[i]
        have OPPOSITE signs (the root cause of cross-talk from the
        anti-correlated overlap).  Same-sign shared features are kept
        intact since they don't cause cross-talk.

        This preserves ~83% of pitch signal and ~87% of duration signal
        (vs ~45%/~57% for cross_concept_sas which zeros ALL shared).

        combined = λ_p · v_pitch_clean + λ_d · v_duration_clean
        """
        combined = {}
        for layer in self.layers:
            vp = self.pitch_vectors[layer].copy()
            vd = self.duration_vectors[layer].copy()
            shared = self._overlap_cache[layer]["shared_mask"]

            if shared.any():
                # Identify opposite-sign shared features
                opposite_sign = shared & (np.sign(vp) != np.sign(vd))
                n_opposite = int(opposite_sign.sum())
                n_same = int(shared.sum()) - n_opposite

                # Zero opposite-sign features in BOTH vectors
                vp[opposite_sign] = 0.0
                vd[opposite_sign] = 0.0

                logger.debug(
                    f"L{layer}: zeroed {n_opposite} opposite-sign shared "
                    f"features, kept {n_same} same-sign"
                )

            combined[layer] = lambda_pitch * vp + lambda_duration * vd
        return combined

    def _gram_schmidt_pitch(
        self, lambda_pitch: float, lambda_duration: float
    ) -> Dict[int, np.ndarray]:
        """Strategy 4: Gram-Schmidt — orthogonalise pitch w.r.t. duration.

        v_pitch⊥ = v_pitch − (v_pitch · v_dur / ||v_dur||²) · v_dur
        Then re-sparsify.

        combined = λ_p · v_pitch⊥ + λ_d · v_dur
        """
        combined = {}
        for layer in self.layers:
            vp = self.pitch_vectors[layer]
            vd = self.duration_vectors[layer]

            denom = np.dot(vd, vd)
            if denom > 1e-12:
                proj = (np.dot(vp, vd) / denom) * vd
                vp_orth = vp - proj
            else:
                vp_orth = vp.copy()

            # Re-sparsify
            orig_nz = vp != 0
            threshold = 0.01 * np.max(np.abs(vp_orth)) if np.any(vp_orth != 0) else 0
            keep = orig_nz | (np.abs(vp_orth) > threshold)
            vp_orth_sparse = np.where(keep, vp_orth, 0.0)

            combined[layer] = lambda_pitch * vp_orth_sparse + lambda_duration * vd
        return combined


# ════════════════════════════════════════════════════════════════════════════
# Convenience loader
# ════════════════════════════════════════════════════════════════════════════


def load_and_create_composer(
    pitch_vectors_path: str,
    duration_vectors_path: str,
) -> SparseVectorComposer:
    """Load SAS vectors from disk and create a composer.

    Args:
        pitch_vectors_path:    Path to average_pitch_sas_vectors.pt
        duration_vectors_path: Path to average_duration_sas_vectors.pt

    Returns:
        SparseVectorComposer
    """
    import torch

    def _load(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        out = {}
        for k, v in data.items():
            arr = v.numpy() if not isinstance(v, np.ndarray) else v
            out[int(k)] = arr
        return out

    pitch_vecs = _load(pitch_vectors_path)
    duration_vecs = _load(duration_vectors_path)

    return SparseVectorComposer(pitch_vecs, duration_vecs)


# ════════════════════════════════════════════════════════════════════════════
# Quick self-test (no model needed)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from config_dual import PITCH_SAS_VECTORS, DURATION_SAS_VECTORS

    logging.basicConfig(level=logging.INFO)

    print("Loading SAS vectors...")
    composer = load_and_create_composer(
        str(PITCH_SAS_VECTORS), str(DURATION_SAS_VECTORS)
    )

    print(f"\nOverlap summary:")
    for layer, info in sorted(composer.get_overlap_summary().items()):
        print(
            f"  Layer {layer:2d}: pitch={info['n_pitch']:3d}, "
            f"dur={info['n_duration']:3d}, shared={info['n_shared']:3d}"
        )

    print("\nTesting all strategies with λ_pitch=0.75, λ_duration=1.0:")
    for strategy in ALL_STRATEGIES:
        combined = composer.compose(0.75, 1.0, strategy)
        norms = {l: float(np.linalg.norm(v)) for l, v in combined.items()}
        l0s = {l: int((v != 0).sum()) for l, v in combined.items()}
        avg_norm = np.mean(list(norms.values()))
        avg_l0 = np.mean(list(l0s.values()))
        print(f"  {strategy:<28s}  avg_norm={avg_norm:7.2f}  avg_L0={avg_l0:.0f}")
