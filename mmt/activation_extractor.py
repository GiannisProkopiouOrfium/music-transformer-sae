import torch
import torch.nn as nn
import numpy as np
import pathlib
import logging
from typing import Dict, List, Optional, Tuple
import h5py
from collections import defaultdict


class ActivationExtractor:
    """Extract and store residual activations from MusicXTransformer."""

    def __init__(self, model, layer_indices: List[int] = None, max_seq_len: int = 1024):
        """
        Initialize the activation extractor.

        Args:
            model: MusicXTransformer model
            layer_indices: Which transformer layers to extract from (0-indexed).
                          If None, extracts from middle layer.
            max_seq_len: Maximum sequence length to store
        """
        self.model = model
        self.max_seq_len = max_seq_len
        self.activations = defaultdict(list)  # layer_idx -> list of tensors
        self.metadata = defaultdict(list)  # layer_idx -> list of metadata dicts
        self.hooks = []

        # The model has 6 transformer layers, but they're organized as:
        # layers[0,1] = layer 0 (attention + ffn)
        # layers[2,3] = layer 1 (attention + ffn)
        # layers[4,5] = layer 2 (attention + ffn)
        # etc.
        total_transformer_layers = len(model.decoder.net.attn_layers.layers) // 2

        if layer_indices is None:
            # Default to middle transformer layer
            middle_layer = total_transformer_layers // 2
            layer_indices = [middle_layer]

        self.layer_indices = layer_indices
        self.total_transformer_layers = total_transformer_layers
        self._register_hooks()

    def store_metadata(self, layer_idx: int, names: List[str], seq_lens: List[int]):
        """
        Store metadata for the current batch.

        Args:
            layer_idx: Layer index
            names: List of source file names for this batch
            seq_lens: List of sequence lengths for this batch
        """
        metadata = {"source_files": names, "sequence_lengths": seq_lens}
        self.metadata[layer_idx].append(metadata)

    def _get_layer_group_indices(self, transformer_layer_idx):
        """
        Convert transformer layer index to actual layer group indices.

        Args:
            transformer_layer_idx: Logical transformer layer (0, 1, 2, ...)

        Returns:
            (attn_layer_idx, ffn_layer_idx): Indices in the actual model layers
        """
        attn_layer_idx = transformer_layer_idx * 2  # Even indices are attention
        ffn_layer_idx = transformer_layer_idx * 2 + 1  # Odd indices are FFN
        return attn_layer_idx, ffn_layer_idx

    def _register_hooks(self):
        """Register forward hooks to capture residual activations."""

        def make_hook(layer_idx, sublayer_name):
            def hook_fn(module, input, output):
                hook_name = f"layer_{layer_idx}_{sublayer_name}"
                logging.info(f"Hook triggered for {hook_name}")

                # The residual module outputs the post-residual activation
                hidden_states = output

                # Check if this looks like valid hidden states
                if (
                    not isinstance(hidden_states, torch.Tensor)
                    or len(hidden_states.shape) != 3
                ):
                    logging.warning(
                        f"{hook_name}: Unexpected output shape {hidden_states.shape if hasattr(hidden_states, 'shape') else type(hidden_states)}"
                    )
                    return

                batch_size, seq_len, hidden_dim = hidden_states.shape
                logging.info(f"{hook_name}: Shape = {hidden_states.shape}")

                # Truncate if sequence is too long
                if seq_len > self.max_seq_len:
                    hidden_states = hidden_states[:, : self.max_seq_len, :]
                    logging.info(f"{hook_name}: Truncated to {hidden_states.shape}")

                # Detach and move to CPU to save memory
                activation_tensor = hidden_states.detach().cpu().clone()
                self.activations[layer_idx].append(activation_tensor)

                logging.info(
                    f"{hook_name}: Stored activation with shape {activation_tensor.shape}"
                )
                logging.info(
                    f"{hook_name}: Total activations stored: {len(self.activations[layer_idx])}"
                )

            return hook_fn

        # Register hooks for specified transformer layers
        logging.info(f"Registering hooks for transformer layers: {self.layer_indices}")
        logging.info(f"Total transformer layers: {self.total_transformer_layers}")

        for transformer_layer_idx in self.layer_indices:
            if transformer_layer_idx >= self.total_transformer_layers:
                logging.error(
                    f"Transformer layer index {transformer_layer_idx} is out of bounds (max: {self.total_transformer_layers-1})"
                )
                continue

            # Get the actual layer group indices
            attn_layer_idx, ffn_layer_idx = self._get_layer_group_indices(
                transformer_layer_idx
            )

            logging.info(
                f"Transformer layer {transformer_layer_idx} maps to layer groups {attn_layer_idx} (attention) and {ffn_layer_idx} (FFN)"
            )

            # Hook the residual connection after FFN (this gives us the full residual stream)
            ffn_layer_group = self.model.decoder.net.attn_layers.layers[ffn_layer_idx]
            if len(ffn_layer_group) >= 3:
                residual_module = ffn_layer_group[2]  # The residual connection
                hook = residual_module.register_forward_hook(
                    make_hook(transformer_layer_idx, "ffn_residual")
                )
                self.hooks.append(hook)
                logging.info(
                    f"Registered hook for transformer layer {transformer_layer_idx}, FFN residual: {type(residual_module)}"
                )
            else:
                logging.error(
                    f"Expected 3 sublayers in FFN layer group {ffn_layer_idx}, got {len(ffn_layer_group)}"
                )

    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_flattened_activations(self, layer_idx: int) -> np.ndarray:
        """
        Get flattened activations for a specific layer.

        Args:
            layer_idx: Layer index to extract activations from

        Returns:
            np.ndarray: Flattened activations [num_tokens, hidden_dim]
        """
        if layer_idx not in self.activations:
            raise ValueError(f"No activations found for layer {layer_idx}")

        # Concatenate all batches and flatten sequence dimension
        layer_activations = []
        for batch_activations in self.activations[layer_idx]:
            # batch_activations: [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = batch_activations.shape
            # Flatten to [batch_size * seq_len, hidden_dim]
            flattened = batch_activations.view(-1, hidden_dim)
            layer_activations.append(flattened.numpy())

        # Concatenate all batches: [total_tokens, hidden_dim]
        return np.concatenate(layer_activations, axis=0)

    def save_activations(self, save_path: pathlib.Path):
        """
        Save all extracted activations and metadata to HDF5 file.

        Args:
            save_path: Path to save the activations
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Debug: Check what we have before saving
        logging.info(f"Preparing to save activations to {save_path}")
        logging.info(f"Layer indices to save: {self.layer_indices}")
        logging.info(f"Activations keys: {list(self.activations.keys())}")

        for layer_idx in self.layer_indices:
            if layer_idx in self.activations:
                num_batches = len(self.activations[layer_idx])
                if num_batches > 0:
                    first_shape = self.activations[layer_idx][0].shape
                    logging.info(
                        f"Layer {layer_idx}: {num_batches} batches, first batch shape: {first_shape}"
                    )
                else:
                    logging.warning(f"Layer {layer_idx}: No activation batches stored")
            else:
                logging.warning(f"Layer {layer_idx}: Not found in activations dict")

        # Only create file if we have activations to save
        has_activations = any(
            layer_idx in self.activations and len(self.activations[layer_idx]) > 0
            for layer_idx in self.layer_indices
        )

        if not has_activations:
            logging.error(
                "No activations to save! Check hook registration and model forward pass."
            )
            return

        with h5py.File(save_path, "w") as f:
            # Save activations
            for layer_idx in self.layer_indices:
                if (
                    layer_idx in self.activations
                    and len(self.activations[layer_idx]) > 0
                ):
                    try:
                        flattened_acts = self.get_flattened_activations(layer_idx)
                        f.create_dataset(
                            f"layer_{layer_idx}",
                            data=flattened_acts,
                            compression="gzip",
                        )
                        logging.info(
                            f"Saved {flattened_acts.shape[0]} tokens from layer {layer_idx}"
                        )
                    except Exception as e:
                        logging.error(f"Error saving layer {layer_idx}: {e}")
                else:
                    logging.warning(f"Skipping layer {layer_idx} - no activations")

            # Save metadata
            if self.metadata:
                metadata_group = f.create_group("metadata")
                for layer_idx in self.layer_indices:
                    if layer_idx in self.metadata and self.metadata[layer_idx]:
                        layer_metadata_group = metadata_group.create_group(
                            f"layer_{layer_idx}"
                        )

                        # Flatten metadata across all batches for this layer
                        all_source_files = []
                        all_seq_lens = []
                        batch_starts = (
                            []
                        )  # Track where each batch starts in the flattened data

                        current_position = 0
                        for batch_metadata in self.metadata[layer_idx]:
                            batch_starts.append(current_position)
                            source_files = batch_metadata["source_files"]
                            seq_lens = batch_metadata["sequence_lengths"]

                            all_source_files.extend(source_files)
                            all_seq_lens.extend(seq_lens)

                            # Each sample contributes seq_len tokens to the flattened data
                            current_position += sum(seq_lens)

                        # Save source files as string dataset
                        string_dtype = h5py.string_dtype(encoding="utf-8")
                        layer_metadata_group.create_dataset(
                            "source_files", data=all_source_files, dtype=string_dtype
                        )
                        layer_metadata_group.create_dataset(
                            "sequence_lengths", data=all_seq_lens
                        )
                        layer_metadata_group.create_dataset(
                            "batch_starts", data=batch_starts
                        )

                        logging.info(
                            f"Saved metadata for layer {layer_idx}: {len(all_source_files)} files"
                        )

        logging.info(f"Activations and metadata saved to {save_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
