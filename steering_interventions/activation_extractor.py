"""Step B: Activation Extraction.

This module:
1. Loads pretrained model
2. Hooks into transformer layers to capture hidden states
3. Runs forward passes on segmented data
4. Extracts "summary" activations (last token's hidden state)
5. Saves activations efficiently for later use
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import dataset as dataset_module
import music_x_transformers
import representation
import utils


class ActivationExtractor:
    """Extract activations from transformer layers using hooks."""

    def __init__(self, model: nn.Module, num_layers: int):
        """Initialize the activation extractor.

        Args:
            model: The MusicXTransformer model
            num_layers: Number of transformer layers
        """
        self.model = model
        self.num_layers = num_layers
        self.activations = {i: [] for i in range(num_layers)}
        self.hooks = []

    def _create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """

        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, dim)
            # We want the last token's hidden state
            # Store detached copy to avoid memory issues
            self.activations[layer_idx].append(output.detach().cpu())

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        # The model structure is: MusicXTransformer -> .decoder (MusicAutoregressiveWrapper) -> .net (MusicTransformerWrapper) -> .attn_layers
        # attn_layers.layers is a ModuleList of transformer layers

        # Navigate through the model structure
        # MusicXTransformer has a 'decoder' attribute (MusicAutoregressiveWrapper)
        if hasattr(self.model, "decoder"):
            decoder_wrapper = self.model.decoder
            if hasattr(decoder_wrapper, "net"):
                transformer = decoder_wrapper.net
            else:
                raise ValueError("Cannot find 'net' in model.decoder")
        elif hasattr(self.model, "net"):
            # Fallback: direct access to net
            transformer = self.model.net
        else:
            raise ValueError(
                "Cannot navigate model structure - no 'decoder' or 'net' attribute"
            )

        # Get the attention layers
        if hasattr(transformer, "attn_layers"):
            attn_layers = transformer.attn_layers
        else:
            raise ValueError("Cannot find attn_layers in transformer")

        # Register hooks on each layer
        if hasattr(attn_layers, "layers"):
            for layer_idx, layer in enumerate(attn_layers.layers):
                hook = layer.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
                logging.debug(f"Registered hook on layer {layer_idx}")
        else:
            raise ValueError("Cannot find layers in attn_layers")

        logging.info(
            f"Registered {len(self.hooks)} hooks on {len(attn_layers.layers)} layers"
        )

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logging.info("Removed all hooks")

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {i: [] for i in range(self.num_layers)}

    def get_summary_activations(self, seq_lengths: List[int]) -> Dict[int, np.ndarray]:
        """Extract summary activations (last valid token) for each layer.

        Args:
            seq_lengths: List of actual sequence lengths (for each batch item)

        Returns:
            Dictionary mapping layer_idx -> array of shape (total_samples, dim)
        """
        summary_activations = {}

        for layer_idx in range(self.num_layers):
            layer_activations = self.activations[layer_idx]

            if not layer_activations:
                logging.warning(f"No activations found for layer {layer_idx}")
                continue

            # Extract last valid token for each batch
            summaries = []
            batch_offset = 0

            for batch_tensor in layer_activations:
                # batch_tensor shape: (batch_size, seq_len, dim)
                batch_size = batch_tensor.shape[0]

                for i in range(batch_size):
                    if batch_offset + i < len(seq_lengths):
                        seq_len = seq_lengths[batch_offset + i]
                        # Get the hidden state at position seq_len - 1
                        summary = batch_tensor[i, seq_len - 1, :].numpy()
                        summaries.append(summary)

                batch_offset += batch_size

            summary_activations[layer_idx] = np.array(summaries)
            logging.debug(f"Layer {layer_idx}: {summary_activations[layer_idx].shape}")

        return summary_activations


def load_segment_as_tokens(
    segment: Dict,
    notes_dir: pathlib.Path,
    encoding: Dict,
    max_seq_len: int = 1024,
    max_beat: int = 256,
) -> Tuple[Optional[np.ndarray], int]:
    """Load a segment and convert to tokens.

    Args:
        segment: Segment dictionary from data_curator
        notes_dir: Directory containing .npy files
        encoding: Encoding dictionary
        max_seq_len: Maximum sequence length
        max_beat: Maximum beat value

    Returns:
        (tokens, actual_length) or (None, 0) if not found
    """
    file_name = segment["file_name"]
    sub_folder_name = file_name.split("-")[0]
    segment_idx = segment.get("segment_idx", 0)
    start_beat = segment.get("start_beat", 0)
    n_beats = segment.get("n_beats")

    # Try to find the .npy file
    # The file structure mirrors the JSON structure
    npy_path = notes_dir / sub_folder_name / f"{file_name}.npy"

    if not npy_path.exists():
        logging.warning(f"NPY file not found: {npy_path}")
        return None, 0

    try:
        # Load the full note sequence (5D format: beat, position, pitch, duration, program)
        notes = np.load(npy_path)

        # Validate shape
        if len(notes.shape) != 2:
            logging.error(f"Invalid shape {notes.shape} for {npy_path}, expected 2D array")
            return None, 0
        
        if notes.shape[1] != 5:
            logging.error(f"Invalid note format {notes.shape} for {npy_path}, expected (seq_len, 5)")
            return None, 0
        
        # Notes format: [beat, position, pitch, duration, program]
        # If we need to segment, extract the relevant beat range FIRST (before encoding)
        if segment_idx > 0 or n_beats is not None:
            beat_values = notes[:, 0]  # Beat is always first dimension in notes
            end_beat = start_beat + n_beats if n_beats else max_beat

            mask = (beat_values >= start_beat) & (beat_values < end_beat)
            notes = notes[mask]

            # Adjust beat values to start from 0
            if len(notes) > 0:
                notes[:, 0] -= start_beat
        
        if len(notes) == 0:
            logging.warning(f"No notes in segment for {npy_path}")
            return None, 0

        # Now convert notes (5D) to codes (6D) using representation.encode_notes
        # This properly handles the instrument dimension
        codes = representation.encode_notes(notes, encoding)
        
        # codes shape: (seq_len, 6) with format [type, beat, position, pitch, duration, instrument]
        
        # Truncate to max_seq_len
        if len(codes) > max_seq_len:
            codes = codes[:max_seq_len]

        actual_length = len(codes)

        return codes, actual_length

    except Exception as e:
        logging.error(f"Error loading {npy_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, 0


def extract_activations_for_segments(
    segments: List[Dict],
    model: nn.Module,
    num_layers: int,
    encoding: Dict,
    notes_dir: pathlib.Path,
    device: torch.device,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    max_beat: int = 256,
) -> Dict[int, np.ndarray]:
    """Extract activations for a list of segments.

    Args:
        segments: List of segment dictionaries
        model: Pretrained model
        num_layers: Number of layers in the model
        encoding: Encoding dictionary
        notes_dir: Directory with .npy files
        device: Device to run on
        batch_size: Batch size for processing
        max_seq_len: Maximum sequence length
        max_beat: Maximum beat value

    Returns:
        Dictionary mapping layer_idx -> activations array
    """
    extractor = ActivationExtractor(model, num_layers)
    extractor.register_hooks()

    model.eval()

    # Prepare data
    all_tokens = []
    all_lengths = []
    valid_indices = []

    logging.info("Loading and tokenizing segments...")
    for idx, segment in enumerate(tqdm(segments, desc="Loading segments")):
        tokens, length = load_segment_as_tokens(
            segment, notes_dir, encoding, max_seq_len, max_beat
        )

        if tokens is not None and length > 0:
            all_tokens.append(tokens)
            all_lengths.append(length)
            valid_indices.append(idx)

    logging.info(f"Loaded {len(all_tokens)} valid segments out of {len(segments)}")

    if len(all_tokens) == 0:
        logging.error("No valid segments found!")
        return {}

    # Process in batches
    logging.info("Extracting activations...")

    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens), batch_size), desc="Processing batches"):
            batch_tokens = all_tokens[i : i + batch_size]
            batch_lengths = all_lengths[i : i + batch_size]

            # Pad to same length
            max_len = max(len(t) for t in batch_tokens)
            padded_batch = []

            for tokens in batch_tokens:
                if len(tokens) < max_len:
                    padding = np.zeros(
                        (max_len - len(tokens), tokens.shape[1]), dtype=tokens.dtype
                    )
                    padded = np.concatenate([tokens, padding], axis=0)
                else:
                    padded = tokens
                padded_batch.append(padded)

            # Convert to tensor
            batch_tensor = torch.from_numpy(np.array(padded_batch)).long().to(device)

            # Create mask
            mask = torch.zeros(
                len(batch_tokens), max_len, dtype=torch.bool, device=device
            )
            for j, length in enumerate(batch_lengths):
                mask[j, :length] = True

            # Forward pass (this will trigger the hooks)
            try:
                _ = model(batch_tensor, mask=mask)
            except Exception as e:
                logging.error(f"Error in forward pass: {e}")
                continue

    # Extract summary activations
    summary_activations = extractor.get_summary_activations(all_lengths)

    # Clean up
    extractor.remove_hooks()

    return summary_activations


def save_activations(
    activations: Dict[int, np.ndarray],
    output_path: pathlib.Path,
    metadata: Dict,
):
    """Save activations to HDF5 file.

    Args:
        activations: Dictionary mapping layer_idx -> activations
        output_path: Output file path
        metadata: Metadata to store
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save metadata
        f.attrs["concept"] = metadata.get("concept", "unknown")
        f.attrs["n_segments"] = metadata.get("n_segments", 0)
        f.attrs["num_layers"] = len(activations)

        # Save metadata as JSON string
        f.attrs["metadata_json"] = json.dumps(metadata)

        # Save activations for each layer
        for layer_idx, layer_activations in activations.items():
            f.create_dataset(
                f"layer_{layer_idx}",
                data=layer_activations,
                compression="gzip",
                compression_opts=4,
            )
            logging.info(f"Saved layer {layer_idx}: shape {layer_activations.shape}")

    logging.info(f"Saved activations to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract activations for steering interventions"
    )
    parser.add_argument(
        "--concept", type=str, default="velocity", help="Which concept to use"
    )
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "datasets",
        help="Directory with curated datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "activations",
        help="Output directory for activations",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Model checkpoint path (default: best_model.pt)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU number to use (e.g., 0, 1). If not specified, uses CPU. Automatically detects CUDA or MPS.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load segments
    high_segments_file = args.dataset_dir / f"high_{args.concept}_segments.json"
    low_segments_file = args.dataset_dir / f"low_{args.concept}_segments.json"
    metadata_file = args.dataset_dir / f"{args.concept}_metadata.json"

    if not high_segments_file.exists() or not low_segments_file.exists():
        logging.error(f"Segment files not found in {args.dataset_dir}")
        logging.error("Please run data_curator.py first!")
        return

    with open(high_segments_file, "r") as f:
        high_segments = json.load(f)

    with open(low_segments_file, "r") as f:
        low_segments = json.load(f)

    with open(metadata_file, "r") as f:
        dataset_metadata = json.load(f)

    logging.info(f"Loaded {len(high_segments)} high segments")
    logging.info(f"Loaded {len(low_segments)} low segments")

    # Setup device
    if args.gpu is not None:
        # User specified a GPU
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logging.warning(
                f"CUDA/MPS not available, falling back to CPU (requested GPU {args.gpu})"
            )
    else:
        # No GPU specified, use CPU
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU specified)")

    logging.info(f"Device: {device}")

    # Load model configuration
    train_args_file = config.MODEL_DIR / "train-args.json"
    if not train_args_file.exists():
        logging.error(f"Training args not found: {train_args_file}")
        return

    train_args = utils.load_json(train_args_file)
    logging.info(f"Loaded training args from: {train_args_file}")

    # Load encoding
    encoding_file = config.NOTES_DIR / "encoding.json"
    encoding = representation.load_encoding(encoding_file)
    logging.info(f"Loaded encoding from: {encoding_file}")

    # Create model
    logging.info("Creating model...")
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    # Load checkpoint
    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f"Loaded checkpoint from: {checkpoint_path}")

    num_layers = train_args["layers"]

    # Extract activations for high segments
    logging.info("=" * 50)
    logging.info("Extracting activations for HIGH segments")
    logging.info("=" * 50)

    high_activations = extract_activations_for_segments(
        high_segments,
        model,
        num_layers,
        encoding,
        config.NOTES_DIR,
        device,
        args.batch_size,
        train_args["max_seq_len"],
        train_args["max_beat"],
    )

    # Save high activations
    high_output_path = args.output_dir / f"high_{args.concept}_activations.h5"
    save_activations(
        high_activations,
        high_output_path,
        {**dataset_metadata, "n_segments": len(high_segments), "type": "high"},
    )

    # Extract activations for low segments
    logging.info("=" * 50)
    logging.info("Extracting activations for LOW segments")
    logging.info("=" * 50)

    low_activations = extract_activations_for_segments(
        low_segments,
        model,
        num_layers,
        encoding,
        config.NOTES_DIR,
        device,
        args.batch_size,
        train_args["max_seq_len"],
        train_args["max_beat"],
    )

    # Save low activations
    low_output_path = args.output_dir / f"low_{args.concept}_activations.h5"
    save_activations(
        low_activations,
        low_output_path,
        {**dataset_metadata, "n_segments": len(low_segments), "type": "low"},
    )

    logging.info("Activation extraction complete!")


if __name__ == "__main__":
    main()
