"""Extraction pipeline for SAE analysis."""

import logging
import torch
import h5py
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from mmt.activation_extractor import ActivationExtractor
from mmt.dataset import MidiDataset
from mmt.music_x_transformers import MusicXTransformer
from utils.gpu import setup_device
from utils.memory import MemoryOptimizer
from utils.storage import HDF5Manager


def load_model(config: Dict[str, Any], device: torch.device) -> MusicXTransformer:
    """Load the pretrained music transformer model."""
    logger = logging.getLogger(__name__)

    model_config = config["model"]
    checkpoint_path = model_config["checkpoint_path"]

    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same config as training
    model = MusicXTransformer(
        num_tokens=2051,  # SOD dataset vocabulary
        dim=512,
        depth=6,
        heads=8,
        max_seq_len=512,
        use_abs_pos_emb=False,
        emb_dropout=0.1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model


def create_data_loader(
    config: Dict[str, Any], memory_optimizer: MemoryOptimizer
) -> torch.utils.data.DataLoader:
    """Create optimized data loader for extraction."""
    logger = logging.getLogger(__name__)

    extraction_config = config["extraction"]
    data_path = extraction_config.get("data_path", "data/sod/processed")

    # Create dataset
    dataset = MidiDataset(
        data_dir=data_path,
        dataset_name="train",  # Use training data for activation extraction
        max_seq_len=512,
    )

    logger.info(f"Created dataset with {len(dataset)} samples")

    # Calculate optimal batch size
    settings = memory_optimizer.optimize_for_sae_training(
        hidden_size=512, sae_expansion=4  # Model hidden size
    )

    batch_size = extraction_config.get("batch_size", settings["batch_size"])
    num_workers = settings["recommended_dataloader_workers"]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for consistent extraction
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        f"Created dataloader with batch_size={batch_size}, num_workers={num_workers}"
    )
    return dataloader


def extract_activations(
    model: MusicXTransformer,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    output_file: str,
    device: torch.device,
    max_samples: int = None,
) -> Dict[str, Any]:
    """Extract activations from the specified layer."""
    logger = logging.getLogger(__name__)

    logger.info(f"Starting activation extraction from layer {layer_idx}")
    logger.info(f"Output file: {output_file}")

    # Initialize extraction components
    extractor = ActivationExtractor(model, [layer_idx])
    hdf5_manager = HDF5Manager()

    all_activations = []
    all_tokens = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and total_samples >= max_samples:
                break

            # Move batch to device
            if isinstance(batch, dict):
                inputs = batch["input_ids"].to(device)
            else:
                inputs = batch.to(device)

            # Extract activations
            _ = model(inputs)  # Forward pass to trigger hooks
            activations = extractor.get_activations()

            if layer_idx in activations:
                # Get activations for this batch
                layer_activations = activations[layer_idx]  # [batch, seq_len, hidden]

                # Reshape to [batch * seq_len, hidden]
                batch_size, seq_len, hidden_size = layer_activations.shape
                flattened_activations = layer_activations.view(-1, hidden_size)

                # Store activations and corresponding tokens
                all_activations.append(flattened_activations.cpu())

                # Flatten tokens similarly
                flattened_tokens = inputs.view(-1)
                all_tokens.append(flattened_tokens.cpu())

                total_samples += flattened_activations.shape[0]

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Processed batch {batch_idx}, total samples: {total_samples}"
                    )

            # Clear activation cache
            extractor.clear_activations()

    # Concatenate all activations
    logger.info("Concatenating activations...")
    final_activations = torch.cat(all_activations, dim=0)
    final_tokens = torch.cat(all_tokens, dim=0)

    logger.info(f"Final activation shape: {final_activations.shape}")

    # Save to HDF5
    logger.info(f"Saving activations to {output_file}")

    with h5py.File(output_file, "w") as f:
        # Create optimized datasets
        activations_dataset = hdf5_manager.create_optimized_dataset(
            f, "activations", final_activations.shape, "float32"
        )
        tokens_dataset = hdf5_manager.create_optimized_dataset(
            f, "tokens", final_tokens.shape, "int32"
        )

        # Write data
        activations_dataset[:] = final_activations.numpy()
        tokens_dataset[:] = final_tokens.numpy()

        # Add metadata
        f.attrs["layer_idx"] = layer_idx
        f.attrs["num_samples"] = total_samples
        f.attrs["hidden_size"] = final_activations.shape[1]
        f.attrs["model_name"] = "MusicXTransformer"

    logger.info(f"Extraction completed: {total_samples} samples saved")

    return {
        "num_samples": total_samples,
        "hidden_size": final_activations.shape[1],
        "output_file": output_file,
        "layer_idx": layer_idx,
    }


def run_extraction_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete extraction pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Initializing extraction pipeline")

    # Setup device and optimization
    device, device_config = setup_device()
    memory_optimizer = MemoryOptimizer()

    # Load model
    model = load_model(config, device)

    # Create data loader
    dataloader = create_data_loader(config, memory_optimizer)

    # Get extraction parameters
    extraction_config = config["extraction"]
    layer_idx = config["model"]["layer_idx"]
    output_file = (
        Path(config["experiment"]["directories"]["activations"])
        / extraction_config["output_file"]
    )
    max_samples = extraction_config.get("max_samples", None)

    # Extract activations
    results = extract_activations(
        model=model,
        dataloader=dataloader,
        layer_idx=layer_idx,
        output_file=str(output_file),
        device=device,
        max_samples=max_samples,
    )

    # Add experiment info to results
    results.update(
        {
            "experiment_name": config["experiment"]["name"],
            "device": str(device),
            "config": extraction_config,
        }
    )

    logger.info("Extraction pipeline completed successfully")
    return results
