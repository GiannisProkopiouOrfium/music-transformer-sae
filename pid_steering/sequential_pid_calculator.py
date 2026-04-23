"""Sequential PID Vector Calculator (Mean-AcT-style).

Computes PID-enhanced steering vectors by accounting for the causal
dependency across layers. After applying the PID correction u(k-1) at
sublayer k-1, the error signal e(k) at sublayer k is re-computed using
the steered source activations.

This is the stronger contribution per the PID paper (Nguyen et al., ICLR 2026):
unlike non-sequential PID which uses static pre-computed DiffMean vectors,
sequential PID captures how each intervention propagates through subsequent
layers before computing the next correction.

Algorithm:
    1. Initialize integral accumulator and prev_error to zero
    2. For each sublayer k = 0..11:
       a. Forward source data through model up to sublayer k
          (with PID corrections at sublayers 0..k-1 applied)
       b. Forward target data through model up to sublayer k (unmodified)
       c. Compute e(k) = mean_target(k) - mean_source_steered(k)
       d. u(k) = Kp*e(k) + Ki*integral + Kd*(e(k) - prev_e)
       e. Update integral (with anti-windup), store prev_error
    3. Save u(k) vectors for inference

The resulting vectors are used identically to non-sequential PID vectors
during inference (via pid_steered_generator.py).
"""

import argparse
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_controller import SpatialPIDController
from pid_steering.pid_vector_calculator import save_pid_vectors
import config_pid

import music_x_transformers
import representation
import utils

logger = logging.getLogger(__name__)


def get_sublayer_modules(model):
    """Extract the ordered list of sublayer modules from the model.

    Returns:
        List of (sublayer_idx, module) tuples for hook registration.
    """
    if hasattr(model, "decoder"):
        transformer = model.decoder.net
    elif hasattr(model, "net"):
        transformer = model.net
    else:
        raise ValueError("Cannot navigate model structure")

    attn_layers = transformer.attn_layers
    modules = []
    for idx, layer in enumerate(attn_layers.layers):
        if isinstance(layer, nn.ModuleList) and len(layer) > 1:
            modules.append((idx, layer[1]))
        else:
            modules.append((idx, layer))

    return modules


class ActivationCapture:
    """Captures activations at a specific sublayer via a forward hook."""

    def __init__(self, position="last"):
        self.activations = []
        self.hook_handle = None
        self.position = position

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            actual_output = output[0]
        else:
            actual_output = output

        if self.position == "last":
            act = actual_output[:, -1, :].detach()  # (batch, dim)
        else:
            act = actual_output.mean(dim=1).detach()  # (batch, dim)

        self.activations.append(act)

    def register(self, module):
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_mean(self):
        """Get the mean activation across all captured samples."""
        all_acts = torch.cat(self.activations, dim=0)  # (N, dim)
        return all_acts.mean(dim=0)  # (dim,)

    def clear(self):
        self.activations = []


class SteeringInjector:
    """Injects a pre-computed steering vector at a specific sublayer."""

    def __init__(self, vector, alpha=1.0, position="last"):
        self.vector = vector
        self.alpha = alpha
        self.position = position
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            actual_output = output[0]
            is_tuple = True
        else:
            actual_output = output
            is_tuple = False

        sv = self.vector.to(actual_output.device)
        if self.position == "last":
            actual_output[:, -1, :] = actual_output[:, -1, :] + self.alpha * sv
        else:
            actual_output = actual_output + self.alpha * sv

        if is_tuple:
            return (actual_output,) + output[1:]
        return actual_output

    def register(self, module):
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def extract_mean_at_layer(
    model,
    data_sequences: List[torch.Tensor],
    layer_idx: int,
    sublayer_modules: list,
    steering_hooks: List[SteeringInjector],
    device: torch.device,
    position: str = "last",
) -> torch.Tensor:
    """Extract mean activation at a specific sublayer.

    Args:
        model: The MMT model.
        data_sequences: List of token sequences to forward.
        layer_idx: Target sublayer index to capture.
        sublayer_modules: List of (idx, module) pairs.
        steering_hooks: Active steering hooks at prior sublayers.
        device: Torch device.
        position: "last" or "mean".

    Returns:
        Mean activation tensor (dim,).
    """
    # Register capture hook at target sublayer
    _, target_module = sublayer_modules[layer_idx]
    capture = ActivationCapture(position=position)
    capture.register(target_module)

    # Register all steering hooks at prior sublayers
    active_handles = []
    for hook in steering_hooks:
        hook.register(sublayer_modules[hook._layer_idx][1])
        active_handles.append(hook)

    try:
        model.eval()
        with torch.no_grad():
            for seq in data_sequences:
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).long()
                seq = seq.unsqueeze(0).to(device)  # (1, seq_len, 6)
                # Forward pass (training mode returns loss, so use net directly)
                model.decoder.net(seq)
    finally:
        capture.remove()
        for hook in active_handles:
            hook.remove()

    return capture.get_mean()


def compute_sequential_pid_vectors(
    model: nn.Module,
    source_sequences: List,
    target_sequences: List,
    Kp: float = 1.0,
    Ki: float = 0.3,
    Kd: float = 0.1,
    max_I: float = 5.0,
    alpha: float = 1.0,
    device: torch.device = None,
    position: str = "last",
) -> Tuple[Dict[int, torch.Tensor], Dict]:
    """Compute sequential PID steering vectors.

    Follows the Mean-AcT sequential computation paradigm: after applying
    u(k-1) at sublayer k-1, re-compute the error at sublayer k using the
    steered source activations.

    Args:
        model: The MMT model.
        source_sequences: Token sequences from the source (e.g., low-pitch) set.
        target_sequences: Token sequences from the target (e.g., high-pitch) set.
        Kp, Ki, Kd, max_I: PID gains.
        alpha: Scaling factor applied to each u(k) during intermediate forwarding.
        device: Torch device.
        position: Activation extraction position ("last" or "mean").

    Returns:
        Tuple of (pid_vectors dict, diagnostics dict).
    """
    if device is None:
        device = next(model.parameters()).device

    sublayer_modules = get_sublayer_modules(model)
    n_layers = len(sublayer_modules)

    controller = SpatialPIDController(
        Kp=Kp, Ki=Ki, Kd=Kd, max_I=max_I, dim=config_pid.MODEL_DIM
    )
    controller.to(device)
    controller.reset()

    pid_vectors = {}
    steering_hooks = []  # Accumulate hooks for prior sublayers
    diagnostics = {"errors": {}, "vectors": {}, "error_norms": [], "vector_norms": []}

    logger.info(f"Computing sequential PID vectors across {n_layers} sublayers")
    logger.info(
        f"Source set: {len(source_sequences)} sequences, "
        f"Target set: {len(target_sequences)} sequences"
    )

    for k in range(n_layers):
        logger.info(f"Processing sublayer {k}/{n_layers - 1}...")

        # 1. Extract mean activation at sublayer k for source data
        #    (with PID corrections at sublayers 0..k-1 applied)
        mu_source = extract_mean_at_layer(
            model,
            source_sequences,
            k,
            sublayer_modules,
            steering_hooks,
            device,
            position,
        )

        # 2. Extract mean activation at sublayer k for target data
        #    (no corrections applied — target represents desired state)
        mu_target = extract_mean_at_layer(
            model,
            target_sequences,
            k,
            sublayer_modules,
            [],
            device,
            position,  # No steering hooks for target
        )

        # 3. Compute error: e(k) = mu_target(k) - mu_source_steered(k)
        e_k = mu_target - mu_source

        # 4. Apply PID control law
        u_k = controller.compute(e_k)

        pid_vectors[k] = u_k.cpu().clone()

        # Store diagnostics
        e_norm = e_k.norm().item()
        u_norm = u_k.norm().item()
        diagnostics["error_norms"].append(e_norm)
        diagnostics["vector_norms"].append(u_norm)
        diagnostics["errors"][k] = e_k.cpu().numpy()
        diagnostics["vectors"][k] = u_k.cpu().numpy()

        logger.info(f"  e(k) norm={e_norm:.4f}, u(k) norm={u_norm:.4f}")

        # 5. Create a steering hook for sublayer k to use in subsequent passes
        injector = SteeringInjector(u_k, alpha=alpha, position=position)
        injector._layer_idx = k  # Store for hook registration
        steering_hooks.append(injector)

    return pid_vectors, diagnostics


def load_contrastive_sequences(
    activations_dir: pathlib.Path,
    concept: str,
) -> Tuple[List, List]:
    """Load contrastive sequence data for source and target sets.

    Tries multiple sources in order:
    1. HDF5 activation files (outputs/activations/)
    2. JSON segment files (outputs/datasets/) → tokenized via activation_extractor

    Args:
        activations_dir: Directory containing activation HDF5 files.
        concept: Concept name (e.g., "average_pitch").

    Returns:
        (source_sequences, target_sequences) lists of token arrays.
    """
    high_h5 = activations_dir / f"high_{concept}_activations.h5"
    low_h5 = activations_dir / f"low_{concept}_activations.h5"

    source_sequences = []
    target_sequences = []

    # Method 1: Try loading from HDF5 activation files
    if high_h5.exists() and low_h5.exists():
        for path, seq_list in [(low_h5, source_sequences), (high_h5, target_sequences)]:
            with h5py.File(path, "r") as f:
                if "segments" in f:
                    for key in sorted(f["segments"].keys()):
                        seq_list.append(f["segments"][key][:])
                elif "metadata_json" in f.attrs:
                    import json

                    meta = json.loads(f.attrs["metadata_json"])
                    if "segment_files" in meta:
                        for seg_file in meta["segment_files"]:
                            seg_path = pathlib.Path(seg_file)
                            if seg_path.exists():
                                seq_list.append(np.load(seg_path))

        if source_sequences and target_sequences:
            logger.info(
                f"Loaded {len(source_sequences)} source + "
                f"{len(target_sequences)} target sequences from HDF5"
            )
            return source_sequences, target_sequences

    # Method 2: Load from JSON segment files and tokenize
    datasets_dir = activations_dir.parent / "datasets"
    high_json = datasets_dir / f"high_{concept}_segments.json"
    low_json = datasets_dir / f"low_{concept}_segments.json"

    if high_json.exists() and low_json.exists():
        import json

        logger.info("Loading contrastive sequences from JSON segment files...")

        encoding = representation.load_encoding(config_pid.ENCODING_PATH)
        notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"

        # Import the tokenization function from activation_extractor
        sys.path.insert(
            0,
            str(config_pid.PROJECT_ROOT / "steering_interventions"),
        )
        from activation_extractor import load_segment_as_tokens

        for json_path, seq_list in [
            (low_json, source_sequences),
            (high_json, target_sequences),
        ]:
            with open(json_path, "r") as f:
                segments = json.load(f)

            # Limit to a manageable number for PID analysis
            max_segments = min(len(segments), 1280)
            for segment in segments[:max_segments]:
                tokens, length = load_segment_as_tokens(
                    segment,
                    notes_dir,
                    encoding,
                    max_seq_len=config_pid.MAX_SEQ_LEN,
                )
                if tokens is not None:
                    seq_list.append(tokens)

        if source_sequences and target_sequences:
            logger.info(
                f"Loaded {len(source_sequences)} source + "
                f"{len(target_sequences)} target sequences from JSON segments"
            )
            return source_sequences, target_sequences

    raise FileNotFoundError(
        f"No contrastive data found for concept '{concept}'. "
        f"Checked: {high_h5}, {low_h5}, {high_json}, {low_json}. "
        f"Run data_curator.py and/or activation_extractor.py first."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute sequential PID steering vectors (Mean-AcT-style)"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="average_pitch",
        help="Concept name (average_pitch or average_duration)",
    )
    parser.add_argument(
        "--activations_dir",
        type=pathlib.Path,
        default=config_pid.ACTIVATIONS_DIR,
        help="Directory with extracted activations / segment data",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_VECTORS_DIR,
        help="Output directory for sequential PID vectors",
    )
    parser.add_argument(
        "--Kp", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Intermediate steering scale"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=pathlib.Path,
        default=config_pid.MODEL_CHECKPOINT,
    )
    parser.add_argument(
        "--train_args",
        type=pathlib.Path,
        default=config_pid.TRAIN_ARGS_PATH,
    )
    parser.add_argument(
        "--encoding",
        type=pathlib.Path,
        default=config_pid.ENCODING_PATH,
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    from pid_steering.pid_steered_generator import load_model

    model, encoding, train_args = load_model(
        args.model_checkpoint, args.train_args, args.encoding, device
    )

    # Load contrastive sequences
    source_seqs, target_seqs = load_contrastive_sequences(
        args.activations_dir, args.concept
    )

    # Compute sequential PID vectors
    pid_vectors, diagnostics = compute_sequential_pid_vectors(
        model,
        source_seqs,
        target_seqs,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        alpha=args.alpha,
        device=device,
    )

    # Save
    metadata = {
        "concept": args.concept,
        "mode": "sequential",
        "variant": "pid",
        "Kp": args.Kp,
        "Ki": args.Ki,
        "Kd": args.Kd,
        "max_I": args.max_I,
        "alpha": args.alpha,
        "n_source": len(source_seqs),
        "n_target": len(target_seqs),
        "error_norms": diagnostics["error_norms"],
        "vector_norms": diagnostics["vector_norms"],
    }

    out_path = args.output_dir / f"{args.concept}_sequential_pid_vectors.pt"
    save_pid_vectors(pid_vectors, out_path, metadata)

    # Print summary
    print("\n" + "=" * 60)
    print("Sequential PID Vector Summary")
    print("=" * 60)
    for k in sorted(pid_vectors.keys()):
        e_norm = diagnostics["error_norms"][k]
        u_norm = diagnostics["vector_norms"][k]
        print(f"Layer {k:2d}: |e(k)|={e_norm:.4f}  |u(k)|={u_norm:.4f}")
    print("=" * 60)

    logger.info("Sequential PID vector computation complete!")


if __name__ == "__main__":
    main()
