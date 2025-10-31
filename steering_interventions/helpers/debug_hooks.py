#!/usr/bin/env python3
"""
Debug script to inspect hook registration and activation capture.

Usage:
    python debug_hooks.py
"""

import pathlib
import sys

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils


def inspect_layer_structure(model):
    """Inspect the layer structure to understand where to hook."""
    print("=" * 80)
    print("INSPECTING LAYER STRUCTURE")
    print("=" * 80)

    # Navigate to attn_layers
    decoder_wrapper = model.decoder
    transformer = decoder_wrapper.net
    attn_layers = transformer.attn_layers

    print(f"\nModel type: {type(model).__name__}")
    print(f"Decoder type: {type(decoder_wrapper).__name__}")
    print(f"Transformer type: {type(transformer).__name__}")
    print(f"Attn layers type: {type(attn_layers).__name__}")
    print(f"Number of layers: {len(attn_layers.layers)}")

    # Inspect first few layers
    print("\nFirst 3 layers structure:")
    for layer_idx in range(min(3, len(attn_layers.layers))):
        layer = attn_layers.layers[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        print(f"    Type: {type(layer).__name__}")

        if isinstance(layer, torch.nn.ModuleList):
            print(f"    Sub-modules: {len(layer)}")
            for sub_idx, sub_module in enumerate(layer):
                print(f"      [{sub_idx}] {type(sub_module).__name__}")

                # If it's also a ModuleList, show its contents
                if isinstance(sub_module, torch.nn.ModuleList):
                    print(f"          Contains {len(sub_module)} modules:")
                    for subsub_idx, subsub_module in enumerate(sub_module):
                        print(
                            f"            [{subsub_idx}] {type(subsub_module).__name__}"
                        )


def test_hook_capture(model, encoding):
    """Test that hooks actually capture activations."""
    print("\n" + "=" * 80)
    print("TESTING HOOK CAPTURE")
    print("=" * 80)

    # Navigate to attn_layers
    decoder_wrapper = model.decoder
    transformer = decoder_wrapper.net
    attn_layers = transformer.attn_layers

    # Storage for captured activations
    captured_activations = {}
    hooks = []

    def create_test_hook(layer_idx):
        def hook_fn(module, input, output):
            print(f"    🎣 Hook fired for layer {layer_idx}!")
            print(f"       Module: {type(module).__name__}")
            print(f"       Output type: {type(output)}")

            if isinstance(output, tuple):
                print(f"       Output is tuple with {len(output)} elements")
                for i, elem in enumerate(output):
                    print(
                        f"         [{i}] {type(elem)} shape: {elem.shape if hasattr(elem, 'shape') else 'N/A'}"
                    )
                actual_output = output[0]
            else:
                actual_output = output
                print(f"       Output shape: {actual_output.shape}")

            if isinstance(actual_output, torch.Tensor):
                captured_activations[layer_idx] = actual_output.detach().cpu()
                print(f"       ✅ Captured tensor of shape {actual_output.shape}")
            else:
                print(f"       ❌ Not a tensor: {type(actual_output)}")

        return hook_fn

    # Register hooks on first 3 layers
    print(f"\nRegistering hooks on first 3 layers...")
    for layer_idx in range(min(3, len(attn_layers.layers))):
        layer_module_list = attn_layers.layers[layer_idx]

        if (
            isinstance(layer_module_list, torch.nn.ModuleList)
            and len(layer_module_list) > 1
        ):
            target_module = layer_module_list[1]
            print(f"  Layer {layer_idx}: Hooking {type(target_module).__name__}")
        else:
            target_module = layer_module_list
            print(
                f"  Layer {layer_idx}: Hooking {type(target_module).__name__} (fallback)"
            )

        hook = target_module.register_forward_hook(create_test_hook(layer_idx))
        hooks.append(hook)

    # Create test input
    print(f"\nCreating test input...")
    # Test with a simple sequence
    # Format: (batch, seq_len, 6) for [type, beat, position, pitch, duration, instrument]
    test_seq = torch.zeros((1, 10, 6), dtype=torch.long)
    # Set type to "note" (value 3)
    test_seq[:, :, 0] = 3
    # Set some reasonable values for other dimensions
    test_seq[:, :, 1] = torch.arange(10)  # beat
    test_seq[:, :, 2] = 6  # position
    test_seq[:, :, 3] = 60  # pitch (middle C)
    test_seq[:, :, 4] = 12  # duration
    test_seq[:, :, 5] = 1  # instrument

    mask = torch.ones((1, 10), dtype=torch.bool)

    print(f"  Test sequence shape: {test_seq.shape}")
    print(f"  Mask shape: {mask.shape}")

    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_seq, mask=mask)
            print(f"  ✅ Forward pass successful")
            print(f"  Output type: {type(output)}")
            if isinstance(output, list):
                print(f"  Output list length: {len(output)}")
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()

    # Check captured activations
    print(f"\nCaptured activations:")
    if captured_activations:
        for layer_idx, activation in captured_activations.items():
            print(f"  Layer {layer_idx}: shape {activation.shape}")
    else:
        print("  ❌ No activations captured!")

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    return len(captured_activations) > 0


def main():
    print("=" * 80)
    print("HOOK DEBUGGING TOOL")
    print("=" * 80)

    # Load model configuration
    train_args_file = config.MODEL_DIR / "train-args.json"
    train_args = utils.load_json(train_args_file)
    print(f"\n✓ Loaded training args from: {train_args_file}")

    # Load encoding
    encoding_file = config.NOTES_DIR / "encoding.json"
    encoding = representation.load_encoding(encoding_file)
    print(f"✓ Loaded encoding from: {encoding_file}")

    # Create model
    print(f"\nCreating model...")
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
    )

    # Load checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    print(f"✓ Loaded checkpoint from: {checkpoint_path}")

    model.eval()

    # Inspect structure
    inspect_layer_structure(model)

    # Test hook capture
    success = test_hook_capture(model, encoding)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if success:
        print("✅ Hooks are working correctly!")
        print("   Activations were successfully captured.")
    else:
        print("❌ Hooks are NOT working!")
        print("   No activations were captured.")
        print("   Check the hook registration logic.")
    print()


if __name__ == "__main__":
    main()
