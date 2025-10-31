#!/usr/bin/env python3
"""
Debug script to inspect the model structure and verify the correct path to attn_layers.

Usage:
    python debug_model_structure.py [--checkpoint PATH]
"""

import argparse
import pathlib
import sys

import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils


def inspect_model_structure(model, prefix="model", max_depth=4, current_depth=0):
    """Recursively inspect model structure."""
    if current_depth >= max_depth:
        return

    print(f"\n{'  ' * current_depth}[{prefix}]")
    print(f"{'  ' * current_depth}Type: {type(model).__name__}")

    # Check for common attributes
    important_attrs = ["decoder", "net", "attn_layers", "layers"]

    for attr in important_attrs:
        if hasattr(model, attr):
            attr_value = getattr(model, attr)
            print(f"{'  ' * current_depth}✓ Has '{attr}': {type(attr_value).__name__}")

            # If it's layers (ModuleList), show count
            if attr == "layers" and hasattr(attr_value, "__len__"):
                print(f"{'  ' * current_depth}  → Contains {len(attr_value)} layers")

            # Recurse for decoder, net, attn_layers
            if (
                attr in ["decoder", "net", "attn_layers"]
                and current_depth < max_depth - 1
            ):
                inspect_model_structure(
                    attr_value, f"{prefix}.{attr}", max_depth, current_depth + 1
                )


def verify_hook_registration_path(model):
    """Verify the exact path to register hooks."""
    print("\n" + "=" * 80)
    print("VERIFYING HOOK REGISTRATION PATH")
    print("=" * 80)

    try:
        # Try the path used in activation_extractor.py (after fix)
        if hasattr(model, "decoder"):
            print("✓ Step 1: model.decoder exists")
            decoder_wrapper = model.decoder

            if hasattr(decoder_wrapper, "net"):
                print("✓ Step 2: model.decoder.net exists")
                transformer = decoder_wrapper.net

                if hasattr(transformer, "attn_layers"):
                    print("✓ Step 3: model.decoder.net.attn_layers exists")
                    attn_layers = transformer.attn_layers

                    if hasattr(attn_layers, "layers"):
                        print(f"✓ Step 4: model.decoder.net.attn_layers.layers exists")
                        print(f"  → Contains {len(attn_layers.layers)} layers")
                        print()
                        print("✅ SUCCESS! The path is correct:")
                        print("   model.decoder.net.attn_layers.layers[i]")
                        print()

                        # Show the first layer's structure
                        if len(attn_layers.layers) > 0:
                            print("First layer structure:")
                            first_layer = attn_layers.layers[0]
                            print(f"  Type: {type(first_layer).__name__}")
                            if hasattr(first_layer, "__len__"):
                                print(f"  Sub-modules: {len(first_layer)}")
                                for idx, submodule in enumerate(first_layer):
                                    print(f"    [{idx}] {type(submodule).__name__}")

                        return True
                    else:
                        print("✗ model.decoder.net.attn_layers.layers NOT FOUND")
                else:
                    print("✗ model.decoder.net.attn_layers NOT FOUND")
            else:
                print("✗ model.decoder.net NOT FOUND")
        else:
            print("✗ model.decoder NOT FOUND")

        return False

    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug model structure")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Model checkpoint path (default: best_model.pt)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MODEL STRUCTURE DEBUGGER")
    print("=" * 80)

    # Load model configuration
    train_args_file = config.MODEL_DIR / "train-args.json"
    if not train_args_file.exists():
        print(f"❌ Training args not found: {train_args_file}")
        return

    train_args = utils.load_json(train_args_file)
    print(f"✓ Loaded training args from: {train_args_file}")

    # Load encoding
    encoding_file = config.NOTES_DIR / "encoding.json"
    encoding = representation.load_encoding(encoding_file)
    print(f"✓ Loaded encoding from: {encoding_file}")

    # Create model
    print("\nCreating model...")
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
    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    print(f"✓ Loaded checkpoint from: {checkpoint_path}")

    model.eval()

    # Inspect structure
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE INSPECTION")
    print("=" * 80)
    inspect_model_structure(model, "model", max_depth=4)

    # Verify hook registration path
    verify_hook_registration_path(model)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("To register hooks on transformer layers, use:")
    print("  1. Get decoder: decoder_wrapper = model.decoder")
    print("  2. Get net: transformer = decoder_wrapper.net")
    print("  3. Get attn_layers: attn_layers = transformer.attn_layers")
    print("  4. Register hooks: attn_layers.layers[i].register_forward_hook(...)")
    print()


if __name__ == "__main__":
    main()
