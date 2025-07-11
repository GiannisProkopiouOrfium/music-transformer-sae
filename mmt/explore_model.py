import torch
import pathlib
import logging
import sys
import utils
import representation
import music_x_transformers

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def explore_model_structure():
    """Explore the detailed structure of the music transformer."""

    # Load minimal configuration
    out_dir = pathlib.Path("exp/sod/ape")
    in_dir = pathlib.Path("data/sod/processed/notes")

    # Load training args
    train_args = utils.load_json(out_dir / "train-args.json")

    # Load encoding
    encoding = representation.load_encoding(in_dir / "encoding.json")

    # Use CPU
    device = torch.device("cpu")

    # Create model
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

    # Load weights
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    print("=" * 60)
    print("DETAILED MODEL STRUCTURE")
    print("=" * 60)

    # Explore the decoder structure
    decoder = model.decoder
    print(f"Decoder type: {type(decoder)}")

    if hasattr(decoder, "net"):
        net = decoder.net
        print(f"Net type: {type(net)}")

        if hasattr(net, "attn_layers"):
            attn_layers = net.attn_layers
            print(f"Attention layers type: {type(attn_layers)}")

            if hasattr(attn_layers, "layers"):
                layers = attn_layers.layers
                print(f"Layers type: {type(layers)}")
                print(f"Number of layer groups: {len(layers)}")

                for i, layer_group in enumerate(layers):
                    print(f"\nLayer group {i}:")
                    print(f"  Type: {type(layer_group)}")

                    if hasattr(layer_group, "__len__"):
                        print(f"  Length: {len(layer_group)}")
                        for j, sublayer in enumerate(layer_group):
                            print(f"    Sublayer {j}: {type(sublayer)}")

                            # Print module names if available
                            if hasattr(sublayer, "_modules"):
                                for name, module in sublayer._modules.items():
                                    if module is not None:
                                        print(f"      {name}: {type(module)}")

    print("\n" + "=" * 60)
    print("TESTING HOOKS ON DIFFERENT LAYERS")
    print("=" * 60)

    # Test hooking different parts
    def test_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                shapes = [x.shape if hasattr(x, "shape") else type(x) for x in output]
                print(f"Hook {name}: tuple output with shapes {shapes}")
            else:
                shape = output.shape if hasattr(output, "shape") else type(output)
                print(f"Hook {name}: output shape {shape}")

        return hook_fn

    hooks = []

    # Hook some layers
    for i in [0, 3, 5]:  # Test a few different layers
        if i < len(model.decoder.net.attn_layers.layers):
            layer_group = model.decoder.net.attn_layers.layers[i]
            if isinstance(layer_group, torch.nn.ModuleList):
                for j, sublayer in enumerate(layer_group):
                    hook = sublayer.register_forward_hook(
                        test_hook(f"layer_{i}_sub_{j}")
                    )
                    hooks.append(hook)

    # Create a test input
    test_input = torch.zeros((1, 10, 6), dtype=torch.long, device=device)  # Small test
    test_mask = torch.ones((1, 10), dtype=torch.bool, device=device)

    print("\nRunning test forward pass...")
    with torch.no_grad():
        output = model(test_input, mask=test_mask)
        print(f"Final output shape: {output.shape}")

    # Remove hooks
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    explore_model_structure()
