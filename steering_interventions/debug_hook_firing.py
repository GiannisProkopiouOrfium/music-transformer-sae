"""Simple debug script to verify hooks are firing during generation."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils


def test_hook_firing():
    """Test if hooks fire during generation."""
    print("Loading model...")

    # Load model
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    device = torch.device("cpu")

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

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Model loaded")

    # Create a test hook
    hook_call_count = [0]

    def test_hook(module, input, output):
        hook_call_count[0] += 1
        if hook_call_count[0] <= 5:
            print(
                f"Hook called: count={hook_call_count[0]}, output shape={output[0].shape if isinstance(output, tuple) else output.shape}"
            )
        return output

    # Register hook on first layer
    layer_module = model.decoder.net.attn_layers.layers[0][1]
    handle = layer_module.register_forward_hook(test_hook)

    print(f"\nHook registered on: {type(layer_module).__name__}")
    print("Generating 20 tokens...\n")

    # Generate
    sos = encoding["type_code_map"]["start-of-song"]
    start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    start_tokens[:, 0, 0] = sos

    generated = model.generate(
        start_tokens,
        20,
        temperature=0.1,
        filter_logits_fn="top_k",
        filter_thres=0.9,
    )

    handle.remove()

    print(f"\nGeneration complete")
    print(f"Total hook calls: {hook_call_count[0]}")
    print(f"Generated shape: {generated.shape}")
    print(f"Expected ~20 calls for 20 tokens")

    if hook_call_count[0] > 0:
        print("\nSUCCESS: Hooks are firing during generation!")
    else:
        print("\nFAILURE: Hooks did not fire")


if __name__ == "__main__":
    test_hook_firing()
