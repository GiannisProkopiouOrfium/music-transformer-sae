#!/usr/bin/env python3
"""Re-run NPY to WAV conversion for missing files with verbose logging."""

import json
import pathlib
import sys
import numpy as np
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))

import representation

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# Try both encoding files
encoding_paths = [
    pathlib.Path("../data/sod/processed/notes/encoding.json"),
    pathlib.Path("../baseline/encoding_mmm.json"),
]

print("Testing encoding files...\n")

for enc_path in encoding_paths:
    print(f"=" * 70)
    print(f"Testing: {enc_path}")
    print(f"Exists: {enc_path.exists()}")

    if not enc_path.exists():
        print("❌ File not found\n")
        continue

    try:
        encoding = representation.load_encoding(enc_path)
        print(f"✅ Loaded successfully")
        print(f"Keys: {list(encoding.keys())[:10]}...")

        # Check for required keys
        required_keys = ["code_type_map", "type_code_map"]
        for key in required_keys:
            if key in encoding:
                print(f"  ✅ Has '{key}'")
            else:
                print(f"  ❌ Missing '{key}'")

        # Try to decode a sample NPY file
        test_npy = pathlib.Path(
            "flamingo_exp/dual/unconditional/low_long/sample_alpha_p-1.5_d0.5_num_0.npy"
        )

        if test_npy.exists():
            print(f"\nTesting decode with: {test_npy.name}")
            tokens = np.load(test_npy)
            print(f"  Token shape: {tokens.shape}, dtype: {tokens.dtype}")

            try:
                music = representation.decode(tokens, encoding)
                print(f"  ✅ Decode successful!")
                print(f"  Music tracks: {len(music.tracks)}")
                print(f"  Music resolution: {music.resolution}")

                # Try to write audio
                test_wav = pathlib.Path("flamingo_eval/test_decode.wav")
                test_wav.parent.mkdir(parents=True, exist_ok=True)
                music.write_audio(str(test_wav))
                print(f"  ✅ WAV written to: {test_wav}")

                # Clean up
                if test_wav.exists():
                    test_wav.unlink()
                    print(f"  🗑️  Cleaned up test file")

            except Exception as e:
                print(f"  ❌ Decode failed: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"\n⚠️  Test NPY not found: {test_npy}")

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        import traceback

        traceback.print_exc()

    print()

print("=" * 70)
print("\nConclusion:")
print("Use the encoding file that successfully decoded the test NPY.")
