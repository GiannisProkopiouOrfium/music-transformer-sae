#!/usr/bin/env python3
"""Force regenerate all missing WAV files from manifest."""

import json
import pathlib
import sys
import numpy as np
import logging
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
import representation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def convert_npy_to_wav(npy_path: pathlib.Path, wav_path: pathlib.Path, encoding: dict) -> bool:
    """Convert NPY to WAV with detailed error logging."""
    try:
        # Load tokens
        tokens = np.load(npy_path)
        
        # Decode to music
        music = representation.decode(tokens, encoding)
        
        # Create output directory
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write audio
        music.write_audio(str(wav_path))
        
        return True
        
    except Exception as e:
        logging.error(f"Failed {npy_path.name}: {e}")
        return False

def main():
    # Load manifest
    manifest_path = pathlib.Path("flamingo_eval/manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Load encoding
    encoding_path = pathlib.Path("../data/sod/processed/notes/encoding.json")
    logging.info(f"Loading encoding from {encoding_path}")
    encoding = representation.load_encoding(encoding_path)
    logging.info("✅ Encoding loaded")
    
    # Find missing WAVs
    missing = []
    for entry in manifest:
        wav_path = pathlib.Path(entry["wav_path"])
        if not wav_path.exists():
            missing.append(entry)
    
    logging.info(f"Found {len(missing)} missing WAV files out of {len(manifest)} total")
    
    if not missing:
        logging.info("✅ All WAV files already exist!")
        return 0
    
    # Regenerate missing WAVs
    success = 0
    failed = 0
    
    for entry in tqdm(missing, desc="Regenerating WAVs"):
        npy_path = pathlib.Path(entry.get("npy_path", ""))
        wav_path = pathlib.Path(entry["wav_path"])
        
        if not npy_path.exists():
            logging.error(f"NPY not found: {npy_path}")
            failed += 1
            continue
        
        if convert_npy_to_wav(npy_path, wav_path, encoding):
            success += 1
        else:
            failed += 1
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Regeneration complete:")
    logging.info(f"  ✅ Success: {success}")
    logging.info(f"  ❌ Failed: {failed}")
    logging.info(f"  Total: {success + failed}")
    
    # Verify
    valid_now = sum(1 for e in manifest if pathlib.Path(e["wav_path"]).exists())
    logging.info(f"\nValid WAV files now: {valid_now}/{len(manifest)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
