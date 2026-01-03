#!/usr/bin/env python3
"""Patch manifest.json to fix WAV path prefixes."""

import json
import pathlib

def main():
    manifest_path = pathlib.Path("flamingo_eval/manifest_original.json")
    
    if not manifest_path.exists():
        print("❌ Original manifest backup not found!")
        print("Please restore it from git or re-run prepare_audios.py")
        return 1
    
    # Load original manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print(f"Loaded {len(manifest)} entries from backup")
    
    # Fix all paths: prepend "flamingo_eval/" if not already there
    fixed = 0
    for entry in manifest:
        old_path = entry["wav_path"]
        
        # Fix wav_path
        if not old_path.startswith("flamingo_eval/"):
            entry["wav_path"] = f"flamingo_eval/{old_path}"
            fixed += 1
    
    print(f"Fixed {fixed} wav_path entries")
    
    # Verify files exist now
    valid = 0
    missing = 0
    for entry in manifest:
        if pathlib.Path(entry["wav_path"]).exists():
            valid += 1
        else:
            missing += 1
    
    print(f"\nAfter fix:")
    print(f"  ✅ Valid WAVs: {valid}")
    print(f"  ❌ Missing WAVs: {missing}")
    
    # Save fixed manifest
    output_path = pathlib.Path("flamingo_eval/manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📝 Saved fixed manifest to: {output_path}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
