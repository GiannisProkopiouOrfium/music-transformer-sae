#!/usr/bin/env python3
"""
Convert generated tensor sequences to audio files for analysis.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import json
from typing import Dict, Any
import representation


def tensor_to_notes(sequence_tensor: torch.Tensor, encoding: Dict[str, Any]) -> list:
    """
    Convert tensor sequence to note representation.

    Args:
        sequence_tensor: Generated sequence tensor [seq_len, 6]
        encoding: Musical encoding dictionary

    Returns:
        List of note events
    """
    # Remove batch dimension if present
    if len(sequence_tensor.shape) == 3:
        sequence_tensor = sequence_tensor.squeeze(0)

    sequence = sequence_tensor.cpu().numpy()
    notes = []

    current_time = 0.0
    current_beat = 0

    # Dimension indices from your encoding
    type_dim = 0
    beat_dim = 1
    position_dim = 2
    pitch_dim = 3
    duration_dim = 4
    instrument_dim = 5

    # Type codes
    note_type_code = encoding["type_code_map"]["note"]
    instrument_type_code = encoding["type_code_map"]["instrument"]

    for i, token in enumerate(sequence):
        token_type = int(token[type_dim])

        if token_type == note_type_code:
            # Extract note information
            beat = int(token[beat_dim])
            position = int(token[position_dim])
            pitch = int(token[pitch_dim])
            duration = int(token[duration_dim])
            instrument = int(token[instrument_dim])

            # Convert to time (simple conversion)
            # Assuming 4/4 time, 120 BPM
            beats_per_second = 2.0  # 120 BPM = 2 beats per second
            note_time = beat / beats_per_second
            note_duration = duration / 16.0  # Duration in fractions of a beat

            # Convert pitch (assuming MIDI pitch)
            midi_pitch = pitch + 21  # Offset to reasonable MIDI range

            notes.append(
                {
                    "time": note_time,
                    "duration": note_duration,
                    "pitch": midi_pitch,
                    "velocity": 80,  # Default velocity
                    "instrument": instrument,
                }
            )

    return notes


def notes_to_audio(
    notes: list, sample_rate: int = 22050, duration: float = 30.0
) -> np.ndarray:
    """
    Convert notes to audio using simple synthesis.

    Args:
        notes: List of note dictionaries
        sample_rate: Audio sample rate
        duration: Total duration in seconds

    Returns:
        Audio array
    """
    # Create audio buffer
    audio_length = int(duration * sample_rate)
    audio = np.zeros(audio_length)

    for note in notes:
        start_sample = int(note["time"] * sample_rate)
        note_duration_samples = int(note["duration"] * sample_rate)
        end_sample = min(start_sample + note_duration_samples, audio_length)

        if start_sample >= audio_length:
            continue

        # Simple sine wave synthesis
        frequency = 440.0 * (2.0 ** ((note["pitch"] - 69) / 12.0))  # A4 = 440Hz
        t = np.linspace(0, note["duration"], end_sample - start_sample)

        # Generate sine wave with envelope
        envelope = np.exp(-t * 3)  # Exponential decay
        wave = envelope * np.sin(2 * np.pi * frequency * t) * (note["velocity"] / 127.0)

        # Add to audio buffer
        if len(wave) > 0:
            audio[start_sample:end_sample] += (
                wave * 0.3
            )  # Scale down to prevent clipping

    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8

    return audio


def convert_sequence_to_audio(
    tensor_path: str,
    encoding: Dict[str, Any],
    output_path: str,
    sample_rate: int = 22050,
    duration: float = 30.0,
) -> str:
    """
    Convert a single tensor sequence to audio.

    Args:
        tensor_path: Path to .pt file containing generated sequence
        encoding: Musical encoding dictionary
        output_path: Output audio file path
        sample_rate: Audio sample rate
        duration: Audio duration in seconds

    Returns:
        Path to generated audio file
    """
    print(f"   Converting {Path(tensor_path).name}...")

    # Load tensor
    data = torch.load(tensor_path, map_location="cpu")
    print(f"     Data keys: {list(data.keys())}")

    # Get sequences from the correct key
    if "generated" in data:
        # New format (like test_feature_interventions.py)
        sequence = data["generated"]
        print(f"     Using 'generated' key with shape: {sequence.shape}")
    else:
        print("     Error: No 'generated' key found in data!")
        return output_path

    # Convert to notes
    notes = tensor_to_notes(sequence, encoding)
    print(f"     Extracted {len(notes)} notes")

    # Convert to audio
    audio = notes_to_audio(notes, sample_rate, duration)

    # Save audio
    sf.write(output_path, audio, sample_rate)
    print(f"     Saved audio: {output_path}")

    return output_path


def convert_all_sequences(
    results_dir: str = "feature_182_test",
    feature_id: str = "182",
    sample_rate: int = 22050,
    duration: float = 30.0,
) -> Dict[str, str]:
    """
    Convert all generated sequences to audio files.

    Args:
        results_dir: Directory containing generated sequences
        feature_id: Feature ID that was tested
        sample_rate: Audio sample rate
        duration: Audio duration in seconds

    Returns:
        Dictionary mapping sequence name to audio file path
    """
    print(f"🎵 Converting Feature {feature_id} sequences to audio...")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    # Load encoding
    encoding_path = "data/sod/processed/notes/encoding.json"
    encoding = representation.load_encoding(encoding_path)
    if encoding is None:
        raise ValueError(f"Could not load encoding from {encoding_path}")

    # Find sequence files
    sequence_files = {
        "suppress_strong": f"feature_{feature_id}_strength_minus2_0.pt",
        "suppress_weak": f"feature_{feature_id}_strength_minus1_0.pt",
        "baseline": f"feature_{feature_id}_strength_plus0_0.pt",
        "promote_weak": f"feature_{feature_id}_strength_plus1_0.pt",
        "promote_strong": f"feature_{feature_id}_strength_plus2_0.pt",
    }

    audio_files = {}

    for name, filename in sequence_files.items():
        tensor_path = results_path / filename

        if not tensor_path.exists():
            print(f"   ⚠️  Skipping {name}: {filename} not found")
            continue

        # Output audio path
        audio_filename = f"feature_{feature_id}_{name}.wav"
        audio_path = results_path / audio_filename

        # Convert to audio
        try:
            convert_sequence_to_audio(
                str(tensor_path), encoding, str(audio_path), sample_rate, duration
            )
            audio_files[name] = str(audio_path)

        except Exception as e:
            print(f"   ❌ Error converting {name}: {e}")

    print(f"✅ Converted {len(audio_files)} sequences to audio")
    return audio_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert generated sequences to audio")
    parser.add_argument(
        "--results-dir",
        default="feature_182_test",
        help="Directory containing generated sequences",
    )
    parser.add_argument(
        "--feature-id", type=str, default="182", help="Feature ID that was tested"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Audio sample rate"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Audio duration in seconds"
    )

    args = parser.parse_args()

    # Convert sequences
    audio_files = convert_all_sequences(
        args.results_dir, args.feature_id, args.sample_rate, args.duration
    )

    # Save audio file list
    audio_list_path = Path(args.results_dir) / "audio_files.json"
    with open(audio_list_path, "w") as f:
        json.dump(audio_files, f, indent=2)

    print(f"\n📁 Audio file list saved to: {audio_list_path}")
    print(f"🎧 Ready for analysis with Audio Flamingo!")


if __name__ == "__main__":
    main()
