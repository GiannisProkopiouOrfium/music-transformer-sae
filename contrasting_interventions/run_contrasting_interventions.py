#!/usr/bin/env python3
"""
Run Interventions on Contrasting Songs

Takes contrasting songs identified by find_contrasting_songs.py and applies
interventions with conditioning from the song prefix, creating dramatic changes.

Usage:
    # Single test run (1 song, 1 feature, all strengths)
    python run_contrasting_interventions.py \
        --layer 3 \
        --feature 182 \
        --contrasting-songs contrasting_songs_results/contrasting_songs_layer3_top20.txt \
        --data-dir data/sod/processed/notes \
        --output-dir contrasting_intervention_results \
        --num-songs 1 \
        --conditioning-length 4

    # Full batch run
    python run_contrasting_interventions.py \
        --layer 3 \
        --features 182 855 997 \
        --contrasting-songs contrasting_songs_results/contrasting_songs_layer3_top20.txt \
        --data-dir data/sod/processed/notes \
        --output-dir contrasting_intervention_results \
        --num-songs 10 \
        --conditioning-length 4 \
        --addition-strengths -2.0 -1.0 0.0 1.0 2.0
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "mmt"))
sys.path.insert(0, str(parent_dir / "baseline"))

import torch
import muspy
from mmt import representation


class ContrastingInterventionRunner:
    """Run interventions on contrasting songs."""

    def __init__(
        self,
        layer: int,
        feature_ids: List[int],
        contrasting_songs_file: Path,
        data_dir: Path,
        output_dir: Path,
        conditioning_length: int = 4,
        seq_len: int = 512,
        addition_strengths: List[float] = None,
        temperature: float = 0.1,
        noise_scale: float = 1.2,
        num_songs: int = 1,
    ):
        self.layer = layer
        self.feature_ids = feature_ids
        self.contrasting_songs_file = contrasting_songs_file
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.conditioning_length = conditioning_length
        self.seq_len = seq_len
        self.addition_strengths = addition_strengths or [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.num_songs = num_songs

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(output_dir / "interventions.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Load contrasting songs
        self.contrasting_songs = self._load_contrasting_songs()

        # Load encoding for JSON->PT conversion if needed
        self.encoding = None

    def _load_encoding(self):
        """Load encoding for JSON to PT conversion."""
        if self.encoding is None:
            encoding_path = Path("data/sod/processed/notes/encoding.json")
            if not encoding_path.exists():
                # Try relative to parent directory
                encoding_path = (
                    Path(__file__).parent.parent
                    / "data/sod/processed/notes/encoding.json"
                )

            if encoding_path.exists():
                self.encoding = representation.load_encoding(encoding_path)
                self.logger.info(f"Loaded encoding from: {encoding_path}")
            else:
                self.logger.warning(f"Encoding not found at {encoding_path}")
        return self.encoding

    def _convert_json_to_pt(self, json_path: Path, pt_path: Path) -> bool:
        """Convert JSON file to PT file on-the-fly."""
        try:
            encoding = self._load_encoding()
            if encoding is None:
                self.logger.error("Cannot convert JSON to PT: encoding not loaded")
                return False

            self.logger.info(f"   Converting JSON to PT: {json_path.name}")

            # Load MusPy JSON
            music = muspy.load(str(json_path))

            # Encode to tokens
            tokens = representation.encode(music, encoding)

            # Convert to tensor and save
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)

            # Save as .pt file
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens_tensor, pt_path)

            self.logger.info(
                f"   ✅ Converted: {json_path.name} -> {pt_path.name} (shape: {tokens_tensor.shape})"
            )
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Failed to convert {json_path.name}: {e}")
            return False

    def _load_contrasting_songs(self) -> List[str]:
        """Load the list of contrasting song names."""
        try:
            with open(self.contrasting_songs_file, "r") as f:
                songs = [line.strip() for line in f if line.strip()]
            self.logger.info(
                f"Loaded {len(songs)} contrasting songs from {self.contrasting_songs_file}"
            )
            return songs[: self.num_songs]  # Limit to requested number
        except Exception as e:
            self.logger.error(f"Failed to load contrasting songs: {e}")
            return []

    def extract_limuf_if_needed(self, feature_id: int) -> Path:
        """Extract LiMuF for feature if not already extracted."""

        # Check if LiMuF already exists
        limuf_dir = (
            self.output_dir.parent
            / "extractions"
            / f"layer{self.layer}"
            / f"limufs_layer{self.layer}_sae_columns"
        )
        limuf_path = limuf_dir / "limufs.pt"

        if limuf_path.exists():
            self.logger.info(f"✅ LiMuF already exists: {limuf_path}")
            return limuf_path

        # Extract LiMuF
        self.logger.info(f"🔄 Extracting LiMuF for Feature {feature_id}...")

        extraction_output_dir = (
            self.output_dir.parent / "extractions" / f"layer{self.layer}"
        )
        extraction_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "extract_limuf_sae_column.py",
            "--layer",
            str(self.layer),
            "--feature-id",
            str(feature_id),
            "--output-dir",
            str(extraction_output_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=Path.cwd().parent
            )
            self.logger.info(f"✅ LiMuF extraction completed for Feature {feature_id}")
            return limuf_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ LiMuF extraction failed: {e.stderr}")
            raise

    def run_intervention(
        self,
        feature_id: int,
        limuf_path: Path,
        song_name: str,
        addition_strengths: List[float],
    ) -> bool:
        """Run interventions for all strengths at once (generates baseline only once)."""

        intervention_name = f"feature{feature_id}_song{song_name}"
        intervention_output_dir = (
            self.output_dir
            / f"layer{self.layer}"
            / f"feature{feature_id}"
            / intervention_name
        )

        # Check if already exists (check for all_wavs folder)
        all_wavs_dir = intervention_output_dir / "all_wavs"
        if all_wavs_dir.exists() and any(all_wavs_dir.glob("*.wav")):
            self.logger.info(f"⏭️  Skipping {intervention_name} (already exists)")
            return True

        self.logger.info(
            f"🎵 Running interventions: Feature {feature_id}, Song {song_name}, Strengths {addition_strengths}"
        )

        # Create custom names file for this specific song
        temp_names_file = self.output_dir / f"temp_song_{song_name}.txt"
        with open(temp_names_file, "w") as f:
            f.write(f"{song_name}\n")

        # Build path to song .pt file
        # Try direct path first, then check Kunstderfuge subfolder
        song_pt_path = self.data_dir / f"{song_name}.pt"

        if not song_pt_path.exists():
            # Try Kunstderfuge subfolder
            song_pt_path = (
                self.output_dir / "pt_convert" / "Kunstderfuge" / f"{song_name}.pt"
            )

        # If .pt doesn't exist, try to convert from JSON
        if not song_pt_path.exists():
            self.logger.info(f"   .pt file not found, looking for JSON...")

            # Try finding JSON file
            json_path = (
                Path("data/sod/processed/json/Kunstderfuge") / f"{song_name}.json"
            )
            if not json_path.exists():
                json_path = (
                    Path(__file__).parent.parent
                    / "data/sod/processed/json/Kunstderfuge"
                    / f"{song_name}.json"
                )

            if json_path.exists():
                # Convert JSON to PT
                song_pt_path = (
                    self.output_dir / "pt_convert" / "Kunstderfuge" / f"{song_name}.pt"
                )
                if not self._convert_json_to_pt(json_path, song_pt_path):
                    if temp_names_file.exists():
                        temp_names_file.unlink()
                    return False
            else:
                self.logger.error(
                    f"❌ Neither .pt nor .json file found for: {song_name}"
                )
                self.logger.error(f"   Tried .pt: {song_pt_path}")
                self.logger.error(f"   Tried .json: {json_path}")
                if temp_names_file.exists():
                    temp_names_file.unlink()
                return False

        self.logger.info(f"   Using song file: {song_pt_path}")

        # Convert to absolute paths for subprocess
        song_pt_path_abs = song_pt_path.resolve()
        intervention_output_dir_abs = intervention_output_dir.resolve()

        # Convert strengths list to comma-separated string
        strengths_str = ",".join(str(s) for s in addition_strengths)

        cmd = [
            "python",
            "contrasting_interventions/song_conditioned_interventions.py",
            "--song-path",
            str(song_pt_path_abs),
            "--feature-limuf-path",
            str(limuf_path),
            "--output-dir",
            str(intervention_output_dir_abs),
            "--intervention-layer",
            str(self.layer),
            "--addition-strengths",
            strengths_str,  # All strengths at once
            "--conditioning-length",
            str(self.conditioning_length),
            "--seq-len",
            str(self.seq_len),
            "--temperature",
            str(self.temperature),
            "--noise-scale",
            str(self.noise_scale),
            "--generation-seed",
            str(42),  # Fixed seed for reproducibility
        ]

        try:
            # Run without capturing output so we can see real-time logs
            subprocess.run(cmd, check=True, cwd=Path.cwd().parent)
            self.logger.info(f"✅ Intervention completed: {intervention_name}")

            # Clean up temp file
            temp_names_file.unlink()

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Intervention failed for {intervention_name}")
            self.logger.error(f"Return code: {e.returncode}")

            # Clean up temp file
            if temp_names_file.exists():
                temp_names_file.unlink()

            return False

    def create_comparison_summary(self, results: Dict):
        """Create a summary document for easy A/B comparison."""

        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "layer": self.layer,
                "features": self.feature_ids,
                "contrasting_songs": self.contrasting_songs,
                "conditioning_length": self.conditioning_length,
                "seq_len": self.seq_len,
                "addition_strengths": self.addition_strengths,
            },
            "interventions": results,
            "listening_guide": self._generate_listening_guide(),
        }

        summary_file = self.output_dir / f"intervention_summary_layer{self.layer}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"✅ Summary saved to {summary_file}")

        # Skip HTML generation - just need WAV files
        # self._create_listening_guide_html(results)

    def _generate_listening_guide(self) -> str:
        """Generate listening instructions."""
        return """
LISTENING GUIDE FOR CONTRASTING INTERVENTIONS
==============================================

These interventions were applied to songs that naturally LACK the target feature properties.
This creates dramatic, obvious changes that demonstrate feature control.

How to listen:
1. Start with the BASELINE (strength 0.0) - this is the contrasting song with natural generation
2. Listen to ABLATION (if available) - feature completely removed
3. Listen to NEGATIVE additions (-2.0, -1.0) - feature pushed in opposite direction
4. Listen to POSITIVE additions (+1.0, +2.0) - feature strongly added

Expected effects:
- Baseline: Should naturally lack the target feature characteristics
- Positive additions: Should add the feature dramatically (e.g., making irregular rhythm steady)
- Negative additions: Should further emphasize the absence or add opposite characteristics

Compare:
- Baseline vs +2.0: Should show maximum contrast
- -2.0 vs +2.0: Should show full range of feature control
- Sequential listening: -2.0 → -1.0 → 0.0 → +1.0 → +2.0 shows gradual transition
"""

    def _create_listening_guide_html(self, results: Dict):
        """Create an HTML listening guide with audio players."""

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Contrasting Interventions - Listening Guide</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .feature-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .song-section { margin: 15px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #4CAF50; }
        .intervention { margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }
        .strength { font-weight: bold; color: #2196F3; }
        audio { width: 100%; margin-top: 5px; }
        .baseline { border-left: 4px solid #FFC107; }
        .positive { border-left: 4px solid #4CAF50; }
        .negative { border-left: 4px solid #F44336; }
        .ablation { border-left: 4px solid #9E9E9E; }
    </style>
</head>
<body>
    <h1>🎵 Contrasting Interventions - Listening Guide</h1>
    <p><strong>Layer {layer}</strong> | Conditioning: {cond_len} beats | Sequence: {seq_len} tokens</p>

    <div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h3>📖 How to Listen:</h3>
        <ol>
            <li><strong>Baseline (0.0)</strong>: Natural generation from contrasting song - lacks target feature</li>
            <li><strong>Negative (-2.0, -1.0)</strong>: Feature pushed away or opposite direction</li>
            <li><strong>Positive (+1.0, +2.0)</strong>: Feature strongly added - expect dramatic change!</li>
            <li><strong>Ablation</strong>: Feature completely removed (if different from baseline)</li>
        </ol>
        <p><strong>💡 Tip:</strong> Compare baseline vs +2.0 for maximum contrast effect!</p>
    </div>
""".format(
            layer=self.layer, cond_len=self.conditioning_length, seq_len=self.seq_len
        )

        # Add sections for each feature
        for feature_id in self.feature_ids:
            feature_results = results.get(feature_id, {})
            html_content += f"""
    <div class="feature-section">
        <h2>Feature {feature_id}</h2>
"""

            # Group by song
            for song_name in self.contrasting_songs:
                song_results = feature_results.get(song_name, {})

                if not song_results:
                    continue

                html_content += f"""
        <div class="song-section">
            <h3>🎼 Song: {song_name}</h3>
"""

                # Sort by strength for logical listening order
                strengths_sorted = sorted(
                    song_results.keys(),
                    key=lambda x: float(x) if x != "ablation" else -999,
                )

                for strength in strengths_sorted:
                    intervention_data = song_results[strength]

                    # Determine class for styling
                    if strength == "ablation":
                        css_class = "ablation"
                        label = "ABLATION"
                    elif float(strength) == 0.0:
                        css_class = "baseline"
                        label = "BASELINE"
                    elif float(strength) > 0:
                        css_class = "positive"
                        label = f"POSITIVE {strength}"
                    else:
                        css_class = "negative"
                        label = f"NEGATIVE {strength}"

                    # Get relative path to audio file
                    audio_path = intervention_data.get("audio_path", "")
                    if audio_path:
                        relative_path = Path(audio_path).relative_to(self.output_dir)

                        html_content += f"""
            <div class="intervention {css_class}">
                <span class="strength">{label}</span>
                <audio controls>
                    <source src="{relative_path}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
"""

                html_content += """
        </div>
"""

            html_content += """
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Save HTML file
        html_file = self.output_dir / f"listening_guide_layer{self.layer}.html"
        with open(html_file, "w") as f:
            f.write(html_content)

        self.logger.info(f"✅ HTML listening guide created: {html_file}")
        self.logger.info(f"   Open in browser: file://{html_file.absolute()}")

    def run(self) -> Dict:
        """Run the complete intervention pipeline."""

        self.logger.info("🚀 Starting Contrasting Intervention Pipeline")
        self.logger.info("=" * 80)
        self.logger.info(f"Layer: {self.layer}")
        self.logger.info(f"Features: {self.feature_ids}")
        self.logger.info(f"Songs: {self.contrasting_songs[:5]}...")
        self.logger.info(f"Conditioning length: {self.conditioning_length} beats")
        self.logger.info(f"Addition strengths: {self.addition_strengths}")

        results = {}

        # Process each feature
        for feature_id in self.feature_ids:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing Feature {feature_id}")
            self.logger.info(f"{'='*80}")

            # Extract LiMuF if needed
            try:
                limuf_path = self.extract_limuf_if_needed(feature_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to extract LiMuF for feature {feature_id}: {e}"
                )
                continue

            feature_results = {}

            # Process each song (run all strengths at once)
            for song_name in self.contrasting_songs:
                # Run intervention with all strengths at once (generates baseline only once!)
                success = self.run_intervention(
                    feature_id, limuf_path, song_name, self.addition_strengths
                )

                if success:
                    # Find all generated audio files in all_wavs folder
                    intervention_name = f"feature{feature_id}_song{song_name}"
                    intervention_dir = (
                        self.output_dir
                        / f"layer{self.layer}"
                        / f"feature{feature_id}"
                        / intervention_name
                    )

                    all_wavs_dir = intervention_dir / "all_wavs"
                    song_results = {}

                    if all_wavs_dir.exists():
                        # Collect all WAV files
                        for wav_file in all_wavs_dir.glob("*.wav"):
                            # Parse filename to get condition (baseline, add_+1.0, etc.)
                            condition = wav_file.stem.replace(f"{song_name}_", "")
                            song_results[condition] = {
                                "success": True,
                                "audio_path": str(wav_file),
                                "intervention_dir": str(intervention_dir),
                            }

                    feature_results[song_name] = song_results

            results[feature_id] = feature_results

        # Create summary and listening guide
        self.create_comparison_summary(results)

        # Print final summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 CONTRASTING INTERVENTION PIPELINE COMPLETE!")
        self.logger.info("=" * 80)

        total_interventions = sum(
            len(song_results)
            for feature_results in results.values()
            for song_results in feature_results.values()
        )
        self.logger.info(f"Total interventions completed: {total_interventions}")
        self.logger.info(f"Output directory: {self.output_dir}")

        self.logger.info("\n🎧 READY TO LISTEN!")
        self.logger.info(f"WAV files saved in: {self.output_dir}")
        self.logger.info(
            "Find audio files organized by: layer/feature/intervention_name/*.wav"
        )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run interventions on contrasting songs for dramatic effects"
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Target layer (1, 3, or 5)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=int,
        required=True,
        help="Target feature IDs (e.g., 182 855 997)",
    )
    parser.add_argument(
        "--contrasting-songs",
        type=Path,
        required=True,
        help="Path to contrasting songs text file from find_contrasting_songs.py",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/sod/processed/notes"),
        help="Directory containing note data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("contrasting_intervention_results"),
        help="Output directory for intervention results",
    )
    parser.add_argument(
        "--num-songs",
        type=int,
        default=1,
        help="Number of songs to process (1 for testing, more for batch)",
    )
    parser.add_argument(
        "--conditioning-length",
        type=int,
        default=3,
        help="Number of TOKENS to use for conditioning prefix (default: 3, use 2-3 for short conditioning)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length to generate"
    )
    parser.add_argument(
        "--addition-strengths",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Intervention strengths to test",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument("--noise-scale", type=float, default=1.2, help="Noise scale")

    args = parser.parse_args()

    # Validate
    if not args.contrasting_songs.exists():
        print(f"❌ Error: Contrasting songs file not found: {args.contrasting_songs}")
        sys.exit(1)

    # Create runner
    runner = ContrastingInterventionRunner(
        layer=args.layer,
        feature_ids=args.features,
        contrasting_songs_file=args.contrasting_songs,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        conditioning_length=args.conditioning_length,
        seq_len=args.seq_len,
        addition_strengths=args.addition_strengths,
        temperature=args.temperature,
        noise_scale=args.noise_scale,
        num_songs=args.num_songs,
    )

    # Run
    results = runner.run()

    if results:
        print(
            "\n✅ Pipeline complete! Check the HTML listening guide to hear the results."
        )
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
