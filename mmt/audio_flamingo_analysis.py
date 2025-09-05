#!/usr/bin/env python3
"""
Audio Flamingo analysis for verifying LiMuF interventions.
"""

import torch
import librosa
import json
from pathlib import Path
from typing import Dict, Any
import argparse
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class AudioFlamingoAnalyzer:
    """Analyze audio using Audio Flamingo for rhythmic properties."""

    def __init__(self, model_name: str = "nvidia/audio-flamingo-2"):
        """
        Initialize Audio Flamingo analyzer.

        Args:
            model_name: HuggingFace model name for Audio Flamingo
        """
        print(f"🤖 Loading Audio Flamingo model: {model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("✅ Audio Flamingo model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading Audio Flamingo: {e}")
            print("   Falling back to simple audio analysis...")
            self.processor = None
            self.model = None

    def analyze_rhythm(self, audio_path: str, question: str) -> str:
        """
        Analyze rhythmic properties of audio using Audio Flamingo.

        Args:
            audio_path: Path to audio file
            question: Question to ask about the audio

        Returns:
            Response from Audio Flamingo
        """
        if self.model is None:
            return self._fallback_analysis(audio_path, question)

        try:
            # Load audio at 16kHz (Audio Flamingo requirement)
            audio, sr = librosa.load(audio_path, sr=16000)

            # Ensure audio is not too long (limit to 30 seconds)
            max_samples = 30 * 16000
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Prepare inputs
            inputs = self.processor(audio=audio, text=question, return_tensors="pt").to(
                self.model.device
            )

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"   ⚠️  Audio Flamingo error: {e}")
            return self._fallback_analysis(audio_path, question)

    def _fallback_analysis(self, audio_path: str, question: str) -> str:
        """
        Fallback analysis using basic audio features.

        Args:
            audio_path: Path to audio file
            question: Question about the audio

        Returns:
            Simple analysis response
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)

            # Compute basic features
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Calculate rhythm regularity
            if len(onset_times) > 1:
                intervals = onset_times[1:] - onset_times[:-1]
                avg_interval = intervals.mean()
                interval_std = intervals.std()
                regularity_score = 1.0 / (interval_std + 0.1)  # Higher = more regular

                if "regular" in question.lower() or "steady" in question.lower():
                    if regularity_score > 2.0:
                        return f"The rhythm appears quite regular with consistent timing (regularity score: {regularity_score:.1f}/10)"
                    elif regularity_score > 1.0:
                        return f"The rhythm has moderate regularity (regularity score: {regularity_score:.1f}/10)"
                    else:
                        return f"The rhythm appears irregular with varied timing (regularity score: {regularity_score:.1f}/10)"

                elif "describe" in question.lower():
                    tempo = 60.0 / avg_interval if avg_interval > 0 else 0
                    return f"The music has {len(onset_times)} note onsets over {len(audio)/sr:.1f}s, estimated tempo ~{tempo:.0f} BPM, timing variability: {interval_std:.2f}s"

                elif (
                    "consistent" in question.lower()
                    or "predictable" in question.lower()
                ):
                    if interval_std < 0.2:
                        return (
                            "Yes, the beat pattern is quite consistent and predictable"
                        )
                    elif interval_std < 0.5:
                        return "The beat pattern has moderate consistency"
                    else:
                        return (
                            "No, the beat pattern is quite irregular and unpredictable"
                        )

            return f"Basic analysis: {len(onset_times)} onsets detected over {len(audio)/sr:.1f} seconds"

        except Exception as e:
            return f"Analysis error: {e}"

    def analyze_sequences(
        self, audio_files: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all sequences for rhythmic properties.

        Args:
            audio_files: Dictionary mapping sequence name to audio file path

        Returns:
            Analysis results for each sequence
        """
        # Questions to ask about rhythm and timing
        questions = [
            "How regular and steady is the rhythm in this music? Rate the regularity from 1 to 10 where 10 is perfectly metronomic.",
            "Does this music have a consistent, predictable beat pattern?",
            "Describe the rhythmic characteristics and timing patterns in this musical piece.",
            "Is the timing in this music irregular and unpredictable, or steady and regular?",
            "How would you describe the pulse consistency and tempo stability in this music?",
        ]

        results = {}

        print(f"\n🔍 Analyzing {len(audio_files)} audio sequences...")

        for name, audio_path in audio_files.items():
            print(f"\n🎵 Analyzing '{name}'...")

            if not Path(audio_path).exists():
                print(f"   ❌ Audio file not found: {audio_path}")
                continue

            results[name] = {"audio_file": audio_path, "responses": {}}

            for i, question in enumerate(questions, 1):
                print(f"   Q{i}: {question[:50]}...")

                try:
                    response = self.analyze_rhythm(audio_path, question)
                    results[name]["responses"][f"question_{i}"] = {
                        "question": question,
                        "response": response,
                    }
                    print(
                        f"        → {response[:100]}{'...' if len(response) > 100 else ''}"
                    )

                except Exception as e:
                    error_msg = f"Error: {e}"
                    results[name]["responses"][f"question_{i}"] = {
                        "question": question,
                        "response": error_msg,
                    }
                    print(f"        → {error_msg}")

        return results


def extract_regularity_scores(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract numerical regularity scores from analysis results.

    Args:
        results: Analysis results from AudioFlamingoAnalyzer

    Returns:
        Dictionary mapping sequence name to regularity score
    """
    scores = {}

    for name, data in results.items():
        # Look for numerical scores in responses
        score = 0.0

        # Check question 1 (regularity rating)
        if "question_1" in data["responses"]:
            response = data["responses"]["question_1"]["response"].lower()

            # Try to extract numerical score
            import re

            numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
            if numbers:
                try:
                    score = float(numbers[0])
                    if score > 10:  # If it's not a 1-10 scale, normalize
                        score = min(score / 10, 10.0)
                except ValueError:
                    pass

            # Fallback: look for descriptive words
            if score == 0.0:
                if any(
                    word in response
                    for word in [
                        "very regular",
                        "highly regular",
                        "perfectly",
                        "metronomic",
                    ]
                ):
                    score = 8.5
                elif any(
                    word in response
                    for word in ["quite regular", "fairly regular", "regular"]
                ):
                    score = 6.5
                elif any(word in response for word in ["moderate", "somewhat"]):
                    score = 5.0
                elif any(
                    word in response
                    for word in ["irregular", "unpredictable", "varied"]
                ):
                    score = 3.0
                else:
                    score = 5.0  # Default

        scores[name] = score

    return scores


def verify_intervention_effectiveness(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify if interventions show expected patterns.

    Args:
        results: Analysis results from AudioFlamingoAnalyzer

    Returns:
        Verification summary
    """
    print("\n📊 VERIFYING INTERVENTION EFFECTIVENESS")
    print("=" * 50)

    # Extract regularity scores
    scores = extract_regularity_scores(results)

    print("\n🎯 Regularity Scores:")
    sequence_order = [
        "suppress_strong",
        "suppress_weak",
        "baseline",
        "promote_weak",
        "promote_strong",
    ]

    for name in sequence_order:
        if name in scores:
            score = scores[name]
            print(f"   {name:15s}: {score:.1f}/10")

    # Check for expected progression
    verification = {"scores": scores, "tests": {}, "overall_success": False}

    # Test 1: Promoted > Baseline > Suppressed
    if all(
        name in scores for name in ["suppress_strong", "baseline", "promote_strong"]
    ):
        promote_score = scores["promote_strong"]
        baseline_score = scores["baseline"]
        suppress_score = scores["suppress_strong"]

        progression_correct = promote_score > baseline_score > suppress_score
        verification["tests"]["basic_progression"] = {
            "expected": "promote > baseline > suppress",
            "actual": f"{promote_score:.1f} > {baseline_score:.1f} > {suppress_score:.1f}",
            "passed": progression_correct,
        }

        print(
            f"\n✅ Basic Progression Test: {'PASS' if progression_correct else 'FAIL'}"
        )
        print(f"   Expected: Promote > Baseline > Suppress")
        print(
            f"   Actual:   {promote_score:.1f} > {baseline_score:.1f} > {suppress_score:.1f}"
        )

    # Test 2: Strength effect (stronger interventions = bigger differences)
    if all(
        name in scores
        for name in [
            "promote_weak",
            "promote_strong",
            "suppress_weak",
            "suppress_strong",
        ]
    ):
        promote_effect = abs(scores["promote_strong"] - scores["baseline"]) > abs(
            scores["promote_weak"] - scores["baseline"]
        )
        suppress_effect = abs(scores["suppress_strong"] - scores["baseline"]) > abs(
            scores["suppress_weak"] - scores["baseline"]
        )

        strength_effect_correct = promote_effect and suppress_effect
        verification["tests"]["strength_effect"] = {
            "expected": "Stronger interventions have bigger effects",
            "promote_effect": promote_effect,
            "suppress_effect": suppress_effect,
            "passed": strength_effect_correct,
        }

        print(
            f"\n✅ Strength Effect Test: {'PASS' if strength_effect_correct else 'FAIL'}"
        )
        print(
            f"   Promote effect: {promote_effect} | Suppress effect: {suppress_effect}"
        )

    # Test 3: Minimum difference threshold
    if "baseline" in scores and "promote_strong" in scores:
        min_difference = 1.0  # Minimum expected difference
        actual_difference = abs(scores["promote_strong"] - scores["baseline"])
        difference_sufficient = actual_difference >= min_difference

        verification["tests"]["sufficient_difference"] = {
            "expected": f"Difference >= {min_difference}",
            "actual": actual_difference,
            "passed": difference_sufficient,
        }

        print(
            f"\n✅ Sufficient Difference Test: {'PASS' if difference_sufficient else 'FAIL'}"
        )
        print(f"   Expected difference: >= {min_difference:.1f}")
        print(f"   Actual difference:   {actual_difference:.1f}")

    # Overall success
    passed_tests = sum(test["passed"] for test in verification["tests"].values())
    total_tests = len(verification["tests"])
    verification["overall_success"] = passed_tests >= total_tests * 0.6  # 60% threshold

    print(
        f"\n🏆 OVERALL RESULT: {'SUCCESS' if verification['overall_success'] else 'NEEDS_INVESTIGATION'}"
    )
    print(f"   Passed: {passed_tests}/{total_tests} tests")

    if verification["overall_success"]:
        print("   ✅ LiMuF interventions appear to be working as expected!")
        print("   ✅ Feature 182 successfully controls steady pulse patterns")
    else:
        print("   ⚠️  Results suggest interventions may need investigation")
        print(
            "   ⚠️  Consider checking model loading, hook placement, or feature extraction"
        )

    return verification


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze generated audio with Audio Flamingo"
    )
    parser.add_argument(
        "--results-dir",
        default="feature_182_test",
        help="Directory containing generated sequences and audio files",
    )
    parser.add_argument(
        "--feature-id", type=int, default=182, help="Feature ID that was tested"
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/audio-flamingo-2",
        help="Audio Flamingo model name",
    )

    args = parser.parse_args()

    results_path = Path(args.results_dir)

    print(f"🔍 AUTOMATED VERIFICATION: Feature {args.feature_id} (Steady Pulse)")
    print("=" * 60)

    # Check for audio files
    audio_list_path = results_path / "audio_files.json"
    if audio_list_path.exists():
        with open(audio_list_path, "r") as f:
            audio_files = json.load(f)
        print(f"📁 Loaded audio file list: {len(audio_files)} files")
    else:
        print("❌ Audio files not found. Run convert_sequences_to_audio.py first")
        return

    # Initialize analyzer
    analyzer = AudioFlamingoAnalyzer(args.model_name)

    # Analyze sequences
    analysis_results = analyzer.analyze_sequences(audio_files)

    # Verify effectiveness
    verification = verify_intervention_effectiveness(analysis_results)

    # Save results
    output_file = (
        results_path / f"audio_flamingo_analysis_feature_{args.feature_id}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "analysis_results": analysis_results,
                "verification": verification,
                "feature_id": args.feature_id,
                "model_used": args.model_name,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
