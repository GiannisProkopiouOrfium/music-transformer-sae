#!/usr/bin/env python3
"""
Batch LLM Evaluation for SAE Feature Interventions

Integrates with existing deterministic analysis pipeline to evaluate
interventions using OpenAI's LLM. Can run independently or compare with
deterministic results.

Usage:
    # Standalone LLM evaluation (unbiased)
    python run_llm_batch_evaluation.py batch_extractions_interventions/interventions \\
        --output-dir llm_evaluation_results
    
    # With deterministic comparison (shows both scores)
    python run_llm_batch_evaluation.py batch_extractions_interventions/interventions \\
        --output-dir llm_evaluation_results \\
        --compare-deterministic mmt_determ_results_integrated
    
    # Include full JSON (more tokens, better context)
    python run_llm_batch_evaluation.py batch_extractions_interventions/interventions \\
        --output-dir llm_evaluation_results \\
        --include-json

Requirements:
    pip install openai python-dotenv
    
Environment:
    OPENAI_API_KEY in .env file or environment variable
"""

import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
import sys
import re
import csv
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_text_evaluation import TextBasedLLMEvaluator, LLMEvaluationResult
from deterministic_analysis.midi_feature_extractors import MIDIFeatureExtractor
import baseline.representation_remi as representation

# Try to import MusPy
try:
    import muspy

    MUSPY_AVAILABLE = True
except ImportError:
    MUSPY_AVAILABLE = False
    logging.warning("MusPy not available. Install with: pip install muspy")


class BatchLLMEvaluator:
    """Batch evaluation of interventions using LLM."""

    def __init__(
        self,
        interventions_dir: str,
        output_dir: str,
        model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.3,
        include_json: bool = False,
        deterministic_dir: Optional[str] = None,
        skip_existing: bool = True,
    ):
        """
        Initialize batch LLM evaluator.

        Args:
            interventions_dir: Directory with intervention .pt files
            output_dir: Directory for LLM evaluation results
            model: OpenAI model to use
            temperature: Sampling temperature
            include_json: Whether to include full JSON (more tokens)
            deterministic_dir: Optional directory with deterministic results
            skip_existing: Skip interventions already evaluated
        """
        self.interventions_dir = Path(interventions_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.temperature = temperature
        self.include_json = include_json
        self.deterministic_dir = Path(deterministic_dir) if deterministic_dir else None
        self.skip_existing = skip_existing

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM evaluator
        self.llm_evaluator = TextBasedLLMEvaluator(model=model, temperature=temperature)

        # Initialize feature extractor
        self.feature_extractor = MIDIFeatureExtractor()

        # Load representation encoding
        self.encoding = representation.get_encoding()
        self.vocabulary = self.encoding["code_event_map"]
        self.logger.info(f"Loaded encoding with {len(self.vocabulary)} tokens")

        # Cache for extracted metrics
        self.metrics_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "total_features": 0,
            "total_interventions": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "skipped_evaluations": 0,
            "total_tokens_used": 0,
            "total_cost_estimate": 0.0,  # USD
        }

        self.logger.info(f"Initialized BatchLLMEvaluator")
        self.logger.info(f"  Interventions: {self.interventions_dir}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Model: {model}")
        self.logger.info(f"  Include JSON: {include_json}")
        if self.deterministic_dir:
            self.logger.info(f"  Deterministic comparison: {self.deterministic_dir}")

    def find_intervention_files(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Find all intervention files organized by feature.

        Returns:
            Dict mapping feature_id to intervention files:
            {
                'feature182': {
                    'baseline': Path(...),
                    'ablation': Path(...),
                    'add_+1.0': Path(...),
                    'add_-1.0': Path(...),
                    ...
                },
                ...
            }
        """
        feature_files = defaultdict(lambda: {"interventions": {}})

        # Scan all layer directories
        for layer_dir in sorted(self.interventions_dir.glob("layer*")):
            if not layer_dir.is_dir():
                continue

            # Scan feature directories
            for feature_dir in sorted(layer_dir.glob("feature*")):
                if not feature_dir.is_dir():
                    continue

                # Extract feature ID from directory name
                feature_match = re.search(r"feature(\d+)", feature_dir.name)
                if not feature_match:
                    continue

                feature_id = feature_match.group(1)

                # Find baseline file
                baseline_pattern = f"baseline_feature{feature_id}_*.pt"
                baseline_files = list(feature_dir.glob(baseline_pattern))

                if baseline_files:
                    feature_files[feature_id]["baseline"] = baseline_files[0]
                    feature_files[feature_id]["feature_dir"] = feature_dir
                else:
                    self.logger.warning(f"No baseline found for feature {feature_id}")
                    continue

                # Find intervention files
                for pt_file in feature_dir.glob("*.pt"):
                    filename = pt_file.name

                    # Skip baseline
                    if filename.startswith("baseline_"):
                        continue

                    # Parse intervention type
                    if filename.startswith("ablation_"):
                        intervention_key = "ablation"
                    elif filename.startswith("add_+"):
                        strength_match = re.search(r"add_\+([0-9.]+)_", filename)
                        if strength_match:
                            strength = strength_match.group(1)
                            intervention_key = f"add_+{strength}"
                    elif filename.startswith("add_-"):
                        strength_match = re.search(r"add_-([0-9.]+)_", filename)
                        if strength_match:
                            strength = strength_match.group(1)
                            intervention_key = f"add_-{strength}"
                    else:
                        continue

                    feature_files[feature_id]["interventions"][
                        intervention_key
                    ] = pt_file

        self.logger.info(f"Found {len(feature_files)} features with interventions")
        return dict(feature_files)

    def extract_metrics_from_pt_file(
        self, pt_file: Path
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Extract metrics from .pt file.

        Args:
            pt_file: Path to .pt file containing generated tokens

        Returns:
            Tuple of (metrics_dict, tokens_array or None)
        """
        # Check cache
        cache_key = str(pt_file)
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]

        try:
            # Load .pt file
            data = torch.load(pt_file, map_location="cpu", weights_only=False)

            # Extract generated tokens
            if "generated" in data:
                tokens = data["generated"]
            elif isinstance(data, torch.Tensor):
                tokens = data
            else:
                self.logger.error(f"Unknown .pt file format: {pt_file}")
                return {}, None

            # Convert tokens to numpy array (expected format)
            import numpy as np

            if torch.is_tensor(tokens):
                tokens = tokens.cpu().numpy()
            elif isinstance(tokens, list):
                tokens = np.array(tokens)

            # Debug: Check shape before processing
            self.logger.info(
                f"Token array shape: {tokens.shape}, dtype: {tokens.dtype}"
            )

            # Handle different token formats
            if tokens.ndim == 3 and tokens.shape[0] == 1:
                # Remove batch dimension: (1, seq_len, features) -> (seq_len, features)
                tokens = tokens[0]

            # Check if this is note format (seq_len, 5 or 6) or token format (seq_len,)
            if tokens.ndim == 2 and tokens.shape[1] >= 5:
                # This is note data: (beat, position, pitch, duration, program[, ...])
                # Use the first 5 columns as notes
                notes = tokens[:, :5]
                self.logger.info(
                    f"Detected note format: {notes.shape}, converting to music directly"
                )

                # Reconstruct music directly from notes
                music = representation.reconstruct(notes, self.encoding["resolution"])

                # For TXT representation, we'll pass the notes directly
                tokens = notes  # Pass notes for TXT extraction
            elif tokens.ndim == 1 or (tokens.ndim == 2 and tokens.shape[1] == 1):
                # This is token code format: (seq_len,) or (seq_len, 1)
                if tokens.ndim == 2:
                    tokens = tokens.flatten()
                self.logger.info(f"Detected token format: {tokens.shape}")

                # Decode tokens to MusPy Music object
                music = representation.decode(tokens, self.encoding, self.vocabulary)
            else:
                self.logger.error(f"Unknown token format: shape {tokens.shape}")
                return {}, None

            # Extract metrics using MusPy
            metrics = self.feature_extractor.extract_all_features(music)

            # Optionally get MusPy JSON for full representation (legacy, not needed with tokens)
            # Cache result (metrics, muspy_json, tokens)
            result = (metrics, tokens)
            self.metrics_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Failed to extract metrics from {pt_file}: {e}")
            return {}, None

    def load_deterministic_result(
        self, feature_id: str, intervention_key: str
    ) -> Optional[Dict[str, Any]]:
        """Load deterministic analysis result if available."""
        if not self.deterministic_dir:
            return None

        # Look for deterministic result file
        result_pattern = (
            f"feature{feature_id}_{intervention_key.replace('.', '_')}_*.json"
        )
        result_files = list(self.deterministic_dir.glob(f"**/{result_pattern}"))

        if not result_files:
            return None

        try:
            with open(result_files[0]) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load deterministic result: {e}")
            return None

    def evaluate_intervention(
        self,
        feature_id: str,
        baseline_file: Path,
        intervention_file: Path,
        intervention_key: str,
        include_deterministic: bool = False,
    ) -> Optional[LLMEvaluationResult]:
        """
        Evaluate a single intervention.

        Args:
            feature_id: Feature ID being evaluated
            baseline_file: Path to baseline .pt file
            intervention_file: Path to intervention .pt file
            intervention_key: Intervention key (e.g., 'add_+1.0', 'ablation')
            include_deterministic: Whether to include deterministic result in prompt

        Returns:
            LLMEvaluationResult or None if evaluation failed
        """
        # Check if already evaluated
        output_file = (
            self.output_dir
            / f"feature{feature_id}_{intervention_key.replace('.', '_')}.json"
        )
        if self.skip_existing and output_file.exists():
            self.logger.info(
                f"Skipping already evaluated: feature {feature_id} - {intervention_key}"
            )
            self.stats["skipped_evaluations"] += 1
            return None

        self.logger.info(f"Evaluating feature {feature_id} - {intervention_key}")

        try:
            # Extract metrics from baseline
            self.logger.info(f"  Extracting baseline metrics...")
            baseline_metrics, baseline_tokens = self.extract_metrics_from_pt_file(
                baseline_file
            )

            if not baseline_metrics:
                self.logger.error(f"  Failed to extract baseline metrics")
                self.stats["failed_evaluations"] += 1
                return None

            # Extract metrics from intervention
            self.logger.info(f"  Extracting intervention metrics...")
            intervention_metrics, intervention_tokens = (
                self.extract_metrics_from_pt_file(intervention_file)
            )

            if not intervention_metrics:
                self.logger.error(f"  Failed to extract intervention metrics")
                self.stats["failed_evaluations"] += 1
                return None

            # Parse intervention type and strength
            if intervention_key == "ablation":
                intervention_type = "ablation"
                strength = 0.0
            elif intervention_key.startswith("add_+"):
                intervention_type = "addition"
                strength = float(intervention_key.replace("add_+", ""))
            elif intervention_key.startswith("add_-"):
                intervention_type = "addition"
                strength = -float(intervention_key.replace("add_-", ""))
            else:
                self.logger.warning(f"Unknown intervention type: {intervention_key}")
                intervention_type = "unknown"
                strength = 0.0

            # Load deterministic result if available
            deterministic_result = None
            if include_deterministic:
                deterministic_result = self.load_deterministic_result(
                    feature_id, intervention_key
                )

            # Run LLM evaluation
            self.logger.info(f"  Calling OpenAI API...")
            result = self.llm_evaluator.evaluate_intervention(
                feature_id=feature_id,
                baseline_metrics=baseline_metrics,
                intervention_metrics=intervention_metrics,
                strength=strength,
                intervention_type=intervention_type,
                include_deterministic=include_deterministic,
                deterministic_result=deterministic_result,
                include_json=self.include_json,
                baseline_tokens=baseline_tokens,
                intervention_tokens=intervention_tokens,
                vocabulary=self.vocabulary,
            )

            # Update statistics
            self.stats["successful_evaluations"] += 1
            self.stats["total_tokens_used"] += result.tokens_used

            # Estimate cost (GPT-4-turbo: $10/1M input, $30/1M output tokens)
            # Rough estimate: assume 50/50 split
            estimated_cost = (result.tokens_used / 1_000_000) * 20
            self.stats["total_cost_estimate"] += estimated_cost

            # Save result
            self.llm_evaluator.save_result(result, output_file)

            # Generate and save report
            report = self.llm_evaluator.generate_report(result)
            report_file = output_file.with_suffix(".md")
            with open(report_file, "w") as f:
                f.write(report)

            self.logger.info(
                f"  ✅ Evaluation complete. Overall score: {result.overall_score:.3f}"
            )
            self.logger.info(
                f"  Tokens used: {result.tokens_used}, Estimated cost: ${estimated_cost:.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"  ❌ Evaluation failed: {e}", exc_info=True)
            self.stats["failed_evaluations"] += 1
            return None

    def run_batch_evaluation(self, include_deterministic: bool = False):
        """
        Run batch evaluation on all interventions.

        Args:
            include_deterministic: Whether to include deterministic results in prompts
                                  (can bias LLM, use for comparison only)
        """
        self.logger.info("Starting batch LLM evaluation...")

        # Find all intervention files
        feature_files = self.find_intervention_files()

        if not feature_files:
            self.logger.error("No intervention files found!")
            return

        # Count total interventions
        total_interventions = sum(
            len(files["interventions"]) for files in feature_files.values()
        )

        self.stats["total_features"] = len(feature_files)
        self.stats["total_interventions"] = total_interventions

        self.logger.info(
            f"Found {len(feature_files)} features with {total_interventions} interventions"
        )

        # Track all results for summary
        all_results: List[LLMEvaluationResult] = []

        # Evaluate each feature's interventions
        for feature_id, files in feature_files.items():
            baseline_file = files["baseline"]

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Feature {feature_id}")
            self.logger.info(f"{'='*60}")

            # Evaluate each intervention
            for intervention_key, intervention_file in files["interventions"].items():
                result = self.evaluate_intervention(
                    feature_id=feature_id,
                    baseline_file=baseline_file,
                    intervention_file=intervention_file,
                    intervention_key=intervention_key,
                    include_deterministic=include_deterministic,
                )

                if result:
                    all_results.append(result)

        # Generate summary
        self.generate_summary(all_results)

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Batch evaluation complete!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total features: {self.stats['total_features']}")
        self.logger.info(f"Total interventions: {self.stats['total_interventions']}")
        self.logger.info(f"Successful: {self.stats['successful_evaluations']}")
        self.logger.info(f"Failed: {self.stats['failed_evaluations']}")
        self.logger.info(f"Skipped: {self.stats['skipped_evaluations']}")
        self.logger.info(f"Total tokens used: {self.stats['total_tokens_used']:,}")
        self.logger.info(f"Estimated cost: ${self.stats['total_cost_estimate']:.2f}")

    def generate_summary(self, results: List[LLMEvaluationResult]):
        """Generate summary CSV and statistics."""

        if not results:
            self.logger.warning("No results to summarize")
            return

        # Save detailed CSV
        csv_file = self.output_dir / "llm_evaluation_summary.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "feature_id",
                    "feature_name",
                    "intervention_type",
                    "intervention_strength",
                    "feature_effectiveness_score",
                    "musical_quality_score",
                    "coherence_score",
                    "musicality_score",
                    "overall_score",
                    "deterministic_effectiveness_score",
                    "deterministic_quality_score",
                    "deterministic_overall_score",
                    "agreement_level",
                    "tokens_used",
                    "timestamp",
                ]
            )

            # Data rows
            for result in results:
                writer.writerow(
                    [
                        result.feature_id,
                        result.feature_name,
                        result.intervention_type,
                        result.intervention_strength,
                        f"{result.feature_effectiveness_score:.3f}",
                        f"{result.musical_quality_score:.3f}",
                        f"{result.coherence_score:.3f}",
                        f"{result.musicality_score:.3f}",
                        f"{result.overall_score:.3f}",
                        (
                            f"{result.deterministic_effectiveness_score:.3f}"
                            if result.deterministic_effectiveness_score
                            else "N/A"
                        ),
                        (
                            f"{result.deterministic_quality_score:.3f}"
                            if result.deterministic_quality_score
                            else "N/A"
                        ),
                        (
                            f"{result.deterministic_score:.3f}"
                            if result.deterministic_score
                            else "N/A"
                        ),
                        result.agreement_level or "N/A",
                        result.tokens_used,
                        result.timestamp,
                    ]
                )

        self.logger.info(f"Saved summary CSV to: {csv_file}")

        # Generate statistics JSON
        stats_summary = {
            "evaluation_stats": self.stats,
            "score_statistics": {
                "feature_effectiveness": {
                    "mean": sum(r.feature_effectiveness_score for r in results)
                    / len(results),
                    "min": min(r.feature_effectiveness_score for r in results),
                    "max": max(r.feature_effectiveness_score for r in results),
                },
                "musical_quality": {
                    "mean": sum(r.musical_quality_score for r in results)
                    / len(results),
                    "min": min(r.musical_quality_score for r in results),
                    "max": max(r.musical_quality_score for r in results),
                },
                "coherence": {
                    "mean": sum(r.coherence_score for r in results) / len(results),
                    "min": min(r.coherence_score for r in results),
                    "max": max(r.coherence_score for r in results),
                },
                "musicality": {
                    "mean": sum(r.musicality_score for r in results) / len(results),
                    "min": min(r.musicality_score for r in results),
                    "max": max(r.musicality_score for r in results),
                },
                "overall": {
                    "mean": sum(r.overall_score for r in results) / len(results),
                    "min": min(r.overall_score for r in results),
                    "max": max(r.overall_score for r in results),
                },
            },
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model,
            "configuration": {
                "include_json": self.include_json,
                "temperature": self.temperature,
                "deterministic_comparison": self.deterministic_dir is not None,
            },
        }

        # Add agreement statistics if deterministic comparison available
        if any(r.deterministic_score is not None for r in results):
            with_deterministic = [
                r for r in results if r.deterministic_score is not None
            ]
            stats_summary["agreement_statistics"] = {
                "high_agreement": len(
                    [r for r in with_deterministic if r.agreement_level == "high"]
                ),
                "moderate_agreement": len(
                    [r for r in with_deterministic if r.agreement_level == "moderate"]
                ),
                "low_agreement": len(
                    [r for r in with_deterministic if r.agreement_level == "low"]
                ),
                "average_score_difference": sum(
                    abs(r.overall_score - r.deterministic_score)
                    for r in with_deterministic
                )
                / len(with_deterministic),
            }

        stats_file = self.output_dir / "evaluation_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats_summary, f, indent=2)

        self.logger.info(f"Saved statistics to: {stats_file}")

        # Print score summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Score Summary")
        self.logger.info("=" * 60)
        for metric, stats in stats_summary["score_statistics"].items():
            self.logger.info(
                f"{metric.capitalize():15s}: "
                f"mean={stats['mean']:.3f}, "
                f"min={stats['min']:.3f}, "
                f"max={stats['max']:.3f}"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch LLM evaluation for SAE feature interventions"
    )

    parser.add_argument(
        "interventions_dir", help="Directory containing intervention .pt files"
    )

    parser.add_argument(
        "--output-dir",
        default="llm_evaluation_results",
        help="Output directory for LLM evaluations (default: llm_evaluation_results)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="OpenAI model to use (default: gpt-4o-mini-2024-07-18)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )

    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Include full JSON representation (more tokens, better context)",
    )

    parser.add_argument(
        "--compare-deterministic",
        help="Directory with deterministic results for comparison",
    )

    parser.add_argument(
        "--include-deterministic-in-prompt",
        action="store_true",
        help="Include deterministic results in LLM prompt (WARNING: can bias LLM)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-evaluate interventions even if results exist",
    )

    args = parser.parse_args()

    # Validate inputs
    interventions_dir = Path(args.interventions_dir)
    if not interventions_dir.exists():
        print(f"❌ Error: Interventions directory not found: {interventions_dir}")
        sys.exit(1)

    # Initialize batch evaluator
    evaluator = BatchLLMEvaluator(
        interventions_dir=str(interventions_dir),
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        include_json=args.include_json,
        deterministic_dir=args.compare_deterministic,
        skip_existing=not args.no_skip_existing,
    )

    # Run batch evaluation
    evaluator.run_batch_evaluation(
        include_deterministic=args.include_deterministic_in_prompt
    )

    print(f"\n✅ Batch evaluation complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Summary: {Path(args.output_dir) / 'llm_evaluation_summary.csv'}")
    print(f"📈 Statistics: {Path(args.output_dir) / 'evaluation_statistics.json'}")


if __name__ == "__main__":
    main()
