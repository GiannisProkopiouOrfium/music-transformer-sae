#!/usr/bin/env python3
"""
Text-Based LLM Evaluation for SAE Feature Interventions

Uses OpenAI API to evaluate musical interventions based on:
- Symbolic music representation (JSON/MusPy)
- Extracted musical metrics (baseline vs intervention)
- Feature-specific expectations

The LLM provides:
- Effectiveness assessment (did it achieve intended effect?)
- Quality evaluation (is the result musically coherent?)
- Detailed reasoning and insights
- Comparison with deterministic analysis

Requirements:
    pip install openai python-dotenv

Environment:
    OPENAI_API_KEY in .env file or environment variable
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import numpy as np

# OpenAI API
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. Install with: pip install openai")

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not installed. Install with: pip install python-dotenv"
    )

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from deterministic_analysis.midi_feature_extractors import MIDIFeatureExtractor
from deterministic_analysis.threshold_manager import ThresholdManager


@dataclass
class LLMEvaluationResult:
    """Result from LLM evaluation."""

    feature_id: str
    feature_name: str
    intervention_strength: float
    intervention_type: str

    # LLM Scores (0-1) - PRIMARY: Feature effectiveness, SECONDARY: Musical quality
    feature_effectiveness_score: float  # PRIMARY (70% weight)
    musical_quality_score: float  # SECONDARY (30% weight)
    coherence_score: float
    musicality_score: float
    overall_score: (
        float  # Calculated: (feature_effectiveness * 0.7) + (musical_quality * 0.3)
    )

    # LLM Analysis with detailed reasoning
    feature_effectiveness_reasoning: str
    musical_quality_reasoning: str
    coherence_reasoning: str
    musicality_reasoning: str
    overall_reasoning: str
    key_observations: List[str]
    concerns: List[str]
    recommendations: List[str]

    # Comparison with deterministic (optional)
    deterministic_score: Optional[float] = None
    deterministic_effectiveness_score: Optional[float] = None
    deterministic_quality_score: Optional[float] = None
    agreement_level: Optional[str] = None  # "high", "moderate", "low"

    # Metadata
    model_used: str = "gpt-4o-mini-2024-07-18"
    timestamp: str = ""
    tokens_used: int = 0


class TextBasedLLMEvaluator:
    """Evaluate musical interventions using text-based LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ):
        """
        Initialize LLM evaluator.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: OpenAI model to use
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai"
            )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize feature extractor and threshold manager
        self.feature_extractor = MIDIFeatureExtractor()
        try:
            self.threshold_manager = ThresholdManager()
            self.logger.info("Loaded calibrated thresholds")
        except Exception as e:
            self.logger.warning(f"Could not load calibrated thresholds: {e}")
            self.threshold_manager = None

        # Feature profiles for context
        self.feature_profiles = self._initialize_feature_profiles()

        self.logger.info(f"Initialized TextBasedLLMEvaluator with model: {model}")

    def _initialize_feature_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize feature profiles with expected effects."""
        return {
            "325": {
                "name": "rhythmic_displacement_syncopation",
                "layer": 1,
                "description": "Detects and controls rhythmic displacement patterns where melodic motifs are shifted by sixteenth notes, creating syncopation effects across instrumental lines. This creates subtle off-beat emphasis and rhythmic tension.",
                "expected_effects_positive": [
                    "Increased syncopation (off-beat emphasis)",
                    "More sixteenth-note rhythmic displacement",
                    "Higher rhythmic complexity and cross-rhythm effects",
                    "More pronounced rhythmic tension between instrumental lines",
                ],
                "expected_effects_negative": [
                    "Reduced syncopation (on-beat emphasis)",
                    "Less rhythmic displacement",
                    "More regular, grid-aligned rhythms",
                    "Decreased rhythmic complexity",
                ],
            },
            "256": {
                "name": "dynamic_contrast_accents",
                "layer": 1,
                "description": "Controls dramatic dynamic contrast patterns with sudden accents followed by immediate soft dynamics, creating forte-piano alternations and expressive ebb-and-flow within the texture.",
                "expected_effects_positive": [
                    "Stronger dynamic contrasts (sudden accent patterns)",
                    "More extreme velocity changes (>30-70 velocity difference)",
                    "Increased dramatic tension through forte-piano alternations",
                    "Higher dynamic range and variance",
                ],
                "expected_effects_negative": [
                    "Reduced dynamic contrast",
                    "More uniform velocity levels",
                    "Softer overall dynamics with less dramatic accents",
                    "Decreased dynamic range",
                ],
            },
            "1323": {
                "name": "wide_pitch_range_texture",
                "layer": 1,
                "description": "Controls wide pitch range utilization across multiple instruments, emphasizing broad harmonic spectrum and textural depth through multi-register utilization.",
                "expected_effects_positive": [
                    "Wider pitch range (broader harmonic spectrum)",
                    "Increased use of extreme registers (high and low)",
                    "More textural depth through multi-register layering",
                    "Higher pitch variance across instrumental lines",
                ],
                "expected_effects_negative": [
                    "Narrower pitch range (compressed register use)",
                    "Reduced extreme register utilization",
                    "Less textural depth and layering",
                    "More compact pitch distribution",
                ],
            },
            "182": {
                "name": "steady_pulse_march_rhythm",
                "layer": 3,
                "description": "Controls steady pulse patterns with metronomic consistency and march-like rhythms. Creates regular, ordered rhythmic structures with consistent motifs that reinforce a sense of temporal regularity.",
                "expected_effects_positive": [
                    "More metronomic consistency and regular pulse",
                    "Stronger march-like rhythmic patterns",
                    "Increased rhythmic regularity and order",
                    "More consistent note durations and inter-onset intervals",
                ],
                "expected_effects_negative": [
                    "Less regular pulse (more rhythmic flexibility)",
                    "Reduced metronomic consistency",
                    "More varied rhythmic patterns",
                    "Decreased march-like characteristics",
                ],
            },
            "855": {
                "name": "dynamic_contrast_tension",
                "layer": 3,
                "description": "Controls dynamic contrast patterns with sudden shifts between high and low velocity ranges, creating dramatic tension and release within short musical passages.",
                "expected_effects_positive": [
                    "Stronger sudden dynamic shifts (high-low velocity alternation)",
                    "More dramatic tension and release patterns",
                    "Increased dynamic range and variance",
                    "More extreme velocity contrasts within passages",
                ],
                "expected_effects_negative": [
                    "Reduced dynamic contrasts",
                    "Less dramatic tension/release",
                    "More uniform dynamic levels",
                    "Decreased velocity range",
                ],
            },
            "997": {
                "name": "antiphonal_call_response",
                "layer": 3,
                "description": "Controls antiphonal textures with call-and-response interactions between instrumental groups. Creates spatial and dialogic effects where musical ideas are echoed or answered between instruments.",
                "expected_effects_positive": [
                    "More pronounced call-and-response patterns",
                    "Stronger antiphonal texture (instrumental dialogue)",
                    "Increased alternating melodic lines between instruments",
                    "More distinct spatial and dialogic effects",
                ],
                "expected_effects_negative": [
                    "Less call-and-response interaction",
                    "Reduced antiphonal texture",
                    "More unified ensemble playing (less dialogue)",
                    "Decreased instrumental alternation",
                ],
            },
            "471": {
                "name": "unison_doubling_octaves",
                "layer": 5,
                "description": "Controls unison doubling patterns where multiple instruments play the same melodic line in octaves, creating a reinforced and powerful unified sound with enhanced textural weight.",
                "expected_effects_positive": [
                    "More unison doubling (multiple instruments on same line)",
                    "Stronger octave reinforcement",
                    "More powerful, unified sound across instruments",
                    "Increased textural weight and density",
                ],
                "expected_effects_negative": [
                    "Less unison doubling (more independent lines)",
                    "Reduced octave reinforcement",
                    "More transparent, less unified texture",
                    "Decreased textural density",
                ],
            },
            "904": {
                "name": "rhythmic_augmentation",
                "layer": 5,
                "description": "Controls rhythmic augmentation patterns where motifs are systematically lengthened in duration, creating expansion, tension-building, and temporal stretching effects over time.",
                "expected_effects_positive": [
                    "More systematic duration lengthening (augmentation)",
                    "Stronger motif expansion and development",
                    "Increased sense of tension and temporal stretching",
                    "Longer average note durations",
                ],
                "expected_effects_negative": [
                    "Less duration lengthening (or rhythmic diminution)",
                    "Reduced motif expansion",
                    "Shorter note durations",
                    "More compact temporal structure",
                ],
            },
            "1950": {
                "name": "dramatic_dynamic_swells",
                "layer": 5,
                "description": "Controls dramatic dynamic swell patterns with abrupt shifts between soft and loud dynamics, creating expressive swells and dramatic textural effects within the ensemble.",
                "expected_effects_positive": [
                    "More abrupt dynamic shifts (soft-to-loud contrasts)",
                    "Stronger expressive dynamic swells",
                    "Increased dramatic textural effects",
                    "Higher dynamic range and variance",
                ],
                "expected_effects_negative": [
                    "Reduced dynamic contrasts",
                    "Less dramatic swells",
                    "More uniform dynamics",
                    "Decreased dynamic variance",
                ],
            },
        }

    def evaluate_intervention(
        self,
        feature_id: str,
        baseline_metrics: Dict[str, Any],
        intervention_metrics: Dict[str, Any],
        strength: float,
        intervention_type: str = "addition",
        include_deterministic: bool = False,
        deterministic_result: Optional[Dict[str, Any]] = None,
        include_json: bool = False,
        baseline_tokens: Optional[np.ndarray] = None,
        intervention_tokens: Optional[np.ndarray] = None,
        vocabulary: Optional[Dict] = None,
    ) -> LLMEvaluationResult:
        """
        Evaluate an intervention using LLM.

        Args:
            feature_id: Feature ID being evaluated
            baseline_metrics: Metrics extracted from baseline
            intervention_metrics: Metrics extracted from intervention
            strength: Intervention strength applied
            intervention_type: Type of intervention ("addition", "ablation")
            include_deterministic: Whether to include deterministic results (can bias LLM)
            deterministic_result: Optional deterministic analysis result for comparison
            include_json: Whether to include symbolic representation (JSON or TXT)
            baseline_json: Optional baseline MusPy JSON (legacy, prefer tokens)
            intervention_json: Optional intervention MusPy JSON (legacy, prefer tokens)
            baseline_tokens: Optional baseline token sequence (for TXT representation - RECOMMENDED)
            intervention_tokens: Optional intervention token sequence (for TXT representation - RECOMMENDED)
            vocabulary: Optional code-to-event vocabulary (required if using tokens)

        Returns:
            LLMEvaluationResult with scores and reasoning
        """
        if feature_id not in self.feature_profiles:
            raise ValueError(f"Unknown feature ID: {feature_id}")

        profile = self.feature_profiles[feature_id]

        # Build prompt
        prompt = self._build_evaluation_prompt(
            feature_id=feature_id,
            profile=profile,
            baseline_metrics=baseline_metrics,
            intervention_metrics=intervention_metrics,
            strength=strength,
            intervention_type=intervention_type,
            include_deterministic=include_deterministic,
            deterministic_result=(
                deterministic_result if include_deterministic else None
            ),
            include_json=include_json,
            baseline_tokens=baseline_tokens,
            intervention_tokens=intervention_tokens,
            vocabulary=vocabulary,
        )

        # Call OpenAI API
        self.logger.info(
            f"Evaluating feature {feature_id} (strength {strength}) with LLM..."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},  # Request JSON response
            )

            # Parse response
            result = self._parse_llm_response(
                response=response,
                feature_id=feature_id,
                profile=profile,
                strength=strength,
                intervention_type=intervention_type,
                deterministic_result=deterministic_result,
            )

            self.logger.info(
                f"LLM evaluation complete. Overall score: {result.overall_score:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert music theorist and AI researcher specializing in evaluating 
musical interventions from neural network features. Your task is to assess whether a feature 
intervention on generated music achieves its intended effect while maintaining musical quality.

EVALUATION PRIORITY:
1. PRIMARY (70% weight): Did the intervention achieve the INTENDED feature effect?
2. SECONDARY (30% weight): Is the musical quality acceptable?

You will analyze:
1. Feature expectations (what the feature should control)
2. Metric changes (baseline vs intervention comparison)
3. Musical quality aspects (coherence, musicality)

Provide:
- Numerical scores (0.0-1.0) for feature_effectiveness, musical_quality, coherence, musicality, overall
- Detailed reasoning for each score
- Key observations about what changed
- Concerns about quality or side effects
- Recommendations for improvement

CRITICAL: The feature_effectiveness score is MOST IMPORTANT. A feature that achieves its intended 
effect (even with some musical tradeoffs) scores higher than a feature that sounds good but 
doesn't achieve its purpose.

Be objective, precise, and grounded in music theory. Your response MUST be valid JSON."""

    def _build_evaluation_prompt(
        self,
        feature_id: str,
        profile: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        intervention_metrics: Dict[str, Any],
        strength: float,
        intervention_type: str,
        include_deterministic: bool,
        deterministic_result: Optional[Dict[str, Any]],
        include_json: bool,
        baseline_tokens: Optional[np.ndarray],
        intervention_tokens: Optional[np.ndarray],
        vocabulary: Optional[Dict],
    ) -> str:
        """Build evaluation prompt for LLM."""

        # Calculate metric changes
        metric_changes = self._calculate_metric_changes(
            baseline_metrics, intervention_metrics
        )

        # Determine expected direction
        if strength > 0 or intervention_type == "addition":
            expected_effects = profile.get("expected_effects_positive", [])
            direction = "positive (increase/enhance)"
        else:
            expected_effects = profile.get("expected_effects_negative", [])
            direction = "negative (decrease/reduce)"

        prompt = f"""# Music Feature Intervention Evaluation

## Feature Information
- **Feature ID**: {feature_id}
- **Feature Name**: {profile['name']}
- **Layer**: {profile['layer']} ({"early" if profile['layer'] == 1 else "mid" if profile['layer'] == 3 else "late"} processing)
- **Description**: {profile['description']}
- **Intervention Strength**: {strength}
- **Intervention Type**: {intervention_type}
- **Expected Direction**: {direction}

## Expected Effects
"""

        for i, effect in enumerate(expected_effects, 1):
            prompt += f"{i}. {effect}\n"

        prompt += "\n## Metric Changes (Baseline → Intervention)\n\n"
        prompt += "### Significant Changes (>5% difference):\n"

        for metric_path, change_info in metric_changes.items():
            if abs(change_info["percent_change"]) > 5:
                prompt += f"- **{metric_path}**: "
                prompt += f"{change_info['baseline']:.3f} → {change_info['intervention']:.3f} "
                prompt += f"({change_info['percent_change']:+.1f}%)\n"

        # Add threshold context if available
        if self.threshold_manager:
            prompt += "\n### Quality Context (Percentile Ranks):\n"
            for metric_path, change_info in metric_changes.items():
                if abs(change_info["percent_change"]) > 5:
                    try:
                        baseline_p = self.threshold_manager.get_metric_percentile_rank(
                            metric_path, change_info["baseline"]
                        )
                        intervention_p = (
                            self.threshold_manager.get_metric_percentile_rank(
                                metric_path, change_info["intervention"]
                            )
                        )
                        if baseline_p and intervention_p:
                            prompt += f"- **{metric_path}**: P{baseline_p:.0f} → P{intervention_p:.0f}\n"
                    except:
                        pass

        # Add deterministic comparison if available (ONLY NUMERIC VALUES - no labels)
        if include_deterministic and deterministic_result:
            prompt += "\n## Deterministic Metric Analysis (numeric reference only)\n"
            prompt += f"- **Effectiveness Score**: {deterministic_result.get('effectiveness_analysis', {}).get('effectiveness_score', 0):.3f}\n"
            prompt += f"- **Quality Score**: {deterministic_result.get('quality_assessment', {}).get('overall_quality_score', 0):.3f}\n"
            prompt += f"- **Overall Score**: {deterministic_result.get('overall_decision', {}).get('overall_score', 0):.3f}\n"
            prompt += "\nNote: These are numeric scores from rule-based analysis. Use only as reference - make your own independent assessment.\n"

        # Add symbolic representation if requested
        if include_json:
            # Prefer TXT representation (token-based) over JSON (more compact and includes full post-prefix region)
            if (
                baseline_tokens is not None
                and intervention_tokens is not None
                and vocabulary is not None
            ):
                prompt += "\n## Musical Representation (TXT - Human-Readable Events)\n"
                prompt += (
                    "### Baseline (post-prefix excerpt where differences occur):\n```\n"
                )
                prompt += self._extract_txt_representation(
                    baseline_tokens, vocabulary, prefix_len=128, excerpt_len=400
                )
                prompt += "\n```\n"
                prompt += "### Intervention (post-prefix excerpt):\n```\n"
                prompt += self._extract_txt_representation(
                    intervention_tokens, vocabulary, prefix_len=128, excerpt_len=400
                )
                prompt += "\n```\n"
                prompt += "\nNote: Skipping first 128 tokens (prefix region) where baseline=intervention. Showing next 400 tokens where intervention effects appear.\n"

        prompt += """

## Your Task

Evaluate this intervention and provide your assessment in JSON format with the following structure:

```json
{
    "feature_effectiveness_score": 0.0-1.0,
    "feature_effectiveness_reasoning": "detailed explanation of whether INTENDED feature effects were achieved",
    "musical_quality_score": 0.0-1.0,
    "musical_quality_reasoning": "detailed explanation of overall musical quality",
    "coherence_score": 0.0-1.0,
    "coherence_reasoning": "explanation of musical coherence and consistency",
    "musicality_score": 0.0-1.0,
    "musicality_reasoning": "explanation of musical appeal and aesthetics",
    "overall_score": 0.0-1.0,
    "overall_reasoning": "summary: feature effectiveness (primary) + musical quality (secondary)",
    "key_observations": ["observation 1", "observation 2", ...],
    "concerns": ["concern 1", "concern 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}
```

EVALUATION CRITERIA (in priority order):

1. **Feature Effectiveness (PRIMARY - 70% weight)**:
   - Do the metric changes align with expected effects?
   - Is the direction of change correct?
   - Is the magnitude appropriate for the intervention strength?
   - Score HIGH if intended effect achieved, even if music has some issues
   - Score LOW if intended effect not achieved, even if music sounds ok

2. **Musical Quality (SECONDARY - 30% weight)**:
   - Is musical quality preserved or enhanced?
   - Are there unexpected side effects?
   - Is the result coherent and musical?
   - Does it sound good?

CALCULATE overall_score as: (feature_effectiveness_score * 0.7) + (musical_quality_score * 0.3)

Provide your analysis in valid JSON format with ALL required fields."""

        return prompt

    def _calculate_metric_changes(
        self, baseline_metrics: Dict[str, Any], intervention_metrics: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate changes between baseline and intervention metrics."""
        changes = {}

        for category, metrics in baseline_metrics.items():
            if not isinstance(metrics, dict):
                continue

            for metric_name, baseline_val in metrics.items():
                if not isinstance(baseline_val, (int, float)):
                    continue

                metric_path = f"{category}.{metric_name}"

                # Get intervention value
                intervention_val = intervention_metrics.get(category, {}).get(
                    metric_name
                )

                if intervention_val is not None and isinstance(
                    intervention_val, (int, float)
                ):
                    # Calculate change
                    if baseline_val != 0:
                        percent_change = (
                            (intervention_val - baseline_val) / baseline_val
                        ) * 100
                    else:
                        percent_change = (
                            intervention_val * 100 if intervention_val != 0 else 0
                        )

                    changes[metric_path] = {
                        "baseline": baseline_val,
                        "intervention": intervention_val,
                        "absolute_change": intervention_val - baseline_val,
                        "percent_change": percent_change,
                    }

        return changes

    def _extract_json_excerpt(self, music_json: Dict, max_notes: int = 20) -> Dict:
        """Extract excerpt from MusPy JSON to reduce token usage."""
        excerpt = {
            "metadata": music_json.get("metadata", {}),
            "resolution": music_json.get("resolution"),
            "tempos": music_json.get("tempos", [])[:2],  # First 2 tempos
            "key_signatures": music_json.get("key_signatures", [])[:2],
            "time_signatures": music_json.get("time_signatures", [])[:2],
            "tracks_summary": [],
        }

        # Add track summaries with limited notes
        for track in music_json.get("tracks", [])[:3]:  # Max 3 tracks
            track_summary = {
                "program": track.get("program"),
                "name": track.get("name"),
                "note_count": len(track.get("notes", [])),
                "notes_excerpt": track.get("notes", [])[:max_notes],  # First N notes
            }
            excerpt["tracks_summary"].append(track_summary)

        return excerpt

    def _extract_txt_representation(
        self, tokens, vocabulary, prefix_len: int = 0, excerpt_len: int = 600
    ) -> str:
        """
        Extract TXT representation for LLM showing full musical content.

        Args:
            tokens: Token sequence (numpy array)
            vocabulary: Code-to-event vocabulary mapping
            prefix_len: Kept for compatibility (not used - we take from start)
            excerpt_len: Maximum number of tokens to include (default 600)

        Returns:
            Human-readable TXT representation with complete musical events
        """
        # Import representation module
        from baseline import representation_remi
        import tempfile
        from pathlib import Path

        # Filter the tokens to remove special markers and get actual musical content
        # Remove start-of-song, end-of-song, and end-of-track tokens
        filtered_tokens = []
        for token in tokens:
            event = vocabulary.get(token, "")
            # Skip special tokens that don't represent musical content
            if event not in ["start-of-song", "end-of-song", "end-of-track"]:
                filtered_tokens.append(token)
        
        # Take up to 600 tokens of actual musical content
        if len(filtered_tokens) > 600:
            excerpt_tokens = np.array(filtered_tokens[:600])
        elif filtered_tokens:
            excerpt_tokens = np.array(filtered_tokens)
        else:
            # Fallback: if no musical content, take original tokens
            excerpt_tokens = tokens[:600] if len(tokens) > 600 else tokens

        # Use save_txt which handles REMI format properly
        # Create temporary file to get the TXT representation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save to temp file using the same method as generate.py
            representation_remi.save_txt(tmp_path, excerpt_tokens, vocabulary)

            # Read back the content
            with open(tmp_path, "r") as f:
                txt_representation = f.read()

            self.logger.info(f"TXT representation: {txt_representation}...")
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

        # Add context header
        header = f"# Musical content excerpt ({len(excerpt_tokens)} tokens, special markers removed)\n"
        header += "# Format: beat_X position_Y instrument_Z pitch_A duration_B\n"
        header += "# Each line represents timing + note events at that position\n\n"

        return header + txt_representation

    def _parse_llm_response(
        self,
        response,
        feature_id: str,
        profile: Dict[str, Any],
        strength: float,
        intervention_type: str,
        deterministic_result: Optional[Dict[str, Any]],
    ) -> LLMEvaluationResult:
        """Parse LLM response into structured result."""

        # Extract response content
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Parse JSON
        try:
            llm_data = json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}")
            self.logger.error(f"Response content: {content}")
            raise

        # Extract deterministic scores if available (for comparison only)
        agreement_level = None
        deterministic_score = None
        deterministic_effectiveness_score = None
        deterministic_quality_score = None

        if deterministic_result:
            deterministic_score = deterministic_result.get("overall_decision", {}).get(
                "overall_score", 0
            )
            deterministic_effectiveness_score = deterministic_result.get(
                "effectiveness_analysis", {}
            ).get("effectiveness_score", 0)
            deterministic_quality_score = deterministic_result.get(
                "quality_assessment", {}
            ).get("overall_quality_score", 0)

            # Calculate agreement based on overall scores
            score_diff = abs(llm_data.get("overall_score", 0) - deterministic_score)
            if score_diff < 0.15:
                agreement_level = "high"
            elif score_diff < 0.30:
                agreement_level = "moderate"
            else:
                agreement_level = "low"

        # Create result with new field names
        result = LLMEvaluationResult(
            feature_id=feature_id,
            feature_name=profile["name"],
            intervention_strength=strength,
            intervention_type=intervention_type,
            feature_effectiveness_score=llm_data.get(
                "feature_effectiveness_score", 0.0
            ),
            musical_quality_score=llm_data.get("musical_quality_score", 0.0),
            coherence_score=llm_data.get("coherence_score", 0.0),
            musicality_score=llm_data.get("musicality_score", 0.0),
            overall_score=llm_data.get("overall_score", 0.0),
            feature_effectiveness_reasoning=llm_data.get(
                "feature_effectiveness_reasoning", ""
            ),
            musical_quality_reasoning=llm_data.get("musical_quality_reasoning", ""),
            coherence_reasoning=llm_data.get("coherence_reasoning", ""),
            musicality_reasoning=llm_data.get("musicality_reasoning", ""),
            overall_reasoning=llm_data.get("overall_reasoning", ""),
            key_observations=llm_data.get("key_observations", []),
            concerns=llm_data.get("concerns", []),
            recommendations=llm_data.get("recommendations", []),
            deterministic_score=deterministic_score,
            deterministic_effectiveness_score=deterministic_effectiveness_score,
            deterministic_quality_score=deterministic_quality_score,
            agreement_level=agreement_level,
            model_used=self.model,
            timestamp=datetime.now().isoformat(),
            tokens_used=tokens_used,
        )

        return result

    def save_result(self, result: LLMEvaluationResult, output_path: str):
        """Save LLM evaluation result to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(asdict(result), f, indent=2)

        self.logger.info(f"Saved LLM evaluation result to: {output_file}")

    def generate_report(self, result: LLMEvaluationResult) -> str:
        """Generate human-readable report from LLM evaluation."""

        report = f"""
# LLM Evaluation Report

## Feature: {result.feature_name} (ID: {result.feature_id})
- **Intervention Strength**: {result.intervention_strength}
- **Intervention Type**: {result.intervention_type}
- **Model Used**: {result.model_used}
- **Timestamp**: {result.timestamp}
- **Tokens Used**: {result.tokens_used}

## Scores

| Metric | Score | Weight |
|--------|-------|--------|
| **Feature Effectiveness** (PRIMARY) | {result.feature_effectiveness_score:.3f} | 70% |
| **Musical Quality** (SECONDARY) | {result.musical_quality_score:.3f} | 30% |
| **Coherence** | {result.coherence_score:.3f} | - |
| **Musicality** | {result.musicality_score:.3f} | - |
| **Overall** | {result.overall_score:.3f} | 100% |

"""

        if result.deterministic_score is not None:
            report += f"""
## Comparison with Deterministic Analysis

| Metric | Deterministic | LLM | Difference |
|--------|---------------|-----|------------|
| **Feature Effectiveness** | {result.deterministic_effectiveness_score:.3f} | {result.feature_effectiveness_score:.3f} | {abs(result.feature_effectiveness_score - result.deterministic_effectiveness_score):.3f} |
| **Musical Quality** | {result.deterministic_quality_score:.3f} | {result.musical_quality_score:.3f} | {abs(result.musical_quality_score - result.deterministic_quality_score):.3f} |
| **Overall** | {result.deterministic_score:.3f} | {result.overall_score:.3f} | {abs(result.overall_score - result.deterministic_score):.3f} |

**Agreement Level**: {result.agreement_level}

"""

        report += f"""
## Feature Effectiveness Reasoning (PRIMARY - 70% weight)

{result.feature_effectiveness_reasoning}

## Musical Quality Reasoning (SECONDARY - 30% weight)

{result.musical_quality_reasoning}

## Coherence Analysis

{result.coherence_reasoning}

## Musicality Analysis

{result.musicality_reasoning}

## Overall Assessment

{result.overall_reasoning}

## Key Observations

"""
        for obs in result.key_observations:
            report += f"- {obs}\n"

        if result.concerns:
            report += "\n## Concerns\n\n"
            for concern in result.concerns:
                report += f"- {concern}\n"

        if result.recommendations:
            report += "\n## Recommendations\n\n"
            for rec in result.recommendations:
                report += f"- {rec}\n"

        return report


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate musical interventions with LLM"
    )
    parser.add_argument("--feature-id", required=True, help="Feature ID to evaluate")
    parser.add_argument(
        "--baseline-metrics", required=True, help="Path to baseline metrics JSON"
    )
    parser.add_argument(
        "--intervention-metrics",
        required=True,
        help="Path to intervention metrics JSON",
    )
    parser.add_argument(
        "--strength", type=float, required=True, help="Intervention strength"
    )
    parser.add_argument(
        "--output", default="llm_evaluation_result.json", help="Output file path"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini-2024-07-18", help="OpenAI model to use"
    )
    parser.add_argument(
        "--include-json", action="store_true", help="Include full JSON (more tokens)"
    )

    args = parser.parse_args()

    # Load metrics
    with open(args.baseline_metrics) as f:
        baseline_metrics = json.load(f)

    with open(args.intervention_metrics) as f:
        intervention_metrics = json.load(f)

    # Initialize evaluator
    evaluator = TextBasedLLMEvaluator(model=args.model)

    # Run evaluation
    result = evaluator.evaluate_intervention(
        feature_id=args.feature_id,
        baseline_metrics=baseline_metrics,
        intervention_metrics=intervention_metrics,
        strength=args.strength,
        include_json=args.include_json,
    )

    # Save result
    evaluator.save_result(result, args.output)

    # Generate and print report
    report = evaluator.generate_report(result)
    print(report)

    # Save report
    report_path = Path(args.output).with_suffix(".md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Evaluation complete!")
    print(f"📄 Result saved to: {args.output}")
    print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
