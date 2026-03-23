#!/usr/bin/env python3
"""
Step 2: Compare Flamingo descriptions using an LLM to mirror the human evaluation.

Loads descriptions.json, builds comparison pairs matching the evaluation blocks,
sends each pair to GPT-4 with the corresponding evaluation question, and
aggregates results into a structured report.

Usage:
    python -m flamingo_evaluation.run_compare \
        --descriptions flamingo_evaluation/results/descriptions.json \
        --output_dir flamingo_evaluation/results \
        --model gpt-4o
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from flamingo_evaluation.audio_catalog import (
    ComparisonPair,
    build_catalog,
    build_all_pairs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Evaluation questions per block, mirroring human eval
# ──────────────────────────────────────────────

EFFECTIVENESS_QUESTIONS = {
    # Direction-specific questions — key is (concept, direction_keyword)
    # The runner will build the question dynamically based on the pair's direction.
}


def _build_effectiveness_question(concept: str, direction: str) -> str:
    """Build a direction-specific effectiveness question.

    The first ~8 seconds of every track is the same conditioning segment,
    so the question focuses on what happens AFTER that shared opening.
    """
    dir_lower = direction.lower()

    if concept == "pitch":
        if "high_to_low" in dir_lower or "low" in dir_lower.split("to_")[-1:]:
            target = "lower average pitch or a shift toward lower registers"
        elif "low_to_high" in dir_lower or "high" in dir_lower.split("to_")[-1:]:
            target = "higher average pitch or a shift toward higher registers"
        else:
            target = "a different pitch profile or shift in pitch register"
        return (
            "Both samples share the same musical opening (~8 seconds). "
            "Focusing on the music AFTER this opening, based on the descriptions below, "
            f"which sample exhibits {target} compared to the other? "
            "Answer with the sample letter."
        )

    elif concept == "duration":
        if "long_to_short" in dir_lower or "short" in dir_lower.split("to_")[-1:]:
            target = "shorter note durations or faster rhythmic movement"
        elif "short_to_long" in dir_lower or "long" in dir_lower.split("to_")[-1:]:
            target = "longer note durations or more sustained notes"
        else:
            target = "a different note duration profile"
        return (
            "Both samples share the same musical opening (~8 seconds). "
            "Focusing on the music AFTER this opening, based on the descriptions below, "
            f"which sample exhibits {target} compared to the other? "
            "Answer with the sample letter."
        )

    else:  # dual
        parts = []
        if "high" in dir_lower and "low" in dir_lower:
            if dir_lower.index("high") < dir_lower.index("low"):
                parts.append("a shift toward lower pitch")
            else:
                parts.append("a shift toward higher pitch")
        if "long" in dir_lower and "short" in dir_lower:
            if dir_lower.index("long") < dir_lower.index("short"):
                parts.append("shorter note durations")
            else:
                parts.append("longer note durations")
        if not parts:
            parts = ["more noticeable changes in both pitch and rhythm"]
        target = " AND ".join(parts)
        return (
            "Both samples share the same musical opening (~8 seconds). "
            "Focusing on the music AFTER this opening, based on the descriptions below, "
            f"which sample exhibits {target} compared to the other? "
            "Answer with the sample letter."
        )


QUALITY_QUESTIONS = [
    {
        "id": "overall_quality",
        "prompt": (
            "Based on the two music descriptions below, rate the overall "
            "musical quality of each sample on a scale of 1-5 "
            "(1=very bad, 5=excellent). Consider coherence, naturalness, "
            "and production quality."
        ),
        "format": "quality_rating",
    },
    {
        "id": "naturalness",
        "prompt": (
            "Based on the two music descriptions below, which sample sounds "
            "more musically natural and human-composed? Answer with the sample letter."
        ),
        "format": "preference",
    },
]

SMOOTH_QUESTIONS = [
    {
        "id": "transition_smoothness",
        "prompt": (
            "Based on the two music descriptions below, which sample has "
            "smoother, more natural-sounding musical transitions and evolution? "
            "Answer with the sample letter."
        ),
        "format": "preference",
    },
    {
        "id": "overall_preference",
        "prompt": (
            "Based on the two music descriptions below, which sample do you "
            "prefer overall in terms of musical quality and coherence? "
            "Answer with the sample letter."
        ),
        "format": "preference",
    },
    {
        "id": "coherence",
        "prompt": (
            "Based on the two music descriptions below, rate the musical "
            "coherence of each sample on a scale of 1-5."
        ),
        "format": "quality_rating",
    },
]

CROSS_METHOD_QUESTIONS = [
    {
        "id": "effect_strength",
        "prompt": (
            "Based on the two music descriptions below, which sample achieves "
            "a more noticeable or stronger musical change/transformation? "
            "Answer with the sample letter."
        ),
        "format": "preference",
    },
    {
        "id": "quality_preservation",
        "prompt": (
            "Based on the two music descriptions below, which sample maintains "
            "higher overall musical quality despite the transformation? "
            "Answer with the sample letter."
        ),
        "format": "preference",
    },
    {
        "id": "overall_preference",
        "prompt": (
            "Based on the two music descriptions below, which sample do you "
            "prefer overall? Answer with the sample letter."
        ),
        "format": "preference",
    },
]

# ──────────────────────────────────────────────
# LLM comparison
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert music analyst evaluating AI-generated music. "
    "You will be given Music Flamingo descriptions of two music samples (A and B). "
    "Answer the evaluation question based ONLY on the provided descriptions. "
    "Be objective and analytical. Always respond in valid JSON."
)

PREFERENCE_TEMPLATE = """\
## Sample A: {label_a}
{desc_a}

## Sample B: {label_b}
{desc_b}

## Question
{question}

Respond in JSON: {{"answer": "A" or "B" or "Equal", "confidence": 1-5, "reasoning": "brief explanation"}}"""

RATING_TEMPLATE = """\
## Sample A: {label_a}
{desc_a}

## Sample B: {label_b}
{desc_b}

## Question
{question}

Respond in JSON: {{"rating_a": 1-5, "rating_b": 1-5, "reasoning": "brief explanation"}}"""


def _call_llm(system: str, user: str, model: str) -> dict:
    """Call OpenAI-compatible LLM and parse JSON response."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    return json.loads(text)


def compare_pair(
    pair: ComparisonPair,
    descriptions: Dict,
    question: str,
    response_format: str,
    model: str,
) -> Optional[dict]:
    """Run one comparison through the LLM."""
    desc_a = descriptions.get(pair.audio_a.filename, {}).get("description")
    desc_b = descriptions.get(pair.audio_b.filename, {}).get("description")

    if not desc_a or not desc_b:
        missing = []
        if not desc_a:
            missing.append(pair.audio_a.filename)
        if not desc_b:
            missing.append(pair.audio_b.filename)
        logger.warning("Missing descriptions for: %s", ", ".join(missing))
        return None

    template = (
        RATING_TEMPLATE if response_format == "quality_rating" else PREFERENCE_TEMPLATE
    )
    user_msg = template.format(
        label_a=pair.label_a,
        desc_a=desc_a,
        label_b=pair.label_b,
        desc_b=desc_b,
        question=question,
    )

    try:
        result = _call_llm(SYSTEM_PROMPT, user_msg, model)
        result["audio_a"] = pair.audio_a.filename
        result["audio_b"] = pair.audio_b.filename
        result["label_a"] = pair.label_a
        result["label_b"] = pair.label_b
        return result
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


# ──────────────────────────────────────────────
# Block runners
# ──────────────────────────────────────────────


def run_block_effectiveness(
    pairs: List[ComparisonPair], descriptions: Dict, model: str
) -> List[dict]:
    results = []
    for pair in pairs:
        concept = pair.concept
        direction = pair.direction or ""
        question = _build_effectiveness_question(concept, direction)
        r = compare_pair(pair, descriptions, question, "preference", model)
        if r:
            r["block"] = "effectiveness"
            r["concept"] = concept
            r["direction"] = direction
            r["method"] = pair.method
            r["song_id"] = pair.song_id
            results.append(r)
    return results


def run_block_quality(
    pairs: List[ComparisonPair], descriptions: Dict, model: str
) -> List[dict]:
    results = []
    for pair in pairs:
        for q in QUALITY_QUESTIONS:
            r = compare_pair(pair, descriptions, q["prompt"], q["format"], model)
            if r:
                r["block"] = "quality"
                r["question_id"] = q["id"]
                r["concept"] = pair.concept
                r["method"] = pair.method
                r["song_id"] = pair.song_id
                results.append(r)
    return results


def run_block_smooth(
    pairs: List[ComparisonPair], descriptions: Dict, model: str
) -> List[dict]:
    results = []
    for pair in pairs:
        for q in SMOOTH_QUESTIONS:
            r = compare_pair(pair, descriptions, q["prompt"], q["format"], model)
            if r:
                r["block"] = "smooth_vs_abrupt"
                r["question_id"] = q["id"]
                r["concept"] = pair.concept
                r["song_id"] = pair.song_id
                results.append(r)
    return results


def run_block_cross_method(
    pairs: List[ComparisonPair], descriptions: Dict, model: str
) -> List[dict]:
    results = []
    for pair in pairs:
        for q in CROSS_METHOD_QUESTIONS:
            r = compare_pair(pair, descriptions, q["prompt"], q["format"], model)
            if r:
                r["block"] = "cross_method"
                r["question_id"] = q["id"]
                r["concept"] = pair.concept
                r["song_id"] = pair.song_id
                results.append(r)
    return results


# ──────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────


def generate_summary(all_results: Dict[str, List[dict]]) -> str:
    lines = ["# Music Flamingo Evaluation — LLM Comparison Results\n"]

    for block_name, results in all_results.items():
        lines.append(f"\n## {block_name.replace('_', ' ').title()}\n")
        if not results:
            lines.append("No comparison pairs available for this block.\n")
            continue

        # Count preferences
        pref_results = [r for r in results if "answer" in r]
        rating_results = [r for r in results if "rating_a" in r]

        if pref_results:
            a_wins = sum(1 for r in pref_results if r["answer"] == "A")
            b_wins = sum(1 for r in pref_results if r["answer"] == "B")
            equal = sum(1 for r in pref_results if r["answer"] == "Equal")
            total = len(pref_results)
            lines.append(f"Preference comparisons: {total}")
            lines.append(f"  - A wins: {a_wins} ({100*a_wins/total:.0f}%)")
            lines.append(f"  - B wins: {b_wins} ({100*b_wins/total:.0f}%)")
            lines.append(f"  - Equal:  {equal} ({100*equal/total:.0f}%)")
            avg_conf = sum(r.get("confidence", 3) for r in pref_results) / total
            lines.append(f"  - Avg confidence: {avg_conf:.1f}/5\n")

            # Per-concept breakdown
            concepts = sorted(set(r.get("concept", "") for r in pref_results))
            for c in concepts:
                sub = [r for r in pref_results if r.get("concept") == c]
                a_w = sum(1 for r in sub if r["answer"] == "A")
                b_w = sum(1 for r in sub if r["answer"] == "B")
                eq = sum(1 for r in sub if r["answer"] == "Equal")
                lines.append(f"  [{c}] A:{a_w} B:{b_w} Equal:{eq}")

        if rating_results:
            avg_a = sum(r["rating_a"] for r in rating_results) / len(rating_results)
            avg_b = sum(r["rating_b"] for r in rating_results) / len(rating_results)
            lines.append(f"\nQuality ratings (avg): A={avg_a:.2f}, B={avg_b:.2f}")

        lines.append("")

        # Detailed results
        lines.append("### Detailed Results\n")
        for r in results:
            a_file = r.get("audio_a", "?")
            b_file = r.get("audio_b", "?")
            if "answer" in r:
                lines.append(
                    f"- **{r.get('concept','')}** {r.get('song_id','')}: "
                    f"{r['answer']} (conf={r.get('confidence','?')}) "
                    f"| A={a_file} vs B={b_file}"
                )
            elif "rating_a" in r:
                lines.append(
                    f"- **{r.get('concept','')}** {r.get('song_id','')}: "
                    f"A={r['rating_a']}/5, B={r['rating_b']}/5 "
                    f"| A={a_file} vs B={b_file}"
                )
            reasoning = r.get("reasoning", "")
            if reasoning:
                lines.append(f"  > {reasoning}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Comprehensive LLM analysis
# ──────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert researcher analyzing results from a Music Flamingo + LLM "
    "evaluation pipeline that mirrors a human listening study on AI music steering. "
    "Two methods are evaluated: Difference-in-Means (DM) and Sparse Activation "
    "Steering (SAS). Both steer a music transformer's generation toward target "
    "attributes (pitch register, note duration, or both). "
    "Smooth steering applies the intervention gradually over beats rather than abruptly.\n\n"
    "Produce a comprehensive scientific analysis suitable for a research paper."
)

ANALYSIS_USER_TEMPLATE = """\
Below are the complete results from an automated Music Flamingo evaluation.
Each audio was first described by Music Flamingo (a music understanding model),
then pairs of descriptions were compared by GPT-4o to mirror human evaluation questions.

## Raw Results
```json
{results_json}
```

## Summary Statistics
{summary_text}

---

Produce a comprehensive analysis covering:

1. **Steering Effectiveness**: For each method (DM, SAS) and concept (pitch, duration, dual),
   does the LLM detect the intended steering direction? How confident are the judgments?
   Which method/concept combinations are most/least effective?

2. **Quality Degradation**: Does steering hurt musical quality? Is there a quality gap
   between steered and baseline? Which method preserves quality better?

3. **Smooth vs Abrupt**: Does smooth (gradual) steering produce more natural transitions?
   Does it reduce quality degradation? Any concept-specific patterns?

4. **Cross-Method Comparison (DM vs SAS)**: When directly compared on the same songs,
   which method produces stronger effects? Which preserves quality better?
   Any concept-specific advantages?

5. **Key Findings & Highlights**: The 3-5 most important takeaways for a research paper.
   Note any surprising results or limitations of the evaluation.

6. **Limitations**: Caveats about using Music Flamingo descriptions as a proxy for
   human perception, potential biases in the LLM comparison, sample size concerns.

Format the analysis as structured markdown with clear section headers."""


def run_comprehensive_analysis(
    all_results: Dict[str, List[dict]],
    summary_text: str,
    output_dir: Path,
    model: str,
) -> str:
    """Send all results to the LLM for a comprehensive research-quality analysis."""
    logger.info("Running comprehensive analysis...")

    # Truncate individual reasonings to fit context window
    compact_results = {}
    for block, results in all_results.items():
        compact_results[block] = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "reasoning"}
            # Keep reasoning but truncate
            reasoning = r.get("reasoning", "")
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            entry["reasoning"] = reasoning
            compact_results[block].append(entry)

    results_json = json.dumps(compact_results, indent=1)
    user_msg = ANALYSIS_USER_TEMPLATE.format(
        results_json=results_json,
        summary_text=summary_text,
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        analysis = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Comprehensive analysis failed: %s", e)
        analysis = f"Analysis generation failed: {e}"

    # Save
    analysis_path = output_dir / "comprehensive_analysis.md"
    analysis_path.write_text(analysis)
    logger.info("Saved comprehensive analysis to %s", analysis_path)

    return analysis


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def run_compare(
    descriptions_path: Path,
    audio_dir: Path,
    output_dir: Path,
    model: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load descriptions
    descriptions = json.loads(descriptions_path.read_text())
    logger.info("Loaded %d descriptions", len(descriptions))

    # Build catalog and pairs
    catalog = build_catalog(audio_dir)
    all_pairs = build_all_pairs(catalog)

    for block, pairs in all_pairs.items():
        logger.info("Block '%s': %d comparison pairs", block, len(pairs))

    # Run each block
    all_results = {}

    logger.info("\n=== Block: Effectiveness ===")
    all_results["effectiveness"] = run_block_effectiveness(
        all_pairs["effectiveness"], descriptions, model
    )

    logger.info("\n=== Block: Quality ===")
    all_results["quality"] = run_block_quality(
        all_pairs["quality"], descriptions, model
    )

    logger.info("\n=== Block: Smooth vs Abrupt ===")
    all_results["smooth_vs_abrupt"] = run_block_smooth(
        all_pairs["smooth_vs_abrupt"], descriptions, model
    )

    logger.info("\n=== Block: Cross-Method ===")
    all_results["cross_method"] = run_block_cross_method(
        all_pairs["cross_method"], descriptions, model
    )

    # Save raw results
    results_file = output_dir / "comparison_results.json"
    results_file.write_text(json.dumps(all_results, indent=2))
    logger.info("Saved raw results to %s", results_file)

    # Generate and save summary
    summary = generate_summary(all_results)
    summary_file = output_dir / "comparison_summary.md"
    summary_file.write_text(summary)
    logger.info("Saved summary to %s", summary_file)

    # Comprehensive LLM analysis
    logger.info("\n=== Comprehensive Analysis ===")
    analysis = run_comprehensive_analysis(all_results, summary, output_dir, model)

    # Print summary + analysis
    print("\n" + summary)
    print("\n" + "=" * 60)
    print(analysis)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Flamingo descriptions via LLM"
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=Path("flamingo_evaluation/results/descriptions.json"),
    )
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path("AUDIO EVAL TRIM/ALL"),
        help="Audio directory (for catalog building)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("flamingo_evaluation/results"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for comparison (default: gpt-4o)",
    )
    args = parser.parse_args()
    run_compare(args.descriptions, args.audio_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()
