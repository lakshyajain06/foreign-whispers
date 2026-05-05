"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

import json
import os
from groq import Groq

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )

def llm_shorten(
    source_text: str,
    baseline_es: str,
    target_chars: int,
    context_prev: str,
    context_next: str,
) -> list[str]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_block = ""
    if context_prev:
        context_block += f"Previous segment: {context_prev}\n"
    if context_next:
        context_block += f"Next segment: {context_next}\n"

    prompt = f"""You are a professional Spanish dubbing translator. Shorten the Spanish translation to fit within {target_chars} characters while preserving meaning.
    Original English: {source_text}
    Baseline Spanish: {baseline_es}
    {context_block}
    Return ONLY a JSON array of 3 shortened Spanish candidates, shortest first. No explanation, no markdown, just the raw JSON array.
    Example: ["short version 1", "short version 2", "short version 3"]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.3,  # l1ow temp for consistency
    )

    raw = response.choices[0].message.content.strip()
    return [c for c in json.loads(raw) if isinstance(c, str)]

import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Assuming TranslationCandidate is imported/defined elsewhere in your file
# from whatever_module import TranslationCandidate

def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
) -> list[TranslationCandidate]:
    """LLM-based shorter translation generation using Groq, mimicking the hybrid pipeline."""
    if not baseline_es or not baseline_es.strip():
        return []

    baseline_len = len(baseline_es)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # We ask the LLM to mimic the distinct strategies of the original pipeline
    prompt = f"""You are a professional Spanish dubbing translator. Your goal is to shorten the baseline Spanish translation while preserving meaning.
    
    Original English: {source_text}
    Baseline Spanish (Length: {baseline_len}): {baseline_es}
    
    Generate exactly 3 shorter candidates using different strategies:
    1. A version that only removes filler words or contracts phrases (Rule-based mimic).
    2. A version that completely rephrases the sentence to be more concise (Alt-translate mimic).
    3. The absolute shortest possible natural-sounding version.
    
    Return ONLY a JSON array of objects with "text" and "rationale" keys. Do NOT wrap in markdown.
    Example: 
    [
      {{"text": "short version", "rationale": "removed filler words"}},
      {{"text": "another version", "rationale": "rephrased entirely"}}
    ]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3, 
        )

        raw = response.choices[0].message.content.strip()

        # Clean up markdown block ticks if the LLM ignores instructions
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
            
        parsed_data = json.loads(raw.strip())
        
        if not isinstance(parsed_data, list):
            logger.warning("[rerank] LLM did not return a list.")
            return []

    except Exception as e:
        logger.error(f"[rerank] Groq LLM generation/parsing failed: {e}")
        return []

    candidates = []
    seen = set()

    for item in parsed_data:
        if isinstance(item, dict) and "text" in item:
            text = item["text"].strip()
            rationale = item.get("rationale", "LLM shortened")
            
            # The ONLY strict requirement: must be shorter than baseline
            if text and text not in seen and len(text) < baseline_len:
                seen.add(text)
                candidates.append(
                    TranslationCandidate(
                        text=text,
                        char_count=len(text),
                        brevity_rationale=rationale
                    )
                )

    # Return sorted by length (shortest first)
    candidates.sort(key=lambda c: c.char_count)
    
    logger.debug(
        "[rerank] baseline=%dc -> %d candidate(s) via Groq",
        baseline_len, len(candidates),
    )
    
    return candidates