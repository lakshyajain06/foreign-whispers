import os
import json
import logging
import math
import threading
import dataclasses
from groq import Groq

log = logging.getLogger(__name__)

# --- SBERT Setup ---
_SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_SBERT_SIM_FLOOR = 0.55
_sbert_state = {"model": None, "tried": False}
_sbert_lock = threading.Lock()

# --- Tuning ---
_DURATION_BUDGET_SLACK = 1.05
_SCORE_LAMBDA = 10.0

@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget."""
    text: str
    char_count: int
    brevity_rationale: str = ""
    predicted_duration_s: float | None = None
    score: float | None = None

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

def _load_sbert():
    """Lazy load SBERT so we don't hang startup."""
    if _sbert_state["tried"]:
        return _sbert_state["model"]
        
    with _sbert_lock:
        if _sbert_state["tried"]:
            return _sbert_state["model"]
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading SBERT ({_SBERT_MODEL_NAME})...")
            _sbert_state["model"] = SentenceTransformer(_SBERT_MODEL_NAME)
        except Exception as e:
            log.warning(f"SBERT failed to load, guard disabled: {e}")
        
        _sbert_state["tried"] = True
        
    return _sbert_state["model"]


def _semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Check cosine similarity to make sure the translation didn't drift."""
    model = _load_sbert()
    if not model:
        return None
        
    try:
        import numpy as np
        # encode and calculate cosine sim
        embs = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
        denom = float(np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
        return float(embs[0] @ embs[1] / denom)
    except Exception as e:
        log.warning(f"SBERT similarity check crashed: {e}")
        return None


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
    target_lang: str = "es",
    duration_slack: float | None = None,
) -> list[TranslationCandidate]:
    """
    LLM-based shortening with SBERT semantic guards and duration math.
    """
    if not baseline_es or not baseline_es.strip():
        return []

    base_len = len(baseline_es)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("Missing GROQ_API_KEY")
        return []
        
    client = Groq(api_key=api_key)

    # 1. Groq LLM Generation
    query = (
        f"English: {source_text}\n"
        f"Spanish Baseline (Length {base_len}): {baseline_es}\n\n"
        "Give me exactly 3 shorter Spanish variations. "
        "Strategy 1: Remove filler words.\n"
        "Strategy 2: Rephrase entirely.\n"
        "Strategy 3: Absolute shortest possible.\n\n"
        'Return ONLY a JSON list of objects with "text" and "rationale" keys. No markdown.'
    )

    try:
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": query}],
            max_tokens=256,
            temperature=0.25,
        )

        raw_out = chat.choices[0].message.content.strip()
        
        # Scrub markdown hacks
        cleaned = raw_out.replace("```json", "").replace("```", "").strip()
        if "[" in cleaned and "]" in cleaned:
            cleaned = cleaned[cleaned.find("["):cleaned.rfind("]")+1]

        suggestions = json.loads(cleaned)
    except Exception as err:
        log.warning(f"Groq shortening failed: {err}")
        suggestions = []

    # Enforce hard length rule & deduplicate
    raw_candidates = []
    used = set()
    
    for s in suggestions:
        if isinstance(s, dict) and "text" in s:
            val = s["text"].strip()
            rat = s.get("rationale", "LLM shortened")
            if val and val not in used and len(val) < base_len:
                used.add(val)
                raw_candidates.append((val, rat))

    # 2. SBERT Semantic Guard
    sim_map = {}
    if source_text.strip() and raw_candidates:
        survivors = []
        for txt, rat in raw_candidates:
            sim = _semantic_similarity(source_text, txt)
            if sim is None:
                # SBERT offline, just accept it to keep pipeline moving
                sim_map[txt] = 1.0
                survivors.append((txt, rat))
            elif sim >= _SBERT_SIM_FLOOR:
                sim_map[txt] = sim
                survivors.append((txt, f"{rat} [sim={sim:.2f}]"))
            else:
                log.debug(f"Dropped candidate (sim={sim:.2f} < {_SBERT_SIM_FLOOR}): {txt[:80]}")
        raw_candidates = survivors

    # 3. Duration Filter & Combined Score
    try:
        from foreign_whispers.alignment import _estimate_duration
    except ImportError:
        log.error("Could not import _estimate_duration, falling back to basic char math")
        def _estimate_duration(t, l): return len(t) / 15.0

    slack = duration_slack if duration_slack is not None else _DURATION_BUDGET_SLACK
    cap = slack * max(target_duration_s, 0.1)
    
    final_out = []
    
    for txt, rat in raw_candidates:
        d_pred = _estimate_duration(txt, target_lang)
        
        if d_pred > cap:
            log.debug(f"Candidate too long ({d_pred:.2f}s > cap {cap:.2f}s): {txt[:40]}")
            continue
            
        sim = sim_map.get(txt, 1.0)
        
        # Combine duration and semantic similarity into a single score.
        # Lower score is better: favors shorter duration and higher similarity.
        score = d_pred + _SCORE_LAMBDA * (1.0 - sim)
        
        final_out.append(TranslationCandidate(
            text=txt,
            char_count=len(txt),
            brevity_rationale=rat,
            predicted_duration_s=d_pred,
            score=score
        ))
        
    # Sort by score ascending (lowest score is best)
    final_out.sort(key=lambda c: c.score if c.score is not None else float('inf'))
    
    return final_out