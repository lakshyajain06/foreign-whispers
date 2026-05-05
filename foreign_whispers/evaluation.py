"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
The original ``clip_evaluation_report`` is kept as the timing-only scorecard;
``dubbing_scorecard`` (NB5 Task 4) extends it across four dimensions:
timing, intelligibility, semantic fidelity, and naturalness.
"""
import logging
import math
import pathlib
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    _count_syllables,
    decide_action,
)

logger = logging.getLogger(__name__)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


# ---------------------------------------------------------------------------
# NB5 Task 4 — multi-dimensional dubbing scorecard
# ---------------------------------------------------------------------------

# Normalisation constants — chosen to map "perfect" to 1.0 and "very bad" to 0.0
# on a clip-by-clip basis.  These are not theoretical maxima; they reflect what
# we have measured on Hormuz and similar 60-Minutes-style content.
_TIMING_MAE_PERFECT  = 0.0    # MAE ≤ 0.0s  → score 1.0
_TIMING_MAE_FLOOR    = 1.0    # MAE ≥ 1.0s  → score 0.0
_PCT_SEVERE_FLOOR    = 30.0   # 30% severe  → score 0.0

_INTEL_WER_FLOOR     = 0.6    # WER ≥ 0.6   → score 0.0 (almost unrecognisable)

_NATURAL_RATE_TARGET = 5.0    # syllables/s — Spanish reference cadence
_NATURAL_RATE_FLOOR  = 4.0    # |stdev|/target above this floor → score 0.0


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein on whitespace-tokenised words.

    Returns a value in [0, 1] (well, technically unbounded for very short
    references, but capped at 1.0 here for stability).  Empty references
    return 0.0 if hypothesis is also empty, else 1.0.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0

    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return min(1.0, prev[m] / n)


_local_whisper_state: dict = {"model": None, "tried": False, "name": None}


def _local_whisper_transcribe(
    audio_path: str, target_lang: str, model_name: str = "base"
) -> tuple[str | None, str]:
    """Transcribe via openai-whisper running in-process.

    Returns ``(text, status_message)``.  ``text`` is ``None`` on failure;
    ``status_message`` is a one-line description (used for debug output).
    """
    if _local_whisper_state["tried"] and _local_whisper_state["model"] is None:
        return None, "openai-whisper previously failed to load"

    if (_local_whisper_state["model"] is None
        or _local_whisper_state["name"] != model_name):
        try:
            import whisper as _whisper  # type: ignore
            _local_whisper_state["model"] = _whisper.load_model(model_name)
            _local_whisper_state["name"]  = model_name
            _local_whisper_state["tried"] = True
        except Exception as exc:
            _local_whisper_state["tried"] = True
            return None, f"openai-whisper load failed: {exc}"

    try:
        result = _local_whisper_state["model"].transcribe(
            audio_path, language=target_lang, fp16=False,
        )
        text = (result.get("text") or "").strip()
        return text, f"local whisper-{model_name}"
    except Exception as exc:
        return None, f"openai-whisper transcribe failed: {exc}"


def _intelligibility_score(
    audio_path:    str | None,
    translations:  list[str],
    speaches_url:  str = "http://localhost:8000",
    target_lang:   str = "es",
    speaches_model: str = "Systran/faster-whisper-small",
    local_model:   str = "base",
    use_local_fallback: bool = True,
) -> tuple[float, dict]:
    """Round-trip intelligibility via Whisper STT.

    Strategy:

    1. Try speaches at ``speaches_url`` with ``speaches_model``.
    2. If that fails (404, model missing, container down, etc.) and
       ``use_local_fallback`` is True, fall back to openai-whisper running
       in-process with ``local_model`` (default "base", ~140 MB download).
    3. Compute WER between the transcription and the concatenated
       translations.  WER 0.0 → score 1.0; WER ≥ ``_INTEL_WER_FLOOR`` → 0.0.

    Returns ``(score, debug_info)``.  Score is 0.0 with ``debug.error`` if
    every backend fails.
    """
    debug: dict = {"speaches_url": speaches_url}
    if not audio_path or not pathlib.Path(audio_path).exists():
        debug["error"] = "audio file not provided or missing"
        return 0.0, debug

    hyp: str | None = None
    backend_used: str | None = None

    # Backend 1: speaches HTTP API
    try:
        import requests  # type: ignore
        with open(audio_path, "rb") as fh:
            resp = requests.post(
                f"{speaches_url}/v1/audio/transcriptions",
                files={"file": (pathlib.Path(audio_path).name, fh, "audio/wav")},
                data={"model": speaches_model, "language": target_lang},
                timeout=180,
            )
        resp.raise_for_status()
        body = resp.json()
        hyp  = (body.get("text") or "").strip()
        backend_used = f"speaches:{speaches_model}"
    except Exception as exc:  # noqa: BLE001
        debug["speaches_error"] = str(exc)[:200]

    # Backend 2: openai-whisper local fallback
    if hyp is None and use_local_fallback:
        text, status = _local_whisper_transcribe(audio_path, target_lang, local_model)
        debug["local_whisper_status"] = status
        if text is not None:
            hyp          = text
            backend_used = f"openai-whisper:{local_model}"

    if hyp is None:
        debug["error"] = "all STT backends failed"
        return 0.0, debug

    ref = " ".join(t for t in translations if t).strip()
    wer = _wer(ref, hyp)
    debug.update({
        "backend":  backend_used,
        "wer":      round(wer, 3),
        "ref_len":  len(ref),
        "hyp_len":  len(hyp),
    })

    score = max(0.0, 1.0 - wer / _INTEL_WER_FLOOR)
    return score, debug


def _semantic_fidelity_score(
    source_texts: list[str], target_texts: list[str]
) -> tuple[float, dict]:
    """Multilingual SBERT cosine similarity between paired source/target texts.

    Pairs are scored independently; the clip score is the mean cosine.
    Returns 0.0 with ``debug.error`` when SBERT is unavailable.
    """
    debug: dict = {"model": "paraphrase-multilingual-MiniLM-L12-v2"}
    if not source_texts or not target_texts:
        debug["error"] = "empty input"
        return 0.0, debug

    try:
        from foreign_whispers.reranking import _load_sbert
        sbert = _load_sbert()
    except Exception as exc:
        debug["error"] = f"sbert load failed: {exc}"
        return 0.0, debug
    if sbert is None:
        debug["error"] = "sbert unavailable"
        return 0.0, debug

    try:
        import numpy as np
        pairs = list(zip(source_texts, target_texts))
        srcs  = [s for s, _ in pairs]
        tgts  = [t for _, t in pairs]
        embs_s = sbert.encode(srcs, convert_to_numpy=True, show_progress_bar=False)
        embs_t = sbert.encode(tgts, convert_to_numpy=True, show_progress_bar=False)
        norms_s = np.linalg.norm(embs_s, axis=1) + 1e-8
        norms_t = np.linalg.norm(embs_t, axis=1) + 1e-8
        cosines = (embs_s * embs_t).sum(axis=1) / (norms_s * norms_t)
        mean_cos = float(np.mean(cosines))
    except Exception as exc:
        debug["error"] = f"sbert encode failed: {exc}"
        return 0.0, debug

    debug.update({"mean_cosine": round(mean_cos, 3), "n_pairs": len(pairs)})
    score = max(0.0, min(1.0, mean_cos))
    return score, debug


def _naturalness_score(
    aligned: list[AlignedSegment], target_lang: str = "es"
) -> tuple[float, dict]:
    """Speaking-rate consistency across segments.

    Computes per-segment syllables/second from the *scheduled* segment
    duration (the time the dub actually occupies on-screen), then scores by
    how close the standard deviation is to zero.  A perfectly consistent
    dub at any rate scores 1.0; a dub whose stdev exceeds
    ``_NATURAL_RATE_FLOOR`` syllables/s scores 0.0.
    """
    debug: dict = {"target_lang": target_lang}
    if not aligned:
        debug["error"] = "no aligned segments"
        return 0.0, debug

    rates = []
    for seg in aligned:
        dur = max(0.05, seg.scheduled_end - seg.scheduled_start)
        syl = _count_syllables(seg.text or "", target_lang)
        rates.append(syl / dur)
    if len(rates) < 2:
        debug.update({"n": len(rates)})
        return 1.0, debug

    rate_mean = _stats.mean(rates)
    rate_std  = _stats.stdev(rates)
    debug.update({
        "rate_mean": round(rate_mean, 3),
        "rate_std":  round(rate_std, 3),
        "n":         len(rates),
    })

    score = max(0.0, 1.0 - rate_std / _NATURAL_RATE_FLOOR)
    return score, debug


def _timing_score(report: dict) -> tuple[float, dict]:
    """Reduce the timing dict to a [0,1] score.

    Combines:
      - Mean absolute duration error: 0s → 1.0, ≥ 1.0s → 0.0.
      - Pct severe stretch: 0% → 1.0, ≥ 30% → 0.0.

    The clip score is the mean of these two sub-scores.
    """
    mae        = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift      = abs(report.get("total_cumulative_drift_s", 0.0))

    mae_score    = max(0.0, 1.0 - (mae - _TIMING_MAE_PERFECT) /
                                  max(_TIMING_MAE_FLOOR - _TIMING_MAE_PERFECT, 1e-6))
    severe_score = max(0.0, 1.0 - pct_severe / _PCT_SEVERE_FLOOR)
    score        = (mae_score + severe_score) / 2.0
    debug = {
        "mae_s":            round(mae,        3),
        "pct_severe":       round(pct_severe, 1),
        "drift_s":          round(drift,      3),
        "mae_subscore":     round(mae_score,    3),
        "severe_subscore":  round(severe_score, 3),
    }
    return score, debug


def dubbing_scorecard(
    metrics:        list[SegmentMetrics],
    aligned:        list[AlignedSegment],
    audio_path:     str | None       = None,
    source_texts:   list[str] | None = None,
    target_texts:   list[str] | None = None,
    target_lang:    str              = "es",
    speaches_url:   str              = "http://localhost:8000",
    weights:        dict[str, float] | None = None,
) -> dict:
    """Compute a four-dimensional dubbing quality scorecard.

    Dimensions (each in [0, 1], higher = better):

    - **timing**         — derived from ``clip_evaluation_report`` (MAE +
      pct_severe_stretch).  Always populated.
    - **intelligibility** — Whisper STT round-trip WER on the dubbed audio.
      Requires ``audio_path`` and a running speaches container.  Returns 0.0
      with ``debug.intelligibility.error`` when unavailable.
    - **semantic_fidelity** — multilingual SBERT cosine between paired source
      and target texts.  Falls back to 0.0 when ``source_texts``/``target_texts``
      aren't supplied or SBERT can't load.
    - **naturalness**    — 1 − stdev(syllables/sec across segments) / floor.
      Always populated.

    The ``overall`` field is the weighted mean of the four dimensions.

    Args:
        metrics: Output of ``compute_segment_metrics``.
        aligned: Output of ``global_align`` or ``global_align_dp``.
        audio_path: Path to the dubbed WAV/MP4 (used for intelligibility).
        source_texts: Per-segment source-language strings (English).
        target_texts: Per-segment target-language strings (Spanish).
        target_lang: BCP-47 code for the target language.
        speaches_url: STT backend URL (defaults to local docker compose).
        weights: Optional per-dimension weights.  Missing dimensions default
            to ``{"timing": 1.0, "intelligibility": 1.0,
                  "semantic_fidelity": 1.0, "naturalness": 1.0}``.

    Returns:
        Dict with keys ``timing``, ``intelligibility``, ``semantic_fidelity``,
        ``naturalness``, ``overall``, and ``debug`` (per-dimension diagnostic
        info).  All dimension scores are floats in [0, 1].
    """
    weights = {**{"timing": 1.0, "intelligibility": 1.0,
                  "semantic_fidelity": 1.0, "naturalness": 1.0},
               **(weights or {})}

    timing_report = clip_evaluation_report(metrics, aligned)
    timing_s,   timing_dbg   = _timing_score(timing_report)
    natural_s,  natural_dbg  = _naturalness_score(aligned, target_lang)

    intel_s, intel_dbg = _intelligibility_score(
        audio_path, target_texts or [], speaches_url, target_lang,
    )

    if source_texts and target_texts:
        sem_s, sem_dbg = _semantic_fidelity_score(source_texts, target_texts)
    else:
        sem_s, sem_dbg = 0.0, {"error": "source_texts/target_texts not provided"}

    parts = {
        "timing":            timing_s,
        "intelligibility":   intel_s,
        "semantic_fidelity": sem_s,
        "naturalness":       natural_s,
    }
    total_w = sum(weights[k] for k in parts) or 1.0
    overall = sum(parts[k] * weights[k] for k in parts) / total_w

    return {
        "timing":            round(timing_s,  3),
        "intelligibility":   round(intel_s,   3),
        "semantic_fidelity": round(sem_s,     3),
        "naturalness":       round(natural_s, 3),
        "overall":           round(overall,   3),
        "debug": {
            "timing":            timing_dbg,
            "intelligibility":   intel_dbg,
            "semantic_fidelity": sem_dbg,
            "naturalness":       natural_dbg,
            "weights":           weights,
            "timing_report":     timing_report,
        },
    }


def plot_scorecard(
    scorecards: dict[str, dict],
    save_to:    str | pathlib.Path | None = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a grouped bar chart comparing multiple scorecards.

    Args:
        scorecards: Mapping of label → ``dubbing_scorecard`` output dict.
            E.g. ``{"greedy": <scorecard1>, "DP": <scorecard2>}``.
        save_to: Optional path to save the figure as PNG.

    Returns:
        The Matplotlib ``Figure`` so the caller can ``plt.show()`` it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    dims   = ["timing", "intelligibility", "semantic_fidelity", "naturalness", "overall"]
    labels = list(scorecards.keys())
    n_lbls = len(labels)
    n_dims = len(dims)

    width = 0.8 / max(n_lbls, 1)
    x     = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lbl in enumerate(labels):
        vals = [float(scorecards[lbl].get(d, 0.0)) for d in dims]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (higher = better)")
    ax.set_title("Dubbing quality scorecard")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(str(save_to), dpi=150)
    return fig
