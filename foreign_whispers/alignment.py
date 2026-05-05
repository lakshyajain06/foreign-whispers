"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum

import pyphen as _pyphen

import json
import pathlib
import math
from typing import Optional

_VOWEL_PATTERNS: dict[str, str] = {
    # Romance — open vowel systems
    "es": "aeiouáéíóúü",       # Spanish
    "fr": "aeiouyàâäéèêëîïôùûüÿœæ",  # French — nasal vowels still cluster
    "it": "aeiouàèéìíîòóùú",   # Italian
    "pt": "aeiouáâãàéêíóôõú",  # Portuguese
    "ro": "aeiouăâî",           # Romanian
    # Germanic
    "de": "aeiouäöüy",
    "nl": "aeiouäöüy",
    "en": "aeiouy",
    # Default fallback (Latin script)
    "_":  "aeiouy",
}

_SYLLABLE_RATES: dict[str, float] = {
    "es": 5.0,
    "fr": 4.6,
    "it": 5.1,
    "pt": 4.8,
    "ro": 4.5,
    "de": 3.8,
    "nl": 4.0,
    "en": 4.0,
    "_":  4.5,   # original library default — keeps backward compat for unknown langs
}

_SCALE          = 1.05
_WORD_OVERHEAD  = 0.018
_PAUSE_OVERHEAD = 0.12
_INTERCEPT      = 0.0


def _load_fitted_coefficients() -> None:
    global _SCALE, _WORD_OVERHEAD, _PAUSE_OVERHEAD, _INTERCEPT

    json_path = pathlib.Path(__file__).parent / "data" / "duration_model.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    _SCALE          = float(data.get("scale",          _SCALE))
    _WORD_OVERHEAD  = float(data.get("word_overhead",  _WORD_OVERHEAD))
    _PAUSE_OVERHEAD = float(data.get("pause_overhead", _PAUSE_OVERHEAD))
    _INTERCEPT      = float(data.get("intercept",      _INTERCEPT))

    rates = data.get("rates")
    if isinstance(rates, dict):
        for k, v in rates.items():
            try:
                _SYLLABLE_RATES[k] = float(v)
            except (TypeError, ValueError):
                continue


_load_fitted_coefficients()

_PYPHEN_LANGS = {"es": "es_ES", "fr": "fr_FR", "it": "it_IT",
                 "pt": "pt_PT", "ro": "ro_RO", "de": "de_DE",
                 "nl": "nl_NL", "en": "en_US"}
_pyphen_cache: dict[str, "_pyphen.Pyphen"] = {}


def _get_pyphen_dic(lang: str) -> "_pyphen.Pyphen":
    """Get or lazily create a pyphen instance for the given language."""
    dic = _pyphen_cache.get(lang)
    if dic is None:
        dic = _pyphen.Pyphen(lang=_PYPHEN_LANGS.get(lang, "en_US"))
        _pyphen_cache[lang] = dic
    return dic

def _get_lang_key(lang: Optional[str]) -> str:
    """Normalise a BCP-47 language tag to a two-letter key we have data for."""
    if not lang:
        return "_"
    prefix = lang.lower()[:2]
    return prefix if prefix in _SYLLABLE_RATES else "_"



def _count_syllables(text: str, lang: Optional[str] = None) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    if not text:
        return 1

    lang_key = _get_lang_key(lang)

    pyphen_obj = _get_pyphen_dic(lang_key)
    if pyphen_obj is not None:
        total = 0
        for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
            hyphenated = pyphen_obj.inserted(word.lower())
            total += hyphenated.count("-") + 1
        return max(1, total)

    if lang_key in _VOWEL_PATTERNS and lang_key != "_":
        # Language has its own accented-vowel pattern → keep accents intact.
        vowels  = _VOWEL_PATTERNS[lang_key]
        working = text.lower()
    else:
        # Unknown / generic — strip diacritics so accented vowels still count.
        # This preserves the pre-notebook_5 behaviour and keeps the legacy
        # ``test_syllable_count_accents`` test passing.
        nfkd    = unicodedata.normalize("NFKD", text.lower())
        working = "".join(c for c in nfkd if not unicodedata.combining(c))
        vowels  = _VOWEL_PATTERNS["_"]

    pattern  = f"[{re.escape(vowels)}]+"
    clusters = re.findall(pattern, working)
    return max(1, len(clusters))

def _count_pause_markers(text: str) -> int:
    return len(re.findall(r"[,;:](?!\s*$)", text))

def _count_words(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(text.split())

def _estimate_duration(text: str, lang: Optional[str] = None) -> float:
    """Estimate TTS duration in seconds using a syllable-rate heuristic."""
    if not text or not text.strip():
        return 0.0

    lang_key  = _get_lang_key(lang)
    rate      = _SYLLABLE_RATES[lang_key]
    syllables = _count_syllables(text, lang)
    n_words   = _count_words(text)
    n_pauses  = _count_pause_markers(text)

    base       = (syllables / rate) * _SCALE
    correction = n_words * _WORD_OVERHEAD + n_pauses * _PAUSE_OVERHEAD
    return max(0.0, base + correction + _INTERCEPT)



@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5–5.1 syllables/second depending on language) and derive three key
    numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        lang: Optional BCP-47 language code for the *target* language.
        predicted_tts_s: Estimated TTS duration from ``_estimate_duration``.
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30 % longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    lang:              Optional[str]          = None
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text, self.lang)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
    lang: Optional[str] = None,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
            lang              = lang,
        ))
    return metrics

def _build_gap_index(silence_regions: list[dict]) -> list[tuple[float, float, float]]:
   
    gaps = []
    for r in silence_regions:
        if r.get("label") == "silence":
            s, e = float(r["start_s"]), float(r["end_s"])
            if e > s:
                gaps.append((s, e, e - s))
    gaps.sort(key=lambda t: t[0])
    return gaps

def _silence_after_fast(end_s: float, gaps: list[tuple[float, float, float]]) -> float:
    # Binary search for the first gap whose start >= end_s - 0.1
    lo, hi = 0, len(gaps)
    target = end_s - 0.1
    while lo < hi:
        mid = (lo + hi) // 2
        if gaps[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(gaps):
        return gaps[lo][2]
    return 0.0


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


# DP 

_W_OVERLAP        = 20.0   # per-second penalty for segment-to-segment overlap
_W_SEVERE_STRETCH = 10.0   # flat penalty each time stretch_factor > max_stretch
_W_DRIFT          = 1.0    # per-second penalty for cumulative timeline drift

_W_REQUEST_SHORTER = 5.0    # downstream re-rank may not find a shorter candidate
_W_FAIL            = 100.0  # last-resort silence fallback — strongly discouraged


_W_STRETCH_GRADIENT = 2.0   # per unit of |stretch_factor − 1.0|

def _segment_cost(
    action:        AlignAction,
    gap_shift_s:   float,
    stretch_factor: float,
    cumulative_drift: float,
    overlap_s:     float,
    max_stretch:   float,
) -> float:
    cost = 0.0
    cost += cumulative_drift * _W_DRIFT
    cost += overlap_s        * _W_OVERLAP
    # Smooth stretch penalty (Bug 2): proportional to deviation from 1.0×.
    cost += abs(stretch_factor - 1.0) * _W_STRETCH_GRADIENT
    # Hard penalty for severe stretch (kept for callers that bypass clamping).
    if stretch_factor > max_stretch:
        cost += _W_SEVERE_STRETCH
    # Action-specific penalties (Bug 1).
    if action == AlignAction.REQUEST_SHORTER:
        cost += _W_REQUEST_SHORTER
    elif action == AlignAction.FAIL:
        cost += _W_FAIL
    return cost

    
@dataclasses.dataclass(order=True)
class _BeamState:
    cost:      float
    drift:     float
    decisions: list[tuple[AlignAction, float, float]] = dataclasses.field(
        default_factory=list, compare=False
    )

def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    beam_width:      int   = 8,
) -> list[AlignedSegment]:
    if not metrics:
        return []

    gaps = _build_gap_index(silence_regions)

    # ------------------------------------------------------------------
    # Candidate generator
    # ------------------------------------------------------------------
    def _candidates(
        m: SegmentMetrics,
        drift: float,
    ) -> list[tuple[AlignAction, float, float]]:
        """Return all feasible (action, gap_shift_s, stretch_factor) triples.

        We always include ACCEPT so the beam is never empty.  We add the
        other actions according to feasibility, not threshold order, so the
        beam can explore sub-optimal local choices that may reduce global cost.
        """
        avail_gap  = _silence_after_fast(m.source_end, gaps)
        sf         = m.predicted_stretch
        candidates = []

        # ACCEPT — only when the segment naturally fits (sf <= 1.1).  For
        # over-budget segments, ACCEPT means "engine will truncate the
        # trailing audio" which causes the user-visible silence problem;
        # exposing it as a candidate here lets the beam pick a free option
        # that secretly trashes audio.  Keeping ACCEPT restricted to the
        # natural-fit band mirrors ``decide_action`` semantics.
        if sf <= 1.1:
            candidates.append((AlignAction.ACCEPT, 0.0, 1.0))

        # MILD_STRETCH — feasible for any stretch > 1.0; this is the floor
        # for over-budget segments (always at least one candidate available).
        # Pass the un-clamped ``sf`` so the cost function can fire the
        # severe-stretch penalty when the segment can't fit under the cap.
        # The engine will still clamp at synthesis time, but beam can now
        # see the truncation cost and prefer REQUEST_SHORTER when warranted.
        if sf > 1.0:
            candidates.append((AlignAction.MILD_STRETCH, 0.0, sf))

        # GAP_SHIFT — feasible only when the gap can absorb the overflow
        if sf > 1.1 and avail_gap >= m.overflow_s:
            candidates.append((AlignAction.GAP_SHIFT, m.overflow_s, 1.0))

        # PARTIAL GAP_SHIFT — borrow only part of the gap, combine with stretch
        # This helps when neither pure GAP_SHIFT nor pure MILD_STRETCH alone is ideal.
        if sf > 1.4 and avail_gap > 0 and avail_gap < m.overflow_s:
            partial = avail_gap
            remaining_overflow = m.overflow_s - partial
            remaining_sf = (m.predicted_tts_s - partial) / m.source_duration_s if m.source_duration_s > 0 else 1.0
            partial_stretch = min(remaining_sf, max_stretch)
            candidates.append((AlignAction.GAP_SHIFT, partial, partial_stretch))

        # REQUEST_SHORTER (Bug 4 fix) — feasible whenever the segment is
        # over-budget AND can't be fully absorbed by silence.  Previously this
        # only fired for sf > 1.8, which left the (1.4, 1.8] band silently
        # falling through to a clamped MILD_STRETCH that the engine would then
        # truncate.  Mirrors the threshold logic in ``decide_action``.
        if sf > 1.4 and avail_gap < m.overflow_s:
            candidates.append((AlignAction.REQUEST_SHORTER, 0.0, 1.0))

        # FAIL — always a last resort (expensive but prevents infinite loops)
        if sf > 2.5:
            candidates.append((AlignAction.FAIL, 0.0, 1.0))

        return candidates

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------
    # Initial beam: one empty hypothesis with zero cost and zero drift.
    beam: list[_BeamState] = [_BeamState(cost=0.0, drift=0.0, decisions=[])]

    for idx, m in enumerate(metrics):
        # Determine the next segment's original start (for overlap detection).
        next_start = metrics[idx + 1].source_start if idx + 1 < len(metrics) else math.inf

        next_beam: list[_BeamState] = []

        for hyp in beam:
            for action, gap_shift_s, stretch_factor in _candidates(m, hyp.drift):
                # Compute the scheduled end of this segment under this hypothesis.
                sched_start = m.source_start + hyp.drift
                sched_end   = sched_start + m.source_duration_s + gap_shift_s

                # Overlap: how far does this segment push into the next one?
                overlap_s = max(0.0, sched_end - next_start)

                # New cumulative drift after this segment.
                new_drift = hyp.drift + gap_shift_s

                # Incremental cost for this choice.
                inc_cost = _segment_cost(
                    action        = action,
                    gap_shift_s   = gap_shift_s,
                    stretch_factor= stretch_factor,
                    cumulative_drift = new_drift,
                    overlap_s     = overlap_s,
                    max_stretch   = max_stretch,
                )

                next_beam.append(_BeamState(
                    cost      = hyp.cost + inc_cost,
                    drift     = new_drift,
                    decisions = hyp.decisions + [(action, gap_shift_s, stretch_factor)],
                ))

        # Prune beam: keep only the B lowest-cost hypotheses.
        next_beam.sort(key=lambda h: h.cost)
        beam = next_beam[:beam_width]

    # ------------------------------------------------------------------
    # Reconstruct the best hypothesis
    # ------------------------------------------------------------------
    best = beam[0]
    aligned: list[AlignedSegment] = []
    cumulative_drift = 0.0

    for m, (action, gap_shift_s, stretch_factor) in zip(metrics, best.decisions):
        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift_s

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift_s,
            stretch_factor  = stretch_factor,
        ))

        cumulative_drift += gap_shift_s

    return aligned

def compare_alignments(
    greedy: list[AlignedSegment],
    dp:     list[AlignedSegment],
) -> dict:
    def _stats(segs: list[AlignedSegment]) -> dict:
        total_drift    = sum(s.gap_shift_s for s in segs)
        severe_stretch = sum(1 for s in segs if s.stretch_factor > 1.4)
        overlaps       = sum(
            1 for a, b in zip(segs, segs[1:])
            if a.scheduled_end > b.scheduled_start + 0.01  # 10 ms tolerance
        )
        req_shorter    = sum(1 for s in segs if s.action == AlignAction.REQUEST_SHORTER)
        fails          = sum(1 for s in segs if s.action == AlignAction.FAIL)
        return {
            "total_drift_s":        round(total_drift, 3),
            "severe_stretch_count": severe_stretch,
            "overlap_count":        overlaps,
            "request_shorter_count":req_shorter,
            "fail_count":           fails,
        }

    return {"greedy": _stats(greedy), "dp": _stats(dp)}