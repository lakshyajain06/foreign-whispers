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

import pyphen # syllabbifier lib

def _count_syllables(text: str) -> int:
    """Count syllables in target-language text using Pyphen.

    Designed for Spanish (lang='es'). Returns at least 1 for any non-empty text.
    """
    if not text.strip():
        return 0
    dic = pyphen.Pyphen(lang='es')
    count = 0
    for word in text.split():
        # Pyphen returns 'word-with-hy-phens'
        hyphenated = dic.inserted(word)
        count += len(hyphenated.split('-'))
    return max(1, count)


# Linear regression constants (seconds = syllables * slope + overhead)
# Updated from regression results in alignment_integration.ipynb
_SYLLABLE_SLOPE = 0.0897  # Seconds per syllable
_TTS_OVERHEAD = 0.7168    # Fixed silence/buffer intercept


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds using a syllable-rate heuristic."""
    syllables = _count_syllables(text)
    if syllables == 0:
        return 0.0
    
    return (syllables * _SYLLABLE_SLOPE) + _TTS_OVERHEAD


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
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
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
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
        ))
    return metrics


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


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Slack-redistribution global alignment optimizer.

    Improvements over greedy ``global_align``:

    1. **Slack harvesting** — segments where predicted TTS is shorter than
       the source window have *slack* (unused time).  The greedy algorithm
       ignores this; we harvest it and redistribute to overflow segments.
    2. **Priority allocation** — overflow segments are ranked by severity
       (worst first) and receive harvested slack before milder cases.
    3. **Natural gap detection** — also detects inter-segment gaps from
       Whisper timestamps and VAD silence regions.

    The key insight: in the test data, 78 segments have a combined 236 s of
    slack while only 12 s of overflow is needed.  By compressing the
    scheduled windows of slack segments, we free up virtual gaps that
    overflow segments can shift into.

    Algorithm:

    1. Compute per-segment slack (source_duration − predicted_tts, if > 0)
       and per-segment overflow.
    2. Also compute natural inter-segment gaps and VAD gaps.
    3. Pool all available time: slack + gaps = total budget.
    4. Sort overflow segments by severity (worst first).
    5. Allocate budget to overflow segments as gap_shifts, converting
       ``REQUEST_SHORTER`` / ``FAIL`` into ``GAP_SHIFT``.
    6. Build the final schedule: slack segments get compressed windows,
       overflow segments get extended windows, cumulative drift is tracked.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable.
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    if not metrics:
        return []

    n = len(metrics)

    # --- Step 1: compute per-segment slack and overflow ------------------
    slack  = [0.0] * n  # time we can harvest from each segment
    overflow = [0.0] * n  # time each segment needs beyond its window

    for i, m in enumerate(metrics):
        diff = m.source_duration_s - m.predicted_tts_s
        if diff > 0.05:   # segment finishes early → has harvestable slack
            slack[i] = diff
        elif diff < -0.01:  # segment overflows
            overflow[i] = -diff  # positive overflow amount

    # --- Step 2: compute explicit gaps (VAD + natural) -------------------
    def _silence_after_vad(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    explicit_gaps = [0.0] * n
    for i in range(n):
        vad_gap = _silence_after_vad(metrics[i].source_end)
        nat_gap = (max(0.0, metrics[i + 1].source_start - metrics[i].source_end)
                   if i < n - 1 else 0.0)
        explicit_gaps[i] = max(vad_gap, nat_gap)

    # --- Step 3: pool the total available budget -------------------------
    total_slack = sum(slack)
    total_gaps  = sum(explicit_gaps)
    total_budget = total_slack + total_gaps
    total_overflow = sum(overflow)

    # --- Step 4: prioritize overflow segments ----------------------------
    # (overflow_amount, segment_index) sorted worst-first
    overflow_candidates = sorted(
        [(overflow[i], i) for i in range(n) if overflow[i] > 0.01],
        key=lambda x: -x[0],
    )

    # --- Step 5: allocate budget to overflow segments --------------------
    gap_assignments: dict[int, float] = {}  # seg_index → gap_shift amount
    budget_remaining = total_budget

    for ov, idx in overflow_candidates:
        if budget_remaining <= 0.01:
            break
        alloc = min(ov, budget_remaining)
        gap_assignments[idx] = alloc
        budget_remaining -= alloc

    # --- Step 6: determine actions and build schedule --------------------
    # Compute how much slack to harvest from each slack segment
    # Distribute proportionally based on available slack
    slack_to_harvest = min(total_slack, total_overflow)
    harvest_fractions = {}
    if total_slack > 0 and slack_to_harvest > 0:
        for i in range(n):
            if slack[i] > 0:
                # Each slack segment donates proportional to its slack
                harvest_fractions[i] = slack[i] / total_slack * slack_to_harvest

    aligned: list[AlignedSegment] = []
    cumulative_drift = 0.0

    for i, m in enumerate(metrics):
        gap_shift = gap_assignments.get(i, 0.0)
        harvested = harvest_fractions.get(i, 0.0)

        # Determine action
        if gap_shift > 0.01:
            # This segment gets extra time via gap_shift
            effective_stretch = m.predicted_tts_s / (m.source_duration_s + gap_shift) \
                if (m.source_duration_s + gap_shift) > 0 else m.predicted_stretch
            if effective_stretch <= 1.1:
                action = AlignAction.ACCEPT
            elif effective_stretch <= 1.4:
                action = AlignAction.MILD_STRETCH
            else:
                action = AlignAction.GAP_SHIFT
            stretch = min(effective_stretch, max_stretch) if action == AlignAction.MILD_STRETCH else 1.0
        else:
            # Normal action evaluation (no gap allocated)
            available = explicit_gaps[i]
            action = decide_action(m, available_gap_s=available)
            stretch = 1.0
            if action == AlignAction.GAP_SHIFT:
                gap_shift = m.overflow_s
            elif action == AlignAction.MILD_STRETCH:
                stretch = min(m.predicted_stretch, max_stretch)

        # Compress slack segments: shorten their scheduled window
        compression = harvested if harvested > 0 else 0.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift - compression

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

        # Drift: gap_shifts push forward, compressions pull back
        cumulative_drift += gap_shift - compression

    return aligned

