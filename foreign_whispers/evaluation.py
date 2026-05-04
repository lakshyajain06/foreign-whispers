"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


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


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    clip_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality evaluation.

    Scores a clip across four independent dimensions, each normalized to
    [0, 1] where 1.0 = perfect:

    **Timing accuracy** — how close predicted TTS durations are to the source
    windows.  Computed as ``1 - clamp(mean_abs_error / 5.0, 0, 1)``.
    A mean error of 0 s → 1.0; ≥ 5 s → 0.0.

    **Action coverage** — what fraction of segments were resolved without
    needing retranslation or failure.  ``(n_accept + n_stretch + n_shift) / n``.

    **Speaking-rate consistency** — how uniform the stretch factors are
    across segments.  Computed as ``1 - clamp(stdev(stretches) / 0.5, 0, 1)``.
    Low variance → 1.0; stdev ≥ 0.5 → 0.0.

    **Drift penalty** — how much cumulative drift the schedule introduces.
    ``1 - clamp(abs(drift) / 10.0, 0, 1)``.  Zero drift → 1.0; ≥ 10 s → 0.0.

    An **overall** score averages the four dimensions.

    Args:
        metrics: Per-segment timing metrics.
        aligned: Aligned segments from ``global_align`` or ``global_align_dp``.
        clip_report: Optional pre-computed ``clip_evaluation_report`` dict.
            If ``None``, it will be computed automatically.

    Returns:
        Dict with keys ``timing_accuracy``, ``action_coverage``,
        ``rate_consistency``, ``drift_penalty``, and ``overall``.
    """
    if not metrics or not aligned:
        return {
            "timing_accuracy":    0.0,
            "action_coverage":    0.0,
            "rate_consistency":   0.0,
            "drift_penalty":      0.0,
            "overall":            0.0,
        }

    # Compute clip report if not provided
    if clip_report is None:
        clip_report = clip_evaluation_report(metrics, aligned)

    n = len(metrics)

    # --- Timing accuracy ---
    mean_err = clip_report.get("mean_abs_duration_error_s", 0.0)
    timing_accuracy = max(0.0, min(1.0, 1.0 - mean_err / 5.0))

    # --- Action coverage ---
    resolved_actions = {AlignAction.ACCEPT, AlignAction.MILD_STRETCH,
                        AlignAction.GAP_SHIFT}
    n_resolved = sum(1 for a in aligned if a.action in resolved_actions)
    action_coverage = n_resolved / max(n, 1)

    # --- Speaking-rate consistency ---
    stretches = [a.stretch_factor for a in aligned]
    if len(stretches) >= 2:
        rate_stdev = _stats.stdev(stretches)
    else:
        rate_stdev = 0.0
    rate_consistency = max(0.0, min(1.0, 1.0 - rate_stdev / 0.5))

    # --- Drift penalty ---
    drift = abs(clip_report.get("total_cumulative_drift_s", 0.0))
    drift_penalty = max(0.0, min(1.0, 1.0 - drift / 10.0))

    # --- Overall ---
    overall = (timing_accuracy + action_coverage
               + rate_consistency + drift_penalty) / 4.0

    return {
        "timing_accuracy":    round(timing_accuracy, 3),
        "action_coverage":    round(action_coverage, 3),
        "rate_consistency":   round(rate_consistency, 3),
        "drift_penalty":      round(drift_penalty, 3),
        "overall":            round(overall, 3),
    }
