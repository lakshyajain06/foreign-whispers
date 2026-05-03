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
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Returns a dict with scores per dimension, each normalised to [0, 1]
    where 1.0 is best (perfect quality) and 0.0 is worst.

    Dimensions:
        timing_accuracy: Inverse of mean absolute duration error, clamped.
            A perfect score means predicted TTS duration equals the source window.
        naturalness: Consistency of speaking rate across segments.
            Measures coefficient of variation of predicted stretch factors;
            lower variance = more natural-sounding pacing.
        severity_distribution: Fraction of segments that are ACCEPT or MILD_STRETCH.
            Higher means fewer segments need drastic intervention.
        overall: Weighted mean of the three dimensions.

    Args:
        metrics: Per-segment timing metrics.
        aligned: Aligned segments from global_align or global_align_dp.

    Returns:
        Dict with keys ``timing_accuracy``, ``naturalness``,
        ``severity_distribution``, and ``overall``, each in [0, 1].
    """
    if not metrics:
        return {
            "timing_accuracy": 1.0,
            "naturalness": 1.0,
            "severity_distribution": 1.0,
            "overall": 1.0,
        }

    # --- Timing accuracy (0 = 3s+ mean error, 1 = 0 error) ------------------
    errors = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    mean_err = _stats.mean(errors)
    timing_accuracy = max(0.0, 1.0 - mean_err / 3.0)

    # --- Naturalness (speaking rate variance) --------------------------------
    stretches = [m.predicted_stretch for m in metrics]
    if len(stretches) >= 2:
        mean_s = _stats.mean(stretches)
        stdev_s = _stats.stdev(stretches)
        cv = stdev_s / mean_s if mean_s > 0 else 0.0
        # CV of 0 = perfect, CV >= 1 = worst
        naturalness = max(0.0, 1.0 - cv)
    else:
        naturalness = 1.0

    # --- Severity distribution -----------------------------------------------
    good_actions = {AlignAction.ACCEPT, AlignAction.MILD_STRETCH}
    n_good = sum(1 for a in aligned if a.action in good_actions)
    severity_distribution = n_good / max(len(aligned), 1)

    # --- Overall (weighted mean) ---------------------------------------------
    overall = 0.4 * timing_accuracy + 0.3 * naturalness + 0.3 * severity_distribution

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "naturalness": round(naturalness, 3),
        "severity_distribution": round(severity_distribution, 3),
        "overall": round(overall, 3),
    }
