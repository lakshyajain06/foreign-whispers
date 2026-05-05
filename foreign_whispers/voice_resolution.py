"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language
and optional speaker ID. The Chatterbox container expects a filename
relative to its /app/voices/ mount point.
"""

from pathlib import Path


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
    gender: str | None = None,
    gender_pool_index: int = 0,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order:
    1. speakers/{lang}/{speaker_id}.wav  (if speaker_id given and file exists)
    2. speakers/{lang}/{gender}/*.wav    (round-robin from gender pool)
    3. speakers/{lang}/default.wav       (language-specific default)
    4. speakers/default.wav              (global fallback)

    Args:
        speakers_dir: Absolute path to the speakers directory.
        target_language: Language code (e.g. "es", "fr").
        speaker_id: Optional speaker identifier (e.g. "SPEAKER_00").
        gender: Optional gender identifier (e.g. "male", "female").
        gender_pool_index: Index for round-robin selection in the gender pool.

    Returns:
        Relative path string for the Chatterbox container (e.g. "es/default.wav").
    """
    speakers_dir = Path(speakers_dir)
    
    # 1. Manual pin: speakers/{lang}/{speaker_id}.wav
    if speaker_id is not None:
        specific = speakers_dir / target_language / f"{speaker_id}.wav"
        if specific.exists():
            return f"{target_language}/{speaker_id}.wav"

    # 2. Gender pool: speakers/{lang}/{gender}/{nth}.wav (round-robin)
    if gender in ("male", "female"):
        pool_dir = speakers_dir / target_language / gender
        if pool_dir.exists():
            wavs = sorted(pool_dir.glob("*.wav"))
            if wavs:
                chosen = wavs[gender_pool_index % len(wavs)]
                return f"{target_language}/{gender}/{chosen.name}"

    # 3. Language default: speakers/{lang}/default.wav
    lang_default = speakers_dir / target_language / "default.wav"
    if lang_default.exists():
        return f"{target_language}/default.wav"

    # 4. Global fallback: speakers/default.wav
    return "default.wav"
