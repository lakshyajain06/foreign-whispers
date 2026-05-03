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
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order:
    1. speakers/{lang}/{speaker_id}.wav  (if speaker_id given and file exists)
    2. speakers/{lang}/default.wav       (language-specific default)
    3. speakers/default.wav              (global fallback)

    Args:
        speakers_dir: Absolute path to the speakers directory.
        target_language: Language code (e.g. "es", "fr").
        speaker_id: Optional speaker identifier (e.g. "SPEAKER_00").

    Returns:
        Relative path string for the Chatterbox container (e.g. "es/default.wav").
    """
    # 1. Speaker-specific: speakers/{lang}/{speaker_id}.wav
    if speaker_id:
        speaker_path = Path(target_language) / f"{speaker_id}.wav"
        if (speakers_dir / speaker_path).exists():
            return str(speaker_path)

    # 2. Language default: speakers/{lang}/default.wav
    lang_default = Path(target_language) / "default.wav"
    if (speakers_dir / lang_default).exists():
        return str(lang_default)

    # 3. Global fallback: speakers/default.wav
    return "default.wav"
