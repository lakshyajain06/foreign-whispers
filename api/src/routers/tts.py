"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(
        None,
        description="Reference voice WAV path relative to pipeline_data/speakers/ "
                    "(e.g. 'es/default.wav'). If omitted, auto-resolves to the "
                    "language default via resolve_speaker_wav().",
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = trans_dir / f"{title}.json"

    # ── Voice selection (Tasks 3 + 4 + gender pooling) ──────────────
    # Three layers, evaluated per segment at synthesis time:
    #   1. speaker_voices (Task 4 — per-speaker map): wins when a segment's
    #      speaker label has an entry. Each entry is built using the
    #      4-tier resolve_speaker_wav() chain: manual pin → gender pool →
    #      language default → global default.
    #   2. speaker_wav (Task 3 — single global voice): fallback applied to
    #      segments whose speaker isn't in the map (or all segments when
    #      no diarization data exists). Auto-resolved if not provided.
    #   3. (in engine) Chatterbox server default if both above are absent.
    target_lang = "es"
    if source_path.exists():
        translated = json.loads(source_path.read_text())
        target_lang = translated.get("target_language", "es")

    if speaker_wav is None:
        speaker_wav = resolve_speaker_wav(settings.speakers_dir, target_lang)

    # Read per-speaker genders from the diarization cache (if present).
    diar_path = settings.diarizations_dir / f"{title}.json"
    gender_map: dict[str, str] = {}
    if diar_path.exists():
        diar_data = json.loads(diar_path.read_text())
        gender_map = diar_data.get("genders", {})

    speaker_voices: dict[str, str] = {}
    if source_path.exists():
        # Compute round-robin pool indices: each speaker of a given gender
        # gets a unique slot in the gender pool (encounter order).
        speaker_pool_index: dict[str, int] = {}
        gender_counters = {"male": 0, "female": 0}
        for seg in translated.get("segments", []):
            sp = seg.get("speaker")
            if sp and sp not in speaker_pool_index:
                gen = gender_map.get(sp)
                if gen in ("male", "female"):
                    speaker_pool_index[sp] = gender_counters[gen]
                    gender_counters[gen] += 1

        unique_speakers = {
            s.get("speaker")
            for s in translated.get("segments", [])
            if s.get("speaker")
        }
        for speaker in unique_speakers:
            speaker_voices[speaker] = resolve_speaker_wav(
                settings.speakers_dir,
                target_lang,
                speaker_id=speaker,
                gender=gender_map.get(speaker),
                gender_pool_index=speaker_pool_index.get(speaker, 0),
            )

    await _run_in_threadpool(
        None, svc.text_file_to_speech, str(source_path), str(audio_dir),
        alignment=alignment,
        speaker_wav=speaker_wav,
        speaker_voices=speaker_voices or None,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
