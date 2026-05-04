from typing import NamedTuple


class _AudioMetaDataShim(NamedTuple):
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = "UNKNOWN"


def apply() -> None:
    try:
        import torchaudio
    except ImportError:
        return
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _AudioMetaDataShim  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "list_audio_backends"):
        # Modern torchaudio auto-selects; pyannote checks this list to pick one.
        # ``soundfile`` is a torchaudio dep so it is always available.
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "info"):
        # torchaudio 2.10 removed top-level ``info``. Implement via soundfile,
        # which is the audio backend pyannote can already use.
        def _info(path, *_args, **_kwargs):
            import soundfile as sf
            sf_info = sf.info(str(path))
            # Parse bits-per-sample from the subtype string, e.g. "PCM_16" -> 16.
            bits = 0
            subtype = sf_info.subtype or ""
            for token in subtype.replace("_", " ").split():
                if token.isdigit():
                    bits = int(token)
                    break
            return _AudioMetaDataShim(
                sample_rate=int(sf_info.samplerate),
                num_frames=int(sf_info.frames),
                num_channels=int(sf_info.channels),
                bits_per_sample=bits,
                encoding=subtype or "UNKNOWN",
            )

        torchaudio.info = _info  # type: ignore[attr-defined]

    # PyTorch 2.6 flipped torch.load's default to weights_only=True. pyannote
    # checkpoints store many pyannote-internal Python objects, so allowlisting
    # them one-by-one is whack-a-mole. Patch torch.load to default
    # weights_only=False, restoring pre-2.6 behavior. Safe here because the
    # pyannote/speaker-diarization-3.1 checkpoint is from a trusted source
    # (the user explicitly accepted its license on HuggingFace).
    try:
        import torch
        
        # We keep this just in case some parts of the code use the basic load
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

        if not getattr(torch.load, "_fw_legacy_weights", False):
            _orig_load = torch.load

            def _legacy_load(*args, **kwargs):
                # FORCE False to override both the new PT 2.6 default 
                # AND any explicit True passed by libraries like HF Hub.
                kwargs["weights_only"] = False 
                return _orig_load(*args, **kwargs)

            _legacy_load._fw_legacy_weights = True
            torch.load = _legacy_load
    except ImportError:
        pass