"""
this file fixes the problems with the torch version not having certain functions and classes
"""
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
        torchaudio.AudioMetaData = _AudioMetaDataShim

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "info"):
        def _info(path, *_args, **_kwargs):
            import soundfile as sf
            sf_info = sf.info(str(path))
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

        torchaudio.info = _info  

    try:
        import torch
        if not getattr(torch.load, "_fw_legacy_weights", False):
            _orig_load = torch.load

            def _legacy_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)

            _legacy_load._fw_legacy_weights = True
            torch.load = _legacy_load
            
        if hasattr(torch.serialization, "add_safe_globals") and hasattr(torch.torch_version, "TorchVersion"):
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except ImportError:
        pass
