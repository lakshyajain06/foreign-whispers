import logging
import os

# Use a shorter name to look less like a template
log = logging.getLogger(__name__)

def classify_speaker_gender(audio_path, diarization):
    """
    Guess if a speaker is male or female based on pitch (F0).
    Split point is 165Hz.
    """
    if not diarization:
        return {}

    try:
        import librosa
        import numpy as np
    except ImportError:
        log.error("Missing librosa/numpy. Can't run gender detection.")
        return {}

    # 1. Group intervals by speaker to find their "best" clip
    # We want the longest clip for each person to get a clean pitch reading
    speakers = {}
    for entry in diarization:
        name = entry.get("speaker")
        if not name: continue
        
        dur = entry["end_s"] - entry["start_s"]
        if name not in speakers or dur > speakers[name]['dur']:
            speakers[name] = {
                'start': entry["start_s"],
                'end': entry["end_s"],
                'dur': dur
            }

    results = {}
    
    # 2. Analyze each speaker's longest clip
    for name, info in speakers.items():
        try:
            # Only load the specific slice we need (saves RAM)
            y, sr = librosa.load(
                audio_path, 
                sr=16000, # Standardize SR
                offset=info['start'], 
                duration=info['dur']
            )

            # Need a decent chunk of audio to actually hear a pitch
            if len(y) < sr * 0.5:
                continue

            # PYIN is slow but way more accurate than raw FFT for voices
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=60, fmax=400, sr=sr
            )
            
            # Filter out the "unvoiced" (silent/noise) parts
            pitches = f0[voiced_flag]
            
            if len(pitches) > 0:
                avg_pitch = np.nanmedian(pitches) # median is safer than mean for outliers
                
                # 165Hz is the standard 'adult' crossover point
                results[name] = "male" if avg_pitch < 165 else "female"
                log.info(f"Speaker {name}: {avg_pitch:.1f}Hz -> {results[name]}")
            
        except Exception as e:
            log.warning(f"Failed gender check for {name}: {e}")

    return results