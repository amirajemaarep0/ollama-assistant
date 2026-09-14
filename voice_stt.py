import os

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import torch
import whisper

# Whisper models cost several seconds to construct. The previous version called
# whisper.load_model() inside transcribe_audio(), so every recording paid that
# cost again. Cache per model name instead.
_MODEL_CACHE: dict[str, "whisper.Whisper"] = {}


def get_model(model_name: str = "base"):
    """Load (and memoise) a Whisper model."""
    if model_name not in _MODEL_CACHE:
        print(f"Loading Whisper model '{model_name}' (first use)...")
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def record_audio(filename="temp_audio.wav", duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    wav.write(filename, fs, recording)
    return filename


def _wav_to_float32(filename: str) -> np.ndarray:
    """Read a WAV file as the mono float32 [-1, 1] array Whisper expects."""
    _sample_rate, audio_data = wav.read(filename)

    if audio_data.dtype == np.int16:
        audio_np = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_np = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_np = audio_data.astype(np.float32)

    # Downmix stereo to mono; Whisper only accepts a 1-D waveform.
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    return audio_np


def transcribe_audio(filename="temp_audio.wav", model_name="base"):
    model = get_model(model_name)

    print("Transcribing audio...")
    # fp16 is only a win on CUDA; on CPU it warns and falls back to fp32 anyway.
    use_fp16 = torch.cuda.is_available()

    try:
        result = model.transcribe(filename, fp16=use_fp16)
    except (FileNotFoundError, RuntimeError):
        # Whisper shells out to FFmpeg to decode. When FFmpeg is missing (common on
        # Windows) decode the WAV ourselves and hand Whisper the raw samples.
        print("FFmpeg unavailable. Falling back to the scipy WAV reader...")
        result = model.transcribe(_wav_to_float32(filename), fp16=use_fp16)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

    return result["text"]


if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    text = transcribe_audio(audio_file)
    print("\n--- Transcription ---")
    print(text)
