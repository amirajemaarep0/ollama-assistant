import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import torch
import torchaudio

def record_audio(filename="temp_audio.wav", duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save as WAV file
    wav.write(filename, fs, recording)
    return filename

def transcribe_audio(filename="temp_audio.wav", model_name="base"):
    print(f"Loading Whisper model '{model_name}'...")
    # Using 'base' or 'tiny' for local speed
    model = whisper.load_model(model_name)
    
    print("Transcribing audio...")
    
    # Since FFmpeg might not be installed, we use a workaround reading the wav with torchaudio/scipy directly 
    # instead of passing the filename to whisper which triggers ffmpeg under the hood
    
    try:
        # standard whisper transcription
        result = model.transcribe(filename, fp16=False) 
    except FileNotFoundError:
        # Fallback for when FFmpeg is not installed on Windows
        print("FFmpeg not found. Attempting scipy backend fallback...")
        sample_rate, audio_data = wav.read(filename)
        
        # Whisper expects 1D array of float32 between -1.0 and 1.0
        if audio_data.dtype == np.int16:
            audio_np = audio_data.astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)
            
        result = model.transcribe(audio_np, fp16=False)

    # Clean up the temp file
    if os.path.exists(filename):
        os.remove(filename)
        
    return result["text"]

if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    text = transcribe_audio(audio_file)
    print("\n--- Transcription ---")
    print(text)
