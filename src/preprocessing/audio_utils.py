import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_audio(file_path, sr_=16000):
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (Hz)
    
    Returns:
        audio: Audio signal as numpy array
        sr: Sample rate
    """
    # TODO: Use librosa to load audio
    # Hint: librosa.load() function
    
    audio, sr = librosa.load(file_path, sr=sr_, mono=True)
    
    return audio, sr

def get_audio_duration(file_path):
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        duration: Duration in seconds
    """
    # TODO: Use librosa to get duration
    # Hint: librosa.get_duration() function
    
    duration = librosa.get_duration(path=file_path)
    
    return duration

def plot_waveform(audio, sr, title="Waveform"):
    """
    Plot audio waveform.
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        title: Plot title
    """
    time = np.arange(len(audio)) / sr
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 4))
    
    # Plot waveform
    plt.plot(time, audio)
    
    # Add labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)  # Optional: adds grid for readability
    plt.tight_layout()