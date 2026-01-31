import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now you can import from src
from src.preprocessing.audio_utils import load_audio, get_audio_duration, plot_waveform

# Pick one of your audio files
# Adjust this path to match your actual LibriSpeech structure
audio_file = project_root / "data" / "raw" / "dev-clean" / "84" / "121123" / "84-121123-0000.flac"

# Check if file exists
if not audio_file.exists():
    print(f"File not found: {audio_file}")
    print("Please update the path to match one of your actual audio files")
    sys.exit(1)

# Test loading
print(f"Loading: {audio_file}")
audio, sr = load_audio(audio_file)
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {get_audio_duration(audio_file):.2f} seconds")

# Test plotting
plot_waveform(audio, sr, title=f"Waveform: {audio_file.name}")
print("Plot saved to test_code/waveform_test.png")
