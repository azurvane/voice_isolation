import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now you can import from src
from src.preprocessing.audio_utils import load_audio, get_audio_duration, plot_waveform

# Find your audio files
data_dir = project_root / "data" / "raw"

# Let's find the first available audio file
audio_file = None
for flac_file in data_dir.rglob("*.flac"):
    audio_file = flac_file
    break  # Take the first one we find

if audio_file is None:
    print("❌ No .flac files found in data/raw/")
    print(f"Please check that your audio files are in: {data_dir}")
    sys.exit(1)

print("="*60)
print("AUDIO FILE TEST")
print("="*60)

# Test loading
print(f"\n📂 Loading: {audio_file.name}")
print(f"   Path: {audio_file}")

audio, sr = load_audio(audio_file)

print(f"\n✅ Audio loaded successfully!")
print(f"   Shape: {audio.shape}")
print(f"   Sample rate: {sr} Hz")
print(f"   Duration: {get_audio_duration(audio_file):.2f} seconds")
print(f"   Min amplitude: {audio.min():.4f}")
print(f"   Max amplitude: {audio.max():.4f}")

# Test plotting
print(f"\n📊 Creating waveform plot...")
plot_waveform(audio, sr, title=f"Waveform: {audio_file.name}")

# Save plot
output_path = project_root / "test_code" / "waveform_test.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   Plot saved to: {output_path}")

# Show plot
plt.show()

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)