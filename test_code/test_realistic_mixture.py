import sys
from pathlib import Path
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generating_data.mixture_generator import create_realistic_mixture, visualize_realistic_mixture

# Collect audio files for 2-3 speakers
data_dir = project_root / "data" / "raw"

# Find speakers (assuming LibriSpeech structure: speakerid/chapterid/*.flac)
speaker_dirs = [d for d in data_dir.rglob("*") if d.is_dir() and list(d.glob("*.flac"))]

# Group by speaker (take first directory level)
speakers_found = {}
for speaker_dir in speaker_dirs:
    # Get speaker ID (assuming structure like: data/raw/dev-clean/84/121123/)
    # Speaker ID is the parent of chapterid
    speaker_id = speaker_dir.parent.name
    
    if speaker_id not in speakers_found:
        speakers_found[speaker_id] = []
    
    audio_files = list(speaker_dir.glob("*.flac"))
    speakers_found[speaker_id].extend(audio_files[:5])  # Take up to 5 files per chapter

# Take 2-3 speakers
selected_speakers = list(speakers_found.keys())[:2]
speaker_audio_files = {
    i: speakers_found[spk_id][:10]  # Max 10 files per speaker
    for i, spk_id in enumerate(selected_speakers)
}

print("="*60)
print("REALISTIC MIXTURE GENERATION TEST")
print("="*60)
print(f"\nSpeakers selected: {len(speaker_audio_files)}")
for spk_id, files in speaker_audio_files.items():
    print(f"  Speaker {spk_id}: {len(files)} utterances")

# Create mixture
output_path = project_root / "test_code" / "generated_data_testcases" / "realistic_mixture_001.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)

metadata = create_realistic_mixture(
    speaker_audio_files=speaker_audio_files,
    output_path=output_path,
    target_duration=30.0,
    min_silence_gap=0.3,
    max_silence_gap=2.0,
    target_overlap_ratio=0.7
)

print(f"\n✅ Mixture created!")
print(f"   Duration: {metadata['duration']:.2f}s")
print(f"   Utterances: {metadata['num_utterances']}")
print(f"   Target overlap: {metadata['target_overlap_ratio']*100:.1f}%")
print(f"   Actual overlap: {metadata['actual_overlap_ratio']*100:.1f}%")

# Visualize
viz_path = project_root / "test_code" / "generated_data_testcases" / "realistic_mixture_timeline.png"
visualize_realistic_mixture(metadata, save_path=viz_path)

print(f"\n✅ Visualization saved to: {viz_path}")
print("="*60)