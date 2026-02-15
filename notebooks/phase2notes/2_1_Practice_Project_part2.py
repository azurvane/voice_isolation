# ============================================
# BONUS CHALLENGE: Load Audio with Labels
# ============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.audio_utils import load_audio

# Device configuration (for your MacBook)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✅ Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Using NVIDIA GPU")
else:
    device = torch.device('cpu')
    print("⚠️ Using CPU")
print(f"Device: {device}")


# Helper function (already provided)
def pad_or_trim_audio(audio_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad with zeros or trim audio to target length."""
    current_length = audio_tensor.shape[0]
    
    if current_length < target_length:
        padding = target_length - current_length
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
    elif current_length > target_length:
        audio_tensor = audio_tensor[:target_length]
    
    return audio_tensor


def load_audio_with_labels(
    manifest_path: Path,
    num_samples: int = 5,
    target_length: int = 48000,  # 3 seconds at 16kHz
    max_speakers: int = 3,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load audio mixtures, clean sources, and diarization labels from manifest.
    
    Args:
        manifest_path: Path to train_manifest.json
        num_samples: How many samples to load
        target_length: Target length in samples (pad/trim to this)
        max_speakers: Maximum number of speakers (for padding speaker dimension)
        device: Device to move tensors to
    
    Returns:
        mixtures: (batch, 1, target_length) - Mixed audio
        sources: (batch, max_speakers, target_length) - Clean sources per speaker
        labels: (batch, max_speakers, num_frames) - Diarization labels
        metadata: List of dict with original info
    
    YOUR TASK:
    Load your generated dataset and prepare it for training!
    """
    
    # TODO 1: Load manifest JSON
    # Hint: json.load()
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Take only num_samples entries
    manifest = manifest[:num_samples]
    
    # Initialize lists
    mixture_tensors = []
    source_tensors = []
    label_tensors = []
    metadata = []
    
    # TODO 2: Loop through each entry in manifest
    for entry in manifest:
        # TODO 2.1: Load mixture audio
        mixture_path = Path(entry['mixture_path'])
        mixture_audio, sr = load_audio(mixture_path)  # Use load_audio
        
        # TODO 2.2: Convert to tensor and pad/trim
        mixture_tensor = torch.from_numpy(mixture_audio)
        mixture_tensor = pad_or_trim_audio(mixture_tensor, target_length)
        mixture_tensor = mixture_tensor.view(1, 1, -1)  # (1, 1, samples)
        mixture_tensors.append(mixture_tensor)
        
        # TODO 2.3: Load clean sources for each speaker
        speaker_sources = []
        for source_path in entry['source_paths']:
            source_audio, _ = load_audio(source_path)
            source_tensor = torch.from_numpy(source_audio)
            source_tensor = pad_or_trim_audio(source_tensor, target_length)
            speaker_sources.append(source_tensor)
        
        # TODO 2.4: Pad sources to max_speakers if needed
        # If mixture has 2 speakers but max_speakers=3, add one zero tensor
        num_speakers = len(speaker_sources)
        while len(speaker_sources) < max_speakers:
            speaker_sources.append(torch.zeros(target_length))
        
        # Stack sources: (max_speakers, samples)
        source_tensor = torch.stack(speaker_sources)
        source_tensors.append(source_tensor)
        
        # TODO 2.5: Load diarization labels
        label_path = Path(entry['label_path'])
        labels = np.load(label_path)
        
        # TODO 2.6: Convert labels to tensor
        # Shape is (num_frames, num_speakers)
        # We need (max_speakers, num_frames) with padding
        labels_tensor = torch.from_numpy(labels)
        
        target_frames = 3000
        labels_tensor = labels_tensor.transpose(0, 1)
        current_speakers = labels_tensor.shape[0]
        current_frames = labels_tensor.shape[1]
        
        pad_s = max_speakers - current_speakers
        pad_f = target_frames - current_frames
        
        labels_tensor = torch.nn.functional.pad(
            labels_tensor,
            (0, pad_f, 0, pad_s)
        )
        
        label_tensors.append(labels_tensor)
        
        # Store metadata
        metadata.append({
            'mixture_id': entry['mixture_id'],
            'duration': entry['duration'],
            'num_speakers': entry['num_speakers'],
            'num_utterances': entry['num_utterances'],
            'overlap_ratio': entry['actual_overlap_ratio']
        })
    
    # TODO 3: Stack all tensors into batches
    mixtures_batch = torch.cat(tensors=mixture_tensors, dim=0)  # torch.cat along dim=0
    sources_batch = torch.stack(source_tensors)   # torch.stack
    labels_batch = torch.stack(label_tensors)    # torch.stack
    
    # TODO 4: Move to device
    mixtures_batch = mixtures_batch.to(device)
    sources_batch = sources_batch.to(device)
    labels_batch = labels_batch.to(device)
    
    return mixtures_batch, sources_batch, labels_batch, metadata


# ============================================
# TEST: Load Audio with Labels
# ============================================

# Path to your training manifest
manifest_path = Path.cwd() / "data" / "processed" / "train" / "train_manifest.json"

print("="*60)
print("LOADING AUDIO BATCH WITH LABELS")
print("="*60)

# Load batch
mixtures, sources, labels, metadata = load_audio_with_labels(
    manifest_path=manifest_path,
    num_samples=5,
    target_length=48000,  # 3 seconds
    max_speakers=3,
    device=device
)

print("\n✅ BATCH LOADED SUCCESSFULLY!")
print("\n📊 TENSOR SHAPES:")
print(f"  Mixtures: {mixtures.shape}")
print(f"  Sources:  {sources.shape}")
print(f"  Labels:   {labels.shape}")
print(f"  Device:   {mixtures.device}")

print("\n📝 METADATA:")
for i, meta in enumerate(metadata):
    print(f"\nSample {i+1}:")
    print(f"  ID: {meta['mixture_id']}")
    print(f"  Duration: {meta['duration']:.2f}s")
    print(f"  Speakers: {meta['num_speakers']}")
    print(f"  Utterances: {meta['num_utterances']}")
    print(f"  Overlap: {meta['overlap_ratio']*100:.1f}%")

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

# Verify shapes
assert mixtures.shape == (5, 1, 48000), f"Mixtures shape should be (5, 1, 48000), got {mixtures.shape}"
assert sources.shape == (5, 3, 48000), f"Sources shape should be (5, 3, 48000), got {sources.shape}"
assert labels.shape[0] == 5, f"Labels batch should be 5, got {labels.shape[0]}"
assert labels.shape[1] == 3, f"Labels should have 3 speakers, got {labels.shape[1]}"

print("✅ All shapes correct!")

# Verify device
assert mixtures.device.type == device.type, f"Should be on {device}"
assert sources.device.type == device.type, f"Should be on {device}"
assert labels.device.type == device.type, f"Should be on {device}"

print("✅ All tensors on correct device!")

# Check data ranges
print(f"\n📈 DATA STATISTICS:")
print(f"  Mixture range: [{mixtures.min():.3f}, {mixtures.max():.3f}]")
print(f"  Sources range: [{sources.min():.3f}, {sources.max():.3f}]")
print(f"  Labels unique values: {torch.unique(labels)}")

print("\n" + "="*60)
print("✅ BONUS CHALLENGE COMPLETE!")
print("="*60)



# ============================================
# VISUALIZE ONE SAMPLE
# ============================================

# Take first sample
sample_idx = 0

mixture_audio = mixtures[sample_idx, 0].cpu().numpy()
source1_audio = sources[sample_idx, 0].cpu().numpy()
source2_audio = sources[sample_idx, 1].cpu().numpy()
sample_labels = labels[sample_idx].cpu().numpy()

# Create figure
fig, axes = plt.subplots(5, 1, figsize=(15, 10))

# Plot mixture
time = np.arange(len(mixture_audio)) / 16000
axes[0].plot(time, mixture_audio)
axes[0].set_title(f"Mixture - {metadata[sample_idx]['mixture_id']}")
axes[0].set_ylabel("Amplitude")

# Plot source 1
axes[1].plot(time, source1_audio)
axes[1].set_title("Clean Source - Speaker 0")
axes[1].set_ylabel("Amplitude")

# Plot source 2
axes[2].plot(time, source2_audio)
axes[2].set_title("Clean Source - Speaker 1")
axes[2].set_ylabel("Amplitude")

# Plot diarization labels (downsampled for visibility)
axes[3].imshow(sample_labels[:, ::10], aspect='auto', cmap='RdYlGn', interpolation='nearest')
axes[3].set_title("Diarization Labels (Speaker Activity)")
axes[3].set_ylabel("Speaker")
axes[3].set_yticks([0, 1, 2])
axes[3].set_xlabel("Time (frames)")

# Plot overlap
overlap = (sample_labels.sum(axis=0) > 1).astype(int)
frame_times = np.arange(len(overlap)) * 0.01  # 10ms frames
axes[4].fill_between(frame_times, overlap, alpha=0.5)
axes[4].set_title("Overlap Regions")
axes[4].set_ylabel("Overlap")
axes[4].set_xlabel("Time (seconds)")
axes[4].set_ylim([0, 1.5])

plt.tight_layout()
plt.savefig(Path.cwd() / "audio_batch_visualization.png", dpi=150)
print("📊 Visualization saved!")
plt.show()
