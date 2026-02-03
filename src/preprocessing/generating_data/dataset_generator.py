import numpy as np
import soundfile as sf
from pathlib import Path
import random
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from src.preprocessing.generating_data.mixture_generator import create_realistic_mixture

def collect_speaker_data(data_dir: Path, max_speakers: int) -> Dict[str, List[Path]]:
    """
    Organize LibriSpeech data by speaker.
    
    YOUR TASK:
    1. Walk through directory structure
    2. Find all .flac files
    3. Group by speaker ID (folder structure: speakerid/chapterid/*.flac)
    4. Return dict: {speaker_id: [list of audio files]}
    
    Hint: Use rglob("*.flac") to find all audio files
    Hint: Extract speaker ID from file path
    """
    
    speaker_files = {}
    
    # YOUR CODE HERE
    def is_numeric_dir(p: Path) -> bool:
        return p.is_dir() and p.name.isdigit()
    
    candidate_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    dataset_root = None
    
    # Case A: data_dir itself contains numeric speaker folders
    if candidate_dirs and all(is_numeric_dir(p) for p in candidate_dirs):
        dataset_root = data_dir
    else:
        # Case B: one of its children is the dataset root
        for p in candidate_dirs:
            subdirs = [c for c in p.iterdir() if c.is_dir()]
            if subdirs and all(is_numeric_dir(c) for c in subdirs):
                dataset_root = p
                break
                
    if dataset_root is None:
        raise RuntimeError(
            "Could not locate dataset root with numeric speaker directories"
        )
    
    speaker_dirs = [
        p for p in dataset_root.iterdir()
        if is_numeric_dir(p)
    ][:max_speakers]
    
    for speaker_dir in speaker_dirs:
        flac_files = sorted(speaker_dir.rglob("*.flac"))
        if flac_files:
            speaker_files[speaker_dir.name] = flac_files
    
    return speaker_files


def create_dataset_split(
    speaker_data: Dict[str, List[Path]],
    output_dir: Path,
    split_name: str,
    num_mixtures: int,
    speakers_per_mixture: Tuple[int, int] = (2, 3),
    min_utterances_per_speaker: int = 3,
    max_utterances_per_speaker: int = 10,
    target_duration: float = 30.0,
    overlap_ratio_range: Tuple[float, float] = (0.2, 0.6)
):
    """
    Generate dataset split.
    
    YOUR TASK:
    1. Create output directory structure
    2. Loop num_mixtures times:
       - Randomly select 2-3 speakers
       - Randomly select 3-10 audio files per speaker
       - Randomly choose overlap ratio
       - Call create_realistic_mixture() - it handles EVERYTHING!
       - Save metadata to manifest
    3. Save manifest.json
    
    NOTE: You DON'T need to save clean sources - 
    create_realistic_mixture() already does it!
    """
    
    # Create output directories
    mixtures_dir = output_dir / "mixtures"
    mixtures_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    
    for mix_idx in tqdm(range(num_mixtures), desc=f"Generating {split_name}"):
        
        # Randomly select speakers and their files
        # YOUR CODE HERE
        # 1. Choose number of speakers: random.randint(...)
        # 2. Choose which speakers: random.sample(...)
        # 3. For each speaker, choose which files: random.sample(...)
        # Result: speaker_audio_files = {0: [file1, file2, ...], 1: [...], ...}
        
        speakers_number = random.randint(speakers_per_mixture[0],speakers_per_mixture[1])
        random_speakers = random.sample(list(speaker_data.keys()), speakers_number)
        # Map real speaker IDs → local contiguous integer IDs
        speaker_id_map = {
            real_id: idx for idx, real_id in enumerate(random_speakers)
        }
        speaker_audio_files = {}
        audio_files_per_speaker = random.randint(
            min_utterances_per_speaker,
            max_utterances_per_speaker
        )
        for real_speaker_id, local_id in speaker_id_map.items():
            speaker_audio_files[local_id] = random.sample(
                speaker_data[real_speaker_id],
                audio_files_per_speaker
            )
        
        # Randomly choose overlap ratio
        # YOUR CODE HERE
        overlap_ratio = random.uniform(overlap_ratio_range[0], overlap_ratio_range[1])
        
        # Create mixture (this handles EVERYTHING including clean sources!)
        mixture_path = mixtures_dir / f"{split_name}_{mix_idx:04d}.wav"
        
        metadata = create_realistic_mixture(
            speaker_audio_files=speaker_audio_files,
            output_path=mixture_path,
            target_duration=target_duration,
            target_overlap_ratio=overlap_ratio  # YOUR VARIABLE
        )
        
        # Add to manifest
        # Convert paths to strings for JSON serialization
        manifest_entry = {
            'mixture_id': f"{split_name}_{mix_idx:04d}",
            'mixture_path': str(metadata['mixture_path']),
            'source_paths': [str(p) for p in metadata['source_paths']],
            'label_path': str(metadata['label_path']),
            'duration': metadata['duration'],
            'num_speakers': metadata['num_speakers'],
            'num_utterances': metadata['num_utterances'],
            'actual_overlap_ratio': metadata['actual_overlap_ratio']
        }
        
        manifest.append(manifest_entry)
    
    # Save manifest
    manifest_path = output_dir / f"{split_name}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ {split_name} set complete: {num_mixtures} mixtures")
    print(f"   Manifest saved to: {manifest_path}")
    
    return manifest


def generate_full_dataset(
    data_dir: Path,
    output_dir: Path,
    train_size: int = 500,
    val_size: int = 50,
    test_size: int = 50
):
    """
    Generate complete dataset.
    
    YOUR TASK:
    1. Collect all speaker data
    2. Split speakers into train/val/test groups (important: no overlap!)
    3. Generate each split
    4. Save dataset statistics
    """
    
    print("="*60)
    print("DATASET GENERATION")
    print("="*60)
    
    # Collect speaker data
    print("\n📂 Collecting speaker data...")
    all_speakers = collect_speaker_data(data_dir, max_speakers=50)
    print(f"   Found {len(all_speakers)} speakers")
    
    # Split speakers (70/15/15)
    # YOUR CODE HERE
    # Hint: 
    # speaker_ids = list(all_speakers.keys())
    # random.shuffle(speaker_ids)
    # Calculate split points
    # train_speakers = {id: all_speakers[id] for id in first_70%}
    # val_speakers = {id: all_speakers[id] for id in next_15%}
    # test_speakers = {id: all_speakers[id] for id in last_15%}
    speaker_ids = list(all_speakers.keys())
    random.shuffle(speaker_ids) 
    num_speakers = len(speaker_ids)
    train_end = int(0.7 * num_speakers)
    val_end = int(0.85 * num_speakers)
    train_speakers = {
        id: all_speakers[id] for id in speaker_ids[:train_end]
    }
    val_speakers = {
        id: all_speakers[id] for id in speaker_ids[train_end:val_end]
    }
    test_speakers = {
        id: all_speakers[id] for id in speaker_ids[val_end:]
    }
    
    # Generate splits
    print("\n📊 Generating train set...")
    train_manifest = create_dataset_split(
        speaker_data=train_speakers,
        output_dir=output_dir / "train",
        split_name="train",
        num_mixtures=train_size
    )
    
    print("\n📊 Generating validation set...")
    # YOUR CODE HERE - similar to train
    val_manifest = create_dataset_split(
        speaker_data=val_speakers,
        output_dir=output_dir / "val",
        split_name="val",
        num_mixtures=train_size
    )
    
    print("\n📊 Generating test set...")
    # YOUR CODE HERE - similar to train
    test_manifest = create_dataset_split(
        speaker_data=test_speakers,
        output_dir=output_dir / "test",
        split_name="test",
        num_mixtures=train_size
    )
    
    # Save dataset statistics
    stats = {
        'total_speakers': len(all_speakers),
        'train_speakers': len(train_speakers),
        'val_speakers': len(val_speakers),
        'test_speakers': len(test_speakers),
        'train_mixtures': train_size,
        'val_mixtures': val_size,
        'test_mixtures': test_size,
        'total_mixtures': train_size + val_size + test_size
    }
    
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"Statistics saved to: {stats_path}")