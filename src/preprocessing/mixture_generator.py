import numpy as np
import soundfile as sf
from pathlib import Path
import random
from typing import List, Tuple, Dict
from src.preprocessing.audio_utils import load_audio, get_audio_duration


def place_utterances_randomly(
    speaker_id: int,
    audio_clips: List[np.ndarray],
    target_duration: float,
    min_gap: float,
    max_gap: float,
    target_sr: int
) -> List[Dict]:
    """
    Helper: Place audio clips randomly across timeline.
    
    Returns:
        List of {'speaker_id': id, 'audio': array, 'start_time': float, 'end_time': float}
    """
    placements = []
    current_time = random.uniform(0, 2)  # Start with small random offset
    
    for audio in audio_clips:
        duration = len(audio) / target_sr
        
        # Stop if we exceed target duration
        if current_time + duration > target_duration:
            break
        
        placements.append({
            'speaker_id':speaker_id,
            'audio': audio,
            'start_time': current_time,
            'end_time': current_time + duration
        })
        
        # Add random gap before next utterance
        gap = random.uniform(min_gap, max_gap)
        current_time += duration + gap
    
    return placements


def create_clean_sources_from_timeline(
    utterance_timeline: List[Dict],
    num_speakers: int,
    total_duration: float,
    sample_rate: int
) -> Dict[int, np.ndarray]:
    """
    Create clean source audio for each speaker from utterance timeline.
    
    Args:
        utterance_timeline: List of utterance dictionaries
        num_speakers: Number of speakers
        total_duration: Total duration in seconds
        sample_rate: Sample rate
    
    Returns:
        Dictionary mapping speaker_id -> clean audio array
        
    YOUR TASK:
    For each speaker, create an audio array containing only their utterances
    at the correct timestamps, with silence everywhere else.
    
    This is almost IDENTICAL to how you created the mixture!
    The only difference: don't add speakers together, keep them separate.
    """
    
    total_samples = int(total_duration * sample_rate)
    
    # Create separate audio array for each speaker
    clean_sources = {}
    
    # YOUR CODE HERE
    # Hint: This is nearly identical to Step 5 in create_realistic_mixture
    # Instead of one mixture array, create one array per speaker
    # 
    # Structure:
    for speaker_id in range(num_speakers):
        clean_sources[speaker_id] = np.zeros(total_samples)
    
    for utterance in utterance_timeline:
        speaker_id = utterance['speaker_id']
        start_sample = int(utterance['start_time'] * sample_rate)
        end_sample = start_sample + len(utterance['audio'])
        clean_sources[speaker_id][start_sample:end_sample] += utterance['audio']
    
    
    return clean_sources


def create_realistic_mixture(
    speaker_audio_files: Dict[int, List[Path]],
    output_path: Path,
    target_duration: float = 30.0,
    min_silence_gap: float = 0.5,
    max_silence_gap: float = 3.0,
    target_overlap_ratio: float = 0.7,
    target_sr: int = 16000
) -> Dict:
    """
    Create realistic conversation-style mixture with multiple utterances per speaker.
    
    Args:
        speaker_audio_files: Dict mapping speaker_id -> list of audio file paths
                           Example: {0: [spk1_file1.flac, spk1_file2.flac],
                                    1: [spk2_file1.flac, spk2_file2.flac]}
        output_path: Where to save mixture
        target_duration: Desired total duration (seconds)
        min_silence_gap: Minimum silence between utterances (seconds)
        max_silence_gap: Maximum silence between utterances (seconds)
        target_overlap_ratio: Approximate overlap ratio (0.0 to 1.0)
        target_sr: Sample rate
    
    Returns:
        metadata: Dictionary with mixture info and frame-level labels
    
    YOUR TASK:
    1. For each speaker, place multiple utterances randomly in time
    2. Control overlap by adjusting how utterances are positioned
    3. Generate frame-level labels (who's speaking at each time frame)
    4. Save mixture and labels
    """
    
    num_speakers = len(speaker_audio_files)
    
    # Step 1: Load all audio clips for each speaker
    # TODO: Create a nested structure to store loaded audio
    # Structure: speaker_audios[speaker_id] = [audio_array1, audio_array2, ...]
    
    speaker_audios = {}
    
    # YOUR CODE HERE
    # Hint: Loop through speaker_audio_files.items()
    # For each speaker, load all their audio files
    for speaker_id, list_audio_files in speaker_audio_files.items():
        speaker_audios[speaker_id] = []
        for audio_file in list_audio_files:
            audio, _ = load_audio(audio_file)
            speaker_audios[speaker_id].append(audio)
    
    
    # Step 2: Create utterance placement plan
    # TODO: For each speaker, decide WHEN each utterance will play
    # Strategy: Randomly space out utterances across target_duration
    #
    # Data structure to create:
    # utterance_timeline = [
    #     {'speaker_id': 0, 'audio': array, 'start_time': 2.5, 'end_time': 5.2},
    #     {'speaker_id': 1, 'audio': array, 'start_time': 1.0, 'end_time': 3.8},
    #     ...
    # ]
    
    utterance_timeline = []
    
    # YOUR CODE HERE
    # For each speaker:
    #   current_time = random starting point
    #   for each utterance:
    #       place at current_time
    #       add silence gap
    #       current_time += utterance_duration + gap
    for speaker_id in speaker_audio_files.keys():
        utterance_timeline.extend(place_utterances_randomly(
            speaker_id=speaker_id,
            audio_clips=speaker_audios[speaker_id],
            target_duration=target_duration,
            min_gap=min_silence_gap,
            max_gap=max_silence_gap,
            target_sr=target_sr
        ))
    
    
    # Step 3: Adjust for target overlap ratio
    # TODO: Optionally shift some utterances to increase/decrease overlap
    # This is ADVANCED - you can skip this initially and just use random placement
    # The natural random placement will give some overlap automatically
    
    # OPTIONAL - Skip for first version
    pass
    
    
    # Step 4: Calculate actual mixture duration
    # TODO: Find the end time of the last utterance
    
    actual_duration = 0.0
    # YOUR CODE HERE
    # Hint: max(utterance['end_time'] for all utterances)
    for utterance in utterance_timeline:
        actual_duration = max(utterance['end_time'], actual_duration)
    
    
    # Step 5: Create mixture array
    # TODO: Initialize empty array and add all utterances
    
    total_samples = int(actual_duration * target_sr)
    mixture = np.zeros(total_samples)
    
    # YOUR CODE HERE
    # For each utterance in utterance_timeline:
    #     start_sample = int(start_time * target_sr)
    #     end_sample = start_sample + len(audio)
    #     mixture[start_sample:end_sample] += audio
    for utterance in utterance_timeline:
        start_sample = int(utterance['start_time'] * target_sr)
        end_sample = start_sample + len(utterance['audio'])
        mixture[start_sample:end_sample] += utterance['audio']
    
    
    # Step 6: Normalize
    # TODO: Prevent clipping
    
    # YOUR CODE HERE
    max_amplitude = np.max(np.abs(mixture))
    
    if max_amplitude > 0:
        mixture = mixture * (0.9 / max_amplitude)
    
    
    # Step 7: Create frame-level labels
    # TODO: Create binary labels for each speaker at each time frame
    # Frame size: typically 10ms or 100ms
    
    frame_shift = 0.01  # 10ms frames
    num_frames = int(actual_duration / frame_shift)
    
    # diarization_labels[frame_idx, speaker_id] = 1 if speaker is active, else 0
    diarization_labels = np.zeros((num_frames, num_speakers), dtype=np.int32)
    
    # YOUR CODE HERE
    # For each utterance:
    #     start_frame = int(start_time / frame_shift)
    #     end_frame = int(end_time / frame_shift)
    #     speaker_id = utterance['speaker_id']
    #     diarization_labels[start_frame:end_frame, speaker_id] = 1
    for utterance in utterance_timeline:
        start_frame = int(utterance['start_time']/frame_shift)
        end_frame = min(int(utterance['end_time']/frame_shift), num_frames)
        speaker_id = utterance['speaker_id']
        diarization_labels[start_frame:end_frame, speaker_id] = 1
    
    
    # Step 8: Calculate overlap statistics
    # TODO: How many frames have 2+ speakers active?
    
    overlap_frames = np.sum(diarization_labels.sum(axis=1) > 1)
    total_active_frames = np.sum(diarization_labels.sum(axis=1) > 0)
    actual_overlap_ratio = overlap_frames / total_active_frames if total_active_frames > 0 else 0.0
    
    
    # Step 9: Save mixture
    # YOUR CODE HERE
    sf.write(file=output_path, data=mixture, samplerate=target_sr)
    
    # Step 9.5: Create and save clean sources
    clean_sources = create_clean_sources_from_timeline(
        utterance_timeline=utterance_timeline,
        num_speakers=num_speakers,
        total_duration=actual_duration,
        sample_rate=target_sr
    )
    
    # Save each clean source
    source_paths = []
    for speaker_id, clean_audio in clean_sources.items():
        source_path = output_path.parent / f"{output_path.stem}_speaker{speaker_id}.wav"
        sf.write(source_path, clean_audio, target_sr)
        source_paths.append(source_path)
    
    # Step 10: Save labels (we'll need these for training!)
    label_path = output_path.parent / (output_path.stem + '_labels.npy')
    np.save(label_path, diarization_labels)
    
    
    # Step 11: Return metadata
    metadata = {
        'mixture_path': output_path,
        'source_paths': source_paths,
        'label_path': label_path,
        'duration': actual_duration,
        'num_speakers': num_speakers,
        'num_utterances': len(utterance_timeline),
        'target_overlap_ratio': target_overlap_ratio,
        'actual_overlap_ratio': actual_overlap_ratio,
        'frame_shift': frame_shift,
        'sample_rate': target_sr,
        'utterance_timeline': utterance_timeline,
        'diarization_labels': diarization_labels
    }
    
    return metadata


def visualize_realistic_mixture(metadata: Dict, save_path: Path = None):
    """
    Visualize the realistic mixture timeline with all utterances.
    
    YOUR TASK:
    Create a detailed timeline showing each utterance and overlap regions.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Create visualization
    # 
    # Suggested layout:
    # - Top subplot: Timeline with horizontal bars for each utterance
    # - Bottom subplot: Overlap indicator (how many speakers active per frame)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Subplot 1: Utterance timeline
    # YOUR CODE HERE
    # Hint: Use different colors for different speakers
    # for utterance in metadata['utterance_timeline']:
    #     speaker_id = utterance['speaker_id']
    #     start = utterance['start_time']
    #     duration = utterance['end_time'] - utterance['start_time']
    #     ax1.barh(speaker_id, duration, left=start, height=0.8, 
    #              label=f"Speaker {speaker_id}" if first occurrence else "")
    
    utterances = metadata['utterance_timeline']
    speaker_ids = sorted(set(u['speaker_id'] for u in utterances))
    colors = plt.cm.tab10.colors
    
    seen_speakers = set()
    
    for utterance in utterances:
        speaker_id = utterance['speaker_id']
        start = utterance['start_time']
        duration = utterance['end_time'] - utterance['start_time']
        
        ax1.barh(
            speaker_id,
            duration,
            left=start,
            height=0.8,
            color=colors[speaker_id % len(colors)],
            label=f"Speaker {speaker_id}" if speaker_id not in seen_speakers else ""
        )
        seen_speakers.add(speaker_id)
    
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Speaker ID")
    ax1.set_yticks(speaker_ids)
    ax1.set_title("Utterance Timeline")
    ax1.legend(loc="upper right")
    
    
    # Subplot 2: Overlap visualization
    # YOUR CODE HERE
    # Hint: Plot number of active speakers per frame
    # frame_times = np.arange(len(metadata['diarization_labels'])) * metadata['frame_shift']
    # active_speakers = metadata['diarization_labels'].sum(axis=1)
    # ax2.plot(frame_times, active_speakers)
    # ax2.fill_between(frame_times, 0, active_speakers, alpha=0.3)
    
    diarization_labels = metadata['diarization_labels']
    frame_shift = metadata['frame_shift']
    
    frame_times = np.arange(len(diarization_labels)) * frame_shift
    active_speakers = diarization_labels.sum(axis=1)
    
    ax2.plot(frame_times, active_speakers)
    ax2.fill_between(frame_times, 0, active_speakers, alpha=0.3)
    
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Active Speakers")
    ax2.set_title("Speaker Overlap Over Time")
    ax2.set_ylim(0, metadata['num_speakers'] + 0.5)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
