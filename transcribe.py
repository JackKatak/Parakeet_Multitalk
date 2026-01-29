#!/usr/bin/env python3
"""
Multitalker Parakeet ASR - Multi-Speaker Speech Recognition with Diarization

This script provides multi-speaker speech recognition using NVIDIA's
Multitalker Parakeet Streaming model combined with speaker diarization.

Features:
    - Multi-speaker transcription (up to 4 speakers)
    - Automatic speaker diarization and labeling
    - Word-level timestamps
    - Multiple output formats:
        - Turn-based with overlap markers (--turns)
        - Colored word-by-word interleaving (--color)
        - Detailed word timestamps (--words)
    - Automatic audio preprocessing (stereo to mono, resampling)

Usage:
    # Use via wrapper script (recommended):
    ./run_transcribe.sh --audio audio.wav --turns -q

    # Direct usage (requires correct LD_LIBRARY_PATH):
    python transcribe.py --audio audio.wav --turns -q

Models:
    - ASR: nvidia/multitalker-parakeet-streaming-0.6b-v1 (600M params)
    - Diarization: nvidia/diar_streaming_sortformer_4spk-v2.1

Requirements:
    - NVIDIA GPU with CUDA support
    - Python 3.10+
    - NeMo toolkit with ASR support

Author: Generated with Claude Code
License: Subject to NVIDIA NeMo licensing terms
"""

import sys
import os
import logging
import warnings

# Check for --quiet/-q early to suppress logging before imports
_QUIET_MODE = '--quiet' in sys.argv or '-q' in sys.argv
if _QUIET_MODE:
    # Set environment variables before any NeMo imports
    os.environ['NEMO_LOGGING_LEVEL'] = 'ERROR'
    os.environ['HYDRA_FULL_ERROR'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Suppress root logger
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.disable(logging.WARNING)
    warnings.filterwarnings('ignore')
    # Suppress specific NeMo loggers
    for name in ['nemo', 'nemo_logger', 'pytorch_lightning', 'omegaconf', 'hydra', 'transformers']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).disabled = True

import torch
import argparse
import tempfile
import soundfile as sf
import numpy as np


class Colors:
    """
    ANSI color codes for terminal output with speaker differentiation.

    Provides foreground and background colors for up to 4 speakers,
    cycling for additional speakers. Colors are chosen for readability
    and visual distinction.

    Attributes:
        RESET: Reset all formatting
        BOLD: Bold text
        SPEAKER_COLORS: List of foreground colors for speakers
        SPEAKER_BG: List of background colors with contrasting text
    """

    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Speaker colors (foreground) - chosen for distinctiveness
    SPEAKER_COLORS = [
        '\033[38;5;39m',   # Blue
        '\033[38;5;208m',  # Orange
        '\033[38;5;34m',   # Green
        '\033[38;5;201m',  # Magenta/Pink
    ]

    # Background versions for the speaker key legend
    SPEAKER_BG = [
        '\033[48;5;39m\033[38;5;15m',   # Blue bg, white text
        '\033[48;5;208m\033[38;5;0m',   # Orange bg, black text
        '\033[48;5;34m\033[38;5;15m',   # Green bg, white text
        '\033[48;5;201m\033[38;5;15m',  # Magenta bg, white text
    ]

    @classmethod
    def speaker(cls, idx):
        """
        Get foreground color for a speaker index.

        Args:
            idx: Speaker index (0-based, cycles for > 4 speakers)

        Returns:
            ANSI escape code for speaker's foreground color
        """
        return cls.SPEAKER_COLORS[idx % len(cls.SPEAKER_COLORS)]

    @classmethod
    def speaker_bg(cls, idx):
        """
        Get background color for a speaker index (used in legends).

        Args:
            idx: Speaker index (0-based, cycles for > 4 speakers)

        Returns:
            ANSI escape code for speaker's background color with text
        """
        return cls.SPEAKER_BG[idx % len(cls.SPEAKER_BG)]


def preprocess_audio(audio_file):
    """
    Preprocess audio file to ensure it's 16kHz mono WAV format.

    The NeMo ASR models require 16kHz mono audio. This function handles:
    - Stereo to mono conversion (averages channels)
    - Resampling from other sample rates to 16kHz

    Args:
        audio_file: Path to the input audio file (WAV, MP3, FLAC, etc.)

    Returns:
        tuple: (processed_file_path, is_temp_file)
            - processed_file_path: Path to the processed audio (may be original if no processing needed)
            - is_temp_file: True if a temporary file was created (caller should clean up)
    """
    data, sample_rate = sf.read(audio_file)

    # Check if processing is needed
    needs_processing = False

    # Convert stereo to mono if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        print(f"  Converting stereo to mono...")
        data = np.mean(data, axis=1)
        needs_processing = True

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        print(f"  Resampling from {sample_rate}Hz to 16000Hz...")
        import resampy
        data = resampy.resample(data, sample_rate, 16000)
        needs_processing = True

    if needs_processing:
        # Create temp file with processed audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, data.astype(np.float32), 16000)
        print(f"  Preprocessed audio saved to: {temp_file.name}")
        return temp_file.name, True

    return audio_file, False


def check_cuda():
    """
    Check CUDA availability and print GPU information.

    Verifies that PyTorch can access CUDA and displays the GPU name
    and CUDA version. This is important as the ASR models are optimized
    for GPU inference.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("CUDA is not available. This model requires a GPU.")
        print("Will attempt to run on CPU (may be slow)...")
        return False


def load_models(device):
    """
    Load the speaker diarization and ASR models from HuggingFace.

    Downloads models on first run (~2GB total) and caches them locally.
    Models are:
    - Diarization: nvidia/diar_streaming_sortformer_4spk-v2.1
    - ASR: nvidia/multitalker-parakeet-streaming-0.6b-v1

    Args:
        device: torch.device to load models onto (cuda or cpu)

    Returns:
        tuple: (diar_model, asr_model) - Speaker diarization and ASR models
    """
    from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel

    print("\nLoading speaker diarization model...")
    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1"
    ).eval().to(device)
    print("Diarization model loaded!")

    print("\nLoading ASR model (multitalker-parakeet-streaming-0.6b-v1)...")
    asr_model = ASRModel.from_pretrained(
        "nvidia/multitalker-parakeet-streaming-0.6b-v1"
    ).eval().to(device)
    print("ASR model loaded!")

    return diar_model, asr_model


def transcribe_audio(audio_file, diar_model, asr_model, device, output_path=None):
    """
    Perform streaming multi-speaker transcription with speaker diarization.

    This is the core transcription function that combines speaker diarization
    with ASR to produce speaker-attributed transcription. It processes audio
    in streaming chunks for memory efficiency.

    The pipeline:
    1. Creates a streaming audio buffer from the input file
    2. Initializes the SpeakerTaggedASR streamer with both models
    3. Processes audio in chunks, running diarization and ASR in parallel
    4. Generates speaker-tagged segments with word-level timestamps
    5. Scales timestamps to actual seconds (from frame indices)

    Args:
        audio_file: Path to the audio file (must be 16kHz mono WAV)
        diar_model: Loaded speaker diarization model (SortformerEncLabelModel)
        asr_model: Loaded ASR model (multitalker-parakeet-streaming)
        device: torch.device for inference (cuda recommended)
        output_path: Optional path for output JSON (default: output_transcription.json)

    Returns:
        tuple: (scaled_results, word_data)
            - scaled_results: List of segment dicts with speaker, start_time, end_time, words
            - word_data: List of word dicts with speaker_idx, word, start, end (in seconds)
    """
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
    from omegaconf import OmegaConf

    # Create configuration with all required parameters
    cfg = OmegaConf.create({
        # Audio input
        'audio_file': audio_file,
        'manifest_file': None,  # Required - None means use audio_file instead
        'output_path': output_path or 'output_transcription.json',

        # Streaming settings
        'online_normalization': True,
        'pad_and_drop_preencoded': True,
        'att_context_size': [70, 13],  # 1.12s latency

        # Speaker/diarization settings
        'single_speaker_mode': False,
        'max_num_of_spks': 4,
        'binary_diar_preds': False,  # Use soft diarization predictions for better speaker separation

        # Processing settings
        'batch_size': 1,
        'fix_prev_words_count': 5,
        'update_prev_words_sentence': 10,
        'ignored_initial_frame_steps': 2,
        'word_window': 8,
        'verbose': False,

        # Enable dynamic speaker detection from diarization
        'cache_gating': True,
        'cache_gating_buffer_size': 2,
        'masked_asr': False,  # Use multitalker mode, not masked ASR

        # Optional settings
        'deploy_mode': False,
        'generate_realtime_scripts': False,
        'log': False,
    })

    print(f"\nProcessing audio file: {audio_file}")

    # Disable CUDA graphs in decoding to avoid compatibility issues
    if hasattr(asr_model, 'decoding') and hasattr(asr_model.decoding, 'decoding'):
        dc = asr_model.decoding.decoding
        if hasattr(dc, 'decoding_computer'):
            dcomp = dc.decoding_computer
            if hasattr(dcomp, 'allow_cuda_graphs'):
                dcomp.allow_cuda_graphs = False
            if hasattr(dcomp, 'disable_cuda_graphs'):
                dcomp.disable_cuda_graphs()
            if hasattr(dcomp, 'cuda_graphs_mode'):
                dcomp.cuda_graphs_mode = None
            print("  CUDA graphs disabled for streaming decoding")

    # Setup streaming audio buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
    streaming_buffer_iter = iter(streaming_buffer)

    # Initialize multispeaker ASR streamer
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    samples = [{'audio_filepath': cfg.audio_file}]

    # Process audio in streaming chunks
    print("\nStreaming transcription in progress...")
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )

        with torch.inference_mode():
            with torch.amp.autocast(device.type, enabled=True):
                with torch.no_grad():
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )

        # Print intermediate results
        if step_num % 10 == 0:
            print(f"  Step {step_num}: Processing...")

    # Generate final output
    multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
    results = multispk_asr_streamer.instance_manager.seglst_dict_list

    # Get audio duration for timestamp scaling
    audio_duration = sf.info(audio_file).duration

    # Extract word-level data from the internal structures
    word_data = []
    frame_to_sec = 0.08  # Each frame = 80ms (10ms hop * 8 subsampling)

    # Calculate segment timestamp scale (different from word timestamps)
    # Segment timestamps are in streaming chunk units
    max_seg_time = 1.0
    for segment in results:
        end_time = segment.get('end_time', 0)
        if end_time > max_seg_time:
            max_seg_time = end_time
    # Scale to match actual audio duration
    seg_to_sec = audio_duration / max_seg_time if max_seg_time > 0 else 1.0

    # Try to get word-level timestamps from the ASR streamer's internal data
    if hasattr(multispk_asr_streamer, 'instance_manager'):
        im = multispk_asr_streamer.instance_manager
        # Check various possible locations for word-level data
        if hasattr(im, '_word_and_ts_seq') and im._word_and_ts_seq:
            for spk_idx, spk_data in enumerate(im._word_and_ts_seq):
                if spk_data and 'words' in spk_data:
                    for word_info in spk_data['words']:
                        if word_info:
                            word_data.append({
                                'speaker': f'speaker_{spk_idx}',
                                'speaker_idx': spk_idx,
                                'word': word_info.get('word', ''),
                                'start': word_info.get('start_offset', 0) * frame_to_sec,
                                'end': word_info.get('end_offset', 0) * frame_to_sec,
                            })

    # If no word-level data, try to extract from segment results
    if not word_data and results:
        for segment in results:
            speaker = segment.get('speaker', 'speaker_0')
            spk_idx = int(speaker.split('_')[-1]) if '_' in speaker else 0
            text = segment.get('words', '')
            # Scale segment times to seconds using calculated scale
            start_time = segment.get('start_time', 0) * seg_to_sec
            end_time = segment.get('end_time', 0) * seg_to_sec

            # Split into words and estimate timing
            words = text.split()
            if words:
                duration = end_time - start_time
                word_duration = duration / len(words) if len(words) > 0 else 0
                for i, word in enumerate(words):
                    word_data.append({
                        'speaker': speaker,
                        'speaker_idx': spk_idx,
                        'word': word,
                        'start': start_time + (i * word_duration),
                        'end': start_time + ((i + 1) * word_duration),
                    })

    # Sort by start time
    word_data.sort(key=lambda x: x['start'])

    # Also scale the segment timestamps in results for display
    scaled_results = []
    for segment in results:
        scaled_segment = dict(segment)
        scaled_segment['start_time'] = segment.get('start_time', 0) * seg_to_sec
        scaled_segment['end_time'] = segment.get('end_time', 0) * seg_to_sec
        scaled_results.append(scaled_segment)

    return scaled_results, word_data


def disable_cuda_graphs(asr_model, verbose=False):
    """
    Disable CUDA graphs at all levels to prevent compatibility issues.

    CUDA graphs can cause issues with streaming decoding and certain GPU
    configurations. This function disables them at:
    - Model config level (prevents re-enabling on reconfigure)
    - Decoding.decoding level
    - Decoding_computer level
    - Joint decoders (for multitalker models with multiple outputs)

    Args:
        asr_model: The ASR model instance
        verbose: If True, print which levels were disabled

    Returns:
        bool: True if any CUDA graphs settings were disabled
    """
    disabled = False

    # 1. Disable at the model config level (prevents re-enabling on reconfigure)
    if hasattr(asr_model, 'cfg') and hasattr(asr_model.cfg, 'decoding'):
        if hasattr(asr_model.cfg.decoding, 'greedy'):
            from omegaconf import open_dict
            with open_dict(asr_model.cfg.decoding.greedy):
                asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
                if verbose:
                    print("  Disabled CUDA graphs in cfg.decoding.greedy")
                disabled = True

    # 2. Disable at the decoding.decoding level
    if hasattr(asr_model, 'decoding') and hasattr(asr_model.decoding, 'decoding'):
        dc = asr_model.decoding.decoding
        # Disable at the greedy decoder level
        if hasattr(dc, 'use_cuda_graph_decoder'):
            dc.use_cuda_graph_decoder = False
            if verbose:
                print("  Disabled use_cuda_graph_decoder in decoding.decoding")
            disabled = True
        # Disable at the decoding_computer level
        if hasattr(dc, 'decoding_computer'):
            dcomp = dc.decoding_computer
            if hasattr(dcomp, 'allow_cuda_graphs'):
                dcomp.allow_cuda_graphs = False
            if hasattr(dcomp, 'disable_cuda_graphs'):
                dcomp.disable_cuda_graphs()
            if hasattr(dcomp, 'cuda_graphs_mode'):
                dcomp.cuda_graphs_mode = None
            if verbose:
                print("  Disabled CUDA graphs in decoding_computer")
            disabled = True

    # 3. Check for any joint decoders (multitalker models have multiple)
    for attr_name in ['joint', 'joint_0', 'joint_1', 'joint_2', 'joint_3']:
        if hasattr(asr_model, attr_name):
            joint = getattr(asr_model, attr_name)
            if hasattr(joint, 'decoding') and hasattr(joint.decoding, 'decoding'):
                jdc = joint.decoding.decoding
                if hasattr(jdc, 'use_cuda_graph_decoder'):
                    jdc.use_cuda_graph_decoder = False
                if hasattr(jdc, 'decoding_computer'):
                    jdcomp = jdc.decoding_computer
                    if hasattr(jdcomp, 'allow_cuda_graphs'):
                        jdcomp.allow_cuda_graphs = False
                    if hasattr(jdcomp, 'cuda_graphs_mode'):
                        jdcomp.cuda_graphs_mode = None
                if verbose:
                    print(f"  Disabled CUDA graphs in {attr_name}")
                disabled = True

    return disabled


def print_colored_interleaved(word_data, speakers_found=None):
    """
    Print transcription with words interleaved chronologically and colored by speaker.

    Displays a color key showing which color represents which speaker,
    then outputs the transcript with consecutive words from the same
    speaker grouped on lines with timestamps.

    Args:
        word_data: List of word dictionaries with keys:
            - 'speaker_idx': int, speaker index (0-3)
            - 'word': str, the transcribed word
            - 'start': float, start time in seconds
            - 'end': float, end time in seconds
        speakers_found: Optional list of speaker indices to display in legend.
            If None, auto-detected from word_data.
    """
    if not word_data:
        print("  No word-level data available for interleaved display.")
        return

    # Find unique speakers
    if speakers_found is None:
        speakers_found = sorted(set(w['speaker_idx'] for w in word_data))

    # Print color key
    print("\n" + "=" * 60)
    print("SPEAKER COLOR KEY:")
    print("=" * 60)
    for idx in speakers_found:
        print(f"  {Colors.speaker_bg(idx)} Speaker {idx} {Colors.RESET}")

    # Print interleaved transcript
    print("\n" + "=" * 60)
    print("INTERLEAVED TRANSCRIPT (chronological):")
    print("=" * 60 + "\n")

    # Group consecutive words by speaker for readability
    current_speaker = None
    line_words = []
    line_start = 0

    for word_info in word_data:
        spk = word_info['speaker_idx']
        word = word_info['word']
        start = word_info['start']

        if current_speaker is None:
            current_speaker = spk
            line_start = start

        if spk != current_speaker:
            # Print the accumulated line
            if line_words:
                color = Colors.speaker(current_speaker)
                text = ' '.join(line_words)
                print(f"[{line_start:5.1f}s] {color}{text}{Colors.RESET}")
            # Start new line
            current_speaker = spk
            line_words = [word]
            line_start = start
        else:
            line_words.append(word)

    # Print final line
    if line_words:
        color = Colors.speaker(current_speaker)
        text = ' '.join(line_words)
        print(f"[{line_start:5.1f}s] {color}{text}{Colors.RESET}")

    print()


def print_turn_based(word_data, speakers_found=None):
    """
    Print transcription with sentence-level turns and overlap markers.

    This format is ideal for conversations and debates. It shows who
    "has the floor" (primary speaker) and marks when others speak over
    them using indented overlap markers.

    Sentence boundaries are detected by:
    - Punctuation: . ? !
    - Long pauses: >0.8s gap between words from same speaker

    Turn transitions occur when:
    - No one currently has the floor
    - Current floor holder's speech ends (with 0.3s grace period)
    - Same speaker continues

    Example output:
        [0:01.35] SPK 0: Ukraine are now they're heavy-handed approach.
                 └─[SPK 1 overlapping]: They're heavy-handed approach.

    Args:
        word_data: List of word dictionaries with speaker_idx, word, start, end
        speakers_found: Optional list of speaker indices for legend
    """
    if not word_data:
        print("  No word-level data available for turn-based display.")
        return

    # Find unique speakers
    if speakers_found is None:
        speakers_found = sorted(set(w['speaker_idx'] for w in word_data))

    # Print color key
    print("\n" + "=" * 70)
    print("SPEAKER KEY:")
    print("=" * 70)
    for idx in speakers_found:
        print(f"  {Colors.speaker_bg(idx)} Speaker {idx} {Colors.RESET}")
    print()
    print("  └─ indicates overlapping speech")
    print("=" * 70)

    # First, group words into sentences for each speaker
    # A sentence ends with . ? ! or after a pause (>0.8s gap to next word from same speaker)
    sentence_endings = {'.', '?', '!'}

    # Build sentences per speaker with time ranges
    speaker_sentences = {idx: [] for idx in speakers_found}

    for idx in speakers_found:
        speaker_words = [w for w in word_data if w['speaker_idx'] == idx]
        if not speaker_words:
            continue

        current_sentence = []
        sentence_start = speaker_words[0]['start']

        for i, word_info in enumerate(speaker_words):
            current_sentence.append(word_info['word'])
            word_end = word_info['end']

            # Check if this ends a sentence
            is_sentence_end = False
            if any(word_info['word'].rstrip().endswith(p) for p in sentence_endings):
                is_sentence_end = True
            elif i < len(speaker_words) - 1:
                # Check for pause (>0.8s gap)
                next_start = speaker_words[i + 1]['start']
                if next_start - word_end > 0.8:
                    is_sentence_end = True

            if is_sentence_end or i == len(speaker_words) - 1:
                # Save this sentence
                speaker_sentences[idx].append({
                    'speaker_idx': idx,
                    'text': ' '.join(current_sentence),
                    'start': sentence_start,
                    'end': word_end,
                })
                current_sentence = []
                if i < len(speaker_words) - 1:
                    sentence_start = speaker_words[i + 1]['start']

    # Flatten all sentences and sort by start time
    all_sentences = []
    for idx in speakers_found:
        all_sentences.extend(speaker_sentences[idx])
    all_sentences.sort(key=lambda x: x['start'])

    # Now display with proper turn detection
    print("\n" + "=" * 70)
    print("TURN-BASED TRANSCRIPT:")
    print("=" * 70 + "\n")

    # Track the current "floor holder" - who is considered to be speaking
    current_floor_holder = None
    floor_end_time = 0

    for sentence in all_sentences:
        spk = sentence['speaker_idx']
        start = sentence['start']
        end = sentence['end']
        text = sentence['text']
        color = Colors.speaker(spk)

        # Format time as MM:SS
        mins = int(start // 60)
        secs = start % 60
        time_str = f"{mins}:{secs:05.2f}"

        # Determine if this is a new turn or overlapping speech
        # A new turn starts if:
        # 1. No one currently has the floor, OR
        # 2. The previous floor holder's speech has ended (with 0.5s grace period), OR
        # 3. This speaker is the current floor holder continuing
        is_new_turn = (
            current_floor_holder is None or
            spk == current_floor_holder or
            start >= floor_end_time - 0.3
        )

        if is_new_turn:
            print(f"[{time_str}] {color}SPK {spk}: {text}{Colors.RESET}")
            current_floor_holder = spk
            floor_end_time = end
        else:
            # Overlapping speech
            print(f"         └─{color}[SPK {spk} overlapping]: {text}{Colors.RESET}")
            # Update floor end time if this overlap extends beyond current
            if end > floor_end_time:
                floor_end_time = end

    print()


def simple_transcribe(audio_file, asr_model, device, with_timestamps=False):
    """
    Perform single-speaker transcription using only the ASR model.

    This bypasses the speaker diarization pipeline for faster processing
    when only one speaker is present or when testing model functionality.

    Handles CUDA graph compatibility by:
    - Disabling CUDA graphs before transcription
    - Monkey-patching change_decoding_strategy when timestamps enabled
      to prevent CUDA graphs from being re-enabled

    Args:
        audio_file: Path to the audio file (should be 16kHz mono WAV)
        asr_model: The loaded ASR model instance
        device: torch.device (kept for API consistency, model already on device)
        with_timestamps: If True, return word-level timestamp information

    Returns:
        List of transcription hypothesis objects with .text attribute
        and optionally .timestamp dict containing word timings
    """
    # Note: device parameter kept for API consistency with other functions
    del device  # Unused - model is already on the correct device
    print(f"\nSimple transcription of: {audio_file}")

    # Disable CUDA graphs at all levels before transcription
    if disable_cuda_graphs(asr_model, verbose=True):
        print("  CUDA graphs disabled for decoding")

    # For timestamps, we need to monkey-patch the model's change_decoding_strategy
    # to prevent CUDA graphs from being re-enabled during reconfiguration
    original_change_decoding = None
    if with_timestamps and hasattr(asr_model, 'change_decoding_strategy'):
        original_change_decoding = asr_model.change_decoding_strategy

        def patched_change_decoding(*args, **kwargs):
            # Modify config to disable CUDA graphs before any decoding change
            if 'decoding_cfg' in kwargs and kwargs['decoding_cfg'] is not None:
                from omegaconf import OmegaConf, open_dict
                dcfg = kwargs['decoding_cfg']
                if hasattr(dcfg, 'greedy'):
                    with open_dict(dcfg):
                        if hasattr(dcfg.greedy, 'use_cuda_graph_decoder'):
                            dcfg.greedy.use_cuda_graph_decoder = False
                        elif not OmegaConf.is_missing(dcfg.greedy, 'use_cuda_graph_decoder'):
                            dcfg.greedy.use_cuda_graph_decoder = False

            result = original_change_decoding(*args, **kwargs)
            # Disable CUDA graphs again after decoding strategy changed
            disable_cuda_graphs(asr_model, verbose=False)
            return result

        asr_model.change_decoding_strategy = patched_change_decoding
        print("  Patched change_decoding_strategy to prevent CUDA graph re-enable")

    try:
        with torch.inference_mode():
            transcription = asr_model.transcribe([audio_file], timestamps=with_timestamps)

            # Disable CUDA graphs again after transcription (in case it was re-enabled)
            if with_timestamps:
                disable_cuda_graphs(asr_model, verbose=False)

    finally:
        # Restore original method if we patched it
        if with_timestamps and original_change_decoding is not None:
            asr_model.change_decoding_strategy = original_change_decoding

    return transcription


def main():
    """
    Main entry point for the multi-speaker transcription CLI.

    Parses command-line arguments, loads models, preprocesses audio,
    and runs transcription with the specified output format options.

    Exit codes:
        0: Success
        1: Error (file not found, model loading failed, etc.)
    """
    parser = argparse.ArgumentParser(
        description="Multi-speaker speech recognition with NVIDIA Parakeet"
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to audio file (16kHz mono WAV)",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for transcription JSON",
        default="output_transcription.json"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple single-speaker transcription (faster, for testing)"
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Show word-level timestamps"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (not recommended, will be slow)"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Show colored interleaved output (multi-speaker mode only)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose NeMo logging output"
    )
    parser.add_argument(
        "--turns",
        action="store_true",
        help="Show turn-based transcript with sentence-level interleaving and overlap markers"
    )

    args = parser.parse_args()

    # Additional logging suppression for any loggers created after import
    if args.quiet:
        for logger_name in ['nemo', 'nemo.collections', 'nemo.core', 'nemo.utils',
                           'pytorch_lightning', 'nemo_logger', 'omegaconf',
                           'transformers', 'hydra']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    print("=" * 60)
    print("NVIDIA Multitalker Parakeet Streaming ASR Test")
    print("=" * 60)

    # Check CUDA availability
    has_cuda = check_cuda()

    if args.cpu or not has_cuda:
        device = torch.device("cpu")
        print("\nUsing CPU (this will be slow)")
    else:
        device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    # Load models
    print("\n" + "-" * 40)
    print("Loading models (this may take a while on first run)...")
    print("-" * 40)

    try:
        diar_model, asr_model = load_models(device)
    except Exception as e:
        print(f"\nError loading models: {e}")
        print("\nTrying to load just the ASR model...")
        from nemo.collections.asr.models import ASRModel
        asr_model = ASRModel.from_pretrained(
            "nvidia/multitalker-parakeet-streaming-0.6b-v1"
        ).eval().to(device)
        diar_model = None
        print("ASR model loaded (diarization model failed)")

    print("\nModels loaded successfully!")
    print(f"ASR Model parameters: ~600M")

    # Process audio if provided
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"\nError: Audio file not found: {args.audio}")
            sys.exit(1)

        print("\n" + "-" * 40)
        print("Preprocessing audio...")
        print("-" * 40)

        # Preprocess audio (convert to mono, resample to 16kHz if needed)
        processed_audio, is_temp = preprocess_audio(args.audio)

        print("\n" + "-" * 40)
        print("Transcribing audio...")
        print("-" * 40)

        try:
            if args.simple or diar_model is None:
                results = simple_transcribe(processed_audio, asr_model, device, with_timestamps=args.words)
                print("\n" + "=" * 60)
                print("TRANSCRIPTION RESULT:")
                print("=" * 60)
                # Extract text from hypothesis objects
                if results and hasattr(results[0], 'text'):
                    print("\n" + results[0].text + "\n")

                    # Show word timestamps if available
                    if args.words and hasattr(results[0], 'timestamp') and results[0].timestamp:
                        print("=" * 60)
                        print("WORD-LEVEL TIMESTAMPS:")
                        print("=" * 60)
                        # NeMo RNNT timestamps are in frame indices
                        # Each frame = 80ms (10ms hop * 8 subsampling factor)
                        frame_to_sec = 0.08
                        for ts in results[0].timestamp['word']:
                            word = ts.get('word', ts.get('char', ''))
                            start = ts.get('start_offset', 0) * frame_to_sec
                            end = ts.get('end_offset', 0) * frame_to_sec
                            print(f"  {start:6.2f}s - {end:6.2f}s : {word}")
                else:
                    print(results)
            else:
                results, word_data = transcribe_audio(
                    processed_audio, diar_model, asr_model, device, args.output
                )

                # Colored interleaved output
                if args.color:
                    print_colored_interleaved(word_data)

                # Turn-based output with overlap markers
                if args.turns:
                    print_turn_based(word_data)

                # Standard speaker-tagged output
                print("\n" + "=" * 60)
                print("TRANSCRIPTION RESULTS (Speaker-Tagged):")
                print("=" * 60)
                for segment in results:
                    speaker = segment.get('speaker', 'unknown')
                    start = segment.get('start_time', 0)
                    end = segment.get('end_time', 0)
                    words = segment.get('words', '')
                    print(f"\n[{speaker}] ({start:.2f}s - {end:.2f}s)")
                    print(f"  {words}")

                if args.words and word_data:
                    print("\n" + "=" * 60)
                    print("WORD-LEVEL TIMESTAMPS (Multi-speaker):")
                    print("=" * 60)
                    for w in word_data:
                        color = Colors.speaker(w['speaker_idx'])
                        print(f"  {w['start']:5.2f}s - {w['end']:5.2f}s : {color}{w['word']}{Colors.RESET} [{w['speaker']}]")
        finally:
            # Clean up temp file if created
            if is_temp and os.path.exists(processed_audio):
                os.unlink(processed_audio)
    else:
        print("\n" + "-" * 40)
        print("No audio file provided. Models are loaded and ready!")
        print("-" * 40)
        print("\nUsage examples:")
        print(f"  python {sys.argv[0]} --audio your_audio.wav")
        print(f"  python {sys.argv[0]} --audio your_audio.wav --simple")
        print(f"  python {sys.argv[0]} --audio your_audio.wav --output result.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
