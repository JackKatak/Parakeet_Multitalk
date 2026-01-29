# Multitalker Parakeet ASR

A multi-speaker speech recognition system using NVIDIA's Multitalker Parakeet Streaming model with speaker diarization.

## Features

- **Multi-speaker transcription**: Identifies and separates up to 4 speakers
- **Speaker diarization**: Automatic speaker identification and labeling
- **Word-level timestamps**: Precise timing for each word
- **Multiple output formats**:
  - Turn-based with overlap markers (best for conversations/debates)
  - Colored word-by-word interleaving
  - Speaker-tagged segments
  - Detailed word timestamps
- **Automatic audio preprocessing**: Converts stereo to mono, resamples to 16kHz
- **Streaming processing**: Efficient chunk-based processing

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **CUDA**: Version 12.x
- **Python**: 3.10+

## Installation

1. Clone or navigate to this directory:
   ```bash
   cd multitalkparakeet
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify the installation:
   ```bash
   python verify_setup.py
   ```

## Usage

### Basic Usage

Use the wrapper script for proper CUDA library handling:

```bash
# Basic multi-speaker transcription
./run_transcribe.sh --audio your_audio.wav

# With quiet mode (suppresses verbose logging)
./run_transcribe.sh --audio your_audio.wav -q
```

### Output Formats

#### Turn-Based with Overlap Markers (Recommended for conversations)

```bash
./run_transcribe.sh --audio your_audio.wav --turns -q
```

Output:
```
[0:01.35] SPK 0: Ukraine are now they're heavy-handed approach.
         └─[SPK 1 overlapping]: They're heavy-handed approach.
[0:04.58] SPK 0: They're heavy-handed approach.
         └─[SPK 1 overlapping]: You both have said Vladimir Putin...
```

#### Colored Word-by-Word Interleaving

```bash
./run_transcribe.sh --audio your_audio.wav --color -q
```

Shows each word in chronological order with speaker colors.

#### Word-Level Timestamps

```bash
./run_transcribe.sh --audio your_audio.wav --words -q
```

Output:
```
   1.35s -  1.89s : Ukraine [speaker_0]
   1.89s -  2.43s : are [speaker_0]
   3.67s -  4.20s : They're [speaker_1]
```

#### Simple Single-Speaker Mode

For faster processing without diarization:

```bash
./run_transcribe.sh --audio your_audio.wav --simple --words -q
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--audio`, `-a` | Path to audio file (required) |
| `--output`, `-o` | Output JSON file path (default: output_transcription.json) |
| `--turns` | Show turn-based transcript with overlap markers |
| `--color` | Show colored interleaved word-by-word output |
| `--words` | Show word-level timestamps |
| `--simple` | Use single-speaker mode (faster, no diarization) |
| `--quiet`, `-q` | Suppress verbose NeMo logging |
| `--cpu` | Force CPU usage (not recommended, very slow) |

### Combining Options

```bash
# Turn-based + word timestamps
./run_transcribe.sh --audio your_audio.wav --turns --words -q

# All visual outputs
./run_transcribe.sh --audio your_audio.wav --turns --color --words -q
```

## Audio Requirements

- **Supported formats**: WAV, MP3, FLAC, and other common formats
- **Automatic preprocessing**:
  - Stereo files are converted to mono
  - Audio is resampled to 16kHz if needed
- **Recommended**: 16kHz mono WAV for best performance

## Models Used

This system uses two NVIDIA NeMo models:

1. **Speaker Diarization**: `nvidia/diar_streaming_sortformer_4spk-v2.1`
   - Identifies up to 4 speakers
   - Streaming-capable for real-time processing

2. **ASR (Speech Recognition)**: `nvidia/multitalker-parakeet-streaming-0.6b-v1`
   - 600M parameter RNNT model
   - Optimized for multi-speaker scenarios
   - Streaming-capable

Models are automatically downloaded from HuggingFace on first run (~2GB total).

## File Structure

```
multitalkparakeet/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── run_transcribe.sh      # Main wrapper script (handles CUDA paths)
├── transcribe.py          # Core transcription logic
├── verify_setup.py        # Installation verification
├── generate_test_audio.py # Generate synthetic test audio
└── venv/                  # Python virtual environment
```

## Troubleshooting

### cuDNN Version Mismatch

If you see errors about cuDNN version incompatibility, the wrapper script `run_transcribe.sh` handles this by setting the correct library paths. Always use the wrapper script instead of running `transcribe.py` directly.

### CUDA Out of Memory

The models require approximately 4-6GB of GPU memory. If you encounter OOM errors:
- Close other GPU-intensive applications
- Try processing shorter audio files
- Use `--simple` mode for reduced memory usage

### Slow First Run

The first run downloads models from HuggingFace (~2GB). Subsequent runs use cached models and start much faster.

## Output Interpretation

### Speaker Labels

- `speaker_0`, `speaker_1`, etc. are automatically assigned
- Labels are consistent within a single audio file
- The model supports up to 4 simultaneous speakers

### Overlap Markers

In `--turns` mode:
- `└─[SPK X overlapping]:` indicates speech that overlaps with the current turn
- Multiple overlapping speakers are shown in chronological order

### Timestamps

- Timestamps are in seconds from the start of the audio
- Word-level timestamps have ~80ms resolution
- Segment timestamps show the full duration of each speaker's contribution

## License

This project uses NVIDIA NeMo models which are subject to NVIDIA's licensing terms.
See [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for details.

## Acknowledgments

- NVIDIA NeMo team for the ASR and diarization models
- HuggingFace for model hosting
