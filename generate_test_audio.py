#!/usr/bin/env python3
"""
Generate Test Audio - Create synthetic audio for pipeline testing.

This creates a simple synthesized audio file with harmonic tones.
Note that this generates tones, not speech - it's only useful for
verifying the audio processing pipeline works, not for testing
actual speech recognition quality.

For real ASR testing, use actual speech recordings.

Usage:
    python generate_test_audio.py
"""

import numpy as np
import soundfile as sf

def generate_test_audio(output_path="test_audio.wav", duration=5.0, sample_rate=16000):
    """
    Generate a simple test audio file.

    For real testing, you should use actual speech audio files.
    This is just to verify the pipeline works.
    """
    print(f"Generating test audio file: {output_path}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")

    # Generate some tones (not speech, but tests the pipeline)
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Mix of frequencies
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)   # A4
    audio += 0.2 * np.sin(2 * np.pi * 554 * t)  # C#5
    audio += 0.15 * np.sin(2 * np.pi * 659 * t) # E5

    # Add some variation
    envelope = np.exp(-t / 2) + 0.3
    audio = audio * envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save as WAV
    sf.write(output_path, audio.astype(np.float32), sample_rate)
    print(f"Test audio saved to: {output_path}")
    print("\nNote: This is a synthetic tone, not speech.")
    print("For real ASR testing, use actual speech recordings in 16kHz mono WAV format.")

    return output_path


if __name__ == "__main__":
    generate_test_audio()
