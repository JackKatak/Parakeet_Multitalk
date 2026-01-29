#!/usr/bin/env python3
"""
Verify Setup - Check all dependencies for Multitalker Parakeet ASR.

This script verifies that all required packages are installed and that
CUDA/GPU support is available. Run this after installation to ensure
everything is correctly configured before running transcription.

Usage:
    python verify_setup.py
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"  [OK] {package_name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {package_name}: {e}")
        return False


def main():
    print("=" * 50)
    print("Verifying Multitalker Parakeet Setup")
    print("=" * 50)

    all_ok = True

    print("\n1. Core Python packages:")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("numpy", "NumPy")
    all_ok &= check_import("soundfile", "SoundFile")

    print("\n2. NeMo Framework:")
    all_ok &= check_import("nemo", "NeMo Toolkit")
    all_ok &= check_import("nemo.collections.asr", "NeMo ASR")

    print("\n3. Additional dependencies:")
    all_ok &= check_import("omegaconf", "OmegaConf")
    all_ok &= check_import("hydra", "Hydra")
    all_ok &= check_import("transformers", "Transformers")

    print("\n4. CUDA/GPU Support:")
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA is available")
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        print(f"       CUDA Version: {torch.version.cuda}")
        print(f"       PyTorch CUDA: {torch.cuda.is_available()}")
    else:
        print("  [WARN] CUDA is NOT available")
        print("         The model will run on CPU (very slow)")
        all_ok = False

    print("\n5. Testing model imports:")
    try:
        from nemo.collections.asr.models import ASRModel
        print("  [OK] ASRModel can be imported")
    except Exception as e:
        print(f"  [FAIL] ASRModel: {e}")
        all_ok = False

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
        print("  [OK] SortformerEncLabelModel can be imported")
    except Exception as e:
        print(f"  [FAIL] SortformerEncLabelModel: {e}")
        all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("  1. Run: ./run_transcribe.sh --audio your_file.wav")
        print("     (This will download the models on first run)")
        print("  2. For quieter output: ./run_transcribe.sh --audio your_file.wav -q")
        print("  3. For turn-based display: ./run_transcribe.sh --audio your_file.wav --turns -q")
    else:
        print("Some checks failed. Please review the errors above.")
    print("=" * 50)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
