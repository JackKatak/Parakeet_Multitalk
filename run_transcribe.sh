#!/bin/bash
# Wrapper script to run transcription with correct library paths
# This fixes cuDNN version conflicts between conda and pip-installed PyTorch

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source venv/bin/activate

# Get PyTorch's library path and SET it as LD_LIBRARY_PATH (override conflicting paths)
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
NVIDIA_LIBS="$TORCH_LIB"

# Also include NVIDIA pip package libs
for pkg in nvidia_cudnn_cu12 nvidia_cublas_cu12 nvidia_cuda_runtime_cu12 nvidia_cufft_cu12 nvidia_curand_cu12 nvidia_cusolver_cu12 nvidia_cusparse_cu12 nvidia_nccl_cu12 nvidia_nvjitlink_cu12; do
    pkg_path=$(python -c "import $pkg; import os; print(os.path.dirname($pkg.__file__))" 2>/dev/null)
    if [ -n "$pkg_path" ] && [ -d "$pkg_path/lib" ]; then
        NVIDIA_LIBS="$pkg_path/lib:$NVIDIA_LIBS"
    fi
done

# Override LD_LIBRARY_PATH to use PyTorch's bundled CUDA libraries
export LD_LIBRARY_PATH="$NVIDIA_LIBS"

# Check if --quiet or -q is in the arguments
QUIET_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--quiet" ] || [ "$arg" = "-q" ]; then
        QUIET_MODE=true
        break
    fi
done

if [ "$QUIET_MODE" = "false" ]; then
    echo "Using CUDA libraries from PyTorch packages"
fi

# Run the transcription script with all passed arguments
# Redirect stderr to /dev/null in quiet mode to suppress NeMo logging
if [ "$QUIET_MODE" = "true" ]; then
    python transcribe.py "$@" 2>/dev/null
else
    python transcribe.py "$@"
fi
