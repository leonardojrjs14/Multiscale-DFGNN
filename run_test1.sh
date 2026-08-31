#!/bin/bash
#SBATCH --job-name=dual_flood_gpu_long
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ==========================================================
# Read arguments
# Usage:
# sbatch run_test1.sh <config> <model> <model_path>
# ==========================================================
CONFIG=$1
MODEL=$2
MODEL_PATH=$3

if [ $# -ne 3 ]; then
    echo "Usage:"
    echo "sbatch run_test1.sh <config> <model> <model_path>"
    exit 1
fi

echo "========================================"
echo "Config      : $CONFIG"
echo "Model       : $MODEL"
echo "Model Path  : $MODEL_PATH"
echo "========================================"

# ==========================================================
# Environment
# ==========================================================
cd ~/projects/multiscale_DFGNN || exit 1

source ~/venvs/dual_flood_gpu/bin/activate

mkdir -p logs
mkdir -p ~/pip_tmp ~/pip_cache ~/cache

export TMPDIR=$HOME/pip_tmp
export TEMP=$HOME/pip_tmp
export TMP=$HOME/pip_tmp
export PIP_CACHE_DIR=$HOME/pip_cache
export XDG_CACHE_HOME=$HOME/cache

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLIS_NUM_THREADS=8
export TORCH_NUM_THREADS=8

# ==========================================================
# System Information
# ==========================================================
echo "===== NODE INFO ====="
hostname
nvidia-smi

echo "===== CUDA CHECK ====="
python - <<'PY'
import torch
import torch_geometric

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("PyTorch Geometric:", torch_geometric.__version__)
PY

# ==========================================================
# Start Testing
# ==========================================================
echo "===== START TESTING ====="

python -u test.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --model_path "$MODEL_PATH" \
    --device cuda

STATUS=$?

echo "========================================"
if [ $STATUS -eq 0 ]; then
    echo "Testing completed successfully."
else
    echo "Testing failed with exit code $STATUS."
fi
echo "========================================"

exit $STATUS
