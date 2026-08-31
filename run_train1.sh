#!/bin/bash
#SBATCH --job-name=dual_flood_gpu_long
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8


cd ~/projects/multiscale_DFGNN
source ~/venvs/dual_flood_gpu/bin/activate

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

echo "===== NODE INFO ====="
hostname
nvidia-smi

echo "===== CUDA CHECK ====="
python - <<'PY'
import torch
import torch_geometric

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("torch_geometric:", torch_geometric.__version__)
PY

echo "===== START TRAINING ====="
python -u train.py --config configs/config.yaml --model MultiScaleDUALFloodGNN --device cuda
