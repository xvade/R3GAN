#!/bin/bash
#SBATCH --job-name=r3gan-score
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/%x-%j.out
# Usage (from repo root): sbatch slurm/score.sbatch
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate trainingslurm
# click is required by legacy.py; install once into a repo-local dir (not home)
DEPS=$PWD/.deps
[ -d "$DEPS/click" ] || pip install --quiet --target "$DEPS" click requests tqdm psutil packaging
export PYTHONPATH=$DEPS:${PYTHONPATH:-}
python score_images.py --network R3GAN-CIFAR10/network-snapshot-final.pkl --indir out/cifar10
