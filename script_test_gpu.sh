#!/bin/bash
#SBATCH --job-name=bert_gpu_tune
#SBATCH --output=bert_complete_gpu.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH  --gpus-per-node=1

module load python/3.10.13

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index transformers pandas torch scikit-learn

python test.py
