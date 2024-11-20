#!/bin/bash
#SBATCH --job-name=bert_cpu_finetuning_job
#SBATCH --output=bert_complete_cpu.txt
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G


# Load an available Python module
module load StdEnv/2020
module load python/3.10.2  # Replace this with an available version from `module spider python`

# Create and activate a virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install required packages
pip install --no-index --upgrade pip
pip install --no-index tensorflow transformers pandas scikit-learn

# Run your Python script
python test.py

