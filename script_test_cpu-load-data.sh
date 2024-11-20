#!/bin/bash
#SBATCH --job-name=dataset_cpu
#SBATCH --output=dataset_cpu.txt
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G


module load python/3.10.2  # Load this only if required; skip if default Python is 3.10+

# Install required packages
pip install --no-index --upgrade pip
pip install --no-index datasets pandas matplotlib --no-deps
pip install tqdm numpy requests fsspec multiprocess

# Run your Python script
python show_dataset.py

