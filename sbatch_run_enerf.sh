#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=65536  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 336:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enerf
python main_nerf.py --config ./configs/mocapDesk2/mocapDesk2_enerf.txt --precompute_evs_poses 0

