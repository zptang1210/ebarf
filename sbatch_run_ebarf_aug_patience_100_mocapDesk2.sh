#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=65536  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 168:00:00  # Job time limit
#SBATCH -o slurm-aug-mocapdesk2-%j.out  # %j = job ID
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate enerf_cloned
python main_barf.py --config ./configs/mocapDesk2/mocapDesk2_ebarf_start_from_5hz_aug_patience_100.txt
