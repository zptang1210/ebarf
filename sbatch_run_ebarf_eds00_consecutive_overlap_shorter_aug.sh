#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=32768  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 168:00:00  # Job time limit
#SBATCH -o slurm-ebarf-eds00-consecutive-overlap-shorter-aug-%j.out  # %j = job ID
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ebarf
python main_barf.py --config ./configs/eds00/eds00_ebarf_consecutive_overlap_shorter_aug.txt

