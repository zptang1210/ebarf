#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=32768  # Requested Memory
#SBATCH -p gypsum-rtx8000  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 168:00:00  # Job time limit
#SBATCH -o slurm-ebarf-mocapdesk2-colmap-consecutive-data-undist-%j.out  # %j = job ID
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ebarf
python main_barf.py --config ./configs/mocapDesk2/mocapDesk2_ebarf_override_poses_hf_colmap_aligned_consecutive_data_undist.txt
