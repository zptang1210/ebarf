#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=32768  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 168:00:00  # Job time limit
#SBATCH -o slurm-ebarf-evo-flyingroom-1-100-override-colmap-imgs-%j.out  # %j = job ID
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ebarf
python main_barf_debug.py --config ./configs/evo/flyingroom/flyingroom_ebarf_1_100_override_colmap_imgs.txt