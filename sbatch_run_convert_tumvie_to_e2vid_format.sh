#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=32768  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
python scripts/convert_tumvie_to_e2vid_input_format.py