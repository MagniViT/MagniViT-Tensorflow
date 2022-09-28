#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/multi-magnification-network/results/combined_k.txt
#SBATCH --error=/home/ofourkioti/Projects/multi-magnification-network/results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/multi-magnification-network/

python run.py --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/features/ --experiment_name combined_k