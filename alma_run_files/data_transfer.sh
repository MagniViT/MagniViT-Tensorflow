#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH --output=/home/ofourkioti/Projects/WSI-preprocessing/datatransfer_test.txt
#SBATCH --partition=data-transfer
#SBATCH --ntasks=1
#SBATCH --time=000:10:00


srun rsync -avP  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/CLAM-features/training/ /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/test_features/
#srun rsync -aP --delete   /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/empty_dir/    /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/vae_patched_data/
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/camelyon_results/gan.txt




#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/camelyon_results/gan.txt
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/camelyon_results/gan.err