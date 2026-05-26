#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=192G
#SBATCH --time=6:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=saminnaji3@gatech.edu

# Load conda environment
source /storage/project/r-fnajafi3-0/saminnaji3/miniconda3/etc/profile.d/conda.sh  # Initialize Conda (put the path your conda is installed in)
conda activate suite2p  # Activate the suite2p environment

# Navigate to project directory
cd /storage/project/r-fnajafi3-0/saminnaji3/Projects/2p_processing_pipeline # put the path of all code directory here

# Run the Python script
python run_suite2p_pipeline.py \
--denoise 1 \
--spatial_scale 1 \
--data_path '/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA21_LG/SA21_20260507_Passive_Short_Long_10000-1488' \ # this is the path to the session you are trying to process
--save_path '/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA21_LG/SA21_20260507' \ # his is the path you want the processed data be stored in
--nchannels 1 \
--functional_chan 2 \
--target_structure 'dendrite'