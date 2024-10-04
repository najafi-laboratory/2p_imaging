#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=6:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=gtg424h@gatech.edu

cd /storage/coda1/p-fnajafi3/0/gtg424h/Projects/2P_Stim/2p_processing_pipeline_202401
source activate suite2p
python run_suite2p_pipeline.py \
--denoise 0 \
--spatial_scale 1 \
--data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/LG04 2P-stim/LG04_PPC_20241001_2pstim_t-001' \
--save_path './results/LG04_PPC_20241001_2pstim_t-001' \
--nchannels 2 \
--functional_chan 2 \
--brain_region 'crbl' \