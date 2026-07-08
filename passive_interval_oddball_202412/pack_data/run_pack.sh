#!/bin/bash
#SBATCH --job-name=PackPassive
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=12:00:00
#SBATCH --output=PackPassive_%A.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/project/r-fnajafi3-0/yhuang887/Projects/passive_interval_oddball_202412
source activate suite2p
python pack_data/pack_results.py \
--results_dir results \
--output_dir results_pack
