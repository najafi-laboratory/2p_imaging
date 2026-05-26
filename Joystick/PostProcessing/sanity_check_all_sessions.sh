#!/bin/bash
#SBATCH --job-name=sanity_check_all_sessions
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/skip_manual_qc_%j.out
#SBATCH --mail-user=saminnaji3@gatech.edu
#SBATCH --mail-type=END,FAIL

# ── EDIT: mouse folder and dates to process ───────────────────────────────────
# Use this when you want to skip manual ROI review and trust the auto QC.
# Runs step 3 (DFF) + step 4 (events) for all listed sessions using ALL
# ROIs that passed auto QC (step 1) as good.
MOUSE_FOLDER='SA16_LG' # the name of mouse folder you are trying to process
DATES=( # list of sessions date you are trying to process
    '20251230'
    '20251231'
    '20260103'
    '20260104'
    '20260105'
    '20260106'
)
# ─────────────────────────────────────────────────────────────────────────────

export KMP_DUPLICATE_LIB_OK=True

BASE_PATH='/storage/project/r-fnajafi3-0/shared/2P_Imaging' # generally the datawill always be in this directory no need to change
PIPELINE_DIR='/storage/project/r-fnajafi3-0/saminnaji3/Projects/Passive_Final_Versions/PostProcessing' # where you ut the postprocessing code folder in
MOUSE_PREFIX=$(echo "$MOUSE_FOLDER" | cut -d'_' -f1)

source /storage/project/r-fnajafi3-0/saminnaji3/miniconda3/etc/profile.d/conda.sh # where yor conda is installed and stored
conda activate suite2p

cd "$PIPELINE_DIR"

python pipeline/sanity_check_all.py "$MOUSE_FOLDER" "${DATES[@]}"

echo "Done: $(date)"

