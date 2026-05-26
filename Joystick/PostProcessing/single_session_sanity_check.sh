#!/bin/bash
#SBATCH --job-name=single_session_sanity_check
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/step1_%j.out
#SBATCH --mail-user=saminnaji3@gatech.edu
#SBATCH --mail-type=END,FAIL

# ── EDIT: mouse folder and dates to process ───────────────────────────────────
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

source /storage/project/r-fnajafi3-0/saminnaji3/miniconda3/etc/profile.d/conda.sh # where yor conda is nstalled and stored
conda activate suite2p


cd "$PIPELINE_DIR"

for DATE in "${DATES[@]}"; do
    SESSION_PATH="${BASE_PATH}/${MOUSE_FOLDER}/${MOUSE_PREFIX}_${DATE}"
    echo "========================================"
    echo "Sanity check: $SESSION_PATH"
    echo "========================================"
    python pipeline/sanity_check.py "$SESSION_PATH"
done

echo "Done: $(date)"
