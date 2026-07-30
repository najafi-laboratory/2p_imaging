# DLC Pipeline — Capstone 4001 (Dr. Najafi)

Complete step-by-step guide for DeepLabCut body-part tracking and neural data analysis.

---

## Overview

The pipeline has two phases:

- **Local PC** — video prep, DLC project creation, frame labeling
- **Cluster (PACE / ICE)** — model training, video analysis, evaluation, and neural data processing

---

## Phase 1: Local PC

### Step 0 — Download Videos from Cluster

**Goal:** Get the raw videos you want to run DLC on.

1. On the cluster web page, go to **Files → Home Directory**
2. Click **Change Directory** and navigate to:
   - Cedar: `/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/video_data`
   - PACE: `/storage/project/r-fnajafi3-0/shared/2P_Imaging/video_data/`
3. Navigate to your project and mouse folder (e.g. `Eyeblink_conditioning/YH24LG/`)
4. Download the videos
5. Copy them to your local PC at: `Capstone_Najafi/Original_videos/`

---

### Step 1 — Crop Videos (Optional)

**Goal:** Reduce noise and speed up training by focusing on the relevant body part.

> Only crop if you are focused on a small body part.

1. Use an online video cropper to manually crop the video
2. Rename the cropped videos appropriately
3. Place them at: `Capstone_Najafi/DLC/Videos_cropped_for_DLC/`

---

### Step 2 — Create DLC Project

**Goal:** Generate the project folder and `config.yaml`.

1. Create a Jupyter notebook at `DLC/dlc_project_local.ipynb`
2. Open it in VS Code and select the **DeepLabCut** kernel
3. In the first cell, import DLC:
   ```python
   import deeplabcut
   ```
4. Create the project:
   ```python
   deeplabcut.create_new_project(
       "Track",
       "GroupName",
       ["path/to/one/cropped/video.avi"],
       working_directory="path/to/DLC/Model",
       copy_videos=False,
       multianimal=False
   )
   ```
5. This generates the project folder with: `config.yaml`, `videos/`, `labeled-data/`, `training-datasets/`, `dlc-models-pytorch/`

---

### Step 3 — Label Body Parts & Create Training Dataset

**Goal:** Manually label keypoints on extracted frames to create training data.

> Label only **1 video** at this stage. Right/left body parts must be labeled relative to the **animal**, not how they appear on screen.

1. Define your config and video paths in `dlc_project_local.ipynb`
2. Add videos:
   ```python
   deeplabcut.add_new_videos(config, [video_file], copy_videos=False)
   ```
3. Extract frames:
   ```python
   deeplabcut.extract_frames(config, mode="automatic", algo="uniform", userfeedback=False)
   ```
   - Extracts ~20 frames per video into `labeled-data/video_name/`
   - If you have multiple videos, move all frames into the first video's folder and delete the empty folders
4. Label frames (opens Napari GUI):
   ```python
   deeplabcut.label_frames(config)
   ```
   - Set labeling mode to **Loop** (go through all frames for one body part, then move to the next)
   - Save with `Cmd+S` / `Ctrl+S`
   - Only label a body part if it is **clearly visible** and you are certain
5. Check labels:
   ```python
   deeplabcut.check_labels(config)
   ```
6. Create training dataset:
   ```python
   deeplabcut.create_training_dataset(config)
   ```
   - Generates `training-datasets/` and `dlc-models-pytorch/` subfolders

---

## Phase 2: Cluster (PACE / ICE)

### Step 4 — Upload DLC Project to Cluster

**Goal:** Transfer your local project to PACE and update config paths.

1. On PACE, go to your home directory
2. Drag and drop (upload) your entire `DLC/` folder to the cluster
   - Double-check that the `videos/` folder is included
3. Open `config.yaml` (line 10) and update `project_path` to the cluster path:
   ```yaml
   project_path: /storage/home/hcoda1/8/yourname/r-yourname-0/DLC/Model/Track-GroupName-YYYY-MM-DD
   ```
4. Also in `config.yaml`, set `batch_size: 32` (line ~65)
5. Open `pytorch_config.yaml` at `dlc-models-pytorch/iteration-0/.../train/` and set `batch_size: 32` in **both** places it appears (lines ~78 and ~179)

---

### Step 5 — Train DLC Model on GPU

**Goal:** Train the model on the cluster using a GPU node.

1. Create `DLC/dlc_project_cluster.ipynb` on the cluster
2. Launch a **VS Code interactive session** on ICE:
   - Go to **Interactive Apps → Microsoft VS Code**
   - Node type: GPU (e.g. L40S, RTX6000)
   - Recommended: 20 CPUs, 3 GPUs, 20 GB RAM, 4 hours
3. Open `dlc_project_cluster.ipynb` and select kernel: **Python (myenv)**
4. Import DLC and define config:
   ```python
   import deeplabcut
   config = "/path/to/DLC/Model/PROJECT/config.yaml"
   ```
5. Train the model:
   ```python
   deeplabcut.train_network(config, autotune=False)
   ```
   - Stage 1: Detector (SSDLite) — 250 epochs
   - Stage 2: Pose model (HRNet) — 200 epochs
   - Snapshots saved to `dlc-models-pytorch/iteration-x/.../train/`

---

### Step 6 — Create Video Analysis Script

**Goal:** Set up the script that will run `analyze_videos` on all videos.

1. Copy all videos from `DLC/Videos_cropped_for_DLC/` to `DLC/Model/PROJECT/videos/`
2. Inside `DLC/Model/PROJECT/`, create `DLC_analyze_videos_script.py`
3. Paste the following:
   ```python
   import deeplabcut

   config = "/path/to/DLC/Model/PROJECT/config.yaml"
   videos_folder_path = "/path/to/DLC/Model/PROJECT/videos"

   deeplabcut.analyze_videos(
       config,
       [videos_folder_path],
       videotype='avi',
       destfolder=videos_folder_path,
       save_as_csv=True
   )
   ```
4. Save the file

---

### Step 7 — Run Analysis via Slurm Job

**Goal:** Submit a batch job to analyze all videos on the cluster.

1. Go to **Jobs → Job Composer** on the ICE web portal
2. Click **+ New Job → From Default Template**
3. Edit `main_job.sh`:
   ```bash
   #!/bin/bash
   #SBATCH -J DLC_Najafilab
   #SBATCH -N1 --gres=gpu:1 -C V100-32GB
   #SBATCH --mem-per-cpu=8G
   #SBATCH --cpus-per-gpu=6
   #SBATCH -t 360
   #SBATCH -o Report-%j.out
   #SBATCH --mail-type=BEGIN,END,FAIL
   #SBATCH --mail-user=YourUsername@gatech.edu
   #SBATCH --account=gts-fnajafi3

   cd "/path/to/DLC/Model/PROJECT"
   module load anaconda3
   module load cuda/11.7.0
   module load cudnn/8.5.0.96-11.7-cuda
   conda activate DeepLabCut
   python DLC_analyze_videos_script.py
   ```
4. Submit the job
5. Results (`.csv` and `.h5` files) are saved to `DLC/Model/PROJECT/videos/`

---

### Step 8 — Evaluate Model Confidence

**Goal:** Plot likelihood scores to assess whether the model is performing well.

1. Create a folder `DLC/Evaluation/` on the cluster
2. Inside it, create `plot_model_likelihoods.ipynb`
3. Copy code from: `https://github.com/najafi-laboratory/Deep-Lab-Cut/blob/main/plot_model_likelihoods.ipynb`
4. Define your paths in the last cell:
   ```python
   path_to_model_videos_folder = "/path/to/DLC/Model/PROJECT/videos"
   path_to_evaluation_folder = "/path/to/DLC/Evaluation"
   ```
5. Run all cells → generates likelihood plots in `DLC/Evaluation/`
6. Download the Evaluation folder to your local PC

**Interpreting results:**
- Confidence > 0.8 for most frames → model is good ✅
- Confidence mostly below 0.8 → retrain (go to Step 9) ❌

---

### Step 9 — Refine Labels & Retrain (If Needed)

**Goal:** Extract poorly-predicted frames, re-label them, and retrain.

**Folder prep:**
1. Download `DLC/Model/PROJECT/videos/` from PACE to local
2. Remove the video used for initial labeling in Step 3 (video 1) from the downloaded folder
3. Download the best model snapshots from `dlc-models-pytorch/.../train/`:
   - `snapshot-best-xx.pt`
   - `snapshot-detector-best-xx.pt`
   - If no "best" file exists, take the snapshot with the largest number

**Label refining (local):**
1. Open `dlc_project_local.ipynb` and add new cells at the bottom
2. Define config, videos folder, and new video paths
3. Add the new video:
   ```python
   deeplabcut.add_new_videos(config, [video_file], copy_videos=False)
   ```
4. Extract outlier frames:
   ```python
   deeplabcut.extract_outlier_frames(config, [videos_folder_path], automatic=True)
   ```
5. Refine labels in Napari:
   ```python
   deeplabcut.refine_labels(config)
   ```
6. Merge refined labels with original dataset:
   ```python
   deeplabcut.merge_datasets(config)
   ```
7. Create new training dataset:
   ```python
   deeplabcut.create_training_dataset(config)
   ```

**Retrain on cluster:**
1. Upload the updated DLC project back to PACE
2. Repeat Steps 5 → 7 → 8 to retrain and re-evaluate

---

### Step 10 — Add More Videos & Repeat (If Still Needed)

If the model still isn't performing well after Step 9:

1. Add a new video (video 3) to the project
2. Run `analyze_videos` on the new video
3. Plot likelihood for the new video
4. Extract outliers for the new video
5. Refine labels for the new video
6. Merge labels from all videos (1, 2, 3)
7. Create new combined training dataset
8. Retrain the model on all 3 videos
9. Re-evaluate

---

### Step 11 — DLC Data Analysis (Summarize Results)

**Goal:** Extract meaningful metrics from DLC output (e.g. pupil area, trial-aligned kinematics).

1. Create a folder `DLC/Data_analysis/` on your local PC
2. Create a Jupyter notebook inside it (remember to select the DeepLabCut kernel)
3. Write your analysis code (see `Pupil_area_compute.py` for a reference example)
4. Define your paths:
   ```python
   csv_dir = "/path/to/DLC/Model/PROJECT/videos/your_best_model_output.csv"
   output_dir = "/path/to/DLC/postproc/Summary_figures"
   ```
5. Run analysis to compute metrics (e.g. pupil area over time, trial-aligned traces)

---

## Phase 3: Neural Data Analysis

### Step 12 — Download Data & Trialization

**Goal:** Align neural recordings to trial events.

**Session data path on Cedar:**
```
/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/processed/joystick/SA16
```

1. Download the `JoystickProcessing2026` code from Cedar:
   ```
   /storage/cedar/.../joystick/code/JoystickProcessing2026
   ```
   Copy it to your home directory on the cluster.

2. Download the session data from Cedar:
   - To run trialization yourself: download the whole session folder
   - Create a folder: `SAxx_LG/` (replace `xx` with subject ID)
   - Place the session folder inside (name format: `SA16_20260119`)

3. **Trialization steps:**
   1. Open VS Code on ICE
   2. Open `SessionTrialization/` folder from the downloaded code
   3. Open `main_jupyter.ipynb`
   4. Set `initial_path` to your session data folder; set `output_dir` to where you want figures
   5. If you get `H5py not found`, run in a new cell: `pip install H5py`
   6. Run the third cell — confirm trialization completed correctly
   7. Two files will be generated in your session data folder:
      - `neural_data.h5`
      - `neural_trials.h5`
   8. Run the last cell to verify both files are readable
   9. Restart the kernel when done

> **Note:** If you skipped trialization due to storage limits, download `neural_data.h5` and `neural_trials.h5` directly from Cedar and place them in the correct session folder.

---

### Step 13 — Basic Alignment & Initial Plotting

**Goal:** Align neural data to events and generate FOV and alignment figures.

1. Place your DLC output CSV in the session folder, renamed to `dlc_output.csv`
2. Request a VS Code interactive session on ICE (same settings as Step 5)
3. Navigate to `JoystickProcessing2026/SessionsTrialization/`
4. Open `main_jupyter.ipynb`, select the `dlc` kernel
5. Edit cell 2: set subject name, ID, and data date
6. Run all cells

**Then for plotting:**
1. Navigate to `JoystickProcessing2026/InitialPlotting/`
2. Open `main_jupyter.ipynb`
3. Edit cell 2 (same subject/date settings)
4. Set `output_dir_onedrive` to your desired output location
5. If you get a `fitz` module error, run:
   ```
   pip uninstall fitz
   pip install pymupdf
   ```
6. Create a `figs/` folder inside your subject folder, with a subfolder named today's date
7. Set today's date in the `date` variable in cell 2
8. Run cells in order:
   - Cell 2: read data and set paths
   - Cell 3: read results and run clustering
   - Next cell: plot FOVs → check output folder for PDFs
   - Last cell: plot alignments

---

### Step 14 — Short/Long Block Separation (Double-Block Sessions Only)

**Goal:** Separate short and long trial plots if your session has a double block design.

1. Copy `InitialPlotting_Short_Long` from Cedar to your home directory:
   ```
   /storage/cedar/.../joystick/last_version_analysis_code/InitialPlotting_Short_Long
   ```
2. Open VS Code on ICE and navigate to the copied folder
3. Open `main_jupyter.ipynb`
4. Edit cell 2 (subject name, ID, date — same as before)
5. Run cells: **1, 2, 3, 5, and the last cell**
6. Short/long separated plots will appear in your output folder

---

### Step 15 — Window Average

**Goal:** Compute and plot average neural activity across trial blocks.

1. Navigate to:
   ```
   /storage/cedar/.../joystick/last_version_analysis_code/average_activity_block_switch
   ```
2. Open `main_trial_wise_complete.ipynb`
3. Run all cells

---

## Troubleshooting

### EOFError on `extract_outlier_frames`
The pickle/metadata file is corrupted or incomplete. Re-run `analyze_videos` for that video to regenerate the metadata before extracting outliers.

### Napari not launching (`TypeError: Shiboken.ObjectType.__new__...`)
Revert to compatible package versions:
```bash
pip install "deeplabcut==3.0.0rc14" "napari==0.6.6" "qtpy==2.4.3" "PySide6==6.10.2" "shiboken6==6.10.2" "napari-deeplabcut==0.2.1.8"
```

### `H5py not found`
```bash
pip install H5py
```

### `fitz` module error in InitialPlotting
```bash
pip uninstall fitz
pip install pymupdf
```

### DLC import taking too long
```python
import os
os.environ["DLClight"] = "True"
import deeplabcut
```

### `FileExistsError` on `add_new_videos` (symlink already exists)
The video symlink already exists in your project. Either skip `add_new_videos` for that video or delete the existing symlink in `DLC/Model/PROJECT/videos/` before re-running.

---

## Quick Reference: Key Code Paths

| Resource | Path |
|---|---|
| All analysis code | `/storage/cedar/.../joystick/code` |
| Raw videos (Cedar) | `/storage/cedar/.../2p_imaging/video_data` |
| Raw videos (PACE) | `/storage/project/r-fnajafi3-0/shared/2P_Imaging/video_data/` |
| JoystickProcessing2026 | `.../joystick/code/JoystickProcessing2026` |
| Short/Long plotting | `.../joystick/last_version_analysis_code/InitialPlotting_Short_Long` |
| Window average | `.../joystick/last_version_analysis_code/average_activity_block_switch` |
| Likelihood plot script | `https://github.com/najafi-laboratory/Deep-Lab-Cut/blob/main/plot_model_likelihoods.ipynb` |
