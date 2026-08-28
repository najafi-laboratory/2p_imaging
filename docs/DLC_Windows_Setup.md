# DeepLabCut Local Setup Guide (Windows)

This guide provides the tested, dependency-pinned installation instructions for running DeepLabCut (v3.0.0rc10) on Windows machines to avoid PyTorch DLL and PySide6/Napari GUI crashes.

## 1. Install Anaconda
Open PowerShell and run:
```
curl.exe -L -o Anaconda3-Windows-x86_64.exe [https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe)
```
```
Start-Process -FilePath ".\Anaconda3-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$env:USERPROFILE\anaconda3" -Wait
```
```
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" "init" "powershell"
```

## Create Conda Environment & Install Dependencies
```
conda create --name DeepLabCut python=3.10 -y
```
```
conda activate DeepLabCut
```
```
conda install -y -c conda-forge mkl intel-openmp
```

## Install DLC and pinned compatible Windows GUI dependencies
```
pip install "deeplabcut[gui,modelzoo]==3.0.0rc10" ipython ipykernel jupyter "numpy<2.0.0" "torch==2.3.1" "torchvision==0.18.1" --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) "pyside6==6.5.3" "shiboken6==6.5.3" "napari==0.5.4" "app-model==0.3.0" "vispy==0.14.3" appdirs numpydoc jsonschema magicgui "napari-svg" "napari-plugin-engine>=0.1.9"
```

## Register Jupyter Kernel
```
python -m ipykernel install --user --name DeepLabCut --display-name "Python (DeepLabCut)"
```

## Verify GUI Launch
```
python -m deeplabcut
```