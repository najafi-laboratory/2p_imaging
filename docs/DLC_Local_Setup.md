# Local Environment Setup for DeepLabCut & VS Code

This guide walks through setting up Anaconda, Python 3.10, DeepLabCut (v3.0.0rc10), and Jupyter/VS Code on your local machine for video annotation.

---

## Step 1: Install Anaconda

### Windows
Open PowerShell and run:
```powershell
curl.exe -L -o Anaconda3-Windows-x86_64.exe https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

Start-Process -FilePath ".\Anaconda3-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$env:USERPROFILE\anaconda3" -Wait

& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" "init" "powershell" 
```

>Close PowerShell and reopen it. You should now see (base) before your command prompt: ```(base) PS C:\Users\xyz ```


### Mac OS/Linux
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-MacOSX-arm64.sh

bash Anaconda3-2024.06-1-MacOSX-arm64.sh

source ~/.bashrc
```

---

## Step 2: Create Conda Environment
Run in terminal/Powershell:
```shell
conda create --name DeepLabCut python=3.10
conda activate DeepLabCut
```
>'DeepLabCut' is your environment name

---

## Step 3: Install DeepLabCut & Dependencies (DLC, ipython, ipykernel, jupyter)

### Windows (Tested Dependency Pins) 

To prevent PyTorch DLL initialization failures, NumPy 2.x breaking changes, and PySide6/Napari GUI crashes, use the pinned versions below:
```powershell
conda install -y -c conda-forge mkl intel-openmp

pip install "deeplabcut[gui,modelzoo]==3.0.0rc10" ipython ipykernel jupyter "numpy<2.0.0" "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu "pyside6==6.5.3" "shiboken6==6.5.3" "napari==0.5.4" "app-model==0.3.0" "vispy==0.14.3" appdirs numpydoc jsonschema magicgui "napari-svg" "napari-plugin-engine>=0.1.9" 
```

### MacOS/Linux
```bash
pip install "deeplabcut[gui,modelzoo]==3.0.0rc10" ipython ipykernel jupyter
```

--- 

## Step 4: Register your Conda environment as a Jupyter kernel (so it appears in Jupyter and VS Code) 
```shell
python -m ipykernel install --user --name DeepLabCut --display-name "Python (DeepLabCut)"
```

--- 

## Step 5: Test GUI
```shell
python -m deeplabcut
```

## Step 6: Install Visual Studio (VS) Code
1. Go to the VS Code Website: https://code.visualstudio.com/
2. Download VS Code: 
    * Click the "Download for [Your Operating System]" button (e.g., Windows, macOS, Linux)
3. Install VS Code: 
    * Run the installer:
        * On Windows, follow the on-screen instructions. Select "Add to PATH" and "Install Code extensions" options. 
        * On macOS, drag the VS Code application to the Applications folder. 
        * On Linux, follow the instructions provided for your specific distribution.
4. Launch VS Code:
    * Open VS Code:
        * Windows/macOS: Search for "Visual Studio Code" in your system's search bar. 
        * Linux: Run ```code``` from the terminal. 
5. Install the Python Extension
    * Open VS Code. 
    * Click on the Extensions icon (on the left sidebar, looks like four squares). 
    * Search for "Python" and click "Install" on the extension published by Microsoft. 