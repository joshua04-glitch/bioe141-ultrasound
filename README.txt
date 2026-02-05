# BIOE 141 – Cardiac Ultrasound Machine Learning

This repository contains code, notebooks, and batch scripts developed for the **BIOE 141 Senior Design** project at Stanford University. The project focuses on applying machine learning methods to **cardiac ultrasound analysis**, including image segmentation, ejection fraction (EF) estimation, and ultrasound quality assessment.

⚠️ **Important:** This repository intentionally contains **code and notebooks only**.  
Large datasets, trained models, checkpoints, logs, and virtual environments are **not tracked** and must be available locally.

---

## Repository Structure


---

## What Is NOT Included

The following items are **ignored by Git** and must exist locally to run the code:

### Datasets
- `data/`
- `database_nifti/`
- `Cactus Dataset/`
- `PanEcho/`

### Models & Outputs
- `weights/`
- `checkpoints/`
- `results/`
- `*.pt`

### Environments & Logs
- `torch-gpu/`
- `venvs/`
- `logs/`
- `__pycache__/`

These files are excluded to keep the repository lightweight and shareable.

---

## Environment Setup (FarmShare)

This project was developed on **Stanford FarmShare** using Python and PyTorch.

Typical workflow:

```bash
cd ~/141
source torch-gpu/bin/activate

cd ~/141
python ML\ train.py

sbatch train_cactus_batch.sh
sbatch train_panecho_batch.sh

export DATA_ROOT=~/141/data
