# BIOE 141 – Cardiac Ultrasound ML Pipeline

This repository contains code, notebooks, and batch scripts developed for the BIOE 141 senior design project. The focus is on machine learning methods for cardiac ultrasound analysis, including image segmentation, ejection fraction (EF) estimation, and ultrasound quality assessment.

⚠️ **Important:** This repository contains **code and notebooks only**.  
Large datasets, trained models, checkpoints, and virtual environments are intentionally **not tracked** and must be provided separately.

---

## Repository Structure


---

## What Is NOT Included (by design)

The following **must exist locally** but are **ignored by Git**:

- Datasets  
  - `data/`
  - `database_nifti/`
  - `Cactus Dataset/`
  - `PanEcho/`
- Trained models and outputs  
  - `weights/`
  - `checkpoints/`
  - `results/`
  - `*.pt`
- Virtual environments  
  - `torch-gpu/`
  - `venvs/`
- Logs and archives  
  - `logs/`
  - `*.zip`

If you clone this repo, you will need to obtain datasets and models separately.

---

## Environment Setup (FarmShare)

This project was developed on Stanford FarmShare using Python + PyTorch.

Typical workflow:

```bash
cd ~/141
source torch-gpu/bin/activate
