# DeepInterpolation for SLAP2 Microscopy

This repository contains the complete codebase and evaluation scripts used for the project:

**"Denoising SLAP2 Two-Photon Microscopy Images of Tadpole Neurons using DeepInterpolation."**

## Overview

This project adapts **DeepInterpolation**, a self-supervised denoising framework developed by the Allen Institute, to high-speed SLAP2 two-photon microscopy data of *Xenopus* tadpole neurons. The goal is to remove random noise and preserve neuronal structure without the need for ground-truth clean images.

Model performance was evaluated across several SLAP2 imaging scenarios:
- Fluorescent dye (static morphology) baseline recordings
- Calcium biosensor activity imaging
- Recordings with structured projector-induced structured artifacts
- Synthetic noisy data with known clean ground truth
- Volumetric recordings

## Repository Structure

```
├── training/
│   ├── train_condition1.py             # Fluorescent dye model training
│   └── train_condition2.py             # Calcium biosensor model training
├── inference/
│   ├── inference_condition1.py
│   └── inference_condition2.py
├── filtering/
│   └── traditional_filtering.py        # Gaussian and Wavelet filtering
├── evaluation/
│   ├── calculate_snr.py
│   ├── calculate_ssim_synthetic_data.py
│   └── losses_plots.py
├── data/
│   └── (not included – add your own SLAP2 movies)
└── README.md
```

## Requirements

- Python 3.7
- TensorFlow 2.4.4
- NumPy, SciPy, scikit-image
- Matplotlib, Seaborn, PyWavelets (`pywt`)
- Tested on:
  - Windows 11 Pro
  - 48 GB RAM
  - NVIDIA RTX 3070 GPU

## Models

Two DeepInterpolation models were trained independently:
- **Condition 1:** Fluorescent dye recordings (baseline morphology)
- **Condition 2:** Calcium biosensor recordings (neuronal activity)

Each model was trained on three 60-second SLAP2 movies and validated on a fourth. Inference was performed on eight test recordings, including ones with structured artifacts and volumetric scans.

## Evaluation

The following evaluation scripts are included:
- **Signal-to-Noise Ratio (SNR)** calculation using Otsu-based region segmentation
- **Structural Similarity Index (SSIM)** for synthetic datasets with known ground truth
- **Qualitative comparisons** across raw, filtered, and DeepInterpolated frames

## How to Run

```bash
# Train model (Condition 1 - Fluorescent Dye)
python training/train_condition1.py

# Train model (Condition 2 - Calcium Biosensor)
python training/train_condition2.py

# Inference using trained model
python inference/inference_condition1.py
python inference/inference_condition2.py

# Apply traditional filters to test movies
python filtering/traditional_filtering.py

# Evaluate SNR across all frames
python evaluation/calculate_snr.py

# Evaluate SSIM for synthetic clean/noisy/denoised movies
python evaluation/calculate_ssim_synthetic_data.py
```

## License

MIT License

## Acknowledgments

- Dr. Kurt Haas and the Haas Lab at UBC for providing SLAP2 microscopy data and research mentorship  
- The Allen Institute for Neural Dynamics for developing the original DeepInterpolation framework  
- The UBC Biomedical Engineering program and BMEG 490B Directed Studies for project support
