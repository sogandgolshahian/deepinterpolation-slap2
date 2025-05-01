# 🧠 DeepInterpolation for SLAP2 Microscopy

This repository contains the complete codebase and evaluation scripts used for the project:

**"Denoising SLAP2 Two-Photon Microscopy Images of Tadpole Neurons using DeepInterpolation."**

## 🔬 Overview

This project adapts **DeepInterpolation**, a self-supervised denoising framework developed by the Allen Institute, to high-speed SLAP2 two-photon microscopy data of *Xenopus* tadpole neurons. The goal is to remove random noise and preserve neuronal structure without the need for ground-truth clean images.

Model performance was evaluated across several SLAP2 imaging scenarios:
- Fluorescent dye (static morphology) baseline recordings
- Calcium biosensor activity imaging
- Recordings with structured projector-induced artifacts
- Synthetic data with known clean-noisy ground truth
- Volumetric recordings with motion artifacts

## 📁 Repository Structure

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
├── results/
│   └── [optional: figures and evaluation outputs]
├── data/
│   └── (not included – add your own SLAP2 movies)
└── README.md
```

## 💻 Requirements

- Python 3.7  
- TensorFlow 2.4.4  
- NumPy, SciPy, scikit-image  
- Matplotlib, Seaborn, PyWavelets (`pywt`)  
- Tested on:
  - Windows 11 Pro
  - 48 GB RAM
  - NVIDIA RTX 3070 GPU

## 🧠 Models

Two DeepInterpolation models were trained independently:
- **Condition 1:** Fluorescent dye recordings (baseline morphology)
- **Condition 2:** Calcium biosensor recordings (neuronal activity)

Each condition has **two versions** of the model:
- One trained using **int16** input data
- One trained using **uint16** input data

This is because DeepInterpolation internally converts all inputs to `float32`, and proper handling of signed vs. unsigned data types during preprocessing and normalization is essential. To ensure compatibility and numerical consistency across different datasets, both versions are provided.

## 📊 Evaluation

The following evaluation scripts are included:
- **Signal-to-Noise Ratio (SNR)** calculation using Otsu-based region segmentation
- **Structural Similarity Index (SSIM)** for synthetic datasets with known ground truth
- **Qualitative comparisons** across raw, filtered, and DeepInterpolated frames

Note: Figures and thesis report are not included in this repository.

## ⚙️ Installation and Setup

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/deepinterpolation_slap2.git
cd deepinterpolation_slap2
```

2. Install the original DeepInterpolation framework:  
   🔗 https://github.com/AllenInstitute/deepinterpolation

3. Refer to this Google Doc for additional installation and configuration steps:  
   📄 [INSERT YOUR GOOGLE DOC LINK HERE]

---

## 🚀 How to Run

After installing DeepInterpolation and its dependencies:

- Edit the training and inference scripts (`train_conditionX.py` and `inference_conditionX.py`) to set the correct file paths for:
  - `train_json_path`
  - `output_file`
  - `input_file_path`

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

---

## 📎 License

MIT License

---

## ✨ Acknowledgments

- Dr. Kurt Haas and the Haas Lab at UBC for providing SLAP2 microscopy data and research mentorship  
- The Allen Institute for Neural Dynamics for developing the original DeepInterpolation framework  
- The UBC Biomedical Engineering program and BMEG 490B Directed Studies for project support
