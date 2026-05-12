# Lung Cancer Detection Model
This section contains preprocessing, training and evaluation for lung cancer proteomics data.

research review name: Stability-Guided Multi-Resolution Spectral Learning for Lung Tumor Classification

## Overview
This project presents a machine learning framework for lung tumor classification using MS1 proteomics mass spectrometry data. The pipeline combines spectral preprocessing, multi-resolution feature engineering, stability-based feature selection, and machine learning models to identify discriminative proteomic patterns associated with tumor samples.

The framework was developed as part of a computational proteomics and cancer biomarker discovery research study.


## Objectives
- Analyze MS1 proteomics spectral data
- Perform lung tumor vs normal tissue classification
- Extract stable spectral signatures
- Investigate candidate proteomic biomarker regions
- Build an interpretable and scalable ML pipeline

## Dataset
Dataset Source:
PRIDE Proteomics Repository

Dataset ID:
PXD002612

Dataset Description:
Publicly available lung adenocarcinoma proteomics mass spectrometry dataset containing tumor and normal tissue spectral samples.

Data Type:
- mzML spectral files
- MS1 spectra
- Tumor and normal tissue samples

## Methodology

### 1. Spectral Preprocessing
- Raw mzML spectral files processed
- MS1 peaks extracted
- m/z-intensity representation generated

### 2. Feature Engineering
- m/z binning across spectral range
- Multi-resolution spectral representation
- Spectral entropy calculation

Bin widths used:
- Δm = 1
- Δm = 5
- Δm = 10

### 3. Stability-Based Feature Selection
Repeated subsampling experiments were performed to identify stable spectral regions consistently associated with tumor samples.

### 4. Machine Learning Models
The following classifiers were evaluated:
- Support Vector Machine (SVM)
- Random Forest (RF)
- Logistic Regression (LR)

### 5. Evaluation
Models were evaluated using:
- Stratified Cross Validation
- ROC-AUC
- Accuracy Analysis

## Results
- Best ROC-AUC achieved: ~0.85
- Stable discriminative m/z regions identified
- Multi-resolution spectral learning improved robustness
- Stability selection reduced noisy features

Example candidate spectral regions:
- 514–515 m/z
- 567–568 m/z
- 846–847 m/z

## Project Structure

```text
project/
│
├── notebook/
│   └── final_experiment.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│
├── models/
│   └── final_model.pkl
│
├── results/
│   ├── roc_curve.png
│   ├── heatmap.png
│
├── data/
│   ├── X.npy
│   └── y.npy
│
├── main.py
├── requirements.txt
└── README.md
