# Point Cloud Semantic Segmentation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/xgboost-latest-green.svg)
![LightGBM](https://img.shields.io/badge/lightgbm-latest-blue.svg)
![CatBoost](https://img.shields.io/badge/catboost-latest-yellow.svg)
![SHAP](https://img.shields.io/badge/shap-explainability-red.svg)

This repository contains the codebase for my Bachelor Thesis focusing on **Point Cloud Semantic Segmentation** using high-performance Tree-based Machine Learning models.

## Overview

The goal of this project is to accurately classify urban point clouds into distinct semantic classes (e.g., Terrain, Vegetation, Buildings) by engineering robust geometric features and training state-of-the-art gradient boosting algorithms. The pipeline includes:
- **Feature Engineering:** Geometric feature extraction performed in CloudCompare.
- **Benchmarking:** Automated evaluation of XGBoost, LightGBM, and CatBoost.
- **Visualization:** High-resolution 3D point cloud rendering.
- **Explainable AI (XAI):** Global and local model explainability using SHAP (SHapley Additive exPlanations).

## Dataset

This project utilizes the **RMIT area dataset**. The raw `.ply` files were processed to extract localized geometric features (linearity, planarity, sphericity, verticality, etc.) at a 0.5m radius.

*Note: Due to GitHub file size limits, the dataset and trained models are not included in this repository.*
- **Download Link:** [FOR-instance Dataset (Zenodo)](https://zenodo.org/records/8287792?preview_file=FORinstance_dataset.zip)
- **Credit:** Full credit is given to the original dataset producers for providing this valuable resource for academic research.

## Project Structure

```text
├── data/
│   ├── raw/                 # Put your raw PLY files here
│   └── processed/           # Put train.ply and test.ply here
├── models/
│   └── saved/               # Trained models (.joblib) are saved here
├── src/
│   ├── data_processing.py   # PLY file parsing logic
│   ├── features.py          # Feature extraction scaffolding
│   ├── benchmark.py         # Main training and evaluation pipeline
│   ├── visualize.py         # 3D Matplotlib visualization generator
│   ├── evaluation.py        # SHAP beeswarm and decision plots
│   └── models/              # Individual model hyperparameters
├── requirements.txt         # Project dependencies
└── README.md
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/burakhfzgl/Interpretation-of-Geometric-Features-Effectiveness-for-Point-Cloud-Semantic-Segmentation-Using-XAI.git
   cd Interpretation-of-Geometric-Features-Effectiveness-for-Point-Cloud-Semantic-Segmentation-Using-XAI
   ```

2. **Install dependencies:**
   It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Placement:**
   Place your exported CloudCompare data into the project:
   - `data/processed/train.ply`
   - `data/processed/test.ply`

## Usage

### 1. Training & Benchmarking
Train XGBoost, CatBoost, and LightGBM and evaluate their Precision, Recall, F1 Score, and Overall Accuracy. This script automatically balances class weights and saves the best model.
```bash
python src/benchmark.py
```

### 2. 3D Visualization
Generate side-by-side, high-resolution 3D point cloud plots for visual comparison of the model predictions.
```bash
python src/visualize.py
```

### 3. SHAP Explainability
Generate Global Beeswarm plots and Local Decision plots to analyze how the geometric features impact the model's predictions.
```bash
python src/evaluation.py
```

## Results

On the RMIT dataset, our optimized models achieved highly robust baseline metrics across 6 highly-imbalanced semantic classes:
- **XGBoost:** ~0.82 Accuracy, ~0.80 Macro F1
- **LightGBM:** ~0.81 Accuracy, ~0.80 Macro F1
- **CatBoost:** ~0.81 Accuracy, ~0.80 Macro F1

## License
[MIT License](LICENSE)
