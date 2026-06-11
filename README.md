# Gaming vs Academic Performance Predictor

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/fastai-2.8-orange?logo=python" alt="fastai 2.8">
  <img src="https://img.shields.io/badge/made_with-Jupyter-orange?logo=jupyter" alt="Made with Jupyter">
  <img src="https://img.shields.io/badge/kaggle-model-20BEFF?logo=kaggle" alt="Kaggle Model">
</p>

<p align="center">
  <b>Try it instantly on Hugging Face:</b>
  <a href="https://huggingface.co/spaces/Anugrha4823/AcademicPerformancePredictor">https://huggingface.co/spaces/Anugrha4823/AcademicPerformancePredictor</a>
</p>

---

A deep learning model that predicts student grades (0–100) from gaming habits, lifestyle, and behavioral data — built with fastai and PyTorch.

Achieves **MAE 4.91**, **R² 0.92** on a held-out test set of 800 students.

## Overview

The project uses a tabular neural network to model how factors like gaming hours, study time, sleep, attendance, and gaming genre affect academic performance. It consists of two notebooks:

- **Training** — EDA, model training, 5-fold cross-validation, permutation feature importance
- **Inference** — single prediction, batch CSV prediction, test set evaluation

## Project Structure

```
gaming-vs-grades-predictor/
├── gaming-vs-grades-train.ipynb   # Training notebook — EDA, model training, k-fold CV, evaluation
├── main.ipynb                     # Inference notebook — single prediction, batch CSV, test analysis
├── data/
│   └── Gaming_Academic_Performance.csv   # Source dataset
├── models/
│   ├── model.pkl                   # Exported fastai learner (used by inference notebook)
│   └── model.pth                   # Model weights
├── requirements.txt
└── README.md
```

## Dataset

**Source:** [Gaming vs Academic Performance](https://www.kaggle.com/datasets/aiexplorer77/gaming-vs-academic-performance) on Kaggle

| Feature | Type | Description |
|---|---|---|
| `gaming_hours` | Continuous | Daily gaming hours |
| `study_hours` | Continuous | Daily study hours |
| `sleep_hours` | Continuous | Daily sleep hours |
| `attendance` | Continuous | Attendance percentage |
| `gaming_genre` | Categorical | Preferred game genre (FPS, RPG, Casual, etc.) |
| `stress_level` | Categorical | Self-reported stress (Low, Medium, High) |
| `gender` | Categorical | Student gender |
| `device_usage` | Continuous | Daily device usage hours |
| `addiction_score` | Continuous | Gaming addiction score |
| `reaction_time_ms` | Continuous | Reaction time in milliseconds |
| `social_activity` | Continuous | Social activity score |
| `grades` | Continuous | Target — academic grade (0–100) |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook gaming-vs-grades-train.ipynb
```

## Model Architecture

| Setting | Value |
|---|---|
| Layers | [300, 150] |
| Dropout | 0.1 |
| Weight decay | 0.1 |
| Batch size | 128 |
| Max epochs | 50 (early stopping, patience=5) |
| Optimizer | AdamW (1cycle policy) |
| Target range | Clipped to (0, 100) |

## Evaluation

**Test set (800 holdout samples):**

| Metric | Value |
|---|---|
| **MAE** | 4.91 |
| **RMSE** | 6.24 |
| **R²** | 0.9205 |
| Within 5 points | 56.9% |
| Within 10 points | 88.9% |
| Within 15 points | 98.4% |

5-fold cross-validation was consistent (avg RMSE ~6.1). The training notebook includes permutation feature importance, residual analysis, and per-fold visualizations.

## Try the Model

You can use the model without cloning this repo:

- **Kaggle:** [Model page](https://www.kaggle.com/models/anugrhabhujel/academic-performance-predictor) with demo code
- **Locally:** Load `models/model.pkl` via `load_learner()` (see Usage below)

## Usage

### Single Student Prediction

```python
student = {
    'age': 18,
    'gender': 'Male',
    'gaming_hours': 2.0,
    'study_hours': 1.0,
    'sleep_hours': 9.0,
    'attendance': 5.0,
    'gaming_genre': 'FPS',
    'social_activity': 3.0,
    'device_usage': 11.0,
    'reaction_time_ms': 90.0,
    'addiction_score': 2.0,
    'stress_level': 'High'
}
```

### Batch Prediction

Point to any CSV with matching column names:

```python
csv_path = 'data/test.csv'
```

## Dependencies

Core: `fastai>=2.8`, `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

See `requirements.txt` for the full pinned list.

## License

MIT
