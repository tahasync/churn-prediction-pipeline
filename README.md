# Churn Prediction Pipeline — Telco Customer Churn

Compares Logistic Regression, Random Forest, and XGBoost classifiers on the Kaggle Telco Customer Churn dataset using Stratified 5-Fold cross-validation, serves the best model via FastAPI, and explains predictions with SHAP.

## What it does

Loads the Telco Customer Churn CSV (7043 customers, 21 features), engineers features (tenure buckets, charges-per-tenure ratio, contract×payment interactions), trains three classifiers with StratifiedKFold CV, selects the best performer, and serializes it to `best_model.pkl`. A FastAPI endpoint (`/predict`) accepts customer data and returns churn probability plus top-3 SHAP drivers. A React frontend provides a form interface with a circular risk gauge and SHAP bar chart. A Dockerfile is included for containerized deployment.

**Note:** SMOTE is imported but is **not actually wired into the training pipeline** — no oversampling is applied during model training.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, joblib
- **ML:** scikit-learn (Logistic Regression, Random Forest), XGBoost, SHAP (TreeExplainer), imbalanced-learn (SMOTE — imported but unused)
- **Frontend:** React 18, Vite 5, Tailwind CSS, Recharts
- **Data:** Pandas, NumPy
- **Infrastructure:** Docker (python:3.11-slim), pytest (3 tests)

## Features

- **Feature engineering** — tenure buckets, charges-per-tenure ratio, contract×payment interaction, fiber-optic/no-support flags
- **Multi-model comparison** — LR/RF/XGB with Stratified 5-Fold CV, auto-selects best model
- **SHAP explanations** — per-prediction top-3 feature importance (uses TreeExplainer for tree models, LinearExplainer for LR)
- **FastAPI endpoint** — `POST /predict` returns churn probability + SHAP values
- **React dashboard** — customer input form, color-coded risk gauge, SHAP bar chart
- **Docker** — containerized API with python:3.11-slim
- **Tests** — 3 unit tests covering feature engineering and data preparation

## Setup

1. Download the Kaggle Telco Customer Churn dataset and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. Install Python deps: `pip install -r requirements.txt`
3. Train the model: `python -m src.train`
4. Build the frontend: `cd frontend && npm install && npm run build && cd ..`
5. Start the API: `uvicorn api.main:app --host 127.0.0.1 --port 8000`

Or on Windows: run `run_all.bat`.

## Model performance (from `models/cv_results.md`)

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.567 ± 0.023 | 0.706 ± 0.030 | 0.628 ± 0.019 | **0.840 ± 0.013** |
| Random Forest | 0.573 ± 0.022 | 0.607 ± 0.020 | 0.590 ± 0.019 | 0.821 ± 0.014 |
| XGBoost | 0.563 ± 0.025 | 0.616 ± 0.018 | 0.588 ± 0.020 | 0.818 ± 0.008 |

XGBoost is saved as the default model for SHAP TreeExplainer interpretability (preferred over LR when within 0.025 ROC-AUC).

## Project structure

```
├── api/main.py             # FastAPI with /predict and /health
├── src/
│   ├── preprocessing.py    # Load, clean, encode (SMOTE defined but not wired)
│   ├── features.py         # Feature engineering
│   ├── train.py            # 3-model comparison + CV + serialization
│   ├── evaluate.py         # SHAP summary + per-prediction plots
│   └── predict.py          # ChurnPredictor singleton
├── frontend/               # React + Vite dashboard
├── models/                 # best_model.pkl, cv_results.md, SHAP plots
├── tests/test_preprocessing.py  # 3 unit tests
├── notebooks/01_eda.ipynb  # Exploratory data analysis
├── Dockerfile
└── requirements.txt
```

## Status

**Functional portfolio project.** The training pipeline, API, and frontend work end-to-end once the dataset is placed in the correct path. The Docker build is correct. Known limitations: SMOTE is imported but not applied during training; the README metrics table previously displayed inflated numbers — the actual scores are in `models/cv_results.md`.