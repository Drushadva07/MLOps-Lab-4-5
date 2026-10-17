# Advanced MLflow and Data Versioning with DVC

## Project Overview
This project demonstrates MLflow experiment tracking, model comparison, model registration, and dataset versioning using Git and DVC. A Random Forest Regressor predicts house prices using area, bedrooms, age, and location score.

## Technologies
- Python 3.11
- MLflow 3.15.1
- DVC 3.67.1
- Scikit-learn 1.9.0
- Pandas and Git

## Project Structure
- `src/train.py` - trains and evaluates three models
- `data/dataset.csv.dvc` - DVC metadata for the dataset
- `data/dataset.csv` - dataset managed by DVC
- `models/metrics.json` - evaluation metrics
- `mlflow.db` - local MLflow tracking database
- `reports/figures/` - screenshots for the report

## Setup
Activate the project environment and install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Training
From the project root:

```powershell
python src/train.py
```

## Open MLflow
Start the MLflow tracking UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000 to compare experiments, metrics, artifacts, and model versions.

## DVC Commands
Check tracking status:

```powershell
dvc status
```

Push tracked data to the configured remote:

```powershell
dvc push
```

Restore the dataset version referenced by the current Git checkout:

```powershell
dvc checkout
```

The original dataset has 30 rows, and the updated dataset has 33 rows. Git tracks DVC metadata, while DVC stores the dataset contents.

## Model Comparison

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| RF_50_Trees | 14193.33 | 16215.89 | 0.9567 |
| RF_100_Trees | 13050.83 | 14615.08 | 0.9648 |
| RF_100_Trees_Depth10 | 12991.67 | 14668.72 | 0.9646 |

The RF_100_Trees model achieved the highest R2 score among these three runs.

## Reproducibility
- Training uses `random_state=42`.
- MLflow runs record parameters, metrics, dataset SHA-256, and Git commit tags.
- The DVC remote is configured on the original machine. Another machine needs an accessible remote containing the dataset.
- The dataset is small and synthetic, so the results demonstrate the workflow rather than real-world house-price prediction performance.

