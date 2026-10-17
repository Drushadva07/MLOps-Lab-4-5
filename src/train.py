
import os
import json
import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

# Project paths
DATA_PATH = "data/dataset.csv"
ARTIFACT_DIR = "models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a reproducible dataset hash
with open(DATA_PATH, "rb") as file:
    dataset_hash = hashlib.sha256(file.read()).hexdigest()

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House_Price_Advanced_MLflow")

# Three candidate models with different hyperparameters
candidates = [
    {"name": "RF_50_Trees", "n_estimators": 50, "max_depth": 5},
    {"name": "RF_100_Trees", "n_estimators": 100, "max_depth": 5},
    {"name": "RF_100_Trees_Depth10", "n_estimators": 100, "max_depth": 10},
]

for config in candidates:
    with mlflow.start_run(run_name=config["name"]):
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        # Track hyperparameters, metrics, and dataset identity
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "random_state": 42,
            "dataset_sha256": dataset_hash,
            "sklearn_version": sklearn.__version__,
        })

        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

        # Save a metrics artifact
        metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
        with open(metrics_path, "w") as file:
            json.dump(
                {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2},
                file,
                indent=4
            )
        mlflow.log_artifact(metrics_path)

        # Log the trained model with input signature and example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="house_price_model",
            signature=signature,
            input_example=input_example
        )

        print(f"\nModel: {config['name']}")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")
        print(f"Dataset SHA-256: {dataset_hash}")

