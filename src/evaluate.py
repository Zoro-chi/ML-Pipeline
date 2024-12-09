import os
import yaml
import pandas as pd
from urllib.parse import urlparse

import pickle
from sklearn.metrics import accuracy_score
import mlflow

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Setting variables
ML_FLOW_TRACKING_URI = os.getenv("ML_FLOW_TRACKING_URI")
ML_FLOW_TRACKING_USERNAME = os.getenv("ML_FLOW_TRACKING_USERNAME")
ML_FLOW_TRACKING_PASSWORD = os.getenv("ML_FLOW_TRACKING_PASSWORD")

# Set the environment variables for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = ML_FLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = ML_FLOW_TRACKING_PASSWORD

print(f"ML_FLOW_TRACKING_URI: {ML_FLOW_TRACKING_URI}")
print(f"ML_FLOW_TRACKING_USERNAME: {ML_FLOW_TRACKING_USERNAME}")
print(f"ML_FLOW_TRACKING_PASSWORD: {ML_FLOW_TRACKING_PASSWORD}")

# Load parameters from params.yaml
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(root_dir, "params.yaml")
params = yaml.safe_load(open(params_path))["train"]


def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)

    # Load the model from the model path
    model = pickle.load(open(model_path, "rb"))

    # Predict the values
    y_pred = model.predict(X)

    # Calculate the accuracy
    accuracy = accuracy_score(y, y_pred)

    # Log the metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")


if __name__ == "__main__":

    data_path = os.path.join(root_dir, params["data"])
    model_path = os.path.join(root_dir, params["model"])

    evaluate(data_path, model_path)


# Add the preprocess stage to the DVC pipeline
"""
dvc stage add -n preprocess \
    -p preprocess.input, preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
"""

# Add the train stage to the DVC pipeline
"""
dvc stage add -n train \
    -p train.data, train.model, train.random_state, train.n_estimators, train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py
"""

# Add the evaluate stage to the DVC pipeline
"""
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
"""
