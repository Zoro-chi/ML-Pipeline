import os
from urllib.parse import urlparse
import requests
from requests.auth import HTTPBasicAuth
import yaml
import pickle

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature

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


def hyperparameter_tuning(X_train, y_train, param_grid):
    # Create a Random Forest Classifier
    clf = RandomForestClassifier()

    # Create grid search object
    clf_grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit on training data
    clf_grid.fit(X_train, y_train)

    return clf_grid


# Load parameters from params.yaml
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(root_dir, "params.yaml")
params = yaml.safe_load(open(params_path))["train"]


def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)

    # Set up the session with basic authentication
    session = requests.Session()
    session.auth = HTTPBasicAuth(ML_FLOW_TRACKING_USERNAME, ML_FLOW_TRACKING_PASSWORD)

    # Set the environment variables for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = ML_FLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = ML_FLOW_TRACKING_PASSWORD

    # Check if the connection is successful
    response = requests.get(
        f"{ML_FLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list",
        auth=HTTPBasicAuth(ML_FLOW_TRACKING_USERNAME, ML_FLOW_TRACKING_PASSWORD),
    )
    print(response.status_code, response.text)

    # Start MLflow run
    with mlflow.start_run():
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        signature = infer_signature(X_train, y_train)

        # Define hyperparameters grid for Model in use (Random Forest Classifier)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param(
            "best_min_samples_split", grid_search.best_params_["min_samples_split"]
        )
        mlflow.log_param(
            "best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"]
        )

        # Log confusion matrix and classification report
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        mlflow.log_text(str(confusion), "confusion_matrix.txt")
        mlflow.log_text(classification_rep, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Register the model
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name="Best Model",
            )
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Create a directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(best_model, open(filename, "wb"))

        print(f"Model saved to {model_path}")


if __name__ == "__main__":

    data_path = os.path.join(root_dir, params["data"])
    model_path = os.path.join(root_dir, params["model"])

    train(
        data_path,
        model_path,
        params["random_state"],
        params["n_estimators"],
        params["max_depth"],
    )
