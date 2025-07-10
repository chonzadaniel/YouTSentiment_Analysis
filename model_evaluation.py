# import numpy as np
# import pandas as pd
# import pickle
# import logging
# import yaml
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# from mlflow.models import infer_signature
#
# # logging configuration
# logger = logging.getLogger('model_evaluation')
# logger.setLevel('DEBUG')
#
# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')
#
# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel('ERROR')
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
#
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)
#
#
# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         df.fillna('', inplace=True)  # Fill any NaN values
#         logger.debug('Data loaded and NaNs filled from %s', file_path)
#         return df
#     except Exception as e:
#         logger.error('Error loading data from %s: %s', file_path, e)
#         raise
#
#
# def load_model(model_path: str):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', model_path)
#         return model
#     except Exception as e:
#         logger.error('Error loading model from %s: %s', model_path, e)
#         raise
#
#
# def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
#     """Load the saved TF-IDF vectorizer."""
#     try:
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
#         logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
#         return vectorizer
#     except Exception as e:
#         logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
#         raise
#
#
# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters loaded from %s', params_path)
#         return params
#     except Exception as e:
#         logger.error('Error loading parameters from %s: %s', params_path, e)
#         raise
#
#
# def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray):
#     """Evaluate the model and log classification metrics and confusion matrix."""
#     try:
#         # Predict and calculate classification metrics
#         y_pred = model.predict(x_test)
#         report = classification_report(y_test, y_pred, output_dict=True)
#         cm = confusion_matrix(y_test, y_pred)
#
#         logger.debug('Model evaluation completed')
#
#         return report, cm
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise
#
#
# def log_confusion_matrix(cm, dataset_name):
#     """Log confusion matrix as an artifact."""
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix for {dataset_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#
#     # Save confusion matrix plot as a file and log it to MLflow
#     cm_file_path = f'confusion_matrix_{dataset_name}.png'
#     plt.savefig(cm_file_path)
#     mlflow.log_artifact(cm_file_path)
#     plt.close()
#
# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         # Create a dictionary with the info you want to save
#         model_info = {
#             'run_id': run_id,
#             'model_path': model_path
#         }
#         # Save the dictionary as a JSON file
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise
#
#
# def main():
#     mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
#     print("Tracking URI:", mlflow.get_tracking_uri())
#
#     mlflow.set_experiment('dvc-pipeline-runs')
#
#     with mlflow.start_run() as run:
#         try:
#             # Load parameters from YAML file
#             root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
#             params = load_params(os.path.join(root_dir, 'params.yaml'))
#
#             # Log parameters
#             for key, value in params.items():
#                 mlflow.log_param(key, value)
#
#             # Load model and vectorizer
#             model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
#             vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
#
#             # Load test data for signature inference
#             test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
#
#             # Prepare test data
#             X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
#             y_test = test_data['category'].values
#
#             # Create a DataFrame for signature inference (using the first few rows as an example)
#             input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  # <--- Added for signature
#
#             # Infer the signature
#             signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature
#
#             # Log model with signature
#             mlflow.sklearn.log_model(
#                 model,
#                 "lgbm_model",
#                 signature=signature,  # <--- Added for signature
#                 input_example=input_example  # <--- Added input example
#             )
#
#             # Save model info
#             artifact_uri = mlflow.get_artifact_uri()
#             model_path = f"{artifact_uri}/lgbm_model"
#             save_model_info(run.info.run_id, model_path, 'experiment_info.json')
#
#             # Log the vectorizer as an artifact
#             mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
#
#             # Evaluate model and get metrics
#             report, cm = evaluate_model(model, X_test_tfidf, y_test)
#
#             # Log classification report metrics for the test data
#             for label, metrics in report.items():
#                 if isinstance(metrics, dict):
#                     mlflow.log_metrics({
#                         f"test_{label}_precision": metrics['precision'],
#                         f"test_{label}_recall": metrics['recall'],
#                         f"test_{label}_f1-score": metrics['f1-score']
#                     })
#
#             # Log confusion matrix
#             log_confusion_matrix(cm, "Test Data")
#
#             # Add important tags
#             mlflow.set_tag("model_type", "LightGBM")
#             mlflow.set_tag("task", "Sentiment Analysis")
#             mlflow.set_tag("dataset", "YouTube Comments")
#
#         except Exception as e:
#             logger.error(f"Failed to complete model evaluation: {e}")
#             print(f"Error: {e}")
#
# if __name__ == '__main__':
#     main()

import os
import json
import pickle
import logging
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature

# ==============================
# Logging Configuration
# ==============================

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ==============================
# Utility Functions
# ==============================

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


def load_model(model_path: str):
    """Load saved model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug(f"Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to load vectorizer from {vectorizer_path}: {e}")
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters from {params_path}: {e}")
        raise


def evaluate_model(model, x_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(x_test)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        acc = accuracy_score(y_test, y_pred)

        logger.debug("Model evaluation completed successfully.")
        return report, cm, acc, y_pred
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def log_confusion_matrix(cm, labels, dataset_name):
    """Log confusion matrix image to MLflow."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file)
    mlflow.log_artifact(cm_file)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str):
    """Save model run info."""
    try:
        info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(info, file, indent=4)
        logger.debug(f"Model info saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model info: {e}")
        raise


def log_classification_report(report, dataset_name):
    """Log classification report JSON to MLflow."""
    report_file = f'classification_report_{dataset_name}.json'
    with open(report_file, 'w') as file:
        json.dump(report, file, indent=4)
    mlflow.log_artifact(report_file)


# ==============================
# Main Function
# ==============================

def main():
    # Update to correct path handling
    current_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "."))  # Project root

    # MLflow Tracking URI
    mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
    print("Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            # Load parameters
            params_path = os.path.join(root_dir, 'params.yaml')
            params = load_params(params_path)
            for section, param_dict in params.items():
                for key, value in param_dict.items():
                    mlflow.log_param(f"{section}_{key}", value)

            # Load artifacts
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Log model to MLflow with signature
            input_example = pd.DataFrame(
                X_test.toarray()[:5],
                columns=vectorizer.get_feature_names_out()
            )
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log vectorizer
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model
            report, cm, acc, y_pred = evaluate_model(model, X_test, y_test)

            # Log accuracy
            mlflow.log_metric("test_accuracy", acc)

            # Log metrics for each label and averages
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    prefix = f"test_{label}"
                    mlflow.log_metrics({
                        f"{prefix}_precision": metrics.get('precision', 0),
                        f"{prefix}_recall": metrics.get('recall', 0),
                        f"{prefix}_f1_score": metrics.get('f1-score', 0)
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, labels=model.classes_, dataset_name="test")

            # Log classification report
            log_classification_report(report, dataset_name="test")

            # Set tags
            mlflow.set_tags({
                "model_type": "LightGBM",
                "task": "Sentiment Analysis",
                "dataset": "YouTube Comments"
            })

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
