# # register model
#
# import json
# import mlflow
# import logging
# import os
#
# # Set up Mlflow tracking URI
# mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
#
# # logging configuration
# logger = logging.getLogger('model_registration')
# logger.setLevel('DEBUG')
#
# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')
#
# file_handler = logging.FileHandler('model_registration_errors.log')
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
# def load_model_info(file_path: str) -> dict:
#     """Load the model info from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             model_info = json.load(file)
#         logger.debug('Model info loaded from %s', file_path)
#         return model_info
#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the model info: %s', e)
#         raise
#
#
# def register_model(model_name: str, model_info: dict):
#     """Register the model to the Mlflow Model Registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
#
#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)
#
#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )
#
#         logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
#     except Exception as e:
#         logger.error('Error during model registration: %s', e)
#         raise
#
#
# def main():
#     # Update to correct path handling
#     current_dir = os.path.abspath(os.path.dirname(__file__))
#     root_dir = os.path.abspath(os.path.join(current_dir, "."))  # Project root
#     try:
#         # Load parameters
#         model_info_path = os.path.join(root_dir, 'experiment_info.json')
#         model_info = load_model_info(model_info_path)
#
#         model_name = "yt_chrome_plugin_model"
#         register_model(model_name, model_info)
#     except Exception as e:
#         logger.error('Failed to complete the model registration process: %s', e)
#         print(f"Error: {e}")
#
#
# if __name__ == '__main__':
#     main()

import os
import json
import mlflow
import logging


# ==============================
# MLflow Tracking URI Setup
# ==============================
mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
print("Tracking URI:", mlflow.get_tracking_uri())

# ==============================
# Logging Configuration
# ==============================
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ==============================
# Utility Functions
# ==============================

def load_model_info(file_path: str) -> dict:
    """Load the model info JSON (contains run_id and model_path)."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f'Model info loaded successfully from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Error while loading model info from {file_path}: {e}')
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to MLflow Model Registry and transition to 'Staging'."""
    try:
        # Extract run_id and model artifact path from the JSON info
        run_id = model_info['run_id']
        artifact_path = model_info['model_path'].split('/')[-1]  # Get last component of the path

        model_uri = f"runs:/{run_id}/{artifact_path}"

        print(f"Registering model from URI: {model_uri}")
        logger.debug(f"Model URI for registration: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        logger.info(f'Model {model_name} registered as version {model_version.version}')

        # Transition model to 'Staging'
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f'Model {model_name} version {model_version.version} transitioned to Staging')
        print(f"Model {model_name} version {model_version.version} registered and moved to Staging")

    except Exception as e:
        logger.error(f'Error during model registration: {e}')
        raise


# ==============================
# Main Function
# ==============================
def main():
    try:
        # Get the absolute path to the current script directory
        current_dir = os.path.abspath(os.path.dirname(__file__))

        # Project root is two levels up: from 'src/model' -> root
        root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

        # Path to experiment_info.json in the project root
        model_info_path = os.path.join(root_dir, 'experiment_info.json')

        # Load model info JSON
        model_info = load_model_info(model_info_path)

        # Model name for MLflow Registry
        model_name = "yt_chrome_plugin_model"

        # Register the model
        register_model(model_name, model_info)

    except Exception as e:
        logger.error(f'Failed to complete model registration: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

