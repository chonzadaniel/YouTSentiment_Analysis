# import os
# import ssl
# import certifi
# import urllib.request
#
# # Fix SSL certificate verification
# # ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
# #
# # # Test the HTTPS connection
# # print(urllib.request.urlopen('https://www.google.com').status)
# # print("Current working directory:", os.getcwd())
#
# import lightgbm
# print(lightgbm.__version__)
# import numpy as np
# import pandas as pd
# import os
# import pickle
# import yaml
# import logging
# import lightgbm as lgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# import mlflow
# import mlflow.sklearn


# # ---------------- Logging Setup ----------------
# logger = logging.getLogger('model_building')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# file_handler = logging.FileHandler('model_building_errors.log')
# file_handler.setLevel('ERROR')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


# # ---------------- Utility Functions ----------------
# def load_params(params_path: str) -> dict:
#     with open(params_path, 'r') as file:
#         params = yaml.safe_load(file)
#     logger.debug('Parameters retrieved from %s', params_path)
#     return params


# def load_data(file_path: str) -> pd.DataFrame:
#     df = pd.read_csv(file_path)
#     df.fillna('', inplace=True)
#     logger.debug('Data loaded and NaNs filled from %s', file_path)
#     return df


# def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
#     vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

#     x_train = train_data['clean_comment'].values
#     y_train = train_data['category'].values

#     x_train_tfidf = vectorizer.fit_transform(x_train)

#     logger.debug(f"TF-IDF transformation complete. Train shape: {x_train_tfidf.shape}")

#     # Save vectorizer
#     with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
#         pickle.dump(vectorizer, f)

#     return x_train_tfidf, y_train


# def train_lgbm(x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
#     model = lgb.LGBMClassifier(
#         objective='multiclass',
#         num_class=3,
#         metric="multi_logloss",
#         is_unbalance=True,
#         class_weight="balanced",
#         reg_alpha=0.1,
#         reg_lambda=0.1,
#         learning_rate=learning_rate,
#         max_depth=max_depth,
#         n_estimators=n_estimators
#     )
#     model.fit(x_train, y_train)
#     logger.debug('LightGBM model training completed')
#     return model


# def save_model(model, file_path: str) -> None:
#     with open(file_path, 'wb') as file:
#         pickle.dump(model, file)
#     logger.debug('Model saved to %s', file_path)


# def get_root_directory() -> str:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     return os.path.abspath(os.path.join(current_dir, '../../'))


# # ---------------- Main with MLflow ----------------
# def main():
#     mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
#     print("Current Tracking URI:", mlflow.get_tracking_uri())

#     mlflow.set_experiment('dvc-pipeline-runs')

#     with mlflow.start_run() as run:
#         print(f"Current MLflow run ID: {run.info.run_id}")
#         try:
#             root_dir = get_root_directory()

#             # Load parameters
#             params = load_params(os.path.join(root_dir, 'params.yaml'))
#             max_features = params['model_building']['max_features']
#             ngram_range = tuple(params['model_building']['ngram_range'])
#             learning_rate = params['model_building']['learning_rate']
#             max_depth = params['model_building']['max_depth']
#             n_estimators = params['model_building']['n_estimators']

#             # Log parameters
#             mlflow.log_param('max_features', max_features)
#             mlflow.log_param('ngram_range', ngram_range)
#             mlflow.log_param('learning_rate', learning_rate)
#             mlflow.log_param('max_depth', max_depth)
#             mlflow.log_param('n_estimators', n_estimators)

#             # Load data
#             train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

#             # TF-IDF
#             x_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

#             # Train model
#             best_model = train_lgbm(x_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

#             # Save model
#             model_path = os.path.join(root_dir, 'lgbm_model.pkl')
#             save_model(best_model, model_path)

#             # Log model to MLflow
#             mlflow.sklearn.log_model(best_model, "lgbm_model")

#             # Log vectorizer
#             mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

#             mlflow.set_tag("model_type", "LightGBM")
#             mlflow.set_tag("task", "Sentiment Analysis")

#             logger.debug('Model training and logging to MLflow completed.')

#         except Exception as e:
#             logger.error('Failed in model building: %s', e)
#             print(f"Error: {e}")


# if __name__ == '__main__':
#     main()
# import mlflow
#
# mlflow.set_tracking_uri("http://ec2-18-208-150-23.compute-1.amazonaws.com:5000/")
#
# mlflow.set_tracking_uri("http://localhost:5000")
#
# with mlflow.start_run() as run:
#     print(f"Current MLflow run ID: {run.info.run_id}")
#     mlflow.log_param("param1", 15)
#     mlflow.log_metric("metric1", 0.90)
#
# mlflow.set_tracking_uri("http://<your-server>:5000")
# print("Tracking URI:", mlflow.get_tracking_uri())
# print("Active Experiment:", mlflow.get_experiment_by_name('dvc-pipeline-runs'))
#
# try:
#     with mlflow.start_run() as run:
#         print("Started MLflow Run:", run.info.run_id)
#         mlflow.log_param("test_param", 123)
#         mlflow.log_metric("test_metric", 0.99)
# except Exception as e:
#     print("MLflow Error:", e)

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-experiment")

with mlflow.start_run() as run:
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    print("Run ID:", run.info.run_id)

print("Tracking URI:", mlflow.get_tracking_uri())

