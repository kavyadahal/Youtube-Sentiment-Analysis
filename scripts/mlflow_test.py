import random

import mlflow

from src.utils.mlflow_config import setup_mlflow

setup_mlflow("dagshub-connection-test")

with mlflow.start_run():
    mlflow.log_param("project", "youtube_sentiment_analysis")
    mlflow.log_param("student", "kavya")
    mlflow.log_metric("test_accuracy", random.uniform(0.80, 0.99))
    mlflow.set_tag("tracking_platform", "DagsHub")

print("Success! Check the Experiments tab in DagsHub.")
