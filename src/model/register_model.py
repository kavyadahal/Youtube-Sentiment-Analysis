import json
import logging

import mlflow

from src.utils.mlflow_config import setup_mlflow

# logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s. Run model evaluation first.", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register the logged model in MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.debug(
            "Model %s version %s registered and moved to Staging.",
            model_name,
            model_version.version,
        )
    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise


def main():
    try:
        setup_mlflow("dvc-pipeline-runs")
        model_info = load_model_info("experiment_info.json")
        register_model("youtube_sentiment_model", model_info)
    except Exception as e:
        logger.error("Failed to complete model registration: %s", e)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
