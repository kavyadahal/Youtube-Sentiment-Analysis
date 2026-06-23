import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import setup_mlflow


@pytest.mark.parametrize("model_name, stage", [
    ("youtube_sentiment_model", "Staging"),
])
def test_load_latest_staging_model(model_name, stage):
    setup_mlflow("dvc-pipeline-runs")
    client = MlflowClient()

    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}'.")
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
