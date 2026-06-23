import pickle

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import setup_mlflow


@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("youtube_sentiment_model", "Staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    setup_mlflow("dvc-pipeline-runs")
    client = MlflowClient()

    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        input_text = "This video is very helpful"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        prediction = model.predict(input_df)

        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} processed the dummy input successfully.")
    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
