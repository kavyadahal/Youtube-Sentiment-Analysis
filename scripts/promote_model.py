import mlflow

from src.utils.mlflow_config import setup_mlflow


def promote_model(model_name: str = "youtube_sentiment_model") -> None:
    """Promote the latest staging model version to production."""
    setup_mlflow("dvc-pipeline-runs")
    client = mlflow.MlflowClient()

    latest_staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_staging:
        raise ValueError(f"No Staging version found for model '{model_name}'.")

    latest_version = latest_staging[0].version

    for version in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived",
        )

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
    )
    print(f"Model version {latest_version} promoted to Production")


if __name__ == "__main__":
    promote_model()
