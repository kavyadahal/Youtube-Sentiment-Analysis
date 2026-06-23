import mlflow
import dagshub

REPO_OWNER = "kavyadahal"
REPO_NAME = "Youtube-Sentiment-Analysis"


def setup_mlflow(experiment_name: str = "dvc-pipeline-runs") -> None:
    """Configure MLflow to log runs to Kavya's DagsHub repository."""
    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True,
    )
    mlflow.set_experiment(experiment_name)
