import mlflow
import mlflow.sklearn
import subprocess
import hashlib


def setup_mlflow():

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Late Delivery Prediction")


def get_git_commit():

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode("utf-8").strip()


def get_dataset_version(dataset_path):

    with open(dataset_path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def log_run_info(dataset_path):

    git_hash = get_git_commit()
    dataset_hash = get_dataset_version(dataset_path)

    mlflow.set_tag("git_commit", git_hash)
    mlflow.set_tag("dataset_version", dataset_hash)


def log_model_artifact(model, model_name="model"):

    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name
    )