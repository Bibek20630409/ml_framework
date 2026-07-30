"""
orchestration/airflow/dags/ml_pipeline.py
──────────────────────────────────────────
Airflow DAG that orchestrates the end-to-end MLOps pipeline:

    dvc_pull → spark_preprocess → train → evaluate_gate → promote_model → trigger_deploy

Each task is a thin wrapper around tooling the framework already provides (DVC, the
Spark job, the ``mlf`` CLI, MLflow). Airflow only schedules and sequences them.

Deploy: copy this file into your Airflow ``dags/`` folder. The project must be
installed (``pip install -e ".[mlops]"``) in the Airflow workers' environment, and
``PROJECT_DIR`` / ``MLFLOW_TRACKING_URI`` set via Airflow Variables or env.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/ml_framework")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "ml-framework")
CONFIG = os.environ.get("TRAIN_CONFIG", "configs/dvc_tabular.yaml")
ACCURACY_GATE = float(os.environ.get("ACCURACY_GATE", "0.6"))

default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _evaluate_gate() -> None:
    """Fail the pipeline if the fresh model doesn't clear the quality bar."""
    metrics_path = os.path.join(PROJECT_DIR, "outputs", "metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    score = metrics.get("test_acc")
    if score is None:  # regression model → gate on (negative) error instead
        score = -metrics.get("test_rmse", float("inf"))
        threshold = -ACCURACY_GATE
    else:
        threshold = ACCURACY_GATE
    if score < threshold:
        raise AirflowFailException(f"quality gate failed: {metrics} (need ≥ {ACCURACY_GATE})")
    print(f"quality gate passed: {metrics}")


def _promote_model() -> None:
    """Alias the newest registered version as @production (MLflow 3 alias API)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
    print(f"promoted {MODEL_NAME} v{latest.version} → @production")


with DAG(
    dag_id="ml_framework_pipeline",
    description="DVC → Spark → train → gate → promote → deploy",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "ml-framework"],
) as dag:
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {PROJECT_DIR} && dvc pull || true",
    )

    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m ml_framework.pipeline.spark_preprocess "
            "--input data/raw/sample.csv --output data/processed --target-col label"
        ),
    )

    # Data contract gate — fails the pipeline before training on bad data.
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m ml_framework.pipeline.contracts "
            "--input data/processed --target-col label"
        ),
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_DIR} && MLFLOW_TRACKING_URI={TRACKING_URI} "
            f"mlf train --config {CONFIG} "
            f"--set logging.mlflow_tracking_uri={TRACKING_URI}"
        ),
    )

    evaluate_gate = PythonOperator(task_id="evaluate_gate", python_callable=_evaluate_gate)
    promote_model = PythonOperator(task_id="promote_model", python_callable=_promote_model)

    trigger_deploy = BashOperator(
        task_id="trigger_deploy",
        # Roll the serving Deployment so it picks up @production (no-op if kubectl absent).
        bash_command=(
            "kubectl -n ml-framework rollout restart deployment/ml-framework-api || "
            "echo 'kubectl not available — deploy step is a no-op here'"
        ),
    )

    (
        dvc_pull
        >> spark_preprocess
        >> validate_data
        >> train
        >> evaluate_gate
        >> promote_model
        >> trigger_deploy
    )
