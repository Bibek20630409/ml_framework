"""
orchestration/airflow/dags/ml_pipeline.py
──────────────────────────────────────────
Airflow DAG that orchestrates the end-to-end MLOps pipeline:

    assert_pinned_config → dvc_pull → spark_preprocess → validate_data
        → train → evaluate_gate → promote_model → trigger_deploy

Each task is a thin wrapper around tooling the framework already provides (DVC, the
Spark job, the ``mlf`` CLI, MLflow). Airflow only schedules and sequences them.

**This DAG retrains; it does not choose.** Choosing a model family is a decision,
retraining it on fresh data is a routine, and the two belong on different clocks.
The bake-off runs out of band — on a laptop, in a one-off job, on whatever cadence
a model review actually has — and writes its answer to a config:

    mlf select --config configs/example_selection.yaml \\
        --emit-config configs/winner.yaml      # or: mlf train --select --emit-config ...

``configs/winner.yaml`` is the effective config of the winning candidate: its
``model.name``, its tuned ``model.params``, and — because a candidate config is
built with the bake-off switched off — ``select.enabled: false``. Commit it, point
``TRAIN_CONFIG`` at it, and this DAG trains that model and only that model.

Running the comparison nightly would not just spend GPU hours on a decision nobody
is going to revisit. It would let the winner *flip* on cross-validation noise, so
the family in production changes without anyone choosing it — and the latency
budget, the explainability story and on-call's mental model all change with it.

``assert_pinned_config`` is what makes that a property rather than a convention:
``mlf train`` reads ``select.enabled`` from the YAML, so deleting a selection stage
from this file is not by itself enough to stop a bake-off happening inside the
``train`` task.

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
# The pinned config: the winner emitted by `mlf select --emit-config`, or any
# config whose `model.name` is already settled. Never one with a bake-off in it.
CONFIG = os.environ.get("TRAIN_CONFIG", "configs/dvc_tabular.yaml")
ACCURACY_GATE = float(os.environ.get("ACCURACY_GATE", "0.6"))


def _raw_input() -> str:
    """Where the preprocess stage reads from — owned by ``params.yaml``.

    Read rather than repeated. This DAG used to spell out the input path, the
    output path and the target column in its Bash commands, which meant editing
    ``params.yaml`` changed what ``dvc repro`` did and changed nothing at all
    about what ran on the schedule. The target and output now come from
    ``TRAIN_CONFIG`` (the stages read it themselves via ``--config``); the raw
    input is the one pipeline-owned value left, and it comes from here.
    """
    import yaml

    path = os.path.join(PROJECT_DIR, "params.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        return (params.get("preprocess") or {})["input"]
    except (OSError, KeyError) as exc:
        # At DAG-parse time a raised exception would break the whole file, not
        # just this DAG. Fail inside the task instead, where it is visible.
        return f"__unresolved__:{exc}"


default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _evaluate_gate() -> None:
    """Fail the pipeline if the fresh model doesn't clear the quality bar."""
    # `runtime.output_dir` is the config's to decide; hardcoding "outputs" here
    # meant changing it in the config silently broke the gate with a FileNotFound
    # at the end of a full training run.
    import yaml

    with open(os.path.join(PROJECT_DIR, CONFIG), encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    output_dir = (config.get("runtime") or {}).get("output_dir", "outputs")

    metrics_path = os.path.join(PROJECT_DIR, output_dir, "metrics.json")
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


def _assert_pinned_config() -> None:
    """Refuse to start if ``TRAIN_CONFIG`` would run a bake-off.

    ``mlf train`` reads ``select.enabled`` from the YAML, so a config with the
    comparison switched on compares families *inside* the ``train`` task — daily,
    under one task id, invisible in the graph. Removing a selection stage from
    this DAG does not prevent that; this check does.

    It is deliberately the first task: the config is on disk, the check costs
    milliseconds, and a misconfigured pipeline should fail before it pulls data
    and starts a Spark job rather than after.
    """
    import yaml

    path = os.path.join(PROJECT_DIR, CONFIG)
    if not os.path.exists(path):
        raise AirflowFailException(f"TRAIN_CONFIG not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if (config.get("select") or {}).get("enabled"):
        raise AirflowFailException(
            f"{CONFIG} has select.enabled: true. This DAG trains a pinned winner and "
            "must not run a bake-off. Choose the family out of band with "
            "`mlf select --config <c> --emit-config configs/winner.yaml`, commit that "
            "file, and point TRAIN_CONFIG at it."
        )

    model = (config.get("model") or {}).get("name")
    if not model:
        raise AirflowFailException(f"{CONFIG} names no model.name, so there is nothing to train")
    print(f"training pinned model '{model}' from {CONFIG}")


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
    # The structural guarantee that this DAG never chooses a model. First in the
    # chain, and cheap, so a config that would start a bake-off fails the run
    # before anything expensive happens.
    assert_pinned_config = PythonOperator(
        task_id="assert_pinned_config",
        python_callable=_assert_pinned_config,
        retries=0,  # a bad config is not transient; retrying it just delays the red square
    )

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {PROJECT_DIR} && dvc pull || true",
    )

    # `--config` rather than `--target-col`/`--output`: the stage reads the target
    # column and the processed-data path out of the same config the train task is
    # pinned to, so those values cannot drift between the two tasks or between
    # this DAG and `dvc repro`.
    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m ml_framework.pipeline.spark_preprocess "
            f"--input {_raw_input()} --config {CONFIG}"
        ),
    )

    # Data contract gate — fails the pipeline before training on bad data.
    # Reads its input path and target from the config too, so the gate is
    # structurally guaranteed to validate the column training will read.
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_DIR} && python -m ml_framework.pipeline.contracts --config {CONFIG}"
        ),
    )

    # Trains exactly what the config names. No --select, and no `--set
    # model.name=...` override: the family is whatever was committed to
    # TRAIN_CONFIG, so the file in git is the single answer to "what is in
    # production", and a run cannot disagree with it.
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

    chain = [
        assert_pinned_config,
        dvc_pull,
        spark_preprocess,
        validate_data,
        train,
        evaluate_gate,
        promote_model,
        trigger_deploy,
    ]
    for upstream, downstream in zip(chain, chain[1:], strict=False):
        upstream >> downstream
