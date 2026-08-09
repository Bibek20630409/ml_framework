"""
orchestration/airflow/dags/ml_pipeline.py
──────────────────────────────────────────
Airflow DAG that orchestrates the end-to-end MLOps pipeline:

    dvc_pull → spark_preprocess → validate_data
        → tune_candidate.expand([...])   ← one mapped task per model family
        → collect_winner
        → train → evaluate_gate → promote_model → trigger_deploy

Each task is a thin wrapper around tooling the framework already provides (DVC, the
Spark job, the ``mlf`` CLI, MLflow). Airflow only schedules and sequences them.

**The fan-out is the point of this shape.** ``CANDIDATES`` families are tuned and
profiled *concurrently*, each in its own worker slot, and a single reduce task
applies the constraints and the decision rule to the reports they wrote. The
framework can do the same thing in-process (``mlf select --max-workers N``), and
that is the right tool on one machine; this is the right tool when the candidates
should be spread across a cluster, retried independently, and shown as separate
rows in a UI when one of them fails.

Set ``SELECT_CANDIDATES=""`` to skip the bake-off entirely and train
``TRAIN_CONFIG``'s configured model — the original linear behaviour, which is
still the right default for a pipeline whose model choice is already settled.
Re-running a bake-off nightly for a decision nobody is going to revisit is just
a way to spend GPU hours.

Deploy: copy this file into your Airflow ``dags/`` folder. The project must be
installed (``pip install -e ".[mlops]"``) in the Airflow workers' environment, and
``PROJECT_DIR`` / ``MLFLOW_TRACKING_URI`` set via Airflow Variables or env.

Dynamic task mapping (``.expand``) needs Airflow >= 2.3.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/ml_framework")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "ml-framework")
CONFIG = os.environ.get("TRAIN_CONFIG", "configs/dvc_tabular.yaml")
ACCURACY_GATE = float(os.environ.get("ACCURACY_GATE", "0.6"))

# The families to compare. Empty disables the bake-off and trains the configured
# model directly.
CANDIDATES = [c.strip() for c in os.environ.get("SELECT_CANDIDATES", "").split(",") if c.strip()]
# Where the mapped tasks write their reports and the reduce task reads them.
REPORT_DIR = os.environ.get("SELECT_REPORT_DIR", "outputs/reports")
# Production limits every candidate must meet. Empty means unconstrained; these
# are interpolated into the CLI flags of each mapped task.
MAX_LATENCY_MS = os.environ.get("SELECT_MAX_LATENCY_MS", "")
MAX_MODEL_MB = os.environ.get("SELECT_MAX_MODEL_MB", "")
MIN_EXPLAINABILITY = os.environ.get("SELECT_MIN_EXPLAINABILITY", "")


def _constraint_flags() -> str:
    """The constraint flags, shared by the fan-out and the reduce task.

    Both halves need them: the mapped tasks so the a-priori gate can skip a
    family it already knows cannot qualify, and the reduce task because the
    *measured* constraints are applied there. Building the string once is what
    keeps the two from disagreeing about the budget.
    """
    flags = []
    if MAX_LATENCY_MS:
        flags.append(f"--max-latency-ms {MAX_LATENCY_MS}")
    if MAX_MODEL_MB:
        flags.append(f"--max-model-mb {MAX_MODEL_MB}")
    if MIN_EXPLAINABILITY:
        flags.append(f"--min-explainability {MIN_EXPLAINABILITY}")
    return " ".join(flags)


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


def _winning_model() -> str:
    """The family the reduce task chose, read from ``selection.json``.

    Pushed to XCom so the ``train`` task can pin ``model.name`` to it. Reading
    the file rather than parsing the reduce task's stdout because the file is the
    artifact the framework commits to, and a log format is not a contract.
    """
    path = os.path.join(PROJECT_DIR, "outputs", "selection.json")
    if not os.path.exists(path):
        raise AirflowSkipException("no selection.json — the bake-off did not run")
    with open(path, encoding="utf-8") as f:
        selection = json.load(f)
    winner = selection.get("winner")
    if not winner:
        raise AirflowFailException(f"selection.json names no winner: {selection.get('reason')}")
    print(f"selected {winner}: {selection.get('reason')}")
    for candidate in selection.get("candidates", []):
        profile = candidate.get("profile") or {}
        print(
            f"  {candidate['model']:<16} "
            f"score={profile.get('score')} "
            f"p95={(profile.get('latency') or {}).get('p95_ms')} "
            f"status={'winner' if candidate['model'] == winner else candidate.get('disqualified') or candidate.get('skipped') or 'ok'}"
        )
    return winner


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

    # ── Model selection: fan out, then reduce ─────────────
    # One mapped task per candidate family. Each tunes and profiles its own model
    # and writes a report; none of them decides anything. Independent retries and
    # independent failures are the reason this is N tasks rather than one task
    # with a loop inside it — a family whose extra is missing on one worker
    # should be one red square, not a dead pipeline.
    if CANDIDATES:
        tune_candidate = BashOperator.partial(
            task_id="tune_candidate",
            # A candidate that gets gated out exits 0 with a note; a genuine
            # crash is what should be retried.
            retries=1,
        ).expand(
            bash_command=[
                (
                    f"cd {PROJECT_DIR} && MLFLOW_TRACKING_URI={TRACKING_URI} "
                    f"mlf select --config {CONFIG} "
                    f"--candidate {candidate} --report-dir {REPORT_DIR} "
                    f"{_constraint_flags()}"
                )
                for candidate in CANDIDATES
            ]
        )

        # The reduce step: read every report, apply the measured constraints, run
        # the decision rule, write selection.json.
        collect_winner = BashOperator(
            task_id="collect_winner",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"mlf select --config {CONFIG} --collect {REPORT_DIR} "
                f"{_constraint_flags()}"
            ),
        )

        announce_winner = PythonOperator(task_id="announce_winner", python_callable=_winning_model)

        # `train` pins the winner rather than re-running the bake-off: the
        # comparison already happened, and repeating it here would double the
        # cost of the pipeline to re-derive an answer sitting in a file.
        train = BashOperator(
            task_id="train",
            bash_command=(
                f"cd {PROJECT_DIR} && MLFLOW_TRACKING_URI={TRACKING_URI} "
                f"mlf train --config {CONFIG} "
                "--set model.name="
                "{{ ti.xcom_pull(task_ids='announce_winner') }} "
                f"--set logging.mlflow_tracking_uri={TRACKING_URI}"
            ),
        )
        selection_stage = [tune_candidate, collect_winner, announce_winner]
    else:
        train = BashOperator(
            task_id="train",
            bash_command=(
                f"cd {PROJECT_DIR} && MLFLOW_TRACKING_URI={TRACKING_URI} "
                f"mlf train --config {CONFIG} "
                f"--set logging.mlflow_tracking_uri={TRACKING_URI}"
            ),
        )
        selection_stage = []

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

    chain = [dvc_pull, spark_preprocess, validate_data]
    chain += selection_stage
    chain += [train, evaluate_gate, promote_model, trigger_deploy]
    for upstream, downstream in zip(chain, chain[1:], strict=False):
        upstream >> downstream
