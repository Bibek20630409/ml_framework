"""
tracking/run_logger.py
──────────────────────
Backend-neutral experiment tracking.

``pipeline/train.py::_build_logger`` returns a *Lightning logger object*, and
``mlflow_utils.log_and_register`` reads ``logger.run_id`` off it. A GBDT or Prophet
run has no Lightning logger, so it cannot consume that — which is why tracking
needs its own protocol, sitting beside the Lightning logger rather than inside it.

Four implementations: ``null``, ``csv``, ``mlflow``, ``wandb``. Every optional
import happens inside the implementation that needs it, so importing this module
on a bare install is free.

Nothing here imports ``config``: a run logger takes plain arguments. That keeps
the dependency arrow pointing one way (pipeline → tracking) and makes the loggers
trivially testable.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

Backend = str  # "none" | "csv" | "mlflow" | "wandb"


@runtime_checkable
class RunLogger(Protocol):
    """What the orchestrator needs from a tracker, and nothing more.

    ``log_artifacts`` takes a **directory** because the unit we track is the whole
    self-contained bundle, not a hardcoded list of filenames.
    """

    backend: Backend

    @property
    def run_id(self) -> str | None: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None: ...

    def finish(self, status: str = "FINISHED") -> None: ...


class _BaseRunLogger:
    """Shared plumbing: context-manager support and metric coercion."""

    backend: Backend = "none"

    @property
    def run_id(self) -> str | None:
        return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        return None

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        return None

    def finish(self, status: str = "FINISHED") -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("FAILED" if exc_type else "FINISHED")

    @staticmethod
    def _numeric(metrics: Mapping[str, Any]) -> dict[str, float]:
        """Drop non-numeric entries instead of raising.

        A tracker must never be the reason a finished training run fails.
        """
        out: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                log.debug("run_logger: skipping non-numeric metric %s=%r", key, value)
        return out


class NullRunLogger(_BaseRunLogger):
    """Tracking disabled. Exists so callers never branch on ``logger is None``."""

    backend = "none"


class CSVRunLogger(_BaseRunLogger):
    """Local, dependency-free tracking: ``params.json`` + ``metrics.csv``.

    The default, because it works everywhere and produces files a human can read
    without starting a server.
    """

    backend = "csv"

    def __init__(self, output_dir: str | Path, *, name: str = "metrics") -> None:
        self.dir = Path(output_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.dir / "metrics.csv"
        self._params_path = self.dir / "params.json"
        self._columns: list[str] = []
        self._rows: list[dict[str, Any]] = []

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._params_path.write_text(
            json.dumps(dict(params), indent=2, default=str), encoding="utf-8"
        )

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        row: dict[str, Any] = {"step": step if step is not None else len(self._rows)}
        row.update(self._numeric(metrics))
        self._rows.append(row)
        for key in row:
            if key not in self._columns:
                self._columns.append(key)
        self._flush()

    def _flush(self) -> None:
        with self._metrics_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self._rows)

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        # Artifacts are already on local disk in the bundle dir; copying them next
        # to themselves would only waste space.
        return None


class MLflowRunLogger(_BaseRunLogger):
    """MLflow tracking via the low-level client.

    Can **attach to an existing run** (``run_id=…``), which is how a Lightning run
    that already created an MLflow run through ``MLFlowLogger`` gets bundle
    artifacts and final metrics logged to that same run rather than a second one.
    """

    backend = "mlflow"

    def __init__(
        self,
        *,
        experiment: str = "ml-framework",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        artifact_location: str | None = None,
        run_id: str | None = None,
    ) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        # The MLflow file store is deprecated in 3.x and refuses to run by default;
        # SQLite is the lightweight local backend that still supports the registry.
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = MlflowClient(tracking_uri=self.tracking_uri)

        if run_id is None:
            exp = self._client.get_experiment_by_name(experiment)
            exp_id = (
                exp.experiment_id
                if exp is not None
                else self._client.create_experiment(experiment, artifact_location=artifact_location)
            )
            run = self._client.create_run(
                exp_id, run_name=run_name, tags={"mlflow.runName": run_name} if run_name else None
            )
            run_id = run.info.run_id
            self._owns_run = True
        else:
            self._owns_run = False
        self._run_id = run_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def client(self) -> Any:
        """The underlying ``MlflowClient``, for registry operations."""
        return self._client

    def log_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            try:
                self._client.log_param(self._run_id, key, value)
            except Exception as exc:  # noqa: BLE001 - a param must not fail a run
                log.debug("mlflow: could not log param %s: %s", key, exc)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        for key, value in self._numeric(metrics).items():
            self._client.log_metric(self._run_id, key, value, step=step or 0)

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        p = Path(path)
        if p.is_dir():
            self._client.log_artifacts(self._run_id, str(p), artifact_path)
        else:
            self._client.log_artifact(self._run_id, str(p), artifact_path)

    def finish(self, status: str = "FINISHED") -> None:
        if self._owns_run:
            self._client.set_terminated(self._run_id, status)


class WandbRunLogger(_BaseRunLogger):
    """Weights & Biases tracking."""

    backend = "wandb"

    def __init__(
        self,
        *,
        project: str = "ml-framework",
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=dict(config or {}))

    @property
    def run_id(self) -> str | None:
        return getattr(self._run, "id", None)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._run.log(self._numeric(metrics), step=step)

    def log_artifacts(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        artifact = self._wandb.Artifact(artifact_path or "bundle", type="model")
        p = Path(path)
        if p.is_dir():
            artifact.add_dir(str(p))
        else:
            artifact.add_file(str(p))
        self._run.log_artifact(artifact)

    def finish(self, status: str = "FINISHED") -> None:
        self._run.finish(exit_code=0 if status == "FINISHED" else 1)


def build_run_logger(
    backend: Backend,
    *,
    output_dir: str | Path,
    experiment: str = "ml-framework",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    artifact_location: str | None = None,
    project: str = "ml-framework",
    run_id: str | None = None,
) -> RunLogger:
    """Build the configured run logger, falling back to ``null`` if unavailable.

    The fallback is **logged at WARNING** rather than raising: losing tracking
    should not destroy a finished model, but it must not be invisible either.
    """
    if not backend or backend == "none":
        return NullRunLogger()
    try:
        if backend == "csv":
            return CSVRunLogger(output_dir)
        if backend == "mlflow":
            return MLflowRunLogger(
                experiment=experiment,
                tracking_uri=tracking_uri,
                run_name=run_name,
                artifact_location=artifact_location,
                run_id=run_id,
            )
        if backend == "wandb":
            return WandbRunLogger(project=project, run_name=run_name)
    except ImportError as exc:
        log.warning("tracking backend '%s' unavailable (%s) — continuing untracked", backend, exc)
        return NullRunLogger()
    raise ValueError(f"Unknown tracking backend '{backend}' (none|csv|mlflow|wandb)")
