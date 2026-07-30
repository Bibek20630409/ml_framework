"""tracking/run_logger.py — backend-neutral experiment tracking."""

from __future__ import annotations

import csv
import json

import pytest

from ml_framework.tracking.run_logger import (
    CSVRunLogger,
    NullRunLogger,
    RunLogger,
    build_run_logger,
)


@pytest.mark.unit
def test_null_and_csv_loggers_satisfy_the_protocol(tmp_path):
    """`RunLogger` is runtime_checkable so the orchestrator can assert its input."""
    assert isinstance(NullRunLogger(), RunLogger)
    assert isinstance(CSVRunLogger(tmp_path), RunLogger)


@pytest.mark.unit
def test_null_logger_accepts_everything_and_does_nothing(tmp_path):
    """Callers must never have to branch on `logger is None`."""
    logger = NullRunLogger()
    logger.log_params({"lr": 1e-3})
    logger.log_metrics({"val/loss": 0.5}, step=1)
    logger.log_artifacts(tmp_path)
    logger.finish()
    assert logger.run_id is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_csv_logger_writes_params_json_and_metrics_csv(tmp_path):
    # Arrange
    logger = CSVRunLogger(tmp_path)

    # Act
    logger.log_params({"lr": 1e-3, "model": "mlp"})
    logger.log_metrics({"val/loss": 0.5}, step=0)
    logger.log_metrics({"val/loss": 0.4, "val/acc": 0.8}, step=1)
    logger.finish()

    # Assert
    params = json.loads((tmp_path / "metrics" / "params.json").read_text(encoding="utf-8"))
    assert params == {"lr": 1e-3, "model": "mlp"}
    with (tmp_path / "metrics" / "metrics.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["step"] for r in rows] == ["0", "1"]
    assert rows[1]["val/acc"] == "0.8"
    assert rows[0]["val/acc"] == ""  # column added later, earlier row stays blank


@pytest.mark.unit
def test_non_numeric_metrics_are_dropped_not_raised(tmp_path):
    """A tracker must never be the reason a finished training run fails."""
    logger = CSVRunLogger(tmp_path)
    logger.log_metrics({"val/loss": 0.5, "note": "best so far"})
    with (tmp_path / "metrics" / "metrics.csv").open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["val/loss"] == "0.5"
    assert "note" not in row


@pytest.mark.unit
def test_csv_logger_params_survive_unserializable_values(tmp_path):
    logger = CSVRunLogger(tmp_path)
    logger.log_params({"path": tmp_path, "fn": len})
    params = json.loads((tmp_path / "metrics" / "params.json").read_text(encoding="utf-8"))
    assert set(params) == {"path", "fn"}


@pytest.mark.unit
def test_logger_is_a_context_manager_that_records_failure(tmp_path):
    class Recorder(NullRunLogger):
        status: str | None = None

        def finish(self, status: str = "FINISHED") -> None:
            self.status = status

    logger = Recorder()
    with pytest.raises(RuntimeError):
        with logger:
            raise RuntimeError("boom")
    assert logger.status == "FAILED"


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["none", ""])
def test_disabled_backends_build_the_null_logger(backend, tmp_path):
    assert isinstance(build_run_logger(backend, output_dir=tmp_path), NullRunLogger)


@pytest.mark.unit
def test_csv_is_buildable_without_any_optional_dependency(tmp_path):
    logger = build_run_logger("csv", output_dir=tmp_path)
    assert logger.backend == "csv"


@pytest.mark.unit
def test_an_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown tracking backend"):
        build_run_logger("tensorboard", output_dir=tmp_path)


@pytest.mark.unit
def test_a_missing_tracking_library_degrades_to_null_with_a_warning(tmp_path, monkeypatch, caplog):
    """Losing tracking must not destroy a finished model — but must not be silent."""
    # Arrange
    import builtins

    real_import = builtins.__import__

    def _no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_wandb)

    # Act
    with caplog.at_level("WARNING"):
        logger = build_run_logger("wandb", output_dir=tmp_path)

    # Assert
    assert isinstance(logger, NullRunLogger)
    assert "unavailable" in caplog.text
