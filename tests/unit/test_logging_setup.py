"""setup_logging tracks handlers per output dir, not with a process-wide flag."""

from __future__ import annotations

import logging

import pytest

from ml_framework.utils.logging import active_handlers, close_logging, setup_logging


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """Run against a root logger holding none of this module's handlers.

    Only *foreign* handlers are restored afterwards (pytest's own, chiefly).
    Re-attaching our handlers would hand the next test a handler that
    ``close_logging`` no longer tracks, which is the very state this fixture
    exists to prevent.
    """
    root = logging.getLogger()
    foreign = [h for h in root.handlers if not getattr(h, "_mlf_owned", False)]
    close_logging()
    yield
    close_logging()
    root.handlers[:] = foreign


@pytest.mark.unit
def test_second_run_logs_into_its_own_file_not_the_first_runs(tmp_path):
    """The v1 bug: a module-global `_FILE_READY` meant the second train() in a
    process appended to the first run's training.log. That becomes load-bearing
    the moment tuning calls train() N times in one process."""
    first, second = tmp_path / "run1", tmp_path / "run2"

    setup_logging(first)
    logging.getLogger("ml_framework.test").info("FIRST-RUN-MESSAGE")
    setup_logging(second)
    logging.getLogger("ml_framework.test").info("SECOND-RUN-MESSAGE")

    first_log = (first / "training.log").read_text(encoding="utf-8")
    second_log = (second / "training.log").read_text(encoding="utf-8")
    assert "FIRST-RUN-MESSAGE" in first_log
    assert "SECOND-RUN-MESSAGE" not in first_log
    assert "SECOND-RUN-MESSAGE" in second_log
    assert "FIRST-RUN-MESSAGE" not in second_log


@pytest.mark.unit
def test_repeated_calls_for_one_directory_do_not_stack_handlers(tmp_path):
    """The CLI calls setup_logging() before train() does; that must stay idempotent."""
    setup_logging(tmp_path)
    setup_logging(tmp_path)
    setup_logging(tmp_path)

    ours = active_handlers()
    files = [h for h in ours if isinstance(h, logging.FileHandler)]
    streams = [h for h in ours if not isinstance(h, logging.FileHandler)]
    assert len(files) == 1
    assert len(streams) == 1

    logging.getLogger("ml_framework.test").info("ONCE")
    assert (tmp_path / "training.log").read_text(encoding="utf-8").count("ONCE") == 1


@pytest.mark.unit
def test_returning_to_an_earlier_directory_reuses_its_handler(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    setup_logging(first)
    setup_logging(second)
    setup_logging(first)
    logging.getLogger("ml_framework.test").info("BACK-IN-A")

    assert "BACK-IN-A" in (first / "training.log").read_text(encoding="utf-8")
    assert "BACK-IN-A" not in (second / "training.log").read_text(encoding="utf-8")


@pytest.mark.unit
def test_only_one_run_file_is_attached_at_a_time(tmp_path):
    setup_logging(tmp_path / "a")
    setup_logging(tmp_path / "b")
    attached = [h for h in active_handlers() if isinstance(h, logging.FileHandler)]
    assert len(attached) == 1


@pytest.mark.unit
def test_setup_without_an_output_dir_configures_only_stdout():
    setup_logging()
    assert not [h for h in active_handlers() if isinstance(h, logging.FileHandler)]


@pytest.mark.unit
def test_close_logging_detaches_everything_it_owns(tmp_path):
    """Without this, Windows refuses to delete a temp dir whose log is still open."""
    setup_logging(tmp_path)
    assert active_handlers()
    close_logging()
    assert active_handlers() == []
