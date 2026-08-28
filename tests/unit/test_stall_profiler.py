"""Measuring how much of a run was spent waiting for data.

Every optimization in P13 — the block shuffle, prefetch depth, pinned memory, a
device decoder — answers a question nobody asks until they know the loader is the
bottleneck. Without this number the usual outcome is a week spent making the model
faster while the GPU idles.

The tests use an injected sleep rather than a real loader, because the property
worth pinning is that the *accounting* is right: time spent between batches lands
in ``data_wait``, time spent inside a batch lands in ``compute``, and the warmup is
genuinely excluded.

The other half is about honesty. A measurement that did not happen must never read
as a measurement of zero — a CPU box has no device to stall, and reporting 0%
there is indistinguishable from a perfectly fed GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ml_framework.backends.staged import StagedDataCallback
from ml_framework.core.stall import (
    FAULTS_FILE,
    STALL_FILE,
    STALL_WARMUP_BATCHES,
    StallProbe,
    StallProfile,
    write_data_reports,
)

pytestmark = pytest.mark.unit


def run_probe(
    probe: StallProbe, *, batches: int, wait_s: float = 0.0, compute_s: float = 0.0
) -> StallProfile:
    """Drive a probe through a loop with known waiting and computing intervals."""
    for _ in range(batches):
        if wait_s:
            time.sleep(wait_s)  # stands in for the loader not having a batch ready
        probe.batch_start()
        if compute_s:
            time.sleep(compute_s)  # stands in for the forward/backward pass
        probe.batch_end()
    return probe.profile()


# ── The accounting ────────────────────────────────────────────────────
def test_a_starved_loop_reports_most_of_its_wall_time_as_waiting():
    """The signal the whole module exists to produce."""
    profile = run_probe(StallProbe(warmup=0, sync=False), batches=8, wait_s=0.02, compute_s=0.002)

    assert profile.measured
    assert profile.n_batches == 8
    assert profile.data_wait_pct >= 45.0, f"got {profile.data_wait_pct:.1f}%"


def test_a_well_fed_loop_reports_almost_no_waiting():
    profile = run_probe(StallProbe(warmup=0, sync=False), batches=8, compute_s=0.02)

    assert profile.measured
    assert profile.data_wait_pct <= 20.0, f"got {profile.data_wait_pct:.1f}%"


def test_compute_and_wait_together_account_for_the_wall_clock():
    """Neither interval may be double-counted or dropped."""
    profile = run_probe(StallProbe(warmup=0, sync=False), batches=6, wait_s=0.01, compute_s=0.01)
    accounted = profile.compute_ms + profile.data_wait_ms

    assert accounted == pytest.approx(profile.wall_ms, rel=0.25)


# ── Warmup ────────────────────────────────────────────────────────────
def test_warmup_batches_are_excluded_from_the_count():
    """Prefetch fill, worker spin-up and cuDNN autotuning are one-time costs that
    would otherwise be attributed to the loader forever."""
    profile = run_probe(StallProbe(warmup=3, sync=False), batches=10, compute_s=0.001)
    assert profile.n_batches == 7


def test_the_wall_window_starts_at_the_first_measured_batch():
    """Otherwise a slow warmup would inflate wall time and deflate every
    percentage computed against it."""
    probe = StallProbe(warmup=2, sync=False)
    # A deliberately slow warmup, then a fast measured stretch.
    run_probe(probe, batches=2, wait_s=0.05)
    profile = run_probe(probe, batches=5, compute_s=0.002)

    assert profile.n_batches == 5
    assert profile.wall_ms < 100.0, "the 100ms warmup must be outside the window"


def test_a_run_shorter_than_the_warmup_reports_unmeasured_with_a_reason():
    """Not zero. A five-batch epoch genuinely cannot be measured with a ten-batch
    warmup, and saying so beats reporting 0% stall."""
    profile = run_probe(StallProbe(warmup=10, sync=False), batches=5, compute_s=0.001)

    assert not profile.measured
    assert "warmup" in profile.error
    assert profile.data_wait_pct == 0.0


def test_the_default_warmup_is_documented_and_nonzero():
    assert STALL_WARMUP_BATCHES == 10


# ── Honesty about what was not measured ───────────────────────────────
def test_gpu_stall_is_none_rather_than_zero_without_a_device():
    """The single most misleading number this module could produce: 0% stall on a
    box with no GPU is indistinguishable from a perfectly fed one."""
    profile = run_probe(StallProbe(warmup=0, sync=False), batches=4, compute_s=0.002)

    assert profile.device_busy_ms is None
    assert profile.gpu_stall_pct is None
    assert profile.gpu_stall_pct is not 0.0  # noqa: F632 - the point is identity


def test_an_unmeasured_profile_reports_zero_percentages_but_says_measured_is_false():
    profile = StallProfile()

    assert not profile.measured
    assert profile.data_wait_pct == 0.0
    assert profile.gpu_stall_pct is None
    assert "not measured" in profile.summary()


def test_the_dict_carries_the_derived_percentages_and_the_measured_flag():
    """`stall.json` and the RunLogger must read the same numbers the summary does."""
    payload = run_probe(StallProbe(warmup=0, sync=False), batches=4, wait_s=0.01).to_dict()

    assert payload["measured"] is True
    assert payload["gpu_stall_pct"] is None
    assert 0.0 <= payload["data_wait_pct"] <= 100.0


def test_the_summary_names_the_missing_device_rather_than_omitting_the_line():
    profile = run_probe(StallProbe(warmup=0, sync=False), batches=4, compute_s=0.002)
    assert "not measured (no CUDA device)" in profile.summary()


def test_a_probe_resets_cleanly_between_epochs():
    probe = StallProbe(warmup=0, sync=False)
    run_probe(probe, batches=5, compute_s=0.002)
    probe.reset()
    profile = run_probe(probe, batches=3, compute_s=0.002)

    assert profile.n_batches == 3


# ── The reports ───────────────────────────────────────────────────────
def test_both_reports_are_always_written_even_when_nothing_happened(tmp_path: Path):
    """An absent file would be indistinguishable from a bundle produced before
    this phase existed, and "no faults" is a different claim from "nobody looked"."""
    write_data_reports(tmp_path)

    stall = json.loads((tmp_path / STALL_FILE).read_text())
    faults = json.loads((tmp_path / FAULTS_FILE).read_text())

    assert stall["n_batches"] == 0
    assert stall["measured"] is False
    assert stall["gpu_stall_pct"] is None
    assert faults["faults"] == 0


def test_only_if_absent_never_clobbers_a_real_measurement(tmp_path: Path):
    """The pipeline fills in defaults for a backend with no epoch loop; it must
    not overwrite what the callback already measured."""
    measured = StallProfile(n_batches=42, wall_ms=100.0, data_wait_ms=50.0)
    write_data_reports(tmp_path, profile=measured, faults=3)
    write_data_reports(tmp_path, only_if_absent=True)

    stall = json.loads((tmp_path / STALL_FILE).read_text())
    assert stall["n_batches"] == 42
    assert json.loads((tmp_path / FAULTS_FILE).read_text())["faults"] == 3


# ── The callback ──────────────────────────────────────────────────────
class _Trainer:
    """The two attributes the callback reads. Not a Lightning Trainer."""

    def __init__(self, epoch: int = 0, sampler=None, dataset=None) -> None:
        self.current_epoch = epoch
        self.datamodule = None
        self.train_dataloader = _Loader(sampler, dataset)


class _Loader:
    def __init__(self, sampler=None, dataset=None) -> None:
        self.sampler = sampler
        self.dataset = dataset


class _Sampler:
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _Recorder:
    def __init__(self) -> None:
        self.metrics: dict = {}

    def log_metrics(self, metrics, step=None) -> None:
        self.metrics.update(metrics)


def test_the_callback_drives_set_epoch_on_the_sampler(tmp_path: Path):
    """Lightning's `DistributedSamplerWrapper` calls `set_epoch` on ITSELF, not on
    the sampler it wraps — so a wrapped sampler would serve the same order every
    epoch and the shuffle would be decorative."""
    sampler = _Sampler()
    trainer = _Trainer(epoch=3, sampler=sampler)
    callback = StagedDataCallback(output_dir=tmp_path)

    callback.on_train_epoch_start(trainer, None)

    assert sampler.epoch == 3


def test_the_callback_prefers_the_datamodule_which_owns_the_staged_sampler(tmp_path: Path):
    class _DataModule:
        def __init__(self):
            self.epoch = -1

        def set_epoch(self, epoch):
            self.epoch = epoch

    sampler = _Sampler()
    trainer = _Trainer(epoch=5, sampler=sampler)
    trainer.datamodule = _DataModule()
    StagedDataCallback(output_dir=tmp_path).on_train_epoch_start(trainer, None)

    assert trainer.datamodule.epoch == 5
    assert sampler.epoch == -1, "the datamodule owns it; do not also poke the sampler"


def test_a_loop_with_no_sampler_still_trains(tmp_path: Path):
    """Best-effort by construction: a run whose sampler cannot be reached does not
    reshuffle, and must not crash for it."""
    StagedDataCallback(output_dir=tmp_path).on_train_epoch_start(_Trainer(), None)


def test_the_callback_reports_faults_to_the_run_logger(tmp_path: Path):
    from ml_framework.data.streaming.integrity import FaultLog, SampleFault

    log = FaultLog(tmp_path / "faults")
    for i in range(3):
        log.record(
            SampleFault(
                index=i,
                shard="s",
                key=f"{i}",
                stage="decode",
                kind="decode_error",
                detail="synthetic",
                decoder="audio.pcm",
                substituted_with=i + 1,
            )
        )
    log.close()

    recorder = _Recorder()
    callback = StagedDataCallback(output_dir=tmp_path, run_logger=recorder)
    callback.on_train_epoch_end(_Trainer(dataset=range(30)), None)

    assert callback.faults == 3
    assert recorder.metrics["data/faults"] == 3.0
    assert recorder.metrics["data/fault_rate"] == pytest.approx(0.1)


def test_a_failing_run_logger_never_breaks_the_epoch(tmp_path: Path):
    """A tracker must never be the reason a finished training run fails — the same
    rule `RunLogger._numeric` encodes one level down."""

    class _Broken:
        def log_metrics(self, metrics, step=None):
            raise RuntimeError("tracker is down")

    callback = StagedDataCallback(output_dir=tmp_path, run_logger=_Broken())
    callback.probe = StallProbe(warmup=0, sync=False)
    run_probe(callback.probe, batches=3, compute_s=0.002)

    callback.on_train_epoch_end(_Trainer(), None)  # must not raise


def test_a_healthy_epoch_reports_no_fault_metrics_at_all(tmp_path: Path):
    """Absence is the signal; logging `faults: 0` every epoch would bury it."""
    recorder = _Recorder()
    callback = StagedDataCallback(output_dir=tmp_path, run_logger=recorder)
    callback.probe = StallProbe(warmup=0, sync=False)
    run_probe(callback.probe, batches=3, wait_s=0.005)

    callback.on_train_epoch_end(_Trainer(), None)

    assert "data/faults" not in recorder.metrics
    assert "stall/data_wait_pct" in recorder.metrics
