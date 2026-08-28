"""P5: mixed precision, strategy, resume, gradient accumulation, cross-validation.

The three phase gates are here: AMP matches FP32 within tolerance, `--resume`
continues from `last.ckpt`, and cross-validation works for GBDT — the last
because CV is an *orchestration* mode driven by the splitter, not a Lightning
feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_framework.backends.base import resolve_precision
from ml_framework.core.types import Capabilities, UnsupportedCapability

pytest.importorskip("pytorch_lightning", reason="the lightning extra is not installed")

NEURAL = Capabilities(supports_mixed_precision=True)
TREES = Capabilities(supports_mixed_precision=False)


# ── Mixed precision resolution ────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "expected"),
    [("16", "16-mixed"), ("bf16", "bf16-mixed"), ("32", "32-true"), ("64", "64")],
)
def test_shorthand_precisions_are_normalized(requested, expected):
    """`16` is what people type; `16-mixed` is what Lightning wants."""
    assert resolve_precision(requested, NEURAL, has_gpu=True) == expected


@pytest.mark.unit
def test_a_backend_without_mixed_precision_is_downgraded_loudly(caplog):
    """The consumer of `supports_mixed_precision`. Silently training in fp32 after
    being asked for fp16 is discovered months later from a wall-clock number that
    never improved."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert resolve_precision("16-mixed", TREES, has_gpu=True) == "32-true"
    assert any("does not support mixed precision" in r.message for r in caplog.records)


@pytest.mark.unit
def test_fp16_without_a_gpu_is_downgraded_and_says_to_use_bf16(caplog):
    """fp16 autocast on CPU has no gradient scaler behind it."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert resolve_precision("16-mixed", NEURAL, has_gpu=False, accelerator="cpu") == "32-true"
    assert any("bf16-mixed" in r.message for r in caplog.records)


@pytest.mark.unit
def test_bf16_survives_on_cpu():
    """bf16 needs no scaler, which is why it is the one that works on a laptop."""
    assert resolve_precision("bf16-mixed", NEURAL, has_gpu=False, accelerator="cpu") == "bf16-mixed"


@pytest.mark.integration
def test_amp_matches_fp32_within_tolerance(tabular_csv, make_config):
    """**Phase gate.** Reduced precision must change the arithmetic, not the model.

    bf16 rather than fp16 because this runs on CPU, where fp16 has no gradient
    scaler; the resolution logic downgrades that case and is tested above.
    """
    from ml_framework.pipeline import train

    fp32 = make_config(tabular_csv, "multiclass", **{"fit.budget.max_epochs": 6})
    amp = make_config(
        tabular_csv,
        "multiclass",
        **{"fit.budget.max_epochs": 6, "runtime.precision": "bf16-mixed"},
    )
    # Distinct output dirs so neither run reads the other's checkpoints.
    fp32 = fp32.with_overrides({"runtime.output_dir": f"{fp32.runtime.output_dir}_fp32"})
    amp = amp.with_overrides({"runtime.output_dir": f"{amp.runtime.output_dir}_amp"})

    fp32_metrics = train(fp32)
    amp_metrics = train(amp)

    assert set(fp32_metrics) == set(amp_metrics)
    # bf16 has ~3 decimal digits of mantissa, so the tolerance is about the format
    # rather than about the model. A wider gap means AMP changed what was trained.
    assert amp_metrics["test_acc"] == pytest.approx(fp32_metrics["test_acc"], abs=0.15)


@pytest.mark.integration
def test_precision_reaches_the_trainer(tabular_csv, make_config):
    from ml_framework.backends.lightning import LightningBackend
    from ml_framework.core.protocols import RunContext

    cfg = make_config(tabular_csv, "multiclass")
    run = RunContext(output_dir=Path(cfg.runtime.output_dir), precision="bf16-mixed")
    assert LightningBackend()._precision(run) == "bf16-mixed"


# ── Gradient accumulation ─────────────────────────────────
@pytest.mark.integration
def test_gradient_accumulation_trains(tabular_csv, make_config):
    """The effective batch size becomes batch_size * N, which is worth knowing
    when comparing runs — but it must first actually run."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass", **{"fit.params.accumulate_grad_batches": 4})
    assert "test_acc" in train(cfg)


# ── Optimizer / scheduler choice ──────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd"])
def test_every_optimizer_builds(optimizer, tabular_csv, make_config):
    from ml_framework.data import build_model

    cfg = make_config(tabular_csv, "multiclass", **{"fit.params.optimizer": optimizer})
    model = build_model(cfg, input_dim=6, output_dim=3)
    assert model.configure_optimizers()


@pytest.mark.unit
@pytest.mark.parametrize("scheduler", ["plateau", "cosine", "step", "none"])
def test_every_scheduler_builds(scheduler, tabular_csv, make_config):
    from ml_framework.data import build_model

    cfg = make_config(tabular_csv, "multiclass", **{"fit.params.scheduler": scheduler})
    model = build_model(cfg, input_dim=6, output_dim=3)
    configured = model.configure_optimizers()
    if scheduler == "none":
        # No schedule at all returns the bare optimizer, not a dict with None in it.
        assert not isinstance(configured, dict)
    else:
        assert configured["lr_scheduler"] is not None


@pytest.mark.unit
def test_the_v1_defaults_are_unchanged(tabular_csv, make_config):
    """Adam + ReduceLROnPlateau on val/loss, exactly as v1 had it."""
    import torch

    from ml_framework.data import build_model

    model = build_model(make_config(tabular_csv, "multiclass"), input_dim=6, output_dim=3)
    configured = model.configure_optimizers()
    assert isinstance(configured["optimizer"], torch.optim.Adam)
    assert isinstance(
        configured["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert configured["lr_scheduler"]["monitor"] == "val/loss"


@pytest.mark.unit
def test_an_unknown_optimizer_is_refused_by_name(tabular_csv, make_config):
    from ml_framework.data import build_model

    cfg = make_config(tabular_csv, "multiclass", **{"fit.params.optimizer": "lbfgs"})
    with pytest.raises(Exception, match="lbfgs|extra_forbidden|Input should be"):
        build_model(cfg, input_dim=6, output_dim=3).configure_optimizers()


# ── Resume ────────────────────────────────────────────────
@pytest.mark.integration
def test_last_checkpoint_lands_in_the_bundle(tabular_csv, make_config):
    """`--resume` reads `<bundle>/model/last.ckpt`, so it has to be there."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    out = Path(cfg.runtime.output_dir)
    assert (out / "model" / "last.ckpt").exists()
    # The manifest still points at the *best* checkpoint: a loader wants the best
    # weights, only a resuming trainer wants the last state.
    from ml_framework.core.bundle import read_manifest

    assert read_manifest(out).model.artifact.endswith("model.ckpt")


@pytest.mark.integration
def test_resume_continues_rather_than_restarting(tabular_csv, make_config):
    """**Phase gate.** The loop picks up where it stopped.

    A resumed run and a fresh one *end* in the same place, so the final epoch
    number proves nothing. What distinguishes them is work done: resuming into an
    already-exhausted budget must train **zero** further steps, where a fresh run
    would train the lot. That is only true if the epoch counter, the optimizer and
    the scheduler all came back.
    """
    import torch

    from ml_framework.pipeline import train

    def state(path: Path) -> tuple[int, int]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["optimizer_states"], "a resumable checkpoint carries optimizer state"
        return ckpt["epoch"], ckpt["global_step"]

    cfg = make_config(tabular_csv, "multiclass", **{"fit.budget.max_epochs": 2})
    train(cfg)
    last = Path(cfg.runtime.output_dir) / "model" / "last.ckpt"
    first_epoch, first_step = state(last)
    assert first_step > 0

    # Same budget, resuming: the checkpoint is already at max_epochs, so nothing
    # more should run. Re-initialising from these weights would train 2 epochs.
    train(cfg, resume=True)
    assert state(last) == (first_epoch, first_step)

    # A longer budget resumes and advances — from where it was, not from zero.
    train(cfg.with_overrides({"fit.budget.max_epochs": 4}), resume=True)
    resumed_epoch, resumed_step = state(last)
    assert resumed_epoch > first_epoch
    assert resumed_step > first_step


@pytest.mark.integration
def test_resume_from_an_explicit_path(tabular_csv, make_config):
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    train(cfg)
    ckpt = Path(cfg.runtime.output_dir) / "model" / "last.ckpt"
    assert "test_acc" in train(cfg, resume=ckpt)


@pytest.mark.integration
def test_resume_without_a_checkpoint_says_so_before_training(tabular_csv, make_config):
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass")
    with pytest.raises(FileNotFoundError, match="--resume found no checkpoint"):
        train(cfg, resume=True)


@pytest.mark.integration
def test_resume_on_a_backend_that_cannot_warns_and_trains_fresh(tabular_csv, make_config, caplog):
    """The consumer of `Capabilities.supports_resume`: a one-shot fit(X, y) has no
    partial state, and saying so beats ignoring the flag."""
    import logging

    pytest.importorskip("xgboost")
    from ml_framework.pipeline import train

    cfg = make_config(
        tabular_csv,
        "multiclass",
        model="xgboost",
        **{"fit.params.n_estimators": 10, "fit.params.early_stopping_rounds": 0},
    )
    with caplog.at_level(logging.WARNING):
        assert "test_acc" in train(cfg, resume=True)
    assert any("cannot resume" in r.message for r in caplog.records)


# ── Cross-validation ──────────────────────────────────────
@pytest.mark.unit
def test_one_fold_is_not_a_split(tabular_csv, make_config):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="1 fold is not a split"):
        make_config(tabular_csv, "multiclass", **{"data.split.folds": 1})


@pytest.mark.unit
def test_cv_folds_partition_the_test_sets_exactly_once(tabular_csv, make_config):
    """Every row is in exactly one fold's test set — otherwise the estimate double
    counts some rows and never sees others."""
    from ml_framework.data.splitters import CrossValidationSplitter

    y = np.repeat([0, 1, 2], 40)
    folds = CrossValidationSplitter(folds=4, seed=0, task="multiclass").split(len(y), y=y)

    assert len(folds) == 4
    combined = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(combined), np.arange(len(y)))

    for fold in folds:
        # No leakage between the three parts of any single fold.
        assert not set(fold.train) & set(fold.test)
        assert not set(fold.train) & set(fold.val)
        assert not set(fold.val) & set(fold.test)


@pytest.mark.unit
def test_regression_folds_do_not_stratify(regression_csv, make_config):
    """Stratifying a continuous target is what crashed the original framework."""
    from ml_framework.data.splitters import CrossValidationSplitter

    y = np.random.default_rng(0).normal(size=120).astype("float32")
    folds = CrossValidationSplitter(folds=3, seed=0, task="regression").split(len(y), y=y)
    assert len(folds) == 3


@pytest.mark.unit
def test_too_few_rows_for_the_fold_count_is_refused(tabular_csv):
    from ml_framework.data.splitters import CrossValidationSplitter, SplitError

    with pytest.raises(SplitError, match="cannot make 10 folds from 4 rows"):
        CrossValidationSplitter(folds=10, task="regression").split(4, y=np.zeros(4))


@pytest.mark.integration
def test_each_fold_gets_its_own_fitted_preprocessor(tabular_csv, make_config):
    """A scaler fitted once on the full data and reused across folds leaks every
    fold's test set into every other fold's preprocessing."""
    from ml_framework.data.builders import build_cv_bundles

    cfg = make_config(tabular_csv, "multiclass", **{"data.split.folds": 3})
    scalers = [b.preprocessor.scaler.mean_ for b in build_cv_bundles(cfg)]
    assert len(scalers) == 3
    # Different training rows → different fitted means. Identical means would mean
    # the preprocessor was fitted once and shared.
    assert not np.allclose(scalers[0], scalers[1])


@pytest.mark.integration
def test_cross_validation_reports_mean_and_spread(tabular_csv, make_config):
    """The spread is the point: 0.84/0.86 and 0.70/1.00 have the same mean and are
    completely different results."""
    from ml_framework.pipeline import train

    cfg = make_config(tabular_csv, "multiclass", **{"data.split.folds": 3})
    metrics = train(cfg)

    assert "cv_acc_mean" in metrics and "cv_acc_std" in metrics
    # The single-holdout score is still reported beside it: they answer different
    # questions, and CV does not replace the shipped model's own evaluation.
    assert "test_acc" in metrics

    cv = json.loads((Path(cfg.runtime.output_dir) / "cv.json").read_text(encoding="utf-8"))
    assert cv["folds"] == 3
    assert len(cv["per_fold"]) == 3
    assert cv["aggregate"]["cv_acc_mean"] == pytest.approx(metrics["cv_acc_mean"])


@pytest.mark.integration
def test_cross_validation_works_for_gbdt_too(tabular_csv, make_config):
    """**Phase gate.** CV is an orchestration mode driven by the splitter, so it
    reaches every backend rather than being a Lightning feature."""
    pytest.importorskip("xgboost")
    from ml_framework.pipeline import train

    cfg = make_config(
        tabular_csv,
        "multiclass",
        model="xgboost",
        **{
            "data.split.folds": 3,
            "fit.params.n_estimators": 15,
            "fit.params.early_stopping_rounds": 0,
        },
    )
    metrics = train(cfg)
    assert "cv_acc_mean" in metrics
    assert (Path(cfg.runtime.output_dir) / "cv.json").exists()
    # And the bundle is still a normal GBDT bundle.
    assert (Path(cfg.runtime.output_dir) / "model" / "model.json").exists()


@pytest.mark.integration
def test_cross_validation_refuses_a_kind_it_cannot_partition(tabular_csv, make_config):
    """Better than silently cross-validating something else.

    Every built-in kind folds now — image over the training directory since P6,
    text since P7, audio and video off the shard index since P13 — so what is
    pinned here is the refusal *mechanism* rather than a gap in coverage. It is the
    contract a third-party source meets: a source whose ``build`` does not accept
    an injected partition is named and refused, not handed one and expected to cope.

    The placeholder kind is deliberately fictional. It used to be ``"audio"``,
    which stopped testing anything the moment audio became foldable.
    """
    from ml_framework.data.builders import build_cv_bundles

    cfg = make_config(tabular_csv, "multiclass", **{"data.split.folds": 3})
    # `model_copy` rather than validation: the point is a kind the CV table does
    # not know, which a registered third-party source is free to introduce.
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"kind": "hologram"})})
    with pytest.raises(NotImplementedError, match="cross-validation is implemented for"):
        list(build_cv_bundles(cfg))


# ── LR finder capability gate ─────────────────────────────
@pytest.mark.unit
def test_lr_finder_refuses_a_model_with_no_learning_rate(tabular_csv, make_config):
    """`mlf lr` on a GBDT used to fail somewhere deep inside torch_lr_finder."""
    pytest.importorskip("xgboost")
    from ml_framework.pipeline.lr_finder import _check_supported

    cfg = make_config(tabular_csv, "multiclass", model="xgboost")
    with pytest.raises(UnsupportedCapability, match="no learning rate to range-test"):
        _check_supported(cfg)


@pytest.mark.unit
def test_lr_finder_accepts_a_neural_model(tabular_csv, make_config):
    from ml_framework.pipeline.lr_finder import _check_supported

    _check_supported(make_config(tabular_csv, "multiclass"))  # must not raise
