"""Zero-config: `mlf train --data x.csv` with no YAML, and the baseline it reports.

The P8 exit gate is the first test here. The rest defend the three properties that
make zero-config trustworthy rather than merely convenient:

* **The precedence chain is ordinary.** Synthesis produces a plain dict that a
  YAML is merged over. Nothing downstream can tell which layer a value came from,
  which is what lets `--data` and `--config` compose instead of being alternatives.
* **An uninstalled family is a refusal, not a substitution.** A score from an MLP
  is not a score from a gradient-booster, and zero-config is exactly the situation
  where the user cannot tell the difference from the output.
* **The trivial baseline is always reported.** `test_acc: 0.91` on a dataset that
  is 91% one class is the most common way a pipeline looks successful while having
  learned nothing, and the user who did not choose the model has nothing else to
  judge it against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_framework import cli
from ml_framework.config.autoconfig import merge, render_config, synthesize


@pytest.fixture
def learnable_csv(tmp_path: Path) -> Path:
    """A separable multiclass problem, so the model can actually beat the baseline."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5)).astype("float32")
    y = (x @ rng.normal(size=(5, 3))).argmax(axis=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    frame["label"] = y
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


# ── The exit gate ─────────────────────────────────────────
@pytest.mark.integration
def test_train_with_only_a_data_path_produces_a_bundle_and_a_baseline(learnable_csv, tmp_path):
    """P8's stated gate: no YAML, no flags beyond the file, a bundle and a
    comparison against doing nothing."""
    out = tmp_path / "outputs"
    assert (
        cli.main(["train", "--data", str(learnable_csv), "--output-dir", str(out), "--no-tune"])
        == 0
    )

    assert (out / "manifest.json").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "test_acc" in metrics
    assert "baseline_acc" in metrics
    # Three roughly balanced classes: a constant prediction lands near a third.
    assert metrics["baseline_acc"] < 0.5
    assert metrics["test_acc"] > metrics["baseline_acc"]


@pytest.mark.integration
def test_the_bundle_records_the_model_the_framework_chose(learnable_csv, tmp_path):
    out = tmp_path / "outputs"
    cli.main(["train", "--data", str(learnable_csv), "--output-dir", str(out), "--no-tune"])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["name"] == "xgboost"
    assert manifest["task"] == "multiclass"


# ── The precedence chain ──────────────────────────────────
@pytest.mark.unit
def test_a_yaml_overrides_exactly_the_fields_it_sets(learnable_csv, tmp_path):
    """A shallow merge would make a YAML that sets only `model.name` discard the
    synthesized `data.kind` and `data.path` alongside it — turning "override one
    field" into "replace the whole block"."""
    overlay = tmp_path / "over.yaml"
    overlay.write_text("model:\n  name: lightgbm\nfit:\n  batch_size: 64\n", encoding="utf-8")

    args = cli.build_parser().parse_args(
        ["train", "--data", str(learnable_csv), "--config", str(overlay)]
    )
    cfg = cli._load_config(args)

    assert cfg.model.name == "lightgbm"  # from the YAML
    assert cfg.fit.batch_size == 64  # from the YAML
    assert cfg.data.kind == "tabular"  # survived from synthesis
    assert cfg.data.target == "label"  # survived from synthesis


@pytest.mark.unit
def test_merge_is_recursive():
    assert merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


@pytest.mark.unit
def test_set_beats_both_layers(learnable_csv, tmp_path):
    args = cli.build_parser().parse_args(
        ["train", "--data", str(learnable_csv), "--set", "runtime.seed=99"]
    )
    assert cli._load_config(args).runtime.seed == 99


# ── Who chose the model ───────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "expected_model", "auto"),
    [
        ([], "xgboost", True),
        (["--model", "lightgbm"], "lightgbm", False),
        (["--set", 'model.name="lightgbm"'], "lightgbm", False),
    ],
)
def test_the_baseline_runs_only_when_the_framework_chose(
    learnable_csv, extra, expected_model, auto
):
    """A user who named the model has their own frame of reference; the baseline
    is about supplying one that is missing. Checked against the *surviving* value,
    because a later layer may have overridden the choice."""
    args = cli.build_parser().parse_args(["train", "--data", str(learnable_csv), *extra])
    cfg = cli._load_config(args)

    assert cfg.model.name == expected_model
    assert args.auto_selected is auto


# ── Refusal rather than substitution ──────────────────────
@pytest.fixture
def no_gbdt(monkeypatch):
    """An install without the `[gbdt]` extra, faked consistently.

    Both `is_available` and `unmet` are patched because in reality they are the
    same fact — `is_available` is literally "nothing unmet". Patching only the
    first would produce a state the code can never actually be in, and the test
    would then exercise a branch that does not exist in production.
    """
    from ml_framework.core.registry import MODELS
    from ml_framework.core.types import Requirement

    missing = (Requirement("xgboost", extra="gbdt", min_version="2.0"),)
    monkeypatch.setattr(MODELS, "is_available", lambda name: False)
    monkeypatch.setattr(MODELS, "unmet", lambda name: missing)
    return MODELS


@pytest.mark.unit
def test_an_uninstalled_family_refuses_instead_of_downgrading(no_gbdt):
    """The decision that makes zero-config trustworthy.

    Silently training an MLP because `[gbdt]` is missing would produce a number
    that looks exactly like the number the framework promised — and the user, who
    did not pick the model, has no way to notice.
    """
    from ml_framework.config.defaults import select_model
    from ml_framework.core.plugins import MissingExtraError

    with pytest.raises(MissingExtraError) as caught:
        select_model("tabular", "multiclass", n_rows=1000)

    message = str(caught.value)
    assert "Refusing to substitute" in message
    # And it says what to do about it, which is the whole point of refusing here
    # rather than three layers down inside a backend.
    assert "pip install 'ml-framework[gbdt]'" in message


@pytest.mark.unit
def test_a_row_threshold_picks_the_simpler_model_on_little_data():
    """The one place the row thresholds do real work: a recurrent forecaster needs
    history before it beats repeating last season."""
    from ml_framework.config.defaults import select_model

    assert select_model("timeseries", "forecasting", n_rows=5000)[0] == "ts.lstm"
    assert select_model("timeseries", "forecasting", n_rows=30)[0] == "ts.naive"


# ── mlf init ──────────────────────────────────────────────
@pytest.mark.unit
def test_init_writes_a_config_with_the_rule_beside_each_inferred_field(learnable_csv, tmp_path):
    """A generated file that just appeared with `task: multiclass` invites the
    reader to assume somebody decided that carefully."""
    destination = tmp_path / "mine.yaml"
    assert cli.main(["init", "--data", str(learnable_csv), "-o", str(destination)]) == 0

    text = destination.read_text(encoding="utf-8")
    assert "task: multiclass  # inferred:" in text
    assert "kind: tabular  # inferred:" in text
    assert "name: xgboost  # inferred:" in text
    assert "Generated by `mlf init`" in text


@pytest.mark.unit
def test_init_output_is_a_config_that_actually_loads(learnable_csv, tmp_path):
    """The comments must not break the YAML, and the values must validate — a
    generated file that cannot be fed back in is a worked example of nothing."""
    from ml_framework.config import ExperimentConfig

    destination = tmp_path / "mine.yaml"
    cli.main(["init", "--data", str(learnable_csv), "-o", str(destination)])

    cfg = ExperimentConfig.from_yaml(destination)
    assert cfg.task == "multiclass"
    assert cfg.model.name == "xgboost"


@pytest.mark.unit
def test_a_weak_inference_is_marked_GUESS(tmp_path):
    """The fallback rules have to look different from the confident ones, in the
    file as well as in the log."""
    frame = pd.DataFrame({"a": [1.0] * 8, "b": [2.0] * 8, "outcome": [0, 1] * 4})
    csv = tmp_path / "noname.csv"
    frame.to_csv(csv, index=False)

    text = render_config(synthesize(csv))
    assert "# GUESS:" in text
    assert "GUESS  data.target" in text  # and again in the header block


@pytest.mark.unit
def test_init_refuses_to_overwrite_without_force(learnable_csv, tmp_path):
    destination = tmp_path / "mine.yaml"
    destination.write_text("existing\n", encoding="utf-8")

    assert cli.main(["init", "--data", str(learnable_csv), "-o", str(destination)]) == 1
    assert destination.read_text(encoding="utf-8") == "existing\n"
    assert cli.main(["init", "--data", str(learnable_csv), "-o", str(destination), "--force"]) == 0


# ── The synthesized config's own choices ──────────────────
@pytest.mark.unit
def test_synthesis_sets_the_imbalance_strategy_the_chosen_model_wants(learnable_csv):
    """The schema default is `smote`, kept so an existing v1 config trains as it
    did. A config synthesized now has no such history, and zero-config always picks
    a tree — which consumes sample weights natively and is hurt by SMOTE. Left at
    the default, every zero-config run would warn about the value this sets."""
    synthesized = synthesize(learnable_csv)

    assert synthesized.config["data"]["params"]["imbalance_strategy"] == "auto"


@pytest.mark.unit
def test_synthesis_returns_a_plain_dict_not_a_validated_config(learnable_csv):
    """The whole mechanism. A validated config could not be merged under a YAML
    without special-casing, and then `--data` and `--config` would be alternatives
    rather than layers."""
    synthesized = synthesize(learnable_csv)

    assert isinstance(synthesized.config, dict)
    assert synthesized.config["data"]["path"] == str(learnable_csv)


# ── The stale-params fix this phase needed ────────────────
@pytest.mark.unit
def test_changing_the_model_drops_the_previous_models_params(learnable_csv, tmp_path):
    """`_resolve_plugin_params` writes every default back into `model.params`, so a
    config validated once carries xgboost's `tree_method`. Re-validating it as
    catboost failed with a wall of "extra inputs are not permitted" — which made
    `--set model.name=…` unusable, and it is a documented layer of the chain.
    """
    from ml_framework.config import ExperimentConfig

    args = cli.build_parser().parse_args(["train", "--data", str(learnable_csv)])
    cfg: ExperimentConfig = cli._load_config(args)
    assert "tree_method" in cfg.model.params  # xgboost's, materialized

    switched = cfg.with_overrides({"model.name": "catboost"})
    assert switched.model.name == "catboost"
    assert "tree_method" not in switched.model.params
    assert "depth" in switched.model.params  # catboost's own default


@pytest.mark.unit
def test_an_explicit_param_survives_the_model_change(learnable_csv):
    """Cleared *before* the overrides are applied, so a value set in the same call
    lands on the new model rather than being wiped by the clearing."""
    args = cli.build_parser().parse_args(["train", "--data", str(learnable_csv)])
    cfg = cli._load_config(args)

    switched = cfg.with_overrides({"model.name": "catboost", "model.params.depth": 5})
    assert switched.model.params["depth"] == 5
