"""
Config-drift guards.

These assert a property no other test can: that the *pipeline definition files*
agree with each other. Neither DVC nor Airflow runs in CI — `dvc.yaml` is only
parsed, and the DAG is only imported — which is exactly how a stale `outs:` list
survived two phases unnoticed. A value duplicated across those files is therefore
unverified by construction, so the guard is that it not be duplicated at all.

The tests read the files as data, not as pipelines, so they need neither DVC nor
Airflow installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DAG = ROOT / "orchestration" / "airflow" / "dags" / "ml_pipeline.py"
TRAIN_CONFIG = ROOT / "configs" / "dvc_tabular.yaml"


def _code_strings(path: Path) -> list[str]:
    """Every string literal the module *executes*, ignoring comments and docstrings.

    Matching raw source text would flag the prose explaining why a value is not
    hardcoded — a guard that fires on its own rationale is a guard people delete.
    Parsing means these tests see only what the DAG actually runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


@pytest.fixture(scope="module")
def params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8")) or {}


@pytest.fixture(scope="module")
def train_config() -> dict:
    return yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8")) or {}


def test_the_train_config_is_the_only_place_the_target_column_is_written_down(params):
    """`params.yaml` must not carry its own copy of the target column.

    Two spellings of the same fact (`preprocess.target_col` and `data.target`)
    can disagree, and when they do the failure lands mid-pipeline as a KeyError
    from the tabular reader rather than at the edit that caused it.
    """
    assert "target_col" not in (params.get("preprocess") or {}), (
        "params.yaml re-declares the target column; it belongs to the train "
        "config's data.target, which every stage now reads via --config"
    )


def test_params_does_not_duplicate_the_processed_data_path(params):
    """`data.path` in the train config is where processed data lives."""
    assert "output" not in (params.get("preprocess") or {}), (
        "params.yaml re-declares the processed-data path; it belongs to the "
        "train config's data.path"
    )


def test_the_dag_hardcodes_no_target_column_or_data_path():
    """The scheduled pipeline must reference the config, not copy its values.

    This is the drift that mattered most: the DAG invoked the stages directly
    with literal values, so `params.yaml` could be edited and committed without
    changing anything the scheduler actually ran.
    """
    executed = " ".join(_code_strings(DAG))

    assert "--target-col" not in executed, (
        "the DAG passes an explicit target column; pass --config so the stage "
        "reads data.target itself"
    )
    for literal in ("data/processed", "data/raw/sample.csv"):
        assert literal not in executed, (
            f"the DAG hardcodes the path {literal!r}; it should come from the "
            "train config (data.path) or params.yaml (preprocess.input)"
        )


def test_the_dag_reads_the_output_directory_from_the_config():
    """`runtime.output_dir` is the config's decision, including for the gate.

    Asserted as "it consults the config" rather than "the word never appears",
    because a literal fallback default is fine — silently ignoring a configured
    value is not.
    """
    assert "output_dir" in _code_strings(DAG), (
        "the DAG never reads runtime.output_dir, so changing it in the config "
        "would break the quality gate with a FileNotFoundError"
    )


def test_dvc_stages_reference_the_config_rather_than_repeating_it():
    """`dvc.yaml` must not re-spell the target column either."""
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8")) or {}
    cmd = dvc["stages"]["preprocess"]["cmd"]

    assert "--target-col" not in cmd, (
        "the DVC preprocess stage passes an explicit target column; pass "
        "--config so it reads data.target from the train config"
    )
    assert "--config" in cmd


def test_dvc_tracks_the_train_config_params_so_a_changed_target_invalidates_the_stage():
    """`vars` substitutes without tracking; `params` is what forces a rerun.

    Without this, editing `data.target` changes what the stage *would* do while
    DVC still considers the old cache entry valid — reproducibility silently
    stops meaning anything.
    """
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8")) or {}
    declared = dvc["stages"]["preprocess"]["params"]

    tracked: set[str] = set()
    for entry in declared:
        if isinstance(entry, dict):
            for file, keys in entry.items():
                tracked.update(f"{file}:{key}" for key in keys)

    assert "configs/dvc_tabular.yaml:data.target" in tracked
    assert "configs/dvc_tabular.yaml:data.path" in tracked


def test_the_stage_resolver_reads_what_the_train_config_actually_declares(train_config):
    """End-to-end on the real config: the resolver returns the committed values."""
    from ml_framework.pipeline.stage_config import load_stage_data

    stage = load_stage_data(TRAIN_CONFIG)

    assert stage.target == train_config["data"]["target"]
    assert stage.path == train_config["data"]["path"]
