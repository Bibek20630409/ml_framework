"""Selecting an engine per run: the config field, the CLI flag, and the refusal.

The promise this feature makes is "for *this* run, use the Spark implementation",
so what is tested here is the selection path — that the choice reaches the source
unchanged, that it is refused by name where it is not implemented, and that the
three pipeline call sites did not have to learn about it.
"""

from __future__ import annotations

import pytest

from ml_framework.config import ExperimentConfig
from ml_framework.core.types import FrameworkError
from ml_framework.data.builders import build_bundle, build_cv_bundles

pytestmark = pytest.mark.unit


def test_the_default_engine_is_local(make_config, tabular_csv):
    """An existing config keeps reading exactly as it did."""
    cfg = make_config(tabular_csv, "multiclass")
    assert cfg.data.backend == "local"
    assert cfg.data.backend_params == {}


def test_every_shipped_config_still_validates():
    """`backend` is defaulted, so no YAML in the repo needs touching."""
    cfg = ExperimentConfig.from_yaml("configs/example_tabular.yaml")
    assert cfg.data.backend == "local"


def test_the_engine_is_selectable_by_dotted_override(make_config, tabular_csv):
    """The same mechanism `--set` and the HPO trial applier use."""
    cfg = make_config(tabular_csv, "multiclass", **{"data.backend": "spark"})
    assert cfg.data.backend == "spark"


def test_backend_params_accepts_keys_the_core_cannot_know(make_config, tabular_csv):
    """`data.backend_params.` is a creatable prefix: the engine owns its knobs, so
    the "key must already exist" rule cannot apply."""
    cfg = make_config(tabular_csv, "multiclass", **{"data.backend_params.shuffle_partitions": 16})
    assert cfg.data.backend_params == {"shuffle_partitions": 16}


def test_a_typo_in_a_fixed_data_key_is_still_refused(make_config, tabular_csv):
    """The new creatable prefix must not have opened the whole `data` block."""
    with pytest.raises(KeyError):
        make_config(tabular_csv, "multiclass", **{"data.backendd": "spark"})


def test_selecting_a_distributed_engine_for_an_unsupported_kind_is_refused_by_name(
    make_config, tabular_csv
):
    """Only `tabular` reads through the backend today. The rest would silently
    collect the whole table and pretend to be distributed, so they refuse."""
    cfg = make_config(tabular_csv, "multiclass", **{"data.backend": "spark"})
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"kind": "text"})})
    with pytest.raises(FrameworkError) as excinfo:
        build_bundle(cfg)
    assert "data.backend 'spark'" in str(excinfo.value)
    assert "tabular" in str(excinfo.value)


def test_the_cross_validation_path_refuses_it_too(make_config, tabular_csv):
    """`build_cv_bundles` calls the source builders directly, so a fold would
    otherwise slip past the refusal the single-holdout path enforces."""
    cfg = make_config(tabular_csv, "multiclass", **{"data.backend": "spark", "data.split.folds": 2})
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"kind": "text"})})
    with pytest.raises(FrameworkError):
        next(iter(build_cv_bundles(cfg)))


def test_the_engine_reaches_the_source_without_the_pipeline_knowing(
    make_config, tabular_csv, monkeypatch
):
    """The point of the whole design: `train.py` calls `build_bundle(config)` and
    the choice rides inside the config. If this passes, no pipeline call site had
    to change.

    Spied at `core.registry.get_data_backend` because that is where *every* path
    resolves an engine — `engine_for` imports it lazily, so one patch covers the
    source and both builders helpers.
    """
    seen: list[tuple[str, dict]] = []
    from ml_framework.core import registry

    real = registry.get_data_backend

    def _spy(name, **params):
        seen.append((name, params))
        return real(name, **params)

    monkeypatch.setattr(registry, "get_data_backend", _spy)
    build_bundle(make_config(tabular_csv, "multiclass"))
    assert seen, "build_bundle resolved no data backend at all"
    assert {name for name, _ in seen} == {"local"}


def test_backend_params_reach_the_engine_not_just_the_config(make_config, tabular_csv, monkeypatch):
    """`engine_for` exists so no source has to remember to forward these.

    Forgetting them would silently drop `max_collect_rows` and turn a guarded
    collect back into an unguarded one — a knob that reads as set and is not.
    """
    seen: list[dict] = []
    from ml_framework.core import registry

    real = registry.get_data_backend

    def _spy(name, **params):
        seen.append(params)
        return real(name)  # drop them; `local` refuses unknown params

    monkeypatch.setattr(registry, "get_data_backend", _spy)
    cfg = make_config(tabular_csv, "multiclass", **{"data.backend_params.max_collect_rows": 99})
    build_bundle(cfg)
    assert all(p == {"max_collect_rows": 99} for p in seen), seen


def test_the_cli_flag_wins_over_the_yaml_file(tmp_path, tabular_csv):
    """Documented precedence: YAML < --set < an explicit flag."""
    import yaml

    from ml_framework.cli import _load_config, build_parser

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "task": "multiclass",
                "data": {
                    "kind": "tabular",
                    "path": str(tabular_csv),
                    "target": "label",
                    "backend": "local",
                },
                "model": {"name": "xgboost"},
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["train", "--config", str(cfg_path), "--data-backend", "spark"]
    )
    assert _load_config(args).data.backend == "spark"


def test_omitting_the_flag_leaves_the_configured_engine_alone(tmp_path, tabular_csv):
    """`None` means "nobody said", which is distinct from asking for local."""
    import yaml

    from ml_framework.cli import _load_config, build_parser

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "task": "multiclass",
                "data": {
                    "kind": "tabular",
                    "path": str(tabular_csv),
                    "target": "label",
                    "backend": "spark",
                },
                "model": {"name": "xgboost"},
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["train", "--config", str(cfg_path)])
    assert _load_config(args).data.backend == "spark"


def test_the_engine_is_part_of_the_bundle_cache_key(make_config, tabular_csv):
    """Flipping the engine mid-search must invalidate: the key serializes
    `config.data`, so this comes for free — and a regression would silently reuse
    a bundle built by the other engine."""
    from ml_framework.pipeline.tune import _BundleCache

    cache = _BundleCache()
    local = make_config(tabular_csv, "multiclass")
    spark = make_config(tabular_csv, "multiclass", **{"data.backend": "spark"})
    assert cache._key(local) != cache._key(spark)
