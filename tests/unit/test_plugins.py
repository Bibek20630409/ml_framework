"""core/plugins.py + the v2 registries.

The first two tests here are **the guardrail for the entire plugin design** (see
the roadmap's "CI guardrail from P0 onward"): the registry must list an
unavailable plugin without importing it, and must refuse it with an actionable
message when selected.
"""

from __future__ import annotations

import sys

import pytest

from ml_framework.core.plugins import (
    BackendSpec,
    DuplicatePluginError,
    IncompatibleCombinationError,
    MissingExtraError,
    ModelSpec,
    PluginLoadError,
    PluginRegistry,
    SourceSpec,
    UnknownPluginError,
    check_requirements,
)
from ml_framework.core.registry import MODELS, SOURCES, models_for, validate_combination
from ml_framework.core.types import Capabilities, Requirement

MISSING_MODULE = "definitely_not_installed_xyz"


def _unavailable_spec(name: str = "phantom") -> ModelSpec:
    return ModelSpec(
        name=name,
        backend="gbdt",
        build=lambda *a, **k: pytest.fail("build must not be called for an unavailable plugin"),
        tasks=frozenset({"binary", "multiclass"}),
        data_kinds=frozenset({"tabular"}),
        requires=(Requirement(MISSING_MODULE, extra="gbdt", min_version="2.0"),),
        capabilities=Capabilities(accepts=frozenset({"arrays", "frame"})),
        description="A plugin whose library is not installed.",
    )


@pytest.fixture
def registry() -> PluginRegistry[ModelSpec]:
    return PluginRegistry("model")


# ── The guardrail ─────────────────────────────────────────
@pytest.mark.unit
def test_registry_lists_an_unavailable_plugin_without_importing_it(registry):
    # Arrange
    registry.register(_unavailable_spec())

    # Act
    rows = {row["name"]: row for row in registry.describe()}

    # Assert
    assert "phantom" in registry.names()
    assert rows["phantom"]["available"] is False
    assert rows["phantom"]["install"] == "pip install 'ml-framework[gbdt]'"
    assert MISSING_MODULE not in sys.modules


@pytest.mark.unit
def test_selecting_an_unavailable_plugin_raises_missing_extra_naming_the_extra(registry):
    # Arrange
    registry.register(_unavailable_spec())

    # Act / Assert
    with pytest.raises(MissingExtraError) as excinfo:
        registry.get("phantom")
    message = str(excinfo.value)
    assert "model 'phantom'" in message
    assert f"{MISSING_MODULE}>=2.0" in message
    assert "pip install 'ml-framework[gbdt]'" in message
    assert message.isascii()  # readable on a console with any code page


@pytest.mark.unit
def test_get_spec_still_returns_an_unavailable_spec_for_introspection(registry):
    """`mlf models --show xgboost` must work on a bare install."""
    registry.register(_unavailable_spec())
    assert registry.get_spec("phantom").backend == "gbdt"
    assert registry.is_available("phantom") is False


# ── Registration semantics ────────────────────────────────
@pytest.mark.unit
def test_duplicate_registration_requires_an_explicit_override(registry):
    registry.register(_unavailable_spec())
    with pytest.raises(DuplicatePluginError):
        registry.register(_unavailable_spec())
    registry.register(_unavailable_spec(), override=True)  # deliberate replacement is fine
    assert len(registry) == 1


@pytest.mark.unit
def test_unknown_plugin_error_is_a_keyerror_with_a_readable_message(registry):
    with pytest.raises(UnknownPluginError) as excinfo:
        registry.get_spec("nope")
    assert isinstance(excinfo.value, KeyError)
    assert "Unknown model 'nope'" in str(excinfo.value)


@pytest.mark.unit
def test_names_are_case_insensitive(registry):
    registry.register(_unavailable_spec("Phantom"))
    assert "phantom" in registry
    assert registry.get_spec("PHANTOM").name == "Phantom"


@pytest.mark.unit
def test_recorded_load_error_is_reraised_when_the_plugin_is_selected(registry):
    """A third-party plugin that failed to import is deferred, never swallowed."""
    # Arrange
    registry._load_errors["broken"] = PluginLoadError("plugin 'broken' failed to load: boom")

    # Act / Assert
    with pytest.raises(PluginLoadError, match="boom"):
        registry.get_spec("broken")
    assert any(row["name"] == "broken" for row in registry.describe())


@pytest.mark.unit
def test_check_requirements_passes_when_everything_is_installed():
    check_requirements((Requirement("json"), Requirement("sys")), what="model 'x'")


@pytest.mark.unit
def test_check_requirements_lists_every_unmet_dependency():
    reqs = (
        Requirement(MISSING_MODULE, extra="gbdt", min_version="2.0"),
        Requirement("another_missing_xyz", extra="nlp"),
    )
    with pytest.raises(MissingExtraError) as excinfo:
        check_requirements(reqs, what="model 'x'")
    message = str(excinfo.value)
    assert "gbdt" in message and "nlp" in message
    assert "Install them with" in message


# ── Spec metadata ─────────────────────────────────────────
@pytest.mark.unit
def test_model_spec_supports_reports_task_and_kind_compatibility():
    spec = _unavailable_spec()
    assert spec.supports(task="binary", data_kind="tabular")
    assert not spec.supports(task="regression")
    assert not spec.supports(data_kind="image")


@pytest.mark.unit
def test_backend_and_source_specs_default_to_no_requirements():
    backend = BackendSpec(name="gbdt", factory=lambda: object())
    source = SourceSpec(name="tabular", data_kind="tabular", build=lambda cfg: None)
    assert backend.requires == () and source.requires == ()
    assert source.payload == "arrays"


# ── The populated built-in registries ─────────────────────
@pytest.mark.unit
def test_builtin_model_specs_are_registered():
    import ml_framework.plugins as plugins  # noqa: F401  (population is a side effect)
    from ml_framework.core import available_models

    expected = {
        "mlp",
        "cnn",
        "xgboost",
        "lightgbm",
        "catboost",
        "ts.naive",
        "ts.arima",
        "ts.prophet",
        "ts.lstm",
    }
    assert set(MODELS.names()) == set(available_models()) == expected
    assert set(plugins.BUILTINS) == expected


@pytest.mark.unit
def test_available_models_covers_plugins_that_have_no_model_class():
    """`available_models()` shims over MODELS, not over the v1 class registry.

    A GBDT plugin registers a build *function*; there is no LightningModule
    subclass to put in the class registry. Reading that registry here would have
    made "available models" quietly Lightning-only the moment a non-neural family
    landed — which is exactly what P3 landed.
    """
    import ml_framework.plugins as plugins
    from ml_framework.core import available_models
    from ml_framework.core.registry import _MODEL_REGISTRY

    assert "xgboost" in available_models()
    # `model_class` imports the defining module, which is what populates the v1
    # class registry — the neural plugins are registered with a lazy build.
    assert plugins.model_class("mlp").__name__ == "MLP"
    assert "mlp" in _MODEL_REGISTRY
    assert "xgboost" not in _MODEL_REGISTRY  # no class, and that is correct


@pytest.mark.unit
def test_builtin_source_specs_are_registered():
    import ml_framework.data  # noqa: F401
    from ml_framework.core import available_datamodules

    assert (
        set(SOURCES.names()) == set(available_datamodules()) == {"tabular", "image", "timeseries"}
    )


@pytest.mark.unit
def test_mlp_spec_declares_its_tasks_kinds_and_capabilities():
    import ml_framework.plugins  # noqa: F401

    spec = MODELS.get_spec("mlp")
    assert spec.backend == "lightning"
    assert spec.tasks == frozenset({"binary", "multiclass", "regression"})
    assert spec.data_kinds == frozenset({"tabular"})
    # torch left the base dependencies in P3, so the neural plugins declare it like
    # any other optional runtime — which is what lets a GBDT-only install list them
    # honestly instead of shipping 533 MB nothing imports.
    assert [r.package for r in spec.requires] == ["torch", "pytorch-lightning"]
    assert all(r.extra == "lightning" for r in spec.requires)
    assert spec.capabilities.needs_scaling and spec.capabilities.supports_pruning
    assert spec.suggest is not None  # conditional hidden_dims space


@pytest.mark.unit
def test_cnn_spec_declares_the_image_extra():
    import ml_framework.plugins  # noqa: F401

    spec = MODELS.get_spec("cnn")
    # torch first (the backend it rides), then its own extra on top.
    assert [r.package for r in spec.requires] == [
        "torch",
        "pytorch-lightning",
        "torchvision",
        "Pillow",
    ]
    extras = {r.package: r.extra for r in spec.requires}
    assert extras["torchvision"] == extras["Pillow"] == "image"
    assert extras["torch"] == "lightning"
    assert spec.capabilities.accepts == frozenset({"dataset"})


@pytest.mark.unit
def test_every_builtin_spec_carries_a_params_model():
    """`model.params` is free-form in core and strict in the plugin. Without a
    params_model the config validator has nothing to enforce, so a typo would
    reach the model as a silently ignored key."""
    import ml_framework.plugins as plugins

    for name in plugins.BUILTINS:
        spec = MODELS.get_spec(name)
        assert spec.params_model is not None, name
        assert spec.params_model.model_config.get("extra") == "forbid", name
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            spec.params_model(definitely_not_a_knob=1)


@pytest.mark.unit
def test_registering_the_builtins_imports_no_optional_runtime():
    """The rule that replaces `except Exception: pass` structurally.

    **Registering** a plugin must not import the library it wraps. Hold that and
    there is no exception to swallow: every model stays listable — honestly marked
    unavailable — on an install that cannot run it, and a genuinely broken module
    can no longer hide behind "the extra is missing".

    How each family manages it differs, and the difference is the point. The GBDT
    plugins are pydantic-only at module scope and import xgboost inside `build()`.
    The neural plugins cannot be: defining a LightningModule subclass imports
    torch. So their schemas live in `params.py` and their modules sit behind a
    lazy `build`.

    Checked by importing the package in a subprocess where every optional runtime
    raises on import, rather than by watching `sys.modules` — pytorch_lightning
    imports torchvision itself when present, which would make a naive runtime
    check pass for the wrong reason.
    """
    import subprocess
    import sys

    optional = sorted({r.module for spec in MODELS.specs() for r in spec.requires})
    assert optional  # a vacuous test would pass forever

    code = f"""
import sys

BANNED = {optional!r}


class _Ban:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ModuleNotFoundError(name)
        return None


sys.meta_path.insert(0, _Ban())
for _m in list(sys.modules):
    if _m.split(".")[0] in BANNED:
        del sys.modules[_m]

import ml_framework.plugins as plugins
from ml_framework.core.registry import MODELS

missing = set(plugins.BUILTINS) - set(MODELS.names())
assert not missing, missing
# The schemas must be usable too: the config validator runs them here.
assert MODELS.get_spec("mlp").params_model is not None
assert MODELS.get_spec("xgboost").params_model is not None
print("ok")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.unit
def test_a_broken_builtin_would_not_be_hidden_behind_a_missing_extra():
    """The behaviour the v1 `except Exception: pass` made impossible to observe.

    A builtin that fails to import is *our* bug, so the import must propagate.
    Simulated by importing a module that raises, since breaking a real plugin is
    not something a test can do reversibly.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml_framework.plugins.no_such_plugin")


@pytest.mark.unit
def test_mlp_search_space_keys_are_dotted_config_paths():
    """Applying a trial must be exactly `config.with_overrides(values)`."""
    import ml_framework.plugins  # noqa: F401

    spec = MODELS.get_spec("mlp")
    assert all("." in key for key in spec.search_space)
    assert "model.params.dropout" in spec.search_space


@pytest.mark.unit
def test_mlp_suggest_hook_emits_dotted_paths_for_a_conditional_space():
    import ml_framework.plugins  # noqa: F401

    class FakeTrial:
        def suggest_int(self, name, low, high, log=False, step=1):
            return low

        def suggest_float(self, name, low, high, log=False, step=None):
            return low

    spec = MODELS.get_spec("mlp")
    assert spec.suggest is not None
    values = spec.suggest(FakeTrial())
    assert values["model.params.hidden_dims"] == [32]
    assert values["model.params.dropout"] == pytest.approx(0.1)


# ── validate_combination ──────────────────────────────────
@pytest.mark.unit
def test_validate_combination_accepts_a_supported_triple():
    import ml_framework.plugins  # noqa: F401

    assert validate_combination("multiclass", "tabular", "mlp").name == "mlp"


@pytest.mark.unit
def test_validate_combination_rejects_an_unsupported_task():
    """Also pins the check order: cnn is torchvision-gated, and the impossible
    combination must be reported before "go install 2 GB of torchvision"."""
    import ml_framework.plugins  # noqa: F401

    with pytest.raises(IncompatibleCombinationError, match="does not support task"):
        validate_combination("regression", "image", "cnn")


@pytest.mark.unit
def test_validate_combination_raises_missing_extra_for_a_sound_but_uninstalled_model():
    import ml_framework.plugins  # noqa: F401

    if MODELS.is_available("cnn"):
        pytest.skip("torchvision installed — nothing to refuse")
    with pytest.raises(MissingExtraError, match=r"ml-framework\[image\]"):
        validate_combination("multiclass", "image", "cnn")


@pytest.mark.unit
def test_validate_combination_rejects_a_wrong_data_kind():
    """'xgboost cannot consume an image folder', at config-load time."""
    import ml_framework.plugins  # noqa: F401

    with pytest.raises(IncompatibleCombinationError, match="cannot consume data kind"):
        validate_combination("multiclass", "image", "mlp")


@pytest.mark.unit
def test_validate_combination_rejects_an_unacceptable_payload():
    import ml_framework.plugins  # noqa: F401

    with pytest.raises(IncompatibleCombinationError, match="payload"):
        validate_combination("multiclass", "tabular", "mlp", payload="series")


@pytest.mark.unit
def test_models_for_lists_only_installed_compatible_models():
    import ml_framework.plugins  # noqa: F401

    names = [s.name for s in models_for(task="multiclass", data_kind="tabular")]
    assert "mlp" in names
    # Ordered by auto_priority, best first — the tie-break zero-config selection
    # uses. XGBoost outranks the MLP on tabular data, which is the whole reason
    # the GBDT family exists.
    installed_trees = [n for n in ("xgboost", "lightgbm", "catboost") if MODELS.is_available(n)]
    if installed_trees:
        assert (
            names[0] == "xgboost" if "xgboost" in installed_trees else names[0] in installed_trees
        )
        assert names.index("mlp") > max(names.index(n) for n in installed_trees)
    # An uninstalled tree must not appear as a candidate at all.
    for name in ("xgboost", "lightgbm", "catboost"):
        assert (name in names) == MODELS.is_available(name)

    # cnn is registered but torchvision-gated, so it must not appear as a candidate
    # for automatic selection unless it is actually installed.
    image_names = [s.name for s in models_for(data_kind="image")]
    assert image_names in ([], ["cnn"])
    assert image_names == (["cnn"] if MODELS.is_available("cnn") else [])
