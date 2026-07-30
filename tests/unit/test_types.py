"""core/types.py — the dependency-free vocabulary."""

from __future__ import annotations

import sys

import pytest

from ml_framework.core.types import (
    Capabilities,
    Requirement,
    _version_tuple,
    unmet_requirements,
)


@pytest.mark.unit
def test_types_module_pulls_in_no_heavy_dependencies():
    """types.py must stay dependency-free.

    Both `config` and `plugins` import from it, so anything heavier here would
    eagerly drag an optional library into a bare install. Loaded by file path in a
    fresh interpreter: importing it as a package member would pull in
    `ml_framework/__init__` (and therefore pydantic) and prove nothing.
    """
    # Arrange
    import subprocess

    import ml_framework.core.types as types_mod

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('t', r'{types_mod.__file__}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        # dataclasses resolves annotations through sys.modules, so register first.
        "sys.modules['t'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "heavy = ('torch','pytorch_lightning','numpy','pydantic','sklearn','pandas','yaml')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )

    # Act
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    # Assert
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.0.3", (2, 0, 3)),
        ("1.1", (1, 1)),
        ("2.0.3.post1", (2, 0, 3)),
        ("4.40.0rc1", (4, 40, 0)),
        ("0.15", (0, 15)),
    ],
)
def test_version_tuple_parses_leading_numeric_components(raw, expected):
    assert _version_tuple(raw) == expected


@pytest.mark.unit
def test_requirement_for_installed_stdlib_module_is_satisfied():
    # Arrange
    req = Requirement("json")

    # Act / Assert
    assert req.is_installed()
    assert req.is_satisfied()
    assert req.unmet_reason() is None


@pytest.mark.unit
def test_requirement_for_missing_module_reports_reason_without_importing():
    # Arrange
    req = Requirement("definitely_not_installed_xyz", extra="gbdt", min_version="2.0")

    # Act
    reason = req.unmet_reason()

    # Assert
    assert req.is_installed() is False
    assert reason is not None and "not installed" in reason
    assert "definitely_not_installed_xyz" not in sys.modules  # find_spec must not import


@pytest.mark.unit
def test_requirement_spec_and_pip_hint_use_the_extra_name():
    req = Requirement("xgboost", extra="gbdt", min_version="2.0")
    assert req.spec() == "xgboost>=2.0"
    assert req.pip_hint() == "pip install 'ml-framework[gbdt]'"


@pytest.mark.unit
def test_requirement_uses_dist_name_when_it_differs_from_import_name():
    req = Requirement("PIL", extra="image", min_version="9.0", dist="Pillow")
    assert req.package == "Pillow"
    assert req.spec() == "Pillow>=9.0"


@pytest.mark.unit
def test_requirement_without_extra_hints_the_bare_package():
    req = Requirement("prophet", min_version="1.1")
    assert req.pip_hint() == "pip install 'prophet>=1.1'"


@pytest.mark.unit
def test_unmet_requirements_preserves_declaration_order():
    reqs = (
        Requirement("json"),
        Requirement("nope_a"),
        Requirement("sys"),
        Requirement("nope_b"),
    )
    assert [r.module for r in unmet_requirements(reqs)] == ["nope_a", "nope_b"]


@pytest.mark.unit
def test_capabilities_are_frozen_and_gate_payloads():
    # Arrange
    caps = Capabilities(accepts=frozenset({"arrays", "frame"}))

    # Act / Assert
    assert caps.can_accept("arrays")
    assert caps.can_accept("frame")
    assert not caps.can_accept("dataset")
    with pytest.raises(AttributeError):
        caps.needs_scaling = False  # type: ignore[misc]
