"""The conftest hook that pins HuggingFace to its local cache.

Testing test-infrastructure is usually not worth it. This is the exception,
because the mechanism turns on two things that are easy to get wrong and silent
when you do:

* **Import ordering.** ``HF_HUB_OFFLINE`` is read into a module constant when
  ``huggingface_hub`` is imported. Set it one line too late and it does nothing —
  no error, just the old behaviour. That is why the pin runs at conftest *import*
  time and resolves the cache directory by hand rather than by asking the library.
* **``"0"`` is truthy.** ``HF_HUB_OFFLINE=0`` means "stay online". A bare
  truthiness check would read it as "already pinned" and skip the pin, giving
  exactly the opposite of what the variable asks for.

What the pin buys, measured: with the hub endpoint unreachable, a single
``from_pretrained`` on a *fully cached* model spends ~37 s in retry backoff and
then raises ``ConnectionError``. Pinned, the same call reads the cache and the NLP
tests pass. One network blip mid-run is what turned six of them into errors once.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# `tests/` is not a package, and pytest puts the root conftest's directory on
# sys.path, so the shared fixtures module imports under its bare name.
from conftest import (  # type: ignore[import-not-found]
    HF_TEST_CHECKPOINTS,
    _hf_cache_dir,
    _pin_hf_to_its_cache,
)


@pytest.fixture
def clean_env(monkeypatch):
    """No inherited HF settings, so each case starts from a known state."""
    for name in ("HF_HUB_OFFLINE", "HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _warm(cache: Path) -> Path:
    """A cache directory containing every checkpoint, in the on-disk layout."""
    for repo in HF_TEST_CHECKPOINTS:
        (cache / f"models--{repo.replace('/', '--')}").mkdir(parents=True)
    return cache


# ── Cache location ────────────────────────────────────────
@pytest.mark.unit
def test_hf_hub_cache_wins(clean_env, tmp_path):
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    clean_env.setenv("HF_HOME", str(tmp_path / "home"))

    assert _hf_cache_dir() == tmp_path / "explicit"


@pytest.mark.unit
def test_hf_home_is_the_fallback_and_gets_the_hub_suffix(clean_env, tmp_path):
    """`HF_HOME` points at the whole HF directory; the blobs live under `hub/`.
    Dropping the suffix would look at a directory that is always empty, so the pin
    would never fire and nobody would notice."""
    clean_env.setenv("HF_HOME", str(tmp_path / "home"))

    assert _hf_cache_dir() == tmp_path / "home" / "hub"


@pytest.mark.unit
def test_the_default_matches_the_librarys_own(clean_env):
    """Checked against `huggingface_hub` itself rather than restated, since a
    silent disagreement here disables the pin without failing anything."""
    constants = pytest.importorskip(
        "huggingface_hub.constants", reason="the nlp extra is not installed"
    )

    assert _hf_cache_dir() == Path(constants.HF_HUB_CACHE)


# ── When it fires ─────────────────────────────────────────
@pytest.mark.unit
def test_a_warm_cache_pins(clean_env, tmp_path):
    clean_env.setenv("HF_HUB_CACHE", str(_warm(tmp_path / "cache")))

    assert _pin_hf_to_its_cache() is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.unit
def test_a_cold_cache_leaves_the_network_alone(clean_env, tmp_path):
    """The first run on a fresh machine has to download. Forcing offline there
    would fail with a cache miss rather than fetching, so absence of the cache is
    the signal to stay online."""
    cold = tmp_path / "cache"
    cold.mkdir()
    clean_env.setenv("HF_HUB_CACHE", str(cold))

    assert _pin_hf_to_its_cache() is False
    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.unit
def test_a_partial_cache_leaves_the_network_alone(clean_env, tmp_path):
    """One checkpoint present and one missing is a cold cache for this purpose —
    pinning would make the missing one fail instead of downloading."""
    cache = tmp_path / "cache"
    first = HF_TEST_CHECKPOINTS[0].replace("/", "--")
    (cache / f"models--{first}").mkdir(parents=True)
    clean_env.setenv("HF_HUB_CACHE", str(cache))

    assert _pin_hf_to_its_cache() is False


# ── The variable's own semantics ──────────────────────────
@pytest.mark.unit
def test_offline_zero_is_not_read_as_already_pinned(clean_env, tmp_path):
    """`HF_HUB_OFFLINE=0` means stay online, and `"0"` is a truthy string.

    A bare truthiness check would return early here, reporting the run as pinned
    while leaving it online — the exact inverse of what was asked for, and
    invisible until a network blip.
    """
    clean_env.setenv("HF_HUB_OFFLINE", "0")
    clean_env.setenv("HF_HUB_CACHE", str(_warm(tmp_path / "cache")))

    # Falls through to the cache check, which is warm, so it pins for real.
    assert _pin_hf_to_its_cache() is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.unit
def test_an_explicit_pin_by_the_caller_is_respected(clean_env, tmp_path):
    """CI or a developer setting it wins, and the cache is not consulted at all —
    which is what lets someone force offline on a machine this code has never
    looked at."""
    clean_env.setenv("HF_HUB_OFFLINE", "1")
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "does-not-exist"))

    assert _pin_hf_to_its_cache() is True
