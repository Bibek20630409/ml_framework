"""
config/defaults.py
──────────────────
Per-backend tuning budgets — the numbers that make "tune by default" a usable
default rather than a broken-feeling one.

The plan's stated requirement was tuning always on. Taken literally with one
uniform trial count that is a bad default: 20 trials × 200 epochs on the Lightning
path is hours, and a ``mlf train`` that appears to hang is worse than no tuning at
all. The budgets below are backend-aware precisely because a boosting trial costs
seconds and a neural trial costs minutes:

    gbdt        30 trials   300 s
    lightning   10 trials   900 s   per-trial epochs capped at 25
    forecast     8 trials   180 s

Two of those axes matter independently. ``max_trials`` bounds how much of the
space gets explored; ``max_seconds`` bounds how long the user waits regardless of
how slow a single trial turns out to be. The **per-trial** epoch cap is the third:
without it a single Lightning trial can consume the entire wall budget and the
search degenerates to one sample. It applies only while tuning — the final fit
runs at the full configured budget.

These are floors to raise, not ceilings to obey: an explicit ``tune.max_trials``
in the YAML, or ``--tune-trials`` / ``--tune-budget`` on the CLI, always wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# The framework-level fallback, used for a backend with no entry of its own — a
# third-party backend, or `forecast` before its models land. Deliberately modest:
# an unknown backend has unknown per-trial cost, and the safe assumption is
# expensive.
DEFAULT_BUDGET_NAME: Final[str] = "default"


@dataclass(frozen=True, slots=True)
class TuneBudget:
    """What a search may spend, before the user's own settings are applied."""

    max_trials: int
    max_seconds: float
    # Epochs a *single trial* may run, for backends with an epoch loop. ``None``
    # means "no per-trial cap" — correct for one-shot fits, where the analogous
    # knob is `n_estimators` and early stopping already bounds it.
    trial_max_epochs: int | None = None

    def with_overrides(
        self, *, max_trials: int | None = None, max_seconds: float | None = None
    ) -> TuneBudget:
        return TuneBudget(
            max_trials=max_trials if max_trials is not None else self.max_trials,
            max_seconds=max_seconds if max_seconds is not None else self.max_seconds,
            trial_max_epochs=self.trial_max_epochs,
        )


TUNE_BUDGETS: Final[dict[str, TuneBudget]] = {
    # Boosting trials are seconds, so the budget buys breadth.
    "gbdt": TuneBudget(max_trials=30, max_seconds=300.0),
    # Neural trials are minutes. Fewer of them, a longer wall clock, and a hard
    # per-trial epoch cap so one slow trial cannot eat the whole search.
    "lightning": TuneBudget(max_trials=10, max_seconds=900.0, trial_max_epochs=25),
    # Fit-per-series: cheap individually, but the count scales with the data.
    "forecast": TuneBudget(max_trials=8, max_seconds=180.0),
    DEFAULT_BUDGET_NAME: TuneBudget(max_trials=10, max_seconds=600.0, trial_max_epochs=25),
}


def budget_for(backend: str) -> TuneBudget:
    """The default budget for ``backend``, falling back to the conservative one."""
    return TUNE_BUDGETS.get(backend, TUNE_BUDGETS[DEFAULT_BUDGET_NAME])


# ── Duration parsing ──────────────────────────────────────
_DURATION = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smh])?\s*$", re.IGNORECASE)
_UNIT_SECONDS: Final[dict[str, float]] = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str | float | int) -> float:
    """``"10m"`` → ``600.0``. Bare numbers are seconds.

    Exists so ``--tune-budget 10m`` reads the way a person would write it. A
    wall-clock budget typed as ``600`` is easy to misread by a factor of sixty,
    and this is a flag people reach for when a run is already taking too long.
    """
    if isinstance(text, (int, float)):
        seconds = float(text)
    else:
        match = _DURATION.match(str(text))
        if not match:
            raise ValueError(
                f"cannot read '{text}' as a duration. Use seconds (900), or a unit: 30s, 10m, 2h"
            )
        seconds = float(match.group("value")) * _UNIT_SECONDS[(match.group("unit") or "s").lower()]
    if seconds <= 0:
        raise ValueError(f"duration must be positive, got '{text}'")
    return seconds


__all__ = [
    "DEFAULT_BUDGET_NAME",
    "TUNE_BUDGETS",
    "TuneBudget",
    "budget_for",
    "parse_duration",
]
