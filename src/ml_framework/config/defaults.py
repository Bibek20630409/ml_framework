"""
config/defaults.py
──────────────────
The numbers zero-config runs on: per-backend tuning budgets, and the model to
pick for a given kind of data.

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

import logging
import re
from dataclasses import dataclass
from typing import Final

log = logging.getLogger(__name__)

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


# ── Model selection ───────────────────────────────────────
# The table zero-config picks from, keyed by (data kind, task).
#
# **Candidates in one row are alternatives of equivalent quality, not a ranking
# by desperation.** The three GBDT entries are one family behind one extra, so
# falling through them costs nothing. What must never happen is falling out of a
# *family*: if `[gbdt]` is not installed, a tabular run raises `MissingExtraError`
# naming the pip command rather than quietly training an MLP and reporting its
# score as though the framework had chosen it. A silent downgrade is the one
# outcome that makes zero-config untrustworthy, because the number it produces
# looks exactly like a number worth acting on.
#
# The single exception is marked `downgrade=True` and logged at WARNING: the plan
# sanctions seasonal-naive as the forecasting fallback, and `ts.naive` needs
# nothing installed, so a forecasting run always has *something* honest to do.


@dataclass(frozen=True, slots=True)
class ModelRule:
    """One candidate, and the dataset shape it applies to."""

    model: str
    # Row thresholds. A recurrent forecaster needs history before it beats
    # repeating last season; below that the simple thing is also the better thing.
    min_rows: int = 0
    max_rows: int | None = None
    # True when choosing this is a real step down in capability rather than an
    # equivalent alternative. Logged loudly; see the note above.
    downgrade: bool = False
    reason: str = ""

    def fits(self, n_rows: int) -> bool:
        if n_rows < self.min_rows:
            return False
        return self.max_rows is None or n_rows <= self.max_rows


# Deliberately no `("tabular", …) -> mlp` row anywhere. Gradient boosting is the
# right default for tabular data at every size this framework will see, and
# listing the MLP as a fallback is exactly the silent downgrade the note forbids.
_GBDT: tuple[ModelRule, ...] = (
    ModelRule("xgboost", reason="gradient boosting is the tabular default"),
    ModelRule("lightgbm", reason="the installed member of the boosting family"),
    ModelRule("catboost", reason="the installed member of the boosting family"),
)

MODEL_RULES: Final[dict[tuple[str, str], tuple[ModelRule, ...]]] = {
    ("tabular", "binary"): _GBDT,
    ("tabular", "multiclass"): _GBDT,
    ("tabular", "regression"): _GBDT,
    ("image", "binary"): (
        ModelRule("cnn", reason="transfer learning beats training from scratch"),
    ),
    ("image", "multiclass"): (
        ModelRule("cnn", reason="transfer learning beats training from scratch"),
    ),
    ("text", "binary"): (ModelRule("nlp.hf_text", reason="a pretrained encoder is the baseline"),),
    ("text", "multiclass"): (
        ModelRule("nlp.hf_text", reason="a pretrained encoder is the baseline"),
    ),
    ("text", "token_classification"): (
        ModelRule("nlp.hf_token", reason="the only per-token head"),
    ),
    ("text", "seq2seq"): (ModelRule("nlp.hf_seq2seq", reason="the only generative head"),),
    ("timeseries", "forecasting"): (
        ModelRule(
            "ts.lstm",
            min_rows=200,
            reason="enough history for a recurrent model to beat repeating last season",
        ),
        ModelRule(
            "ts.naive",
            downgrade=True,
            reason="too little history to justify a learned model",
        ),
    ),
}


def select_model(data_kind: str, task: str, *, n_rows: int = 0) -> tuple[str, str]:
    """``(model name, the reason)`` for this data, or a refusal with a pip command.

    Availability is checked through the registry rather than by importing
    anything, so this answers correctly on an install that has none of the
    optional runtimes — which is precisely the install where the answer matters
    most, because it is the one that has to produce the ``pip install`` line.
    """
    # Populating the registry is an import side effect, and the plugins package is
    # required to be importable with zero optional dependencies — so this stays
    # answerable on a bare install, which is the install that needs the pip line.
    import ml_framework.plugins  # noqa: F401

    from ..core.plugins import MissingExtraError
    from ..core.registry import MODELS

    candidates = MODEL_RULES.get((data_kind, task))
    if not candidates:
        raise MissingExtraError(
            f"no default model for {data_kind}/{task}. Set model.name explicitly."
        )

    applicable = [rule for rule in candidates if rule.fits(n_rows)] or [candidates[-1]]
    for rule in applicable:
        # `in` rather than `is_available`, which raises for a name the registry has
        # never heard of — possible if a builtin is renamed and this table is not.
        if rule.model in MODELS and MODELS.is_available(rule.model):
            if rule.downgrade:
                log.warning(
                    "falling back to '%s' (%s) — this is a weaker model, not an equivalent one",
                    rule.model,
                    rule.reason,
                )
            return rule.model, rule.reason

    # Nothing in the family is installed. Refuse, naming the fix — never train
    # something else and report its score as the framework's choice.
    best = applicable[0]
    unmet = MODELS.unmet(best.model) if best.model in MODELS else ()
    hint = unmet[0].pip_hint(MODELS.dist) if unmet else f"install {best.model}"
    raise MissingExtraError(
        f"{data_kind}/{task} data wants '{best.model}' and it is not installed. {hint}. "
        f"Refusing to substitute a different model family — a score from one is not "
        f"a score from the other."
    )


__all__ = [
    "DEFAULT_BUDGET_NAME",
    "MODEL_RULES",
    "TUNE_BUDGETS",
    "ModelRule",
    "TuneBudget",
    "budget_for",
    "parse_duration",
    "select_model",
]
