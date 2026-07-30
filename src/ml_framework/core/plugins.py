"""
core/plugins.py
───────────────
The generic, spec-carrying plugin registry — plus the specs themselves and the
errors that make an unavailable plugin *actionable* instead of mysterious.

Three rules, each of which exists because of a specific failure mode:

1. **A plugin module must be importable with zero optional dependencies
   installed.** Heavy imports go inside ``build()``/``fit()``, never at module
   scope. That is what lets ``mlf models`` list every plugin on a bare install.
2. **Availability is answered by ``importlib.util.find_spec``** — no import, no
   exception. This replaces ``try: … except Exception: pass`` *structurally*
   rather than with a better except clause: there is no longer an exception to
   swallow, so a genuinely broken builtin can no longer hide behind "torchvision
   is missing".
3. **Nothing fails silently.** A builtin that fails to import is our bug and
   re-raises. A third-party plugin that fails to load is recorded, warned once,
   shown in ``mlf models --all``, and re-raised *chained* if the user actually
   selects it.

Selecting an unavailable plugin raises :class:`MissingExtraError` naming the pip
extra::

    model 'xgboost' requires xgboost>=2.0. Install it with:
    pip install 'ml-framework[gbdt]'

That message is the difference between a framework that feels finished and one
that does not.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .protocols import SearchSpace
from .types import (
    DIST_NAME,
    Capabilities,
    DataKind,
    FrameworkError,
    Payload,
    Requirement,
    Task,
    unmet_requirements,
)

if TYPE_CHECKING:
    from pydantic import BaseModel as PydanticModel

log = logging.getLogger(__name__)

# The entry-point group third-party plugins advertise themselves under. Built now
# even though this release ships as a single package with extras, so splitting
# later stays possible.
ENTRY_POINT_GROUP = "ml_framework.plugins"


# ── Errors ────────────────────────────────────────────────
class MissingExtraError(FrameworkError):
    """A plugin was selected whose optional dependencies are not installed."""


class PluginLoadError(FrameworkError):
    """A plugin module failed to import. Never swallowed — only deferred."""


class DuplicatePluginError(FrameworkError):
    """Two plugins claimed the same name without an explicit override."""


class UnknownPluginError(FrameworkError, KeyError):
    """No plugin registered under that name.

    Also a ``KeyError`` so that ``except KeyError`` around the old
    ``get_model_class`` keeps working.
    """

    def __str__(self) -> str:  # KeyError.__str__ would repr() the message
        return self.args[0] if self.args else ""


class IncompatibleCombinationError(FrameworkError):
    """A (task, data_kind, model) combination that cannot work.

    Raised from the config validator so the failure lands at load time rather than
    40 seconds into data loading.
    """


def check_requirements(
    requires: tuple[Requirement, ...],
    *,
    what: str,
    dist: str = DIST_NAME,
) -> None:
    """Raise :class:`MissingExtraError` if anything in ``requires`` is unmet.

    ``what`` is the subject of the message, e.g. ``"model 'xgboost'"``. The
    resulting text is the documented contract:

        model 'xgboost' requires xgboost>=2.0. Install it with:
        pip install 'ml-framework[gbdt]'

    Deliberately pure ASCII: this text is the first thing a user sees on a bare
    install, and a Windows console whose code page lacks an em dash would render a
    typographic separator as a replacement glyph.
    """
    unmet = unmet_requirements(requires)
    if not unmet:
        return
    needs = ", ".join(r.spec() for r in unmet)
    hints: list[str] = []
    for r in unmet:
        hint = r.pip_hint(dist)
        if hint not in hints:
            hints.append(hint)
    verb = "Install it with" if len(hints) == 1 else "Install them with"
    raise MissingExtraError(f"{what} requires {needs}. {verb}: {'; '.join(hints)}")


# ── Specs ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A registered model: metadata first, constructor second.

    The v1 registry mapped name → class and nothing else, so every downstream
    question ("can this model do regression?", "what does it need installed?",
    "what should HPO tune?") was answered by branching somewhere else. Those
    answers live here now.
    """

    name: str
    # Which fit-loop shape trains this model: "lightning" | "gbdt" | "forecast".
    backend: str
    build: Callable[..., Any]
    """Constructs the model.

    P0: the legacy model class, called ``(input_dim, output_dim, config,
    class_weights)``. P2 switches this to ``build(BuildContext) -> Any`` once the
    v2 config lands; the signature is loose here only for that transition.
    """
    tasks: frozenset[Task] = frozenset()
    data_kinds: frozenset[DataKind] = frozenset()
    requires: tuple[Requirement, ...] = ()
    capabilities: Capabilities = field(default_factory=Capabilities)
    # Keys are DOTTED CONFIG PATHS against the v2 schema, so applying a trial is
    # `config.with_overrides(values)`. Consumed by pipeline/tune.py (P4).
    search_space: SearchSpace = field(default_factory=dict)
    # Escape hatch for spaces a declarative mapping cannot express (conditional
    # dimensions: n_layers → n_units_l{i}). Overrides `search_space` when set.
    suggest: Callable[..., Mapping[str, Any]] | None = None
    # Pydantic model validating `model.params`; frozen + extra="forbid" so typos
    # still error at config-load time. Resolved lazily in P2.
    params_model: type[PydanticModel] | None = None
    # Tie-break for zero-config model selection (higher wins).
    auto_priority: int = 0
    description: str = ""

    def supports(self, task: str | None = None, data_kind: str | None = None) -> bool:
        if task is not None and self.tasks and task not in self.tasks:
            return False
        if data_kind is not None and self.data_kinds and data_kind not in self.data_kinds:
            return False
        return True


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """A registered training backend (one per fit-loop shape).

    ``factory`` is a zero-arg callable returning the
    :class:`~ml_framework.core.protocols.TrainingBackend`; it imports its heavy
    dependencies inside itself, so registering a backend never imports torch.
    """

    name: str
    factory: Callable[[], Any]
    capabilities: Capabilities = field(default_factory=Capabilities)
    requires: tuple[Requirement, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A registered data source — the extension point that replaces datamodules.

    Datamodules stop being an extension point (there is now exactly one, the
    Lightning adapter); *sources* are what users plug in.
    """

    name: str
    data_kind: DataKind
    build: Callable[..., Any]
    """Materializes the data.

    P0: returns the legacy ``FrameworkDataModule`` for ``data.kind``. P1 switches
    this to ``build(config) -> DataBundle``.
    """
    payload: Payload = "arrays"
    requires: tuple[Requirement, ...] = ()
    description: str = ""


SpecT = TypeVar("SpecT", bound="ModelSpec | BackendSpec | SourceSpec")


# ── Registry ──────────────────────────────────────────────
class PluginRegistry(Generic[SpecT]):
    """Name → spec, with availability, discovery and honest error reporting.

    Unlike the v1 registries this holds *specs*, not classes, so a plugin can be
    listed, described and version-checked without importing the library it wraps.
    """

    def __init__(self, kind: str, *, dist: str = DIST_NAME) -> None:
        self.kind = kind  # "model" | "backend" | "source" — used in messages
        self.dist = dist
        self._specs: dict[str, SpecT] = {}
        self._load_errors: dict[str, PluginLoadError] = {}
        self._discovered = False

    # ── registration ──
    def register(self, spec: SpecT, *, override: bool = False) -> SpecT:
        key = spec.name.lower()
        if key in self._specs and not override:
            raise DuplicatePluginError(
                f"{self.kind} '{key}' is already registered "
                f"(pass override=True to replace it deliberately)"
            )
        self._specs[key] = spec
        return spec

    def unregister(self, name: str) -> None:
        """Remove a plugin. Exists for tests; production code registers once."""
        self._specs.pop(name.lower(), None)
        self._load_errors.pop(name.lower(), None)

    # ── lookup ──
    def get_spec(self, name: str) -> SpecT:
        """The spec, **without** checking availability. Safe on a bare install."""
        key = name.lower()
        if key in self._specs:
            return self._specs[key]
        if key in self._load_errors:
            # Recorded at discovery, re-raised chained now that it actually matters.
            raise self._load_errors[key]
        raise UnknownPluginError(f"Unknown {self.kind} '{name}'. Registered: {self.names()}")

    def get(self, name: str) -> SpecT:
        """The spec, refusing with :class:`MissingExtraError` if unavailable."""
        spec = self.get_spec(name)
        check_requirements(spec.requires, what=f"{self.kind} '{spec.name}'", dist=self.dist)
        return spec

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[SpecT]:
        return iter(self.specs())

    # ── introspection ──
    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> tuple[SpecT, ...]:
        return tuple(self._specs[k] for k in self.names())

    def is_available(self, name: str) -> bool:
        return not unmet_requirements(self.get_spec(name).requires)

    def unmet(self, name: str) -> tuple[Requirement, ...]:
        return unmet_requirements(self.get_spec(name).requires)

    def available_names(self) -> list[str]:
        return [n for n in self.names() if self.is_available(n)]

    def load_errors(self) -> Mapping[str, PluginLoadError]:
        return dict(self._load_errors)

    def describe(self) -> list[dict[str, Any]]:
        """Rows for ``mlf models`` / ``mlf backends``: one dict per plugin.

        Includes unavailable plugins with the reason and the fix — listing only
        what happens to be installed would hide the framework's own capabilities.
        """
        rows: list[dict[str, Any]] = []
        for spec in self.specs():
            unmet = unmet_requirements(spec.requires)
            rows.append(
                {
                    "name": spec.name,
                    "kind": self.kind,
                    "available": not unmet,
                    "requires": [r.spec() for r in spec.requires],
                    "missing": [r.unmet_reason() for r in unmet],
                    "install": unmet[0].pip_hint(self.dist) if unmet else None,
                    "description": getattr(spec, "description", ""),
                }
            )
        for name, err in sorted(self._load_errors.items()):
            rows.append(
                {
                    "name": name,
                    "kind": self.kind,
                    "available": False,
                    "requires": [],
                    "missing": [f"failed to load: {err}"],
                    "install": None,
                    "description": "",
                }
            )
        return rows

    # ── third-party discovery ──
    def discover(self, group: str = ENTRY_POINT_GROUP, *, force: bool = False) -> None:
        """Load third-party plugins advertised under ``group``.

        Each entry point is expected to be a callable that performs its own
        registration. A failure is recorded and warned **once** — never swallowed:
        it shows up in ``describe()`` and is re-raised chained by
        :meth:`get_spec` if that plugin is actually selected.
        """
        if self._discovered and not force:
            return
        self._discovered = True
        from importlib.metadata import entry_points

        for ep in entry_points(group=group):
            try:
                loader = ep.load()
                if _takes_registry(loader):
                    loader(self)
                else:
                    loader()
            except Exception as exc:  # noqa: BLE001 - recorded, warned, re-raised later
                err = PluginLoadError(f"plugin '{ep.name}' ({ep.value}) failed to load: {exc}")
                err.__cause__ = exc
                self._load_errors[ep.name.lower()] = err
                warnings.warn(str(err), RuntimeWarning, stacklevel=2)
                log.warning("%s", err, exc_info=exc)


def _takes_registry(fn: Callable[..., Any]) -> bool:
    """Whether an entry-point loader wants the registry passed to it."""
    import inspect

    try:
        return len(inspect.signature(fn).parameters) >= 1
    except (TypeError, ValueError):  # builtins / C callables
        return False
