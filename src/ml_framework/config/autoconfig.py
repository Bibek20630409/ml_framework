"""
config/autoconfig.py
────────────────────
``mlf train --data x.csv`` with no YAML — the synthesis half of zero-config.

**This module produces a plain ``dict``, never a validated config.** That single
choice is the whole mechanism, because it lets synthesized values sit in an
ordinary precedence chain instead of being special-cased anywhere downstream:

    plugin/backend defaults < synthesis < YAML file < --set < explicit CLI flags
                                  → ExperimentConfig.model_validate(merged)

Nothing below this line knows a value was inferred rather than typed. ``--config``
and ``--data`` therefore compose freely: point at a CSV, keep a YAML that overrides
two fields, and the two merge by the same rule as any other layer.

The other half is honesty. Every inferred value carries the rule that produced it
(:class:`~ml_framework.data.sniff.Inference`), the CLI logs each one, and
``mlf init`` writes them into the generated file as comments. A framework that
decides things for you and will not say why is worse than one that makes you type.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.sniff import Inference, Sniffed, sniff
from .defaults import select_model

log = logging.getLogger(__name__)

# Where a zero-config run writes its bundle when nobody said otherwise.
DEFAULT_OUTPUT_DIR = "outputs"


@dataclass(frozen=True, slots=True)
class Synthesized:
    """A config dict, and the reasoning that produced every inferred field."""

    config: dict[str, Any]
    sniffed: Sniffed
    inferences: tuple[Inference, ...]

    def log(self) -> None:
        for inference in self.inferences:
            (log.warning if inference.weak else log.info)("inferred %s", inference)


def synthesize(
    path: str | Path,
    *,
    target: str | None = None,
    time_col: str | None = None,
    text_col: str | None = None,
    model: str | None = None,
    output_dir: str | None = None,
) -> Synthesized:
    """Look at ``path`` and build the smallest config that would train on it.

    The model is chosen from a rules table rather than from a default in the
    schema, and an uninstalled family is a refusal with a ``pip install`` line —
    never a substitution. See :func:`~ml_framework.config.defaults.select_model`
    for why that matters more here than anywhere else: a zero-config score looks
    exactly like a score worth acting on.
    """
    found = sniff(path, target=target, time_col=time_col, text_col=text_col)
    inferences = list(found.inferences)

    if model is None:
        model, reason = select_model(found.kind, found.task, n_rows=found.n_rows)
        inferences.append(Inference("model.name", model, reason))
    else:
        inferences.append(Inference("model.name", model, "given explicitly"))

    config: dict[str, Any] = {
        "task": found.task,
        "runtime": {"output_dir": output_dir or DEFAULT_OUTPUT_DIR},
        "data": {"kind": found.kind, "path": found.path},
        "model": {"name": model},
    }
    if found.target is not None:
        config["data"]["target"] = found.target
    if found.time_col is not None:
        # `auto` would resolve to temporal for this kind anyway; naming the column
        # is what the splitter actually needs.
        config["data"]["split"] = {"time_col": found.time_col}
    if found.text_col is not None:
        config["data"]["params"] = {"text_col": found.text_col}
    if found.kind == "tabular":
        # The schema's default is `smote`, kept so an existing v1 tabular config
        # trains exactly as it did. A config synthesized *now* has no such history,
        # and zero-config always picks a tree — which consumes sample weights
        # natively and is measurably hurt by synthetic neighbours. Left at the
        # default, every zero-config run would emit a warning telling the user to
        # set the value this line sets.
        config["data"]["params"] = {"imbalance_strategy": "auto"}
        inferences.append(
            Inference(
                "data.params.imbalance_strategy",
                "auto",
                "resolves against the chosen model: sample weights for a tree, SMOTE otherwise",
            )
        )
    if found.kind in ("image", "audio", "video"):
        # A class-directory corpus needs a held-back directory and there is no way
        # to infer one, so the training folder is reused and the split carves
        # validation and test out of it. Recorded as a weak inference because it is
        # a real compromise: `test_dir` is the honest way to hold samples back.
        config["data"]["params"] = {"test_dir": found.path}
        inferences.append(
            Inference(
                "data.params.test_dir",
                found.path,
                f"no held-out {found.kind} directory was given, so the training "
                "folder is reused",
                weak=True,
            )
        )

    return Synthesized(config=config, sniffed=found, inferences=tuple(inferences))


def merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` onto ``base``; ``overlay`` wins at every leaf.

    Recursive on purpose. A shallow merge would make a YAML that sets only
    ``data.target`` discard the synthesized ``data.kind`` and ``data.path``
    alongside it — turning "override one field" into "replace the whole block",
    which is the behaviour nobody expects from a config layer.
    """
    out = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = merge(current, value)
        else:
            out[key] = value
    return out


def render_config(synthesized: Synthesized) -> str:
    """The synthesized config as YAML, **with the rule beside every inferred field**.

    This is what stops zero-config from being a black box. A generated file that
    just appeared with `task: multiclass` in it invites the reader to assume
    somebody decided that carefully; a comment saying *why* invites them to check.

    Comments are attached by walking the dumped lines and tracking the path stack
    by indentation, rather than by templating the file. Templating would mean every
    new inferred field needs a matching edit here — and the one that gets forgotten
    is the one that silently loses its explanation.
    """
    import yaml

    rules = {inference.field: inference for inference in synthesized.inferences}
    body = yaml.safe_dump(synthesized.config, sort_keys=False, default_flow_style=False)

    out: list[str] = []
    stack: list[tuple[int, str]] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("-"):
            out.append(line)
            continue
        indent = len(line) - len(stripped)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key = stripped.split(":", 1)[0].strip()
        stack.append((indent, key))
        path = ".".join(name for _, name in stack)

        inference = rules.get(path)
        if inference is None:
            out.append(line)
        else:
            marker = "GUESS" if inference.weak else "inferred"
            out.append(f"{line}  # {marker}: {inference.rule}")

    weak = [i for i in synthesized.inferences if i.weak]
    header = [
        f"# Generated by `mlf init` from {synthesized.sniffed.path}",
        "#",
        "# Every inferred field carries the rule that produced it. Check them —",
        "# especially any marked GUESS, which are fallbacks rather than evidence.",
    ]
    if weak:
        header += ["#"] + [f"#   GUESS  {i.field}: {i.rule}" for i in weak]
    return "\n".join([*header, "", *out, ""])


__all__ = ["DEFAULT_OUTPUT_DIR", "Synthesized", "merge", "render_config", "synthesize"]
