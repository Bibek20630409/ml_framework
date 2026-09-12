"""
cli.py
──────
`mlf` command-line entry point. All commands load a validated YAML config and
accept dotted ``--set key=value`` overrides.

    mlf models         [--all] [--show]
    mlf backends       [--all] [--show]
    mlf init           --data data/raw/sample.csv -o configs/mine.yaml
    mlf train          --data data/raw/sample.csv          # no YAML at all
    mlf lr             --config configs/example_tabular.yaml
    mlf tune           --config configs/example_tabular.yaml --emit-config configs/tuned.yaml
    mlf select         --config configs/example_tabular.yaml --max-latency-ms 20
    mlf train          --config configs/example_tabular.yaml --set fit.budget.max_epochs=5
    mlf train          --config configs/example_gbdt.yaml --no-tune
    mlf train          --config configs/example_gbdt.yaml --tune-budget 10m --tune-trials 50
    mlf train          --config configs/example_tabular.yaml --select
    mlf serve          --artifacts outputs --host 0.0.0.0 --port 8000
    mlf export         --artifacts outputs --format onnx -o model.onnx
    mlf dockerfile     --artifacts outputs -o Dockerfile.serve
    mlf migrate-config -i configs/old.yaml -o configs/new.yaml

``mlf train`` **tunes by default**, under a per-backend budget (see
``config/defaults.py``) — 300 s for trees, 900 s for neural nets. ``--no-tune``
skips it; ``--tune-budget``/``--tune-trials`` turn it up just as easily.

``mlf select`` **does not run by default**, and that asymmetry is deliberate: a
bake-off costs one tuning budget *per candidate family*, so it is a decision to
make rather than one to inherit. It compares every compatible family on score,
measured latency, artifact size, explainability and fold stability, and prints
the table. ``mlf train --select`` does the same and then trains the winner.
``--candidate``/``--collect`` split the same work across an orchestrator — see
``orchestration/airflow/dags/ml_pipeline.py``.

**Zero-config.** ``--data`` synthesizes a config by looking at the file, so
``mlf train --data x.csv`` needs no YAML. Every inferred field is logged with the
rule that produced it, and ``--data`` composes with ``--config``: synthesis is
just one layer of an ordinary precedence chain,

    plugin defaults < synthesis < YAML file < --set < explicit CLI flags

so a YAML that sets two fields overrides exactly those two. A run whose model the
*framework* chose also scores the trivial baseline and warns if the model fails to
beat it — see ``core/baseline.py`` for why that is the guard that matters most
here.

``mlf models`` and ``mlf backends`` take no config at all: they answer "what can
this install train, and what would it take to widen that" from the plugin registry
alone. It deliberately lists models whose optional extra is **missing**, with the
``pip install`` line that fixes each — a listing of only what happens to be
installed would describe the machine rather than the framework, and would make an
uninstalled extra indistinguishable from a model that does not exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .core.export import EXPORT_FORMATS
from .core.types import DECODER_STAGES, FrameworkError
from .utils import setup_logging

log = logging.getLogger(__name__)


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got '{raw}'")
    key, value = raw.split("=", 1)
    try:
        parsed = json.loads(value)  # numbers, bools, lists, null
    except json.JSONDecodeError:
        parsed = value  # plain string
    return key.strip(), parsed


def _add_data_args(sub: argparse.ArgumentParser) -> None:
    """The zero-config surface: point at data, let the framework read it."""
    sub.add_argument(
        "--data",
        "-d",
        default=None,
        help="Dataset to infer a config from (CSV/Parquet/JSONL, or an image folder)",
    )
    sub.add_argument("--target", default=None, help="Target column (else inferred)")
    sub.add_argument("--time-col", default=None, help="Time column, forcing a time series")
    sub.add_argument("--text-col", default=None, help="Text column, forcing a text corpus")
    sub.add_argument("--model", default=None, help="Model name (else chosen from the data)")
    sub.add_argument("--output-dir", default=None, help="Where the bundle is written")


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the config from whichever layers were supplied.

    The precedence chain is the whole zero-config mechanism, and it is deliberately
    ordinary: synthesis produces a plain dict that a YAML file is merged *over*,
    exactly as if the user had typed the synthesized values first. Nothing below
    here can tell which layer a value came from — that is what lets ``--data`` and
    ``--config`` compose instead of being alternatives.
    """
    import yaml

    layers: list[dict[str, Any]] = []
    chosen: str | None = None
    if getattr(args, "data", None):
        from .config.autoconfig import synthesize

        synthesized = synthesize(
            args.data,
            target=getattr(args, "target", None),
            time_col=getattr(args, "time_col", None),
            text_col=getattr(args, "text_col", None),
            model=getattr(args, "model", None),
            output_dir=getattr(args, "output_dir", None),
        )
        synthesized.log()
        chosen = synthesized.config["model"]["name"]
        layers.append(synthesized.config)

    if getattr(args, "config", None):
        with open(args.config, encoding="utf-8") as handle:
            layers.append(yaml.safe_load(handle) or {})
    elif not layers:
        raise SystemExit("nothing to load: pass --config, or --data to infer one")

    from .config.autoconfig import merge

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = merge(merged, layer)

    cfg = ExperimentConfig.model_validate(merged)
    if args.set:
        cfg = cfg.with_overrides(dict(args.set))

    # After `--set`, per the documented precedence: an explicit flag is the last
    # word. `None` means the flag was not passed at all, which is distinct from a
    # user asking for `local` -- the same "None means unset" rule the tune flags use.
    if getattr(args, "data_backend", None):
        cfg = cfg.with_overrides({"data.backend": args.data_backend})

    # The staged-read and transport flags, folded in the same way. Table-driven
    # rather than six `if`s, borrowed from `_apply_select_args`: the mapping IS the
    # information, and a flag added without a config path is then a visibly missing
    # row rather than a silently absent branch.
    transport = {
        flag: key
        for flag, key in (
            ("decoder", "data.params.decoder"),
            ("on_corrupt", "data.integrity.on_corrupt"),
            ("shuffle", "data.shards.shuffle"),
            ("num_workers", "runtime.num_workers"),
            ("prefetch_factor", "runtime.prefetch_factor"),
            ("pin_memory", "runtime.pin_memory"),
            ("device_transform", "runtime.device_transform"),
        )
        if getattr(args, flag, None) is not None
    }
    if transport:
        cfg = cfg.with_overrides({key: getattr(args, flag) for flag, key in transport.items()})

    # Whether the *framework* chose the model, which is what decides if the trivial
    # baseline is worth scoring. Compared against the surviving value rather than
    # set when synthesis ran: a later layer -- a YAML, a --set, an explicit --model
    # -- may have overridden the choice, and then the user has their own frame of
    # reference and the baseline is not the point. Recorded on `args` rather than
    # in the config because it is a fact about this invocation, not the experiment.
    args.auto_selected = (
        chosen is not None and getattr(args, "model", None) is None and cfg.model.name == chosen
    )
    return cfg


def _add_config_args(sub: argparse.ArgumentParser) -> None:
    # Not required: `--data` can supply the config instead, and both together is a
    # valid and useful combination rather than a conflict.
    sub.add_argument("--config", "-c", default=None, help="Path to YAML config")
    _add_data_args(sub)
    sub.add_argument(
        "--set",
        action="append",
        type=_parse_override,
        default=[],
        metavar="key=value",
        help="Override a config value (dotted key), repeatable",
    )
    # Here rather than in `_add_data_args` because that is the *synthesis* surface
    # (also used by `mlf init`), and which engine to run on is a choice about this
    # invocation, not a fact inferred from the data file.
    sub.add_argument(
        "--data-backend",
        default=None,
        metavar="NAME",
        help="Data-processing engine for this run: local (default), polars or spark",
    )
    # Staged-read controls. Like `--data-backend`, these are choices about *this
    # invocation* rather than facts inferred from the data, so they live here and
    # not on the synthesis surface. Every one defaults to None so "the user asked
    # for this" stays distinguishable from "nobody said".
    sub.add_argument(
        "--decoder",
        default=None,
        metavar="NAME",
        # No `choices=`: the set is open and validated by the registry, exactly as
        # `--data-backend` is. `mlf decoders` lists what is available.
        help="Media decoder for this run. Resolved from the corpus when omitted",
    )
    sub.add_argument(
        "--on-corrupt",
        default=None,
        choices=("substitute", "raise"),
        help="A corrupt sample is substituted (default) or stops the run. Never skipped",
    )
    sub.add_argument(
        "--shuffle",
        default=None,
        choices=("block", "global", "none"),
        help="Shard shuffle strategy: block (default, keeps reads local), global, none",
    )
    sub.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="DataLoader workers (-1 = auto: 0 on Windows, else 4)",
    )
    sub.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        metavar="N",
        help="Batches each worker prefetches (default 2). Ignored without workers",
    )
    sub.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Page-lock staging buffers for an async H2D copy (CUDA only)",
    )
    sub.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Never page-lock, even on CUDA",
    )
    sub.add_argument(
        "--device-transform",
        dest="device_transform",
        action="store_true",
        default=None,
        help="Run the transform stage after H2D, on the device (CUDA only; the default)",
    )
    sub.add_argument(
        "--no-device-transform",
        dest="device_transform",
        action="store_false",
        help="Keep the transform in the collate, in a worker, even on CUDA",
    )


def _add_tune_args(sub: argparse.ArgumentParser) -> None:
    """Tuning controls. Turning it *up* is as easy as turning it off, by design."""
    sub.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        default=None,
        help="Skip hyperparameter search and train the configured parameters",
    )
    sub.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        metavar="N",
        help="Trials to run (default: per-backend, see config/defaults.py)",
    )
    sub.add_argument(
        "--tune-budget",
        default=None,
        metavar="DURATION",
        help="Wall-clock budget for the search: 900, 30s, 10m, 2h",
    )


def _apply_tune_args(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Fold the tuning flags into the config, so there is one source of truth.

    Flags default to ``None`` rather than to the schema's values: that is what
    distinguishes "the user asked for this" from "nobody said", which is exactly
    the distinction the per-backend budget defaults need.
    """
    from .config.defaults import parse_duration

    overrides: dict[str, Any] = {}
    if getattr(args, "tune", None) is False:
        overrides["tune.enabled"] = False
    if getattr(args, "tune_trials", None) is not None:
        overrides["tune.max_trials"] = args.tune_trials
    if getattr(args, "tune_budget", None) is not None:
        overrides["tune.max_seconds"] = parse_duration(args.tune_budget)
    return cfg.with_overrides(overrides) if overrides else cfg


def _add_select_args(sub: argparse.ArgumentParser) -> None:
    """Bake-off controls. Every flag maps to exactly one ``select.*`` config key,
    so the CLI and the YAML cannot describe different runs."""
    sub.add_argument(
        "--candidates",
        default=None,
        metavar="A,B,C",
        help="Comma-separated model families to compare (default: every compatible one)",
    )
    sub.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        metavar="N",
        help="Cap the pool after gating, in priority order",
    )
    sub.add_argument(
        "--objective",
        choices=("tolerance", "weighted"),
        default=None,
        help="How the winner is chosen (default: tolerance)",
    )
    sub.add_argument(
        "--tolerance",
        type=float,
        default=None,
        metavar="D",
        help="Score difference counted as noise (default: one std error of the CV mean)",
    )
    sub.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        metavar="MS",
        help="Disqualify a candidate whose measured p95 single-row latency exceeds this",
    )
    sub.add_argument(
        "--max-model-mb",
        type=float,
        default=None,
        metavar="MB",
        help="Disqualify a candidate whose serialized artifact exceeds this",
    )
    sub.add_argument(
        "--min-explainability",
        type=float,
        default=None,
        metavar="S",
        help="Disqualify below this attribution tier (1.0 native, 0.8 shap, 0.5 permutation)",
    )
    sub.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate this many candidates concurrently, as separate processes",
    )
    sub.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        default=None,
        help="Skip latency/size/explainability measurement and compare on score alone",
    )


def _apply_select_args(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Fold the bake-off flags into the config. Same ``None``-means-unset rule as
    :func:`_apply_tune_args`."""
    overrides: dict[str, Any] = {}
    if getattr(args, "candidates", None):
        overrides["select.candidates"] = [
            c.strip() for c in args.candidates.split(",") if c.strip()
        ]
    for flag, key in (
        ("max_candidates", "select.max_candidates"),
        ("objective", "select.objective"),
        ("tolerance", "select.tolerance"),
        ("max_workers", "select.max_workers"),
        ("profile", "select.profile"),
        ("max_latency_ms", "select.constraints.max_latency_p95_ms"),
        ("max_model_mb", "select.constraints.max_model_mb"),
        ("min_explainability", "select.constraints.min_explainability"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[key] = value
    return cfg.with_overrides(overrides) if overrides else cfg


def _format_search_space(spec: Any) -> list[str]:
    """The model's own declared search space, one line per knob.

    Deliberately *not* the effective space: merging in the backend's would mean
    importing the backend, and importing the Lightning backend imports torch — on
    a bare install that would turn `mlf models --show` into the one command that
    cannot run. The lines say which half they are.
    """
    if spec.suggest is not None:
        return ["search space: defined in code (conditional on other values)"]
    if not spec.search_space:
        return ["search space: none of its own (the backend's still applies)"]
    return ["search space (this model's own; the backend adds lr/batch_size):"] + [
        f"  {key} = {value}" for key, value in sorted(spec.search_space.items())
    ]


def _print_plugins(registry: Any, *, include_failed: bool, show_detail: bool) -> int:
    """``mlf models`` / ``mlf backends`` — a plugin registry, rendered.

    Reads the registry and nothing else, so it works on an install with no optional
    extra present at all. That is the property the whole plugin design protects:
    registering a plugin must not import its runtime, and these commands are what
    make the property visible.

    One renderer for both because the interesting columns are the same ones —
    what it is, whether it is ready, and what would make it ready. Only the second
    column differs, and a spec answers that about itself.
    """
    rows = registry.describe()
    failed = set(registry.load_errors())
    if not include_failed:
        # A plugin that failed to *import* is a different thing from one whose
        # extra is missing, and mixing them would make a genuine bug look like an
        # uninstalled dependency. Hidden by default, never swallowed.
        rows = [row for row in rows if row["name"] not in failed]

    if not rows:
        # Derived, not hardcoded: this printed "no models registered" for every
        # registry, including `mlf backends`.
        print(f"no {registry.kind}s registered")
        return 0

    label = registry.kind.upper()
    # The header counts toward the column width: "DATA BACKEND" is longer than any
    # engine name, and sizing on the names alone ragged-edged every row under it.
    width = max(*(len(row["name"]) for row in rows), len(label))
    pad = " " * (width + 9)
    second = {"model": "BACKEND", "data backend": "ENGINE", "decoder": "INTEGRITY"}.get(
        registry.kind, "ACCEPTS"
    )
    # 14 wide: a backend accepting two payloads prints "arrays,dataset", and a
    # narrower column would ragged-edge every description beside it.
    print(f"{label.ljust(width)}  READY  {second:<14}  DESCRIPTION")
    for row in sorted(rows, key=lambda r: r["name"]):
        spec = None if row["name"] in failed else registry.get_spec(row["name"])
        detail = "-" if spec is None else _second_column(registry.kind, spec)
        ready = "yes  " if row["available"] else "NO   "
        print(f"{row['name'].ljust(width)}  {ready}  {detail:<14}  {row['description']}")

        for reason in row["missing"]:
            print(f"{pad}{reason}")
        if row["install"]:
            print(f"{pad}fix: {row['install']}")

        if show_detail and spec is not None:
            for line in _format_detail(registry.kind, spec):
                print(f"{pad}{line}")

    hidden = len(failed) if not include_failed else 0
    if hidden:
        # The command name, derived from the registry rather than assumed to be one
        # of two: "data backend" -> `mlf data-backends --all`.
        verb = registry.kind.replace(" ", "-") + "s"
        print(f"\n{hidden} plugin(s) failed to load and are hidden; `mlf {verb} --all` shows them")
    return 0


def _second_column(kind: str, spec: Any) -> str:
    """What a model rides on, what a backend consumes, or what an engine runs on."""
    if kind == "model":
        return str(spec.backend)
    if kind == "data backend":
        # A DataBackendSpec carries no `capabilities`: every flag on that class is
        # about a model or a fit loop. Reaching for one here raised AttributeError.
        return str(spec.engine) or "-"
    if kind == "decoder":
        # Integrity, not `lands_in`: this is the column the staged pipeline exists
        # for, and "how does this format fail" is what an operator scanning the
        # table needs to see without asking for --show. Every value fits in 14.
        return str(spec.integrity) or "-"
    return ",".join(sorted(spec.capabilities.accepts)) or "-"


def _format_detail(kind: str, spec: Any) -> list[str]:
    if kind == "model":
        return [
            f"tasks: {', '.join(sorted(spec.tasks)) or 'any'}",
            f"data:  {', '.join(sorted(spec.data_kinds)) or 'any'}",
            *_format_search_space(spec),
        ]
    if kind == "data backend":
        return [f"engine: {spec.engine}"]
    if kind == "decoder":
        lines = [
            f"kind:   {spec.data_kind}",
            f"lands:  {spec.lands_in} memory",
            f"output: {spec.output_dtype} {spec.output_layout}",
            # Sorted read/demux/decode rather than alphabetically: the pipeline
            # order is the information, and "decode,demux,read" reads backwards.
            # The decoder's HEAD stages only -- the tail (construct, transform,
            # h2d, gpu_transform) belongs to the preprocessor and the transport
            # layer, and listing it per decoder would attribute it to the wrong
            # thing.
            f"stages: {','.join(s for s in DECODER_STAGES if s in spec.stages)}",
        ]
        if spec.suffixes:
            lines.append(f"files:  {' '.join(spec.suffixes)}")
        if spec.oracle:
            lines.append(f"oracle: {spec.oracle} (cross-checked at materialization)")
        if spec.integrity_note:
            # Verbatim, wrapped. This is the sentence the whole command exists to
            # deliver -- an operator should be able to learn from it that MP3
            # resyncs silently and that a token shard has no integrity at all.
            lines.extend(textwrap.wrap(spec.integrity_note, width=76))
        return lines
    caps = spec.capabilities
    supported = [
        name
        for name, value in (
            ("gpu", caps.supports_gpu),
            ("mixed-precision", caps.supports_mixed_precision),
            ("pruning", caps.supports_pruning),
            ("resume", caps.supports_resume),
            ("sample-weight", caps.supports_sample_weight),
            ("lr-range-test", caps.supports_lr_range_test),
        )
        if value
    ]
    return [f"supports: {', '.join(supported) or 'nothing beyond a plain fit'}"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlf", description="ML Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # The one command that takes no config: it describes the framework, not a run.
    models_p = sub.add_parser("models", help="List registered models and what they need")
    models_p.add_argument(
        "--all",
        action="store_true",
        help="Also list plugins that failed to import (a bug, not a missing extra)",
    )
    models_p.add_argument(
        "--show",
        action="store_true",
        help="Add each model's tasks, data kinds and its own search space",
    )

    backends_p = sub.add_parser("backends", help="List registered backends and what they need")
    backends_p.add_argument("--all", action="store_true", help="Also list failed imports")
    backends_p.add_argument("--show", action="store_true", help="Add each backend's capabilities")

    # Its own verb rather than a flag on `backends`: that command answers "what
    # fit-loop shapes exist", and a flag that silently made it answer a different
    # question would be worse than a second command. This one also shows up in
    # `mlf --help`, which a flag does not.
    data_backends_p = sub.add_parser(
        "data-backends", help="List data-processing engines (local, polars, spark)"
    )
    data_backends_p.add_argument("--all", action="store_true", help="Also list failed imports")
    data_backends_p.add_argument("--show", action="store_true", help="Add each engine's detail")

    # Sibling of `data-backends`, and the same shape: a registry rendered from
    # specs alone, so it lists the H.264 path and its pip line on an install with
    # no FFmpeg binding. The second column is INTEGRITY — how each format fails —
    # because that is the column the staged pipeline exists for.
    decoders_p = sub.add_parser(
        "decoders", help="List media decoders, what they output, and how each fails"
    )
    decoders_p.add_argument("--all", action="store_true", help="Also list failed imports")
    decoders_p.add_argument(
        "--show",
        action="store_true",
        help="Add each decoder's stages, output dtype/layout and integrity note",
    )

    # The offline pass. It exists because two of the four integrity classes are
    # invisible at training time: an MP3 resyncs past damage and a hardware
    # decoder emits green frames, both producing correctly-shaped tensors that no
    # handler will ever see. Probing once, here, is the only way to catch them.
    materialize_p = sub.add_parser(
        "materialize",
        help="Decode-probe a corpus, write its shard index, and report what is unreadable",
    )
    materialize_p.add_argument("--data", "-d", metavar="PATH", help="Corpus directory")
    materialize_p.add_argument("--config", "-c", metavar="PATH", help="Config naming the corpus")
    materialize_p.add_argument(
        "--decoder",
        default=None,
        metavar="NAME",
        # No `choices=`: the set is open and validated by the registry, exactly as
        # `--data-backend` is. An unknown name lists what is registered.
        help="Decoder to probe with. Inferred from the corpus's suffixes when omitted",
    )
    materialize_p.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        metavar="F",
        help="Probe this fraction of the corpus for a quick check (default 1.0 = all)",
    )
    materialize_p.add_argument(
        "--max-fault-rate",
        type=float,
        default=None,
        metavar="F",
        help="Exit non-zero above this fraction (default: data.integrity.max_fault_rate)",
    )
    materialize_p.add_argument(
        "--force", action="store_true", help="Rewrite an index that already exists"
    )
    # The walk above covers read/demux/decode. This covers the other four stages,
    # which are per-batch and therefore invisible to a per-sample probe -- and it
    # needs a config, because the preprocessor that owns them is a property of the
    # run's data kind rather than of the corpus on disk.
    materialize_p.add_argument(
        "--probe-full",
        action="store_true",
        help="Also push one batch through construct/transform/h2d/gpu_transform (needs --config)",
    )

    init_p = sub.add_parser("init", help="Write a config inferred from a dataset")
    _add_data_args(init_p)
    init_p.add_argument("--output", "-o", required=True, metavar="PATH", help="Where to write it")
    init_p.add_argument("--force", action="store_true", help="Overwrite an existing file")

    export_p = sub.add_parser("export", help="Convert a trained bundle to another format")
    export_p.add_argument("--artifacts", "-a", default="outputs", help="Artifact bundle dir")
    export_p.add_argument(
        "--format",
        "-f",
        required=True,
        choices=list(EXPORT_FORMATS),
        help="Target format. Refused, never substituted, where the backend cannot produce it",
    )
    export_p.add_argument("--output", "-o", required=True, metavar="PATH", help="Where to write")

    docker_p = sub.add_parser(
        "dockerfile", help="Write a Dockerfile tailored to one bundle's dependencies"
    )
    docker_p.add_argument("--artifacts", "-a", default="outputs", help="Artifact bundle dir")
    docker_p.add_argument("--output", "-o", default="Dockerfile.serve", metavar="PATH")
    docker_p.add_argument("--force", action="store_true", help="Overwrite an existing file")

    lr = sub.add_parser("lr", help="Run the LR range test")
    _add_config_args(lr)

    train_p = sub.add_parser("train", help="Select (if enabled), tune (unless disabled) and train")
    _add_config_args(train_p)
    _add_tune_args(train_p)
    train_p.add_argument(
        "--select",
        dest="select",
        action="store_true",
        default=None,
        help="Compare every compatible model family and train the winner",
    )
    _add_select_args(train_p)
    train_p.add_argument(
        "--emit-config",
        default=None,
        metavar="PATH",
        help="Write the effective (post-tuning) config as YAML, for committing back",
    )
    train_p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="CKPT",
        help="Continue from a checkpoint (default: <output_dir>/model/last.ckpt)",
    )
    train_p.add_argument(
        "--folds",
        type=int,
        default=None,
        metavar="K",
        help="Cross-validate over K folds before the final fit (0 = holdout only)",
    )

    # `tune` searches and reports without fitting the winner at full budget.
    # `hpo` is kept as an alias: removing a verb people have in scripts is a
    # gratuitous break, and the new driver answers the same question.
    for name, help_text in (
        ("tune", "Search hyperparameters and report the best config"),
        ("hpo", "Alias for `tune` (the v1 name)"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_config_args(p)
        _add_tune_args(p)
        p.add_argument("--emit-config", default=None, metavar="PATH", help="Write the winner")

    # `select` runs the bake-off and reports, without fitting the winner at full
    # budget or writing a bundle — the same relationship `tune` has to `train`.
    select_p = sub.add_parser(
        "select", help="Compare model families on score, latency, size and explainability"
    )
    _add_config_args(select_p)
    _add_tune_args(select_p)
    _add_select_args(select_p)
    select_p.add_argument(
        "--emit-config",
        default=None,
        metavar="PATH",
        help="Write the winning config as YAML, for committing back",
    )
    select_p.add_argument(
        "--candidate",
        default=None,
        metavar="MODEL",
        help=(
            "Evaluate exactly one family and write its report to --report-dir, "
            "without deciding. The fan-out half of a parallel orchestration."
        ),
    )
    select_p.add_argument(
        "--collect",
        default=None,
        metavar="DIR",
        help="Decide from the reports already written under DIR. The fan-in half.",
    )
    select_p.add_argument(
        "--report-dir",
        default=None,
        metavar="DIR",
        help="Where --candidate writes its report (default: <output_dir>/reports)",
    )

    serve = sub.add_parser("serve", help="Serve a trained model via FastAPI")
    serve.add_argument("--artifacts", "-a", default="outputs", help="Artifact bundle dir")
    serve.add_argument("--registry-model", default=None, help="Load from MLflow registry by name")
    serve.add_argument(
        "--registry-stage", default="production", help="Registry stage/alias/version"
    )
    serve.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1, help="Server worker processes")
    serve.add_argument("--api-key", default=None, help="API key (prefer the MLF_API_KEY env var)")
    serve.add_argument("--rate-limit", default="60/minute", help="Per-client rate limit")
    serve.add_argument("--max-instances", type=int, default=10_000, help="Max instances/request")

    migrate = sub.add_parser("migrate-config", help="Convert a v1 YAML config to the v2 schema")
    migrate.add_argument("--input", "-i", required=True, help="Path to the v1 YAML config")
    migrate.add_argument("--output", "-o", required=True, help="Where to write the v2 config")
    migrate.add_argument(
        "--force", action="store_true", help="Overwrite the output file if it exists"
    )
    migrate.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help=(
            "Skip validating the result. Use when the config selects a model whose "
            "optional extra is not installed on this machine."
        ),
    )

    return parser


def _run_select(cfg: ExperimentConfig, args: argparse.Namespace) -> int:
    """``mlf select`` — the whole bake-off, or one half of a distributed one.

    Three modes, and the split exists so the same decision rule runs whether the
    candidates were evaluated in one process or across an Airflow fan-out:

    ``--candidate M``  evaluate M alone, write ``<report-dir>/M.json``, decide
                       nothing. One mapped task's worth of work.
    ``--collect DIR``  read every report in DIR, apply the constraints and the
                       decision rule, print the winner. The reduce step.
    *(neither)*        do both here, sequentially or across ``--max-workers``.
    """
    import json

    from .pipeline.select import (
        SelectionError,
        SelectionResult,
        apply_constraints,
        candidate_models,
        decide,
        evaluate_candidate,
        select,
    )

    cfg = _apply_select_args(cfg, args)
    out = Path(cfg.runtime.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else out / "reports"

    try:
        if args.candidate:
            report = evaluate_candidate(cfg, args.candidate, output_dir=out)
            report_dir.mkdir(parents=True, exist_ok=True)
            destination = report_dir / f"{args.candidate.replace('.', '_')}.json"
            destination.write_text(
                json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
            )
            print(f"wrote {destination}")
            if not report.eligible:
                # A gated-out candidate is a normal result of a fan-out task, not
                # a failed one: exiting non-zero would fail the whole DAG because
                # one family of five was too slow.
                print(f"note: {args.candidate} produced no usable profile")
            return 0

        if args.collect:
            reports = _load_reports(Path(args.collect))
            if not reports:
                log.error("no candidate reports found under %s", args.collect)
                return 1
            reports = apply_constraints(reports, cfg)
            winner, reason = decide(reports, cfg)
            result = SelectionResult(
                config=_winning_config(cfg, winner),
                tuning=select(_winning_config(cfg, winner)).tuning,
                winner=winner.model,
                reason=reason,
                objective=cfg.select.objective,
                metric=winner.profile.primary_metric if winner.profile else "",
                reports=reports,
            )
        else:
            cfg = cfg.with_overrides({"select.enabled": True})
            if not cfg.select.candidates:
                # Resolve now so the log names the pool before spending on it.
                log.info("candidate pool: %s", ", ".join(candidate_models(cfg)))
            result = select(cfg)
    except (SelectionError, FrameworkError) as exc:
        log.error("%s", exc)
        return 1

    print()
    print(result.table())
    print()
    print(f"winner: {result.winner} — {result.reason}")
    if result.reports and result.winner:
        for path, value in sorted(
            next(r for r in result.reports if r.model == result.winner).best_params.items()
        ):
            print(f"  {path}: {value}")

    if args.emit_config:
        from .pipeline.train import _emit_config

        _emit_config(result.config, args.emit_config)
        print(f"\nwrote {args.emit_config}")
    else:
        print("\nRe-run `mlf train --select` to train the winner, or --emit-config to save it.")
    return 0


def _load_reports(directory: Path) -> list[Any]:
    """Rehydrate ``CandidateReport``s written by ``--candidate`` runs.

    Reads plain JSON rather than a pickle: the writer and the reader are separate
    processes on separate machines in the orchestrated case, and a pickle would
    couple them to one interpreter version and one framework version.
    """
    import json

    from .core.profile import (
        CostProfile,
        LatencyProfile,
        MaintainabilityProfile,
        ModelProfile,
    )
    from .pipeline.select import CandidateReport

    reports: list[Any] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping unreadable report %s (%s)", path, exc)
            continue
        if "model" not in raw:
            continue  # not a candidate report (selection.json, metrics.json, …)

        profile = None
        if raw.get("profile"):
            p = raw["profile"]
            explain = p.get("explainability") or {}
            profile = ModelProfile(
                model=p.get("model", raw["model"]),
                backend=p.get("backend", ""),
                primary_metric=p.get("primary_metric", ""),
                score=_nan(p.get("score")),
                score_std=_nan(p.get("score_std")) if p.get("score_std") is not None else 0.0,
                n_folds=int(p.get("n_folds", 0)),
                metrics={k: _nan(v) for k, v in (p.get("metrics") or {}).items()},
                latency=LatencyProfile(**_only(p.get("latency"), LatencyProfile)),
                cost=CostProfile(**_only(p.get("cost"), CostProfile)),
                explainability=float(explain.get("score", 0.0)),
                explain_method=str(explain.get("method", "none")),
                maintainability=MaintainabilityProfile(
                    **_only(p.get("maintainability"), MaintainabilityProfile)
                ),
            )
        reports.append(
            CandidateReport(
                model=raw["model"],
                backend=raw.get("backend", ""),
                profile=profile,
                best_params=raw.get("best_params") or {},
                n_trials=int(raw.get("n_trials", 0)),
                tune_skipped=raw.get("tune_skipped"),
                elapsed=float(raw.get("elapsed_seconds", 0.0)),
                skipped=raw.get("skipped"),
                disqualified=raw.get("disqualified"),
            )
        )
    return reports


def _only(raw: Any, cls: type) -> dict[str, Any]:
    """``raw`` narrowed to ``cls``'s own fields.

    The serialized form carries derived values (``artifact_mb``, ``score``) that
    are properties, not fields. Filtering rather than popping known extras means
    a future derived field does not break the reader.
    """
    from dataclasses import fields

    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (raw or {}).items() if k in names}


def _nan(value: Any) -> float:
    """JSON ``null`` back to ``NaN`` — the inverse of ``profile._jsonable``."""
    return float("nan") if value is None else float(value)


def _winning_config(cfg: ExperimentConfig, winner: Any) -> ExperimentConfig:
    return cfg.with_overrides(
        {"model.name": winner.model, "select.enabled": False, **winner.best_params}
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.command in ("models", "backends", "data-backends", "decoders"):
        # Importing the plugin package is what populates the registry, and it is
        # required to be dependency-free — see plugins/__init__.py.
        import ml_framework.plugins  # noqa: F401

        from . import backends as _backends  # noqa: F401  (registers the backends)
        from .core.registry import BACKENDS, DATA_BACKENDS, DECODERS, MODELS
        from .data import backends as _data_backends  # noqa: F401  (registers the engines)
        from .data import streaming as _streaming  # noqa: F401  (registers the decoders)

        registry = {
            "models": MODELS,
            "backends": BACKENDS,
            "data-backends": DATA_BACKENDS,
            "decoders": DECODERS,
        }[args.command]
        return _print_plugins(registry, include_failed=args.all, show_detail=args.show)

    if args.command == "materialize":
        from .data.streaming.materialize import MaterializeError, materialize
        from .data.streaming.shards import ShardIndex

        # `--data` alone is enough (this is a corpus-level operation, not a run),
        # but a config supplies the decoder params and the fault ceiling.
        cfg = None
        if args.config:
            cfg = ExperimentConfig.from_yaml(args.config)
        path = args.data or (cfg.data.path if cfg else None)
        if not path:
            log.error("materialize needs --data PATH or a --config naming data.path")
            return 1

        index_dir = ShardIndex.location(path, cfg.data.shards.index_dir if cfg else None)
        if ShardIndex.exists(index_dir) and not args.force:
            log.error(
                "a shard index already exists at %s. It is a property of the corpus, not of "
                "a run, so two runs are meant to share it -- pass --force to rebuild.",
                index_dir,
            )
            return 1

        ceiling = args.max_fault_rate
        if ceiling is None:
            ceiling = cfg.data.integrity.max_fault_rate if cfg else 0.01

        # The tail stages belong to a preprocessor, and which preprocessor is a
        # property of the *run* rather than of the bytes on disk -- so unlike the
        # decode probe, this one cannot be inferred from the corpus.
        tail_preprocessor = None
        if args.probe_full:
            if cfg is None:
                log.error(
                    "--probe-full needs --config: the construct/transform stages belong to "
                    "the data kind's preprocessor, which a corpus directory does not state."
                )
                return 1
            from .data.sources import preprocessor_for

            try:
                tail_preprocessor = preprocessor_for(cfg)
            except ValueError as exc:
                log.error("%s", exc)
                return 1

        # Not `dict(cfg.data.decoder_params)`: a video decoder needs the clip
        # geometry up front so it can stop early, and that lives in `data.params`.
        # Passing the raw dict made every video materialization decode at the
        # decoder's default geometry rather than the configured one.
        from .data.sources import decoder_params_for

        try:
            _, report = materialize(
                path,
                index_dir=index_dir,
                decoder=args.decoder,
                decoder_params=decoder_params_for(cfg) if cfg else None,
                data_kind=cfg.data.kind if cfg else "tabular",
                max_fault_rate=ceiling,
                sample_rate=args.sample_rate,
                seed=cfg.runtime.seed if cfg else 42,
                tail_preprocessor=tail_preprocessor,
            )
        except MaterializeError as exc:
            # The one place the fault ceiling raises. Safe here and nowhere else:
            # this is a single process, running before any collective exists.
            log.error("%s", exc)
            return 1
        except FrameworkError as exc:
            log.error("%s", exc)
            return 1

        print(report.render())
        print(f"\nwrote {index_dir}")
        return 0

    if args.command == "export":
        from .core.export import UnsupportedExportError
        from .core.inference import Inferencer
        from .core.registry import get_backend

        try:
            inf = Inferencer.from_artifacts(args.artifacts)
            backend = get_backend(inf.backend_name)
            result = backend.export(
                inf.estimator, Path(args.output), args.format, manifest=inf.manifest
            )
        except (UnsupportedExportError, FrameworkError, FileNotFoundError) as exc:
            log.error("%s", exc)
            return 1

        print(f"wrote {result.path} ({result.format})")
        if result.notes:
            # Not decoration: "preprocessing is NOT included" is the difference
            # between an artifact that works and one that scores nonsense.
            print(f"note: {result.notes}")
        return 0

    if args.command == "dockerfile":
        from .deploy import render_dockerfile

        destination = Path(args.output)
        if destination.exists() and not args.force:
            log.error("%s exists; pass --force to overwrite", destination)
            return 1
        try:
            text = render_dockerfile(args.artifacts)
        except (FrameworkError, FileNotFoundError) as exc:
            log.error("%s", exc)
            return 1
        destination.write_text(text, encoding="utf-8")
        print(f"wrote {destination}")
        return 0

    if args.command == "init":
        from .config.autoconfig import render_config, synthesize

        if not args.data:
            log.error("`mlf init` needs --data: there is nothing to infer a config from")
            return 1
        try:
            synthesized = synthesize(
                args.data,
                target=args.target,
                time_col=args.time_col,
                text_col=args.text_col,
                model=args.model,
                output_dir=args.output_dir,
            )
        except FrameworkError as exc:
            log.error("%s", exc)
            return 1
        synthesized.log()

        destination = Path(args.output)
        if destination.exists() and not args.force:
            log.error("%s exists; pass --force to overwrite", destination)
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_config(synthesized), encoding="utf-8")
        print(f"wrote {destination}")
        return 0

    if args.command == "migrate-config":
        from pydantic import ValidationError

        from .config import MigrationError, migrate_file

        try:
            migrate_file(args.input, args.output, validate=args.validate, overwrite=args.force)
        except (MigrationError, FileExistsError, FileNotFoundError, FrameworkError) as exc:
            log.error("%s", exc)
            return 1
        except ValidationError as exc:
            # The migration itself is mechanical and faithful; the *result* can
            # still be invalid, and the commonest reason is a v1 key that never
            # did anything (model.dropout on a cnn). Report it, do not write.
            log.error("the migrated config is not valid:\n%s", exc)
            log.error("remove the offending key(s), or re-run with --no-validate")
            return 1
        print(f"wrote {args.output}")
        return 0

    if args.command == "serve":
        import uvicorn

        # Secret prefers the environment; the flag is a convenience for local use.
        api_key = args.api_key or os.environ.get("MLF_API_KEY")

        if args.workers and args.workers > 1:
            # Multi-worker needs an import string → configure the env-based factory.
            os.environ["MLF_ARTIFACTS"] = args.artifacts
            if args.registry_model:
                os.environ["MLF_REGISTRY_MODEL"] = args.registry_model
            os.environ["MLF_REGISTRY_STAGE"] = args.registry_stage
            if args.tracking_uri:
                os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
            if api_key:
                os.environ["MLF_API_KEY"] = api_key
            os.environ["MLF_RATE_LIMIT"] = args.rate_limit
            os.environ["MLF_MAX_INSTANCES"] = str(args.max_instances)
            uvicorn.run(
                "ml_framework.serving.asgi:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
            )
            return 0

        from .serving.api import create_app

        app = create_app(
            args.artifacts,
            registry_model=args.registry_model,
            registry_stage=args.registry_stage,
            tracking_uri=args.tracking_uri,
            api_key=api_key,
            rate_limit=args.rate_limit,
            max_instances=args.max_instances,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    cfg = _load_config(args)
    if args.command == "lr":
        from .pipeline import find_lr

        find_lr(cfg)
        return 0

    cfg = _apply_tune_args(cfg, args)
    if args.command in ("tune", "hpo"):
        if args.command == "hpo":
            log.warning("`mlf hpo` is the v1 name; use `mlf tune`")
        from .pipeline import tune

        result = tune(cfg)
        if not result.ran:
            log.warning("no search ran: %s", result.skipped)
            return 1
        print(f"\nbest {result.metric} = {result.best_value:.4f}  ({result.n_trials} trials)")
        for path, value in sorted(result.best_params.items()):
            print(f"  {path}: {value}")
        if args.emit_config:
            from .pipeline.train import _emit_config

            _emit_config(result.config, args.emit_config)
            print(f"\nwrote {args.emit_config}")
        else:
            print("\nRe-run `mlf train` to apply these, or pass --emit-config to save them.")
        return 0

    if args.command == "select":
        return _run_select(cfg, args)

    if args.command == "train":
        from .pipeline import train

        cfg = _apply_select_args(cfg, args)
        if getattr(args, "select", None):
            cfg = cfg.with_overrides({"select.enabled": True})
        if args.folds is not None:
            cfg = cfg.with_overrides({"data.split.folds": args.folds})
        train(
            cfg,
            emit_config=args.emit_config,
            resume=args.resume,
            # Only when the *framework* picked the model. A user who named it has
            # their own frame of reference; a zero-config run has none, which is
            # exactly when a trivial baseline is worth the milliseconds.
            baseline=getattr(args, "auto_selected", False),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
