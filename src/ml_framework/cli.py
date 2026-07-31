"""
cli.py
──────
`mlf` command-line entry point. All commands load a validated YAML config and
accept dotted ``--set key=value`` overrides.

    mlf lr             --config configs/example_tabular.yaml
    mlf tune           --config configs/example_tabular.yaml --emit-config configs/tuned.yaml
    mlf train          --config configs/example_tabular.yaml --set fit.budget.max_epochs=5
    mlf train          --config configs/example_gbdt.yaml --no-tune
    mlf train          --config configs/example_gbdt.yaml --tune-budget 10m --tune-trials 50
    mlf serve          --artifacts outputs --host 0.0.0.0 --port 8000
    mlf migrate-config -i configs/old.yaml -o configs/new.yaml

``mlf train`` **tunes by default**, under a per-backend budget (see
``config/defaults.py``) — 300 s for trees, 900 s for neural nets. ``--no-tune``
skips it; ``--tune-budget``/``--tune-trials`` turn it up just as easily.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from .config import ExperimentConfig
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


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.set:
        cfg = cfg.with_overrides(dict(args.set))
    return cfg


def _add_config_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", "-c", required=True, help="Path to YAML config")
    sub.add_argument(
        "--set",
        action="append",
        type=_parse_override,
        default=[],
        metavar="key=value",
        help="Override a config value (dotted key), repeatable",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlf", description="ML Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    lr = sub.add_parser("lr", help="Run the LR range test")
    _add_config_args(lr)

    train_p = sub.add_parser("train", help="Tune (unless disabled) and train")
    _add_config_args(train_p)
    _add_tune_args(train_p)
    train_p.add_argument(
        "--emit-config",
        default=None,
        metavar="PATH",
        help="Write the effective (post-tuning) config as YAML, for committing back",
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.command == "migrate-config":
        from pydantic import ValidationError

        from .config import MigrationError, migrate_file
        from .core.types import FrameworkError

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

    if args.command == "train":
        from .pipeline import train

        train(cfg, emit_config=args.emit_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
