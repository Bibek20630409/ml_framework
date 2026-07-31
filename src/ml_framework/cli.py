"""
cli.py
──────
`mlf` command-line entry point. All commands load a validated YAML config and
accept dotted ``--set key=value`` overrides.

    mlf lr             --config configs/example_tabular.yaml
    mlf hpo            --config configs/example_tabular.yaml
    mlf train          --config configs/example_tabular.yaml --set fit.budget.max_epochs=5
    mlf serve          --artifacts outputs --host 0.0.0.0 --port 8000
    mlf migrate-config -i configs/old.yaml -o configs/new.yaml
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlf", description="ML Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("lr", "hpo", "train"):
        p = sub.add_parser(name, help=f"Run the {name} stage")
        _add_config_args(p)

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
    elif args.command == "hpo":
        from .pipeline import run_hpo

        run_hpo(cfg)
    elif args.command == "train":
        from .pipeline import train

        train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
