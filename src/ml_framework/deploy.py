"""
deploy.py
─────────
A Dockerfile for **one bundle**, built from what that bundle actually declares.

The repository Dockerfile has fixed targets — `serve` for the deep-learning stack,
`serve-gbdt` without torch. They cover the common cases and they are guesses about
which case you are in. A bundle already knows: `manifest.requires` lists exactly
the libraries its backend needs, recorded at training time by the plugin that
needed them.

So this reads the manifest and emits an image that installs *those* extras and no
others. The payoff is the same one `serve-gbdt` demonstrated in P3 — a booster
image without torch is roughly 500 MB lighter, with a cold start to match — except
it now follows from the bundle rather than from the user picking the right target.

Generated, not templated into the repo: a Dockerfile committed next to the code
would go stale the moment a plugin's requirements changed, and the staleness would
be invisible until an image failed to serve.

**The build context is part of the contract, and it used to be implicit.** The
bundle was copied by its *basename*, which silently assumed the context was the
bundle's parent — while the same file copies `pyproject.toml` and `src/`, which
assumes the context is the repository root. Those two assumptions coincide only
for a bundle sitting directly in the root, i.e. the default `outputs/` — and that
is exactly the path this repository's own `.dockerignore` strips. So the *default*
configuration produced a Dockerfile whose `COPY` found nothing, and the failure
surfaced as a container that could not serve rather than as a build that stopped.

Both halves are now checked here, at generation time: the path is emitted relative
to the context, and a bundle the context would strip is refused with the line that
strips it. That is the same reasoning that makes this module a generator rather
than a committed file — an invisible failure is the one worth spending code on.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from .core.types import DIST_NAME, FrameworkError

log = logging.getLogger(__name__)

# The base image the repository Dockerfile uses. Kept in step deliberately —
# generating an image on a different Python than CI tests against is a way to
# discover a version-specific bug in production.
BASE_IMAGE = "python:3.11-slim"

# Always needed to serve anything, whatever the model is.
SERVING_EXTRAS: tuple[str, ...] = ("serve",)

DOCKERIGNORE = ".dockerignore"
# The generated file copies this from the context. Its absence does not make the
# context wrong — a caller may know better — but it is worth one warning, because
# the resulting failure is another `COPY` that finds nothing.
CONTEXT_MARKER = "pyproject.toml"


class DeployError(FrameworkError):
    """A deployable image could not be described for this bundle."""


def required_extras(manifest: Any) -> list[str]:
    """The extras this bundle's model needs, from its own recorded requirements.

    Reads `manifest.requires` — written at training time from the plugin's
    `ModelSpec.requires` — rather than mapping the backend name to a guess. A
    third-party plugin declaring its own extra therefore gets a correct image
    without this module knowing the plugin exists.
    """
    extras = {ref.extra for ref in manifest.requires if getattr(ref, "extra", None)}
    return sorted(extras | set(SERVING_EXTRAS))


def _ignore_rules(context: Path) -> list[tuple[str, bool]]:
    """``(pattern, negated)`` from the context's ``.dockerignore``, in file order."""
    path = context / DOCKERIGNORE
    if not path.exists():
        return []
    rules: list[tuple[str, bool]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        rules.append((line[1:].strip() if negated else line, negated))
    return rules


def excluded_by(rel_path: str, context: Path) -> str | None:
    """The ``.dockerignore`` pattern that strips ``rel_path``, or ``None``.

    Two of Docker's rules matter here and both are easy to get wrong by hand:
    **last match wins**, so a later ``!pattern`` re-includes what an earlier one
    excluded; and **excluding a directory excludes what is under it**, so a
    pattern is tested against the path *and* against each of its parent prefixes.

    An approximation of Docker's matcher rather than a reimplementation, and
    deliberately biased toward reporting a collision. A false positive costs a
    message naming a real line in a real file, which a reader can check in
    seconds. A false negative costs a build that copies nothing, an image that
    looks fine, and a failure at serving time — which is the whole class of
    outcome this module exists to prevent.
    """
    parts = PurePosixPath(rel_path).parts
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]

    hit: str | None = None
    for pattern, negated in _ignore_rules(context):
        if any(fnmatch(prefix, pattern) for prefix in prefixes):
            hit = None if negated else pattern
    return hit


def render_dockerfile(
    bundle_dir: str | Path,
    *,
    base_image: str = BASE_IMAGE,
    context: str | Path | None = None,
) -> str:
    """A Dockerfile that serves ``bundle_dir`` and installs nothing more.

    The bundle is **copied in** rather than mounted: an image that needs a volume
    to be useful is not a deployable artifact, and the whole point of the bundle
    format is that it is self-contained.

    ``context`` is the directory ``docker build`` will be pointed at; it defaults
    to the bundle's parent. The bundle must live inside it — ``COPY`` cannot
    reach outside a build context — and the emitted path is relative to it rather
    than being the bundle's basename, so a bundle nested any deeper than one level
    is copied by a path the context can actually resolve.
    """
    from .core.bundle import read_manifest

    source = Path(bundle_dir)
    if not source.is_dir():
        raise DeployError(f"no bundle directory at {source}")

    root = Path(context) if context is not None else source.parent
    try:
        relative = source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise DeployError(
            f"the bundle at {source} is not inside the build context {root}. "
            f"COPY cannot reach outside a build context, so pass a context that "
            f"contains both the bundle and src/, or move the bundle under one."
        ) from None

    stripped = excluded_by(relative, root)
    if stripped:
        raise DeployError(
            f"{root / DOCKERIGNORE} strips '{relative}' via the pattern '{stripped}', so "
            f"`docker build` would receive a context with no bundle in it and this file's "
            f"COPY would find nothing — an image that builds and then cannot serve. "
            f"Add '!{relative}' to {DOCKERIGNORE} (last match wins), or generate against "
            f"a context that does not exclude the bundle."
        )

    if not (root / CONTEXT_MARKER).exists():
        # Not fatal: the caller may be assembling a context deliberately. But the
        # generated file copies this, so saying nothing means a second COPY that
        # finds nothing, discovered the same way as the first.
        log.warning(
            "build context %s has no %s, which this Dockerfile copies — "
            "`docker build` from it will fail unless you add one",
            root,
            CONTEXT_MARKER,
        )

    manifest = read_manifest(source)
    extras = required_extras(manifest)
    spec = f"{DIST_NAME}[{','.join(extras)}]"

    torch_free = not any(extra in ("lightning", "image", "nlp") for extra in extras)
    weight_note = (
        "# No torch in this image: this model does not need it. That is roughly\n"
        "# 500 MB and a cold start the deep-learning image pays and this one does not."
        if torch_free
        else "# This model needs the deep-learning runtime, so torch is installed\n"
        "# CPU-only -- the CUDA build adds several GB of nvidia wheels."
    )
    torch_install = (
        ""
        if torch_free
        else "\n# CPU-only torch, in its own layer so it caches across rebuilds above it.\n"
        "RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu\n"
    )

    lines = f"""# Generated by `mlf dockerfile` for the bundle at {source.name}
#
# Tailored to what this bundle *declares* it needs -- `manifest.requires`, written
# at training time by the plugin that needed them -- rather than to a fixed target
# somebody had to pick correctly.
#
#   model    {manifest.model.name} ({manifest.model.backend} backend)
#   task     {manifest.task} on {manifest.data_kind} data
#   extras   {", ".join(extras)}
#
{weight_note}

FROM {base_image}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
{torch_install}
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir '.[{",".join(extras)}]'

# The bundle is copied in, not mounted: an image that needs a volume to be useful
# is not a deployable artifact. The path is relative to the build context, which
# must therefore be: {root}
COPY {relative}/ ./bundle/

# Serving reads MLF_* from the environment -- MLF_API_KEY (secret),
# MLF_RATE_LIMIT, MLF_MAX_INSTANCES. See serving/asgi.py.
ENV MLF_ARTIFACTS=/app/bundle
EXPOSE 8000

# Health: GET /health   Metrics: GET /metrics
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \\
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["mlf", "serve", "--artifacts", "/app/bundle", "--host", "0.0.0.0", "--port", "8000"]
"""
    log.info("dockerfile for %s: extras=%s (%s)", manifest.model.name, extras, spec)
    return lines


__all__ = [
    "BASE_IMAGE",
    "CONTEXT_MARKER",
    "DOCKERIGNORE",
    "DeployError",
    "excluded_by",
    "render_dockerfile",
    "required_extras",
]
