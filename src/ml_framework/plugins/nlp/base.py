"""
plugins/nlp/base.py
───────────────────
What the three HuggingFace plugins share: how they are built from a checkpoint,
and how they are written to and read from a bundle.

Three models, three ``Auto*`` classes, one file format. Without this the
``save_pretrained``/``from_pretrained`` pair would be copied three times — and the
copies would drift, because the interesting part of that pair is a *comment* about
why a Lightning checkpoint is the wrong format here, and comments drift fastest.

The one thing subclasses must supply is :attr:`AUTO_CLASS`. Everything else is
either shared or expressed as a hook with a default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, cast

from ...core.bundle import head_width
from ...core.lit_model import BaseModel
from ...core.protocols import ArtifactRef

log = logging.getLogger(__name__)

# The subdirectory inside `model/` holding a `save_pretrained` dump.
HF_MODEL_DIR = "hf_model"
# Tracked separately from the directory name so the layout can change without
# breaking readers — the same reason every other artifact records a format.
HF_FORMAT = "huggingface-pretrained"


class HFPretrainedModel(BaseModel):
    """A ``transformers`` model wearing the framework's contract.

    **Why these models own their file format.** A Lightning checkpoint would
    round-trip the weights, but rebuilding the architecture to put them in calls
    ``from_pretrained(model_name)`` — which needs the hub, or a warm HF cache, *at
    load time*. A bundle that only loads on a machine with network access is not a
    bundle, and the failure surfaces in a serving container rather than in CI. So
    :meth:`save_artifact` writes a checkpoint directory and :meth:`load_artifact`
    reads one. The backend discovers both by name and falls back to the checkpoint
    path for every model that defines neither.
    """

    # Name of the `transformers.Auto*` class this model builds. A string rather
    # than the class itself so this module stays importable without transformers.
    AUTO_CLASS: ClassVar[str] = ""

    def from_pretrained_kwargs(self) -> dict[str, Any]:
        """Extra arguments for ``from_pretrained``. Overridden per head shape."""
        return {}

    def build_network(self) -> Any:
        import transformers

        factory = getattr(transformers, self.AUTO_CLASS, None)
        if factory is None:  # pragma: no cover - guards a typo in a subclass
            raise ValueError(f"transformers has no '{self.AUTO_CLASS}'")

        net = factory.from_pretrained(self.params.model_name, **self.from_pretrained_kwargs())
        if getattr(self.params, "freeze_encoder", False):
            for parameter in net.base_model.parameters():
                parameter.requires_grad = False
            log.info("encoder frozen: training the head only")
        return net

    @property
    def hf_model(self) -> Any:
        """``self.network``, with the type it actually has.

        torch types a module attribute as ``Tensor | Module``, so calling a
        transformers method on one is a type error even though that is precisely
        what the attribute is. Narrowed once here rather than at each use.
        """
        return cast("Any", self.network)

    # ── self-serialization ──
    def save_artifact(self, dest: str | Path) -> ArtifactRef:
        """Write the checkpoint directory; return its bundle-relative path.

        ``dest`` is the bundle's ``model/`` directory, so the returned path is
        relative to its parent — the bundle root, which is what the manifest
        records.
        """
        target_dir = Path(dest)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.hf_model.save_pretrained(target_dir / HF_MODEL_DIR)
        return ArtifactRef(path=f"{target_dir.name}/{HF_MODEL_DIR}", format=HF_FORMAT)

    @classmethod
    def load_artifact(cls, path: str | Path, manifest: Any) -> HFPretrainedModel:
        """Rebuild from ``model/hf_model/`` — no hub, no cache, no network.

        ``model_name`` is redirected at the bundle's own directory. The manifest
        keeps the original checkpoint name in ``model.params``, so the audit trail
        still records what was fine-tuned; only where the weights are read from
        changes.
        """
        params = dict(manifest.model.params)
        params["model_name"] = str(path)
        return cls(
            # A token sequence has no fixed input width; the checkpoint's own
            # config carries the shapes that matter and is loaded from `path`.
            input_dim=0,
            output_dim=head_width(manifest),
            task=manifest.task,
            params=params,
        )


__all__ = ["HF_FORMAT", "HF_MODEL_DIR", "HFPretrainedModel"]
