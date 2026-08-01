"""Export adapters, and the parity that makes an exported artifact worth having.

The P9 exit gate is `test_exported_onnx_matches_native_predictions`. The rest
defend two properties:

* **A refusal is never a substitution.** Asking a Prophet bundle for ONNX raises.
  Writing *some* file would be discovered at deployment time by a runtime that
  cannot load it — or worse, by one that loads it and scores differently.
* **The caveats are stated.** A traced graph does not carry the bundle's scaler,
  and an ONNX file fed raw unscaled features produces confident nonsense with no
  error. The notes say so and the CLI prints them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_framework import cli
from ml_framework.core.export import UnsupportedExportError, example_input_shape
from ml_framework.core.inference import Inferencer
from ml_framework.core.registry import get_backend

ONNX_TOLERANCE = 1e-5
"""Float32 through two different runtimes will not be bit-identical.

Stated rather than tuned until green: 1e-5 is far tighter than any decision
boundary this would change, and looser than the ~6e-08 actually observed.
"""


def _table(tmp_path: Path, name: str = "d.csv") -> Path:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)).astype("float32")
    y = (x @ rng.normal(size=(6, 3))).argmax(axis=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["label"] = y
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _train(tmp_path: Path, model: str) -> Path:
    """A trained bundle, via the CLI so the whole path is exercised."""
    out = tmp_path / f"run_{model.replace('.', '_')}"
    assert (
        cli.main(
            [
                "train",
                "--data",
                str(_table(tmp_path)),
                "--model",
                model,
                "--output-dir",
                str(out),
                "--no-tune",
                "--set",
                "fit.budget.max_epochs=2",
                "--set",
                "runtime.accelerator=cpu",
            ]
        )
        == 0
    )
    return out


# ── The exit gate ─────────────────────────────────────────
@pytest.mark.integration
def test_exported_onnx_matches_native_predictions(tmp_path):
    """P9's stated gate.

    Seven rows through a graph traced at batch size one — which also proves the
    dynamic batch axis, without which the artifact could only ever score a single
    row and would fail on the second.
    """
    pytest.importorskip("onnxruntime", reason="the export extra is not installed")
    import onnxruntime as ort

    bundle = _train(tmp_path, "mlp")
    inf = Inferencer.from_artifacts(bundle)
    destination = tmp_path / "model.onnx"

    get_backend(inf.backend_name).export(inf.estimator, destination, "onnx", manifest=inf.manifest)

    probe = np.random.default_rng(1).normal(size=(7, 6)).astype("float32")
    native = inf.estimator.predict_proba(probe)

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    logits = session.run(None, {"input": probe})[0]
    exported = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    assert exported.shape == native.shape
    assert np.allclose(native, exported, atol=ONNX_TOLERANCE)
    # The assertion that would actually matter in production: same decision.
    assert (native.argmax(1) == exported.argmax(1)).all()


@pytest.mark.integration
def test_the_exported_file_declares_the_opset_we_promised(tmp_path):
    """The pin is a promise to a deployment target, so assert it on the artifact.

    The dynamo exporter builds at opset 18 and *down-converts* to the requested
    17, warning that conversion "may not be successful". It succeeds for the ops
    an MLP uses — but that is op-dependent, not a blanket guarantee. Reading
    `opset_import` off the produced file means an op that fails to down-convert
    fails the build, instead of silently shipping an opset-18 artifact to a
    runtime that was promised 17.
    """
    pytest.importorskip("onnx", reason="the export extra is not installed")
    import onnx

    from ml_framework.backends.lightning import ONNX_OPSET

    bundle = _train(tmp_path, "mlp")
    inf = Inferencer.from_artifacts(bundle)
    destination = tmp_path / "model.onnx"

    get_backend(inf.backend_name).export(inf.estimator, destination, "onnx", manifest=inf.manifest)

    model = onnx.load(str(destination))
    default_domain = {i.domain or "ai.onnx": i.version for i in model.opset_import}
    assert default_domain["ai.onnx"] == ONNX_OPSET

    # And the batch axis is a symbol, not the width it was traced at. Without this
    # the artifact scores one row and fails on the second.
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param, "batch axis was frozen to a constant"


@pytest.mark.integration
def test_export_survives_a_console_that_cannot_encode_a_check_mark(tmp_path):
    """A successful export must not be reported as a failure by its own logging.

    The dynamo exporter prints U+2705 on success. On Windows the console is
    cp1252, so encoding it raises `UnicodeEncodeError` *after the artifact is
    already correct* — the export fails for a reason that has nothing to do with
    the model. `verbose=False` is what suppresses it, and this test is what stops
    someone removing that as noise.

    Reproduced everywhere rather than only on Windows: the stream is what matters,
    not the platform.
    """
    pytest.importorskip("onnxruntime", reason="the export extra is not installed")
    import contextlib
    import io

    bundle = _train(tmp_path, "mlp")
    inf = Inferencer.from_artifacts(bundle)
    destination = tmp_path / "model.onnx"

    cp1252 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with contextlib.redirect_stdout(cp1252):
        get_backend(inf.backend_name).export(
            inf.estimator, destination, "onnx", manifest=inf.manifest
        )

    assert destination.exists() and destination.stat().st_size > 0


@pytest.mark.integration
def test_torchscript_round_trips_through_torch(tmp_path):
    import torch

    bundle = _train(tmp_path, "mlp")
    inf = Inferencer.from_artifacts(bundle)
    destination = tmp_path / "model.pt"

    get_backend(inf.backend_name).export(
        inf.estimator, destination, "torchscript", manifest=inf.manifest
    )

    probe = np.random.default_rng(2).normal(size=(4, 6)).astype("float32")
    loaded = torch.jit.load(str(destination)).eval()
    with torch.no_grad():
        exported = loaded(torch.tensor(probe)).numpy()
    native = inf.estimator.logits(probe).numpy()

    assert np.allclose(native, exported, atol=ONNX_TOLERANCE)


# ── Refusals ──────────────────────────────────────────────
@pytest.mark.integration
def test_a_booster_refuses_onnx_and_says_what_it_can_do(tmp_path):
    """Not a gap. A booster's own format is what every serving runtime for that
    library reads, and an ONNX conversion would introduce a second implementation
    of the same tree traversal that can disagree at a split boundary."""
    pytest.importorskip("xgboost")
    bundle = _train(tmp_path, "xgboost")
    inf = Inferencer.from_artifacts(bundle)

    with pytest.raises(UnsupportedExportError) as caught:
        get_backend(inf.backend_name).export(inf.estimator, tmp_path / "m.onnx", "onnx")

    message = str(caught.value)
    assert "cannot export 'onnx'" in message
    assert "native" in message  # says what it *can* do


@pytest.mark.integration
def test_a_booster_exports_its_native_format(tmp_path):
    pytest.importorskip("xgboost")
    bundle = _train(tmp_path, "xgboost")
    inf = Inferencer.from_artifacts(bundle)
    destination = tmp_path / "model.json"

    result = get_backend(inf.backend_name).export(inf.estimator, destination, "native")

    assert destination.exists() and destination.stat().st_size > 0
    assert result.format == "native"
    # And it really is xgboost's own file, not something shaped like one.
    import xgboost as xgb

    booster = xgb.XGBClassifier()
    booster.load_model(str(destination))


@pytest.mark.unit
def test_an_unknown_format_is_refused_rather_than_guessed():
    from ml_framework.backends.base import BaseBackend

    with pytest.raises(UnsupportedExportError, match="no export formats"):
        BaseBackend().export(object(), Path("x"), "onnx")


# ── The shape it refuses to invent ────────────────────────
@pytest.mark.unit
def test_a_text_bundle_refuses_tracing_with_the_reason():
    """Tracing a tokenized batch would bake this batch's sequence length into the
    graph, so every future request would be silently truncated or padded to it."""
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        signature=SimpleNamespace(input=SimpleNamespace(payload="dataset", n_features=0)),
        preprocessor=SimpleNamespace(params={"model_name": "bert", "max_length": 128}),
    )

    with pytest.raises(UnsupportedExportError, match="sequence length"):
        example_input_shape(manifest)


@pytest.mark.unit
def test_an_image_bundle_traces_at_the_recorded_size():
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        signature=SimpleNamespace(input=SimpleNamespace(payload="dataset", n_features=0)),
        preprocessor=SimpleNamespace(params={"img_size": 64}),
    )

    assert example_input_shape(manifest) == (1, 3, 64, 64)


@pytest.mark.unit
def test_a_tabular_bundle_traces_at_its_feature_count():
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        signature=SimpleNamespace(input=SimpleNamespace(payload="arrays", n_features=11)),
        preprocessor=None,
    )

    assert example_input_shape(manifest) == (1, 11)


# ── The CLI ───────────────────────────────────────────────
@pytest.mark.integration
def test_mlf_export_writes_the_file_and_prints_the_caveat(tmp_path, capsys):
    """The note is not decoration: an ONNX file fed raw unscaled features scores
    nonsense with no error, so "preprocessing is NOT included" has to reach the
    person who is about to deploy it."""
    pytest.importorskip("onnxruntime", reason="the export extra is not installed")
    bundle = _train(tmp_path, "mlp")
    destination = tmp_path / "out.onnx"
    capsys.readouterr()

    assert (
        cli.main(["export", "--artifacts", str(bundle), "--format", "onnx", "-o", str(destination)])
        == 0
    )

    out = capsys.readouterr().out
    assert destination.exists()
    assert "preprocessing is NOT included" in out


@pytest.mark.integration
def test_mlf_export_returns_nonzero_when_refused(tmp_path, caplog):
    pytest.importorskip("xgboost")
    bundle = _train(tmp_path, "xgboost")

    assert (
        cli.main(
            ["export", "--artifacts", str(bundle), "--format", "onnx", "-o", str(tmp_path / "x")]
        )
        == 1
    )
    assert "cannot export" in caplog.text
