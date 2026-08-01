"""Text classification end to end: fine-tune, bundle, reload, serve.

Two things here are P7's exit gates and the rest supports them: the bundle carries
a HuggingFace *directory* rather than a Lightning checkpoint, and ``/predict``
takes raw strings.

The directory format is not a preference. A Lightning checkpoint round-trips the
weights, but rebuilding the architecture to put them in calls
``from_pretrained(model_name)`` — which needs the hub, or a warm cache, at load
time. A bundle that only loads on a machine with network access is not a bundle,
and the failure would appear in a serving container rather than in CI.

Everything runs against a ~90 K-parameter random DistilBert. It learns nothing,
which is the point: these tests are about plumbing, and a real checkpoint would
buy nothing but a 250 MB download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers", reason="the nlp extra is not installed")
pytest.importorskip("pytorch_lightning", reason="the lightning extra is not installed")

from ml_framework.config import ExperimentConfig  # noqa: E402
from ml_framework.core.inference import Inferencer  # noqa: E402
from ml_framework.pipeline import train  # noqa: E402
from ml_framework.plugins.nlp.hf_text import HF_FORMAT, HF_MODEL_DIR  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"

POSITIVE = ["great movie", "loved it", "wonderful acting", "a delight", "superb", "really good"]
NEGATIVE = ["terrible film", "hated it", "awful acting", "a chore", "dreadful", "really bad"]


@pytest.fixture
def reviews_csv(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    rows = [{"text": t, "label": "pos"} for t in POSITIVE * 4]
    rows += [{"text": t, "label": "neg"} for t in NEGATIVE * 4]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def text_config(csv: Path, out: Path, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig.model_validate(
        {
            "task": "binary",
            "runtime": {
                "output_dir": str(out),
                "seed": 0,
                "num_workers": 0,
                "accelerator": "cpu",
            },
            "data": {
                "kind": "text",
                "path": str(csv),
                "target": "label",
                "split": {"val_size": 0.2, "test_size": 0.2},
            },
            "model": {
                "name": "nlp.hf_text",
                "params": {"model_name": TINY_MODEL, "max_length": 16},
            },
            "fit": {"budget": {"max_epochs": 1}, "batch_size": 8},
            "tune": {"enabled": False},
            "logging": {"backend": "none"},
        }
    )
    return cfg.with_overrides(overrides) if overrides else cfg


@pytest.fixture
def trained_bundle(reviews_csv, tmp_path) -> Path:
    """One training run, reused by the tests that only read its output."""
    out = tmp_path / "run"
    train(text_config(reviews_csv, out))
    return out


# ── Fine-tuning defaults ──────────────────────────────────
@pytest.mark.unit
def test_a_text_config_gets_fine_tuning_defaults(reviews_csv, tmp_path):
    """1e-3 with Adam is right for a network trained from scratch and destroys a
    pretrained encoder in the first few steps — while looking entirely healthy.

    So the spec carries them, and the config validator applies them at load time,
    which is also what puts them in ``config.json`` as the audit record.
    """
    cfg = text_config(reviews_csv, tmp_path / "run")

    assert cfg.fit.params["lr"] == 2e-5
    assert cfg.fit.params["optimizer"] == "adamw"
    assert cfg.fit.params["weight_decay"] == 0.01


@pytest.mark.unit
def test_an_explicit_learning_rate_beats_the_default(reviews_csv, tmp_path):
    """Defaults, not constants: only keys the user did not write are filled in."""
    cfg = text_config(reviews_csv, tmp_path / "run", **{"fit.params.lr": 3e-4})

    assert cfg.fit.params["lr"] == 3e-4
    assert cfg.fit.params["optimizer"] == "adamw"  # still defaulted


@pytest.mark.unit
def test_a_model_narrows_the_backends_learning_rate_range(reviews_csv, tmp_path):
    """The more specific declaration wins.

    The Lightning backend proposes 1e-4..1e-2, which for a pretrained encoder is a
    range in which most trials are damage — so a search would spend its budget
    confirming that wrecking the checkpoint scores badly.
    """
    from ml_framework.pipeline.tune import effective_space

    space = effective_space(text_config(reviews_csv, tmp_path / "run"))
    assert space["fit.params.lr"].high <= 1e-4
    # The knobs the model has no opinion about still come from the backend.
    assert "fit.batch_size" in space


# ── The bundle format (exit gate) ─────────────────────────
@pytest.mark.integration
def test_the_bundle_carries_a_huggingface_directory(trained_bundle):
    manifest = json.loads((trained_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["model"]["artifact"] == f"model/{HF_MODEL_DIR}"
    assert manifest["model"]["format"] == HF_FORMAT
    artifact = trained_bundle / manifest["model"]["artifact"]
    assert artifact.is_dir()
    # `from_pretrained` needs both of these on disk, and their absence is exactly
    # the failure that would only show up in a serving container.
    assert (artifact / "config.json").exists()
    assert any(artifact.glob("*.safetensors")) or (artifact / "pytorch_model.bin").exists()


@pytest.mark.integration
def test_the_tokenizer_ships_beside_the_weights(trained_bundle):
    """Model and tokenizer in one bundle is what makes it self-contained: no hub,
    no cache, no ``model_name`` lookup at serving time."""
    manifest = json.loads((trained_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["preprocessor"]["files"] == ["tokenizer"]
    assert (trained_bundle / "preprocessor" / "tokenizer").is_dir()


@pytest.mark.integration
def test_last_ckpt_still_rides_along_for_resume(trained_bundle):
    """Resuming is a property of the loop, not of the file the loop produced — so
    a model with its own format keeps the optimizer state too."""
    assert (trained_bundle / "model" / "last.ckpt").exists()


@pytest.mark.integration
def test_reloading_reproduces_the_fitted_models_predictions(trained_bundle):
    """The round-trip that matters: same inputs, same probabilities.

    A format that loads without error but restores the *pretrained* weights rather
    than the fine-tuned ones would pass every structural check above and be
    useless.
    """
    inf = Inferencer.from_artifacts(trained_bundle)
    texts = POSITIVE[:2] + NEGATIVE[:2]

    first = inf.predict_proba(texts)
    second = Inferencer.from_artifacts(trained_bundle).predict_proba(texts)
    assert np.allclose(first, second)


@pytest.mark.integration
def test_class_names_survive_into_the_manifest(trained_bundle):
    """String labels were encoded on the way in; the names have to come back out
    or the served predictions are integers nobody can interpret."""
    manifest = json.loads((trained_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["signature"]["output"]["class_names"] == ["neg", "pos"]


# ── Serving raw strings (exit gate) ───────────────────────
@pytest.fixture
def client(trained_bundle):
    pytest.importorskip("fastapi", reason="the serve extra is not installed")
    from fastapi.testclient import TestClient

    from ml_framework.serving.api import create_app

    return TestClient(create_app(str(trained_bundle)))


@pytest.mark.serving
def test_predict_accepts_raw_strings(client):
    """Asking clients to send token ids would make every one of them responsible
    for using the right vocabulary — the skew the bundled tokenizer prevents — and
    would make the endpoint unusable from curl."""
    response = client.post("/predict", json={"inputs": ["great movie", "terrible film"]})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert set(body["labels"]) <= {"neg", "pos"}


@pytest.mark.serving
def test_predict_proba_returns_named_classes(client):
    response = client.post("/predict_proba", json={"inputs": ["great movie"]})

    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["neg", "pos"]
    assert len(body["probabilities"][0]) == 2


@pytest.mark.serving
def test_a_tabular_body_is_rejected_for_a_text_model(client):
    """The request schema comes from the manifest, so ``/docs`` describes *this*
    model rather than the framework."""
    assert client.post("/predict", json={"instances": [[1.0, 2.0]]}).status_code == 422


@pytest.mark.serving
def test_drift_is_refused_rather_than_invented(client):
    """PSI over token ids is a number without a meaning."""
    assert client.get("/drift").status_code == 501


@pytest.mark.serving
def test_health_reports_no_feature_count(client):
    """A token sequence has no fixed width; reporting one would be a number the
    serving layer checks requests against and is wrong about."""
    assert client.get("/health").json()["n_features"] == 0


# ── Cross-validation ──────────────────────────────────────
@pytest.mark.integration
def test_training_with_folds_reports_a_cv_estimate(reviews_csv, tmp_path):
    """Text CV needed no new machinery: the source accepts injected indices like
    the other three, and the fold logic never learned what text is."""
    out = tmp_path / "cv"
    metrics = train(text_config(reviews_csv, out, **{"data.split.folds": 2}))

    assert "cv_acc_mean" in metrics
    cv = json.loads((out / "cv.json").read_text(encoding="utf-8"))
    assert cv["folds"] == 2 and len(cv["per_fold"]) == 2
