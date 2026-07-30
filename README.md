# ML Framework

Production-grade PyTorch Lightning framework for **tabular**, **image**, and mixed
data. Supports binary classification, multi-class classification, and regression —
driven end-to-end by a single validated YAML config and a `mlf` CLI.

```
lr finder → HPO (Optuna) → train → serve (FastAPI)
```

## Install

```bash
pip install -e ".[dev,serve,hpo,image,diagnostics]"   # full toolkit
# or minimal training only:  pip install -e .
mlf --help
```

Extras: `image`, `serve`, `hpo`, `diagnostics`, `logging`, `dev`.

## Quickstart

1. **Write one YAML config** (copy `configs/example_tabular.yaml`). Set `task`,
   `data.csv_path`, `data.target_col`. That's the only file you edit per project.

2. **Learning rate** — find a good LR, paste into `optim.lr`:
   ```bash
   mlf lr --config configs/example_tabular.yaml
   ```

3. **HPO** — search architecture, paste `model.hidden_dims`/`dropout`/`lr`/`weight_decay`:
   ```bash
   mlf hpo --config configs/example_tabular.yaml
   ```

4. **Train** — final run, writes the artifact bundle:
   ```bash
   mlf train --config configs/example_tabular.yaml
   # override anything inline:
   mlf train --config configs/example_tabular.yaml --set train.epochs=5 --set optim.lr=3e-4
   ```

5. **Serve**:
   ```bash
   mlf serve --artifacts outputs --port 8000
   curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
        -d '{"instances": [[0.1, 0.2, 0.3, 0.4]]}'
   ```

## Project structure

```
src/ml_framework/
├── config/schema.py     Pydantic ExperimentConfig (validated, frozen)
├── core/                framework internals — you rarely touch these
│   ├── lit_model.py     BaseModel: steps, metrics, loss, optimizer
│   ├── lit_data.py      Tabular/Image DataModules: split, scale, imbalance
│   ├── evaluate.py      report.txt · predictions.csv · confusion_matrix.txt
│   ├── inference.py     Inferencer.from_artifacts(dir)
│   └── registry.py      @register_model / @register_datamodule
├── models/              mlp.py (tabular), cnn.py (image transfer learning)
├── data/builders.py     config → datamodule/model factory
├── pipeline/            train · hpo · lr_finder
├── serving/api.py       FastAPI: /health /predict /predict_proba
├── utils/               logging, seed, platform-aware workers
└── cli.py               `mlf` entry point
configs/                 example_tabular.yaml · example_image.yaml
tests/                   unit · integration · serving
```

## Extending (scalability)

Add a new model or data source without touching the pipeline — just register it:

```python
from ml_framework.core import register_model, BaseModel

@register_model("my_net")
class MyNet(BaseModel):
    def build_network(self):
        ...   # uses self.input_dim, self.output_dim, self.config.model
```

Then set `model.name: my_net` in the YAML. Same pattern for `@register_datamodule`.

## What runs automatically

- Train/val/test split (holdout for large data, KFold-derived for small; **KFold for
  regression** to avoid stratification crashes)
- Scaling (`StandardScaler`, fit on train only), persisted for inference
- Imbalance handling — one explicit strategy: `smote | class_weights | none`
- Gradient clipping, early stopping, best-checkpointing, LR scheduling
- Metrics + CSV/WandB logging; final report, predictions, confusion matrix
- Self-contained artifact bundle: `model.ckpt · scaler.pkl · metadata.json`

## Task reference

| task | loss | metrics |
|---|---|---|
| `binary` | `BCEWithLogitsLoss` (scalar `pos_weight`) | acc, F1 |
| `multiclass` | `CrossEntropyLoss` (class weights) | acc, macro-F1 |
| `regression` | `MSELoss` | MAE, RMSE |

## Production MLOps

Beyond training + serving, the framework ships a full, self-hostable MLOps stack —
**MLflow** (tracking + model registry), **DVC** (data versioning), **Apache Spark**
(preprocessing) orchestrated by **Apache Airflow**, deployed on **Kubernetes**
(autoscaled FastAPI, KServe optional), and monitored with **Prometheus + Grafana**.

```bash
pip install -e ".[dev,serve,mlops]"
docker compose up --build      # api + mlflow + prometheus + grafana + minio
```

See **[docs/MLOPS.md](docs/MLOPS.md)** for the architecture and per-component guide.

## Development

```bash
make install   # editable install with dev extras
make lint      # ruff + black --check
make type      # mypy
make test      # pytest
make cov       # pytest with coverage
```

CI runs lint + type-check + tests on Python 3.10–3.12. Licensed MIT.
