# Production MLOps Guide

This framework ships a complete, self-hostable MLOps stack around the training/serving
core. Each tool has one job; together they take a model from raw data → versioned →
orchestrated → tracked → deployed → scaled → monitored.

```
 DVC-tracked raw data ──▶ Airflow DAG:
    ├─ (Spark) preprocess → processed Parquet (DVC-tracked)
    ├─ (mlf)  train       → logs + registers model in MLflow
    ├─ evaluate gate      → promote version to @production
    └─ trigger deploy     → roll the serving Deployment
                                │
 Serving: FastAPI (loads models:/ml-framework@production) ─▶ Docker ─▶ Kubernetes
    │  /predict /predict_proba /predict_with_confidence           (Deployment/Svc/Ingress/HPA)
    │  /health /metrics /drift
    ▼
 Monitoring: Prometheus scrapes /metrics ─▶ Grafana ("ML Framework — Serving")
```

| Concern | Tool | Where |
|---|---|---|
| Experiment tracking + model registry | **MLflow** | `tracking/mlflow_utils.py`, `logging.backend: mlflow` |
| Data version control | **DVC** | `dvc.yaml`, `params.yaml` |
| Data processing engine (per run) | **pandas · Polars · Spark** | `data/backends/`, `data.backend` / `--data-backend` |
| Data processing (batch stage) | **Apache Spark** | `pipeline/spark_preprocess.py` |
| Orchestration | **Apache Airflow** | `orchestration/airflow/dags/ml_pipeline.py` |
| Serving | **FastAPI** (+ KServe option) | `serving/api.py`, `deploy/k8s/api.yaml` |
| Infra / scaling | **Kubernetes** | `deploy/k8s/` manifests, `deploy/kustomization.yaml` |
| Monitoring | **Prometheus + Grafana** | `deploy/monitoring/`, `deploy/k8s/monitoring.yaml` |

Install everything: `pip install -e ".[dev,serve,mlops]"`. Add `fast` for the Polars
engine — it is a peer of pandas, not part of the `mlops` stack.

## Fastest path: the whole stack locally

```bash
docker compose up --build          # api + mlflow + prometheus + grafana + minio
docker compose run --rm train train --config configs/example_tabular.yaml
# api → :8000  mlflow → :5000  prometheus → :9090  grafana → :3000 (admin/admin)
```

## 1. MLflow — tracking + registry

```bash
mlf train --config configs/example_tabular.yaml \
  --set logging.backend=mlflow \
  --set logging.mlflow_tracking_uri=http://localhost:5000 \
  --set logging.registered_model_name=ml-framework
```

Every run's params, metrics, and the whole bundle v2 directory (`manifest.json` +
`model/` + `preprocessor/` + `config.json`, under `bundle/`) are logged; a new
**registered model version** is created. Serving loads it back:

```bash
mlf serve --registry-model ml-framework --registry-stage production \
          --tracking-uri http://localhost:5000
```

Local default backend is `sqlite:///mlflow.db` (the file store is deprecated in
MLflow 3). Production uses an HTTP tracking server + S3/MinIO artifacts.

## 2. DVC — data versioning

```bash
dvc remote add -d minio s3://dvc -f
dvc remote modify minio endpointurl http://localhost:9000
dvc repro          # preprocess (Spark) → train, content-addressed
dvc metrics show   # reads outputs/metrics.json
```

The pipeline (`dvc.yaml`) has two stages: `preprocess` runs the Spark job, `train`
runs `mlf train` on the processed Parquet.

`params.yaml` holds what belongs to the *pipeline* — the raw input path and which
config the run is pinned to. It deliberately does not carry a target column or an
output path: those describe the dataset, so they live in the train config's
`data:` block and every stage reads them from there via `--config`. One value, one
owner. See [§3](#3-spark--preprocessing).

## 3. Spark — preprocessing

In the pipeline, the stage is handed the config and reads `data.target` /
`data.path` itself — so the target column is written down exactly once:

```bash
python -m ml_framework.pipeline.spark_preprocess \
  --input data/raw/sample.csv --config configs/dvc_tabular.yaml
```

For a one-off run, the explicit flags still work and still win over the config:

```bash
python -m ml_framework.pipeline.spark_preprocess \
  --input data/raw/sample.csv --output data/processed --target-col label
```

Reads raw CSV/Parquet, cleans + feature-engineers at scale, writes processed Parquet
that the datamodule reads directly (`read_table` handles CSV, Parquet files, and
Spark's Parquet directories). Requires Java 11/17.

The engine is a flag, and `spark` is only the default:

```bash
python -m ml_framework.pipeline.spark_preprocess \
  --input data/raw/sample.csv --output data/processed --target-col label \
  --data-backend local        # identical cleaning logic in pandas — no JVM needed
```

Same code path, same output, different executor — which is what makes the stage
testable on a laptop and on a cluster without two implementations to keep in step.

### The data backend, per run

`data.backend` (or `--data-backend` on `train`, `lr`, `tune` and `select`) chooses the
engine that reads and reduces the table:

| Engine | Extra | Use when |
|---|---|---|
| `local` | none — always available | The default. pandas; `read_table -> pd.DataFrame` is public API |
| `polars` | `[fast]` | Parse-bound runs on wide or long CSVs |
| `spark` | `[mlops]` + a JVM | The table does not fit on one machine |

```bash
mlf data-backends --show            # every engine, its extra, and whether it is ready
mlf train -c configs/example_gbdt.yaml --data-backend polars
```

Registering an engine costs a bare install nothing — none of the three imports its
runtime until selected, which the `gbdt-no-torch` CI job asserts explicitly. **Only
`tabular` is backend-aware**; image, text and timeseries read through pandas and refuse
a non-local engine by name rather than silently ignoring it. See
[`choose.md`](../choose.md) for why Polars was gated on a measurement, and
`benchmarks/data_backends.py` for the measurement itself.

## 4. Airflow — orchestration

```bash
cd orchestration/airflow
docker compose -f docker-compose.airflow.yml up airflow-init
docker compose -f docker-compose.airflow.yml up      # UI :8080 (airflow/airflow)
```

The `ml_framework_pipeline` DAG runs: `assert_pinned_config → dvc_pull →
spark_preprocess → validate_data → train → evaluate_gate → promote_model →
trigger_deploy`. The gate fails the run if accuracy is below `ACCURACY_GATE`;
promotion sets the `@production` alias in MLflow.

### The DAG retrains; it does not choose

`train` fits whatever `TRAIN_CONFIG` names — no bake-off, no `model.name`
override. Choosing a family is a decision, retraining it on fresh data is a
routine, and running the comparison nightly lets the winner flip on
cross-validation noise, changing the production family with nobody choosing it.

Select out of band, emit the winner, commit it, point the DAG at it:

```bash
mlf select --config configs/example_selection.yaml \
    --emit-config configs/winner.yaml     # model.name + tuned params + select.enabled: false

export TRAIN_CONFIG=configs/winner.yaml
```

`assert_pinned_config` is what makes this structural rather than conventional:
`mlf train` reads `select.enabled` from the YAML, so a config with the comparison
switched on would run a bake-off *inside* the `train` task. The guard fails the
run before `dvc pull`, naming the file and what to do about it.

To distribute a bake-off you do want, `mlf select --candidate` / `--collect` fan
out across dynamic task mapping in a **separate** DAG whose output is a proposed
`configs/winner.yaml`. See
[MODEL_SELECTION.md](MODEL_SELECTION.md#6-running-candidates-in-parallel).

## 5. Kubernetes — deploy + scale

```bash
kubectl kustomize deploy              # render/validate without applying
kubectl apply -k deploy               # namespace, api (Deploy/Svc/Ingress/HPA),
                                      # mlflow, minio, prometheus, grafana
```

The kustomization root is `deploy/`, not `deploy/k8s/`: it generates the Grafana
dashboard ConfigMap from `deploy/monitoring/`, and kustomize will not read a file
outside its root. `--load-restrictor` is not an option — `kubectl kustomize` accepts
it but `kubectl apply -k` does not, so it would fix validation and leave the deploy
broken. The manifests themselves stay in `deploy/k8s/`.

The API `Deployment` loads `models:/ml-framework@production` from MLflow and
autoscales via the `HorizontalPodAutoscaler` (2–10 pods at 70% CPU). For serverless
inference (scale-to-zero, canary) apply `deploy/k8s/kserve-inferenceservice.yaml`
instead (requires KServe).

## 6. Prometheus + Grafana — monitoring

The API exposes `/metrics` (request rate, latency histogram, error rate, and
`mlf_predictions_total{predicted_class}` for drift). Prometheus scrapes pods
annotated `prometheus.io/scrape=true`; Grafana auto-loads the **ML Framework —
Serving** dashboard. Local: `http://localhost:3000` (admin/admin).

## 7. Production hardening

The hardening layer that makes the platform safe for live traffic:

| Concern | Implementation |
|---|---|
| **Secrets** | K8s `Secret` (`secrets.example.yaml` → copy to gitignored `secrets.yaml`) referenced via `secretKeyRef`; `.env` for local compose. Use sealed-secrets / External Secrets / Vault in prod. |
| **Auth** | API key (`X-API-Key`) on `/predict*`; key from `MLF_API_KEY` env (never a file). Auth off when unset (local dev). |
| **Rate limiting** | `slowapi` per-client (`--rate-limit`, default `60/minute`) → 429 on burst. |
| **Request caps** | `--max-instances` per request → 413. |
| **Multi-worker** | `gunicorn` + uvicorn workers (image); `mlf serve --workers N`. |
| **TLS** | `ingress-tls.yaml` — cert-manager + Let's Encrypt, terminates at the ingress. |
| **Drift** | `reference_stats.json` saved at train; serving exposes `mlf_feature_psi{feature}` + `GET /drift` (PSI vs training reference). |
| **Alerting** | `deploy/monitoring/alerts.yml` — error rate, p95 latency, PSI>0.2, no-predictions, target-down. |
| **Model quality** | `python -m ml_framework.monitoring.model_quality` joins predictions↔delayed labels → live accuracy to MLflow. |
| **Data contracts** | `pipeline/contracts.py` (Pandera) as the gated `validate_data` Airflow task. |
| **K8s posture** | `policies.yaml` — ResourceQuota, LimitRange, PDB, default-deny NetworkPolicy; non-root `securityContext`, dropped caps. |
| **Supply chain** | `.github/workflows/cd.yml` — build → Trivy scan → Syft SBOM → cosign sign → push; gated deploy. |
| **Load / chaos** | `tests/load/` — Locust + k6 SLO gate (p95<200ms, err<1%) + chaos/pod-kill notes. |

### Go-live checklist

- [ ] Real secrets provisioned (sealed-secrets/Vault); `secrets.yaml` **not** committed.
- [ ] `MLF_API_KEY` set; clients send `X-API-Key`; rate limit tuned to capacity.
- [ ] TLS cert issued (cert-manager) and HTTP→HTTPS redirect on.
- [ ] `HPA` limits validated against a **k6 load test**; PDB keeps ≥1 pod.
- [ ] Prometheus scraping the API; Grafana dashboard live; AlertManager wired and alerts firing on breach.
- [ ] Drift (`/drift`, `mlf_feature_psi`) and delayed-label quality job scheduled.
- [ ] `validate_data` contract gate green in the Airflow DAG.
- [ ] CD pipeline builds, **Trivy-clean**, SBOM archived, image cosign-signed.
- [ ] Chaos test passed (kill a pod, SLOs hold).

## Verification boundary

What CI actually proves, by job — worth stating precisely, because "validated" covers
three very different strengths of evidence here.

| Job | Proves | How |
|---|---|---|
| `test` | MLflow tracking/registry round-trip, DVC pipeline parse, `read_table`, `/metrics` + `/drift`, auth / rate-limit / size caps, PSI/KS drift, Pandera contracts | The suite, on Python 3.10–3.14, at a coverage floor of 80% |
| `gbdt-no-torch` | A GBDT bundle trains **and serves** with no deep-learning stack present; unavailable plugins and data backends are listed rather than hidden, and refuse with a `pip install` line | torch, pyspark and polars are genuinely absent — no meta-path trickery |
| `spark-contract` | The Spark backend **executes**: the collects, the `orderBy` tie-break, cross-engine bundle equality, and `spark_preprocess` against a live session | `setup-java` temurin 17 + a real `SparkSession`. The job asserts the gate is open *before* running, so it cannot go green by skipping |
| `mlops-validate` | The DAG, the pipeline modules and the load script import; every YAML parses; the kustomization renders **non-empty**; the Grafana dashboards are valid JSON | `py_compile`, YAML parsing, `kubectl kustomize` with a resource count check |

Spark moved out of the static-check column when `spark-contract` was added: its sort
tie-break is the feature's worst failure mode — Spark fixes the order of keys but not of
*equal* keys, so duplicate timestamps would silently produce different folds and
different scores run to run — and a test that only ever skips does not catch that.

What remains genuinely unproven by CI, and runs on your infrastructure: **Airflow and
Kubernetes** (both need clusters), **Trivy and cosign** against a real registry, and
**k6/Locust**, which need a live endpoint. The manifests are rendered but never applied;
the DAG is compiled but never scheduled.
