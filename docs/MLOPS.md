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
    │  /predict  /predict_proba  /health  /metrics            (Deployment/Svc/Ingress/HPA)
    ▼
 Monitoring: Prometheus scrapes /metrics ─▶ Grafana ("ML Framework — Serving")
```

| Concern | Tool | Where |
|---|---|---|
| Experiment tracking + model registry | **MLflow** | `tracking/mlflow_utils.py`, `logging.backend: mlflow` |
| Data version control | **DVC** | `dvc.yaml`, `params.yaml` |
| Data processing (at scale) | **Apache Spark** | `pipeline/spark_preprocess.py` |
| Orchestration | **Apache Airflow** | `orchestration/airflow/dags/ml_pipeline.py` |
| Serving | **FastAPI** (+ KServe option) | `serving/api.py`, `deploy/k8s/api.yaml` |
| Infra / scaling | **Kubernetes** | `deploy/k8s/` (kustomize) |
| Monitoring | **Prometheus + Grafana** | `deploy/monitoring/`, `deploy/k8s/monitoring.yaml` |

Install everything: `pip install -e ".[dev,serve,mlops]"`.

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

Every run's params, metrics, and the portable bundle (`model.ckpt` + `scaler.pkl` +
`metadata.json`) are logged; a new **registered model version** is created. Serving
loads it back:

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
runs `mlf train` on the processed Parquet. `params.yaml` holds the tunables.

## 3. Spark — preprocessing

```bash
python -m ml_framework.pipeline.spark_preprocess \
  --input data/raw/sample.csv --output data/processed --target-col label
```

Reads raw CSV/Parquet, cleans + feature-engineers at scale, writes processed Parquet
that the datamodule reads directly (`read_table` handles CSV, Parquet files, and
Spark's Parquet directories). Requires Java 11/17.

## 4. Airflow — orchestration

```bash
cd orchestration/airflow
docker compose -f docker-compose.airflow.yml up airflow-init
docker compose -f docker-compose.airflow.yml up      # UI :8080 (airflow/airflow)
```

The `ml_framework_pipeline` DAG runs: `dvc_pull → spark_preprocess → train →
evaluate_gate → promote_model → trigger_deploy`. The gate fails the run if accuracy
is below `ACCURACY_GATE`; promotion sets the `@production` alias in MLflow.

## 5. Kubernetes — deploy + scale

```bash
kubectl kustomize deploy/k8s          # render/validate without applying
kubectl apply -k deploy/k8s           # namespace, api (Deploy/Svc/Ingress/HPA),
                                      # mlflow, minio, prometheus, grafana
```

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

The Python integrations (MLflow tracking/registry round-trip, DVC pipeline parse,
`read_table`, `/metrics` + `/drift`, auth / rate-limit / size caps, PSI/KS drift,
Pandera contracts) are covered by the test suite and run in CI. The cluster
components (Spark needs a JVM, Airflow/K8s need clusters, Trivy/cosign need a
registry, k6/Locust need a live endpoint) are validated by static checks —
`py_compile`, YAML parsing, `kustomize build` — and run on your infrastructure.
