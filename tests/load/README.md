# Load & chaos testing

Prove the autoscaling and latency SLOs **before** trusting them.

These scripts are **excluded from the default pytest run** (`addopts =
--ignore=tests/load` in `pyproject.toml`), so nothing here executes in CI. They need
a live endpoint and they take minutes; both are invoked by hand against a running API.

## Load (Locust — interactive)
```bash
pip install locust
locust -f tests/load/locustfile.py --host http://localhost:8000
# open http://localhost:8089 and set users / spawn rate
```

Headless, as a pass/fail gate in a pipeline:
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 \
       --headless -u 100 -r 10 -t 2m --only-summary
```

## Load (k6 — SLO gate, CI-friendly)
```bash
k6 run -e HOST=http://localhost:8000 -e API_KEY=$MLF_API_KEY tests/load/k6-script.js
```
Fails if p95 > 200ms or error rate > 1% (thresholds in the script). Watch the
Grafana "ML Framework — Serving" dashboard while it runs; confirm the HPA scales
pods up under load and back down after.

## Feature width and auth

Both scripts synthesize request bodies, so they have to know how wide an instance is.
The default is **6**, matching `data/raw/sample.csv` — point them at a bundle trained on
anything else without setting this and every request comes back 422, which reads like a
load failure and is not one.

| Setting | Locust | k6 |
|---|---|---|
| Feature count | `MLF_N_FEATURES` env | `-e N_FEATURES=` |
| API key (sent as `X-API-Key`) | `MLF_API_KEY` env | `-e API_KEY=` |
| Target host | `--host` | `-e HOST=` |

```bash
MLF_N_FEATURES=20 MLF_API_KEY=secret \
  locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 1m

k6 run -e HOST=http://localhost:8000 -e N_FEATURES=20 -e API_KEY=secret tests/load/k6-script.js
```

## Chaos
Verify resilience by injecting failure and watching recovery (PDB keeps ≥1 pod):
```bash
# kill a serving pod — the Deployment should reschedule, traffic keep flowing
kubectl -n ml-framework delete pod -l app=ml-framework-api --field-selector=status.phase=Running --grace-period=0 | head -1
```
For structured experiments (network latency, CPU stress, pod-kill schedules) use
**LitmusChaos** or **Chaos Mesh** and assert the SLOs hold via the k6 gate above.
