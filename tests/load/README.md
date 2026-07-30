# Load & chaos testing

Prove the autoscaling and latency SLOs **before** trusting them.

## Load (Locust — interactive)
```bash
pip install locust
locust -f tests/load/locustfile.py --host http://localhost:8000
# open http://localhost:8089
```

## Load (k6 — SLO gate, CI-friendly)
```bash
k6 run -e HOST=http://localhost:8000 -e API_KEY=$MLF_API_KEY tests/load/k6-script.js
```
Fails if p95 > 200ms or error rate > 1% (thresholds in the script). Watch the
Grafana "ML Framework — Serving" dashboard while it runs; confirm the HPA scales
pods up under load and back down after.

## Chaos
Verify resilience by injecting failure and watching recovery (PDB keeps ≥1 pod):
```bash
# kill a serving pod — the Deployment should reschedule, traffic keep flowing
kubectl -n ml-framework delete pod -l app=ml-framework-api --field-selector=status.phase=Running --grace-period=0 | head -1
```
For structured experiments (network latency, CPU stress, pod-kill schedules) use
**LitmusChaos** or **Chaos Mesh** and assert the SLOs hold via the k6 gate above.
