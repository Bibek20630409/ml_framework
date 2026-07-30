"""
tests/load/locustfile.py
────────────────────────
Load test for the inference API — proves the autoscaling + latency SLOs before
you trust them.

    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089 and set users / spawn rate
    # headless SLO gate:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --headless -u 100 -r 10 -t 2m --only-summary

Env: MLF_API_KEY (sent as X-API-Key), MLF_N_FEATURES (default 6).
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

N_FEATURES = int(os.environ.get("MLF_N_FEATURES", "6"))
API_KEY = os.environ.get("MLF_API_KEY", "")


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @property
    def _headers(self) -> dict:
        return {"X-API-Key": API_KEY} if API_KEY else {}

    def _instances(self, n: int = 1) -> dict:
        return {"instances": [[random.gauss(0, 1) for _ in range(N_FEATURES)] for _ in range(n)]}

    @task(5)
    def predict(self) -> None:
        with self.client.post(
            "/predict", json=self._instances(), headers=self._headers, catch_response=True
        ) as r:
            if r.status_code == 429:
                r.success()  # rate limiting is expected under load, not a failure
            elif r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @task(1)
    def predict_batch(self) -> None:
        self.client.post("/predict", json=self._instances(16), headers=self._headers)

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
