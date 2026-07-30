// tests/load/k6-script.js
// SLO gate with k6: ramps request rate and FAILS the run if the latency/error
// SLOs are breached — usable directly in CI or a canary check.
//
//   k6 run -e HOST=http://localhost:8000 -e API_KEY=secret tests/load/k6-script.js
import http from "k6/http";
import { check } from "k6";

const HOST = __ENV.HOST || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "";
const N_FEATURES = parseInt(__ENV.N_FEATURES || "6", 10);

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-arrival-rate",
      startRate: 10,
      timeUnit: "1s",
      preAllocatedVUs: 50,
      maxVUs: 200,
      stages: [
        { target: 50, duration: "1m" },
        { target: 200, duration: "2m" },
        { target: 0, duration: "30s" },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"], // < 1% errors
    http_req_duration: ["p(95)<200"], // p95 < 200ms
  },
};

function instance() {
  const row = [];
  for (let i = 0; i < N_FEATURES; i++) row.push(Math.random() * 2 - 1);
  return { instances: [row] };
}

export default function () {
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  const res = http.post(`${HOST}/predict`, JSON.stringify(instance()), { headers });
  check(res, { "status is 200 or 429": (r) => r.status === 200 || r.status === 429 });
}
