import http from 'k6/http';
import { check, sleep } from 'k6';

// Load profile for the SSE chat endpoint.
//
// Honest accounting of what this measures:
//   * k6's http.post buffers the whole response, so http_req_duration is the
//     duration of the ENTIRE stream, not time-to-first-token. Asserting TTFT on
//     it (as this script previously did) measures the wrong thing entirely.
//     http_req_waiting is time-to-first-byte, which is a defensible TTFT proxy.
//   * VUS below is the real concurrency. It was set to 200 while the comment and
//     README both claimed 1000; raise VUS if you want to substantiate a larger
//     number, and note Render's free tier will not sustain it.
//   * Each iteration sleeps 1s, so sustained RPS is roughly VUS / (stream_time + 1).
const VUS = Number(__ENV.VUS || 200);

export const options = {
  vus: VUS,
  duration: __ENV.DURATION || '60s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    // Time to first byte -- the actual TTFT proxy.
    http_req_waiting: ['p(95)<400'],
  },
};

export default function () {
  const url = __ENV.API_URL || 'http://localhost:8000/api/v1/chat/stream';
  const payload = JSON.stringify({
    message: 'What is first-line therapy for hypertension with CKD?',
    // Reuse a small pool of threads per VU. A unique thread per iteration grew
    // the thread store without bound for the whole run, so the test measured
    // memory growth as much as request latency.
    thread_id: `load_${__VU}`,
  });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      // Sent only when provided; required if the target has ENABLE_AUTH=true,
      // otherwise the run would just be measuring 401s.
      ...(__ENV.API_KEY ? { 'X-API-Key': __ENV.API_KEY } : {}),
    },
    timeout: '30s',
  };
  const res = http.post(url, payload, params);
  check(res, {
    'status 200': (r) => r.status === 200,
    'is event-stream': (r) => (r.headers['Content-Type'] || '').includes('text/event-stream'),
    'has citation': (r) => r.body && r.body.includes('[1]'),
    'not rate limited': (r) => r.status !== 429,
  });
  sleep(1);
}
