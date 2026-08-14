import http from 'k6/http';
import { check, sleep } from 'k6';

// 1000 concurrent SSE (as close as k6 can) — hits POST streaming
export const options = {
  vus: 200,
  duration: '60s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<400'], // TTFT proxy
  },
};

export default function () {
  const url = __ENV.API_URL || 'http://localhost:8000/api/v1/chat/stream';
  const payload = JSON.stringify({
    message: 'What is first-line therapy for hypertension with CKD?',
    thread_id: `load_${__VU}_${__ITER}`,
  });
  const params = { headers: { 'Content-Type': 'application/json' }, timeout: '10s' };
  const res = http.post(url, payload, params);
  check(res, {
    'status 200': (r) => r.status === 200,
    'is event-stream': (r) => (r.headers['Content-Type'] || '').includes('text/event-stream'),
    'has citation': (r) => r.body && r.body.includes('[1]'),
  });
  sleep(1);
}
