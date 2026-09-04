"""
test_endpoint.py
----------------
Quick sanity check for the local /v1/chat/completions endpoint.
Run with:  python test_endpoint.py
"""

import json
import time
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def http(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": "Bearer not-used"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


print("1. GET /health ...")
try:
    h = http("GET", "/health")
    print(f"   ok. model={h['base_model']}, adapter={h['adapter']}")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

SYSTEM = ("You are a Non-RT RIC rApp steering a fan of uplink beams to follow a moving crowd. "
          "Return ONLY one JSON object with keys: fan_center (-49..49), tilt (3..45), "
          "action (follow|widen|allocate), reason (short). No prose, JSON only.")
USER = "per_beam_counts=[1, 4, 12, 6, 2] (beam azimuths [-40, -20, 0, 20, 40] deg); current fan_center=0, tilt=20"

body = {
    "model": "beam-lora", "temperature": 0, "max_tokens": 150,
    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    "chat_template_kwargs": {"enable_thinking": False},
}

print()
print("2. POST /v1/chat/completions ...")
t0 = time.time()
try:
    resp = http("POST", "/v1/chat/completions", body)
    print(f"   ok in {time.time() - t0:.1f}s, usage={resp['usage']}")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

content = resp["choices"][0]["message"]["content"]
print()
print("3. Assistant content:")
print(f"   {content}")
try:
    parsed = json.loads(content)
    missing = {"fan_center", "tilt", "action", "reason"} - set(parsed.keys())
    if missing:
        print(f"   MISSING KEYS: {missing}")
    else:
        print(f"   all keys present. fan_center={parsed['fan_center']} tilt={parsed['tilt']} action={parsed['action']}")
except Exception as e:
    print(f"   NOT valid JSON: {e}")
