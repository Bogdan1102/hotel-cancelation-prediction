"""
predict-test.py — Test the local Flask service end-to-end.

Starts serve.py as a subprocess, waits for it to come up,
sends a /health check and a /predict POST with sample_request.json,
and asserts the response shape.

Usage:
    python predict-test.py

Exit codes:
    0 = all tests passed
    1 = a test failed
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 9696
BASE = f"http://127.0.0.1:{PORT}"
SAMPLE_FILE = "sample_request.json"


def wait_for_server(timeout=40):
    """Wait for /health to return 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def test_health():
    """GET /health must return 200 with status=ok."""
    print("Test 1: GET /health ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
            body = json.loads(r.read().decode())
            assert r.status == 200, f"expected 200, got {r.status}"
            assert body.get("status") == "ok", f"unexpected body: {body}"
        print("PASS")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_predict():
    """POST /predict must return cancellation_probability and is_canceled."""
    print("Test 2: POST /predict ...", end=" ", flush=True)

    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    req = urllib.request.Request(
        f"{BASE}/predict",
        data=json.dumps(sample).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode())
            assert r.status == 200, f"expected 200, got {r.status}"
            assert "cancellation_probability" in body, f"missing key: {body}"
            assert "is_canceled" in body, f"missing key: {body}"
            p = body["cancellation_probability"]
            assert 0.0 <= p <= 1.0, f"probability out of [0,1]: {p}"
            assert isinstance(body["is_canceled"], bool), \
                f"is_canceled should be bool: {body['is_canceled']}"
        print(f"PASS (probability={p:.4f}, is_canceled={body['is_canceled']})")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    # Start server in subprocess
    print("Starting serve.py ...")
    proc = subprocess.Popen(
        [sys.executable, "serve.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            print("ERROR: server did not come up within 40s")
            return 1

        results = [test_health(), test_predict()]
        ok = all(results)
        print()
        print(f"{sum(results)}/{len(results)} tests passed")
        return 0 if ok else 1
    finally:
        # Clean shutdown (cross-platform: terminate() works on Windows + Linux/Mac)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
