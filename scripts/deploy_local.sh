#!/bin/bash
# Build the model-serving Docker image locally and smoke-test its endpoints.
#
# Prerequisites: run notebooks/01_training.ipynb and notebooks/02_deployment.ipynb
# first - they produce models/model (the exported pyfunc model) and
# artifacts/drift_baseline_stats.json, which the image bakes in. Both are
# gitignored, so the image can only be built after the notebooks have run.
#
# Usage:
#   ./scripts/deploy_local.sh            # build, run on :5001, smoke-test
#   ./scripts/deploy_local.sh stop       # stop and remove the container
#   PORT=5005 ./scripts/deploy_local.sh  # use another host port
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:-vial-model-serve:local}"
CONTAINER_NAME="${CONTAINER_NAME:-vial-model-serve-local}"
PORT="${PORT:-5001}"

if [ "${1:-}" = "stop" ]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo "Container $CONTAINER_NAME removed (if it existed). Image $IMAGE_TAG kept."
    exit 0
fi

if [ ! -f models/model/MLmodel ]; then
    echo "ERROR: models/model not found. Run notebooks/01_training.ipynb and"
    echo "       notebooks/02_deployment.ipynb first (the export cell writes models/model)."
    exit 1
fi
if [ ! -f artifacts/drift_baseline_stats.json ]; then
    echo "ERROR: artifacts/drift_baseline_stats.json not found. Run notebooks/01_training.ipynb first."
    exit 1
fi

echo "==> Building image $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f docker/Dockerfile.model-serve .

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo "==> Starting container $CONTAINER_NAME on http://localhost:$PORT"
docker run -d --name "$CONTAINER_NAME" -p "$PORT:$PORT" "$IMAGE_TAG"

echo "==> Waiting for /health (importing torch takes ~30-60s)..."
healthy=0
for _ in $(seq 1 90); do
    body="$(curl -s -m 2 "http://localhost:$PORT/health" || true)"
    if echo "$body" | grep -q '"status": "ok"'; then healthy=1; break; fi
    sleep 2
done
if [ "$healthy" != "1" ]; then
    echo "ERROR: container did not become healthy. Recent logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
fi
echo "==> Healthy: $(curl -s "http://localhost:$PORT/health")"

echo "==> Smoke-testing endpoints"
python3 - "$PORT" <<'PYEOF'
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

port = sys.argv[1]
base = f"http://localhost:{port}"


def get(path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def post(path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read().decode()


s, body = get("/version")
print(f"GET /version -> {s} {body.strip()[:160]}")
s, body = get("/drift")
print(f"GET /drift   -> {s} {body.strip()[:160]}...")

images = sorted(Path("data/test").glob("*.jpg"))
if images:
    payload = {"instances": [base64.b64encode(images[0].read_bytes()).decode()]}
    for path in ("/invocations", "/infer"):
        s, body = post(path, payload)
        print(f"POST {path} ({images[0].name}) -> {s} {body[:160]}")
else:
    print("No data/test/*.jpg found - skipping POST smoke test (run 01_training.ipynb to download them)")
PYEOF

echo
echo "==> Done. Endpoints: http://localhost:$PORT/{health,invocations,infer,version,drift}"
echo "    Stop with: ./scripts/deploy_local.sh stop   (or: docker rm -f $CONTAINER_NAME)"
