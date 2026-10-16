#!/usr/bin/env bash
# Build the BrowserGym sandbox image and push it to GHCR.
#
# Usage:
#   GHCR_OWNER=my-org TAG=0.1.0 ./build_and_push_ghcr.sh
#
# Auth (one of):
#   - `docker login ghcr.io` beforehand with a PAT that has write:packages
#   - export GHCR_TOKEN=ghp_...  (this script logs in for you)
#
# The image must be world-readable (public package) unless your flash-sandbox
# orchestrator pulls GHCR with credentials -- flash-sandbox passes
# already-qualified refs (ghcr.io/...) through unchanged, no registry rewrite.
set -euo pipefail
cd "$(dirname "$0")"

: "${GHCR_OWNER:?Set GHCR_OWNER to your GitHub user/org (the GHCR namespace)}"
TAG="${TAG:-0.1.0}"
PLATFORM="${PLATFORM:-linux/amd64}"   # flash-sandbox backends are x86_64
IMAGE="ghcr.io/${GHCR_OWNER}/browsergym:${TAG}"

if [[ -n "${GHCR_TOKEN:-}" ]]; then
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_OWNER" --password-stdin
fi

echo "==> building $IMAGE ($PLATFORM)"
docker build --platform "$PLATFORM" -f Containerfile.browsergym -t "$IMAGE" .

echo "==> smoke-testing the image locally (bridge CLI + task registry)"
docker run --rm --platform "$PLATFORM" "$IMAGE" \
  python /opt/bgym/bridge.py health --port 1 || true   # client path is stdlib; nonzero expected w/o daemon
docker run --rm --platform "$PLATFORM" "$IMAGE" \
  python -c "import browsergym.miniwob, gymnasium as gym; print('miniwob tasks:', len([i for i in gym.envs.registry if i.startswith('browsergym/miniwob.')]))" || true

echo "==> pushing $IMAGE"
docker push "$IMAGE"

echo "==> also pushing :latest"
docker tag "$IMAGE" "ghcr.io/${GHCR_OWNER}/browsergym:latest"
docker push "ghcr.io/${GHCR_OWNER}/browsergym:latest"

cat <<EOF

Done. Point rollouts at it, e.g.:

  python -m uni_agent.tasks.browsergym.preprocess \\
      --benchmark browsergym.miniwob \\
      --image $IMAGE \\
      --output browsergym_miniwob.parquet

or set BROWSERGYM_IMAGE=$IMAGE for ad-hoc runs.
EOF
