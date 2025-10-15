#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: DOCKER_IMAGE=<username/repo> [DOCKER_TAG=latest] [MODEL_URL=...] [MODEL_NAME=...] ./docker/publish.sh

Environment variables:
  DOCKER_IMAGE  (required) Docker Hub repository, e.g. myuser/vosk-gpu-ws
  DOCKER_TAG    (optional) Tag to apply/push (default: latest)
  DOCKER_PLATFORM (optional) Target platform for buildx (default: linux/amd64)
  MODEL_URL     (optional) Override model download URL at build time
  MODEL_NAME    (optional) Folder name inside the model archive

The script uses 'docker buildx build --platform=...' and pushes the image.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  echo "[publish] DOCKER_IMAGE env var is required (e.g. export DOCKER_IMAGE=myuser/vosk-gpu-ws)" >&2
  usage
  exit 1
fi

DOCKER_TAG=${DOCKER_TAG:-latest}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-linux/amd64}
declare -a BUILD_ARGS=()

if [[ -n "${MODEL_URL:-}" ]]; then
  BUILD_ARGS+=("--build-arg" "MODEL_URL=${MODEL_URL}")
fi

if [[ -n "${MODEL_NAME:-}" ]]; then
  BUILD_ARGS+=("--build-arg" "MODEL_NAME=${MODEL_NAME}")
fi

IMAGE_REF="${DOCKER_IMAGE}:${DOCKER_TAG}"

echo "[publish] Building ${IMAGE_REF} for platform ${DOCKER_PLATFORM} (buildx)"
if (( ${#BUILD_ARGS[@]} )); then
  docker buildx build \
    "${BUILD_ARGS[@]}" \
    --platform="${DOCKER_PLATFORM}" \
    --tag "${IMAGE_REF}" \
    --file docker/Dockerfile \
    --push \
    .
else
  docker buildx build \
    --platform="${DOCKER_PLATFORM}" \
    --tag "${IMAGE_REF}" \
    --file docker/Dockerfile \
    --push \
    .
fi

echo "[publish] Done"
