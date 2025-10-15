#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: DOCKER_IMAGE=<username/repo> [DOCKER_TAG=latest] [DOCKER_PLATFORM=linux/amd64] ./docker/publish.sh

Environment variables:
  DOCKER_IMAGE     (required) Docker Hub repository, e.g. myuser/sherpa-asr
  DOCKER_TAG       (optional) Tag to apply/push (default: latest)
  DOCKER_PLATFORM  (optional) Target platform for buildx (default: linux/amd64)

The script uses 'docker buildx build --platform=...' and pushes the image.
Ensure you're logged in with 'docker login'.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  echo "[publish] DOCKER_IMAGE env var is required (e.g. export DOCKER_IMAGE=myuser/sherpa-asr)" >&2
  usage
  exit 1
fi

DOCKER_TAG=${DOCKER_TAG:-latest}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-linux/amd64}

IMAGE_REF="${DOCKER_IMAGE}:${DOCKER_TAG}"

echo "[publish] Building ${IMAGE_REF} for platform ${DOCKER_PLATFORM} (buildx)"
docker buildx build \
  --platform="${DOCKER_PLATFORM}" \
  --tag "${IMAGE_REF}" \
  --file docker/Dockerfile \
  --push \
  .

echo "[publish] Done"
