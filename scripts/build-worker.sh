#!/usr/bin/env sh
# Build a worker image, smoke-test it without a GPU, and push it only if the test passes.
#   scripts/build-worker.sh image v18     # then set worker_images.<family>.tag in policy
#   scripts/build-worker.sh train v4
# The registry comes from MW_REGISTRY (e.g. docker.io/yourname); images are public, so they must
# never contain secrets: the Dockerfiles copy only code and policy, and .dockerignore drops .env.
set -eu
family="${1:?family: image or train}"
tag="${2:?tag, e.g. v18}"
registry="${MW_REGISTRY:?set MW_REGISTRY, e.g. docker.io/yourname}"
registry="${registry#docker.io/}"

case "$family" in
  image) file=docker/worker.Dockerfile; args="--build-arg EXTRA=image" ;;
  train) file=docker/worker-train.Dockerfile; args="" ;;
  *) echo "unknown family $family" >&2; exit 2 ;;
esac
image="$registry/mw-worker-$family:$tag"

# shellcheck disable=SC2086
docker build -q -f "$file" $args -t "$image" .
docker run --rm --entrypoint python "$image" -m manyworlds.worker.smoke "$family"
docker push -q "$image"
echo "pushed $image"
