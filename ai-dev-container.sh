#!/usr/bin/env bash
set -euo pipefail

ai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${ai_root}/char-completion-transformer"
image_name="ai-model-dev"
action="${1:-shell}"
[[ $# -gt 0 ]] && shift

mkdir -p "${ai_root}/models" "${ai_root}/cache/huggingface"

build_image() {
    docker build --tag "${image_name}" \
        --file "${ai_root}/.devcontainer/Dockerfile" "${ai_root}"
}

if [[ "${action}" == "build" ]]; then
    build_image
    exit
fi

if [[ "${action}" == "help" || "${action}" == "--help" || "${action}" == "-h" ]]; then
    echo "Usage: ./ai-dev-container.sh {build|shell|verify|jupyter} [arguments]"
    echo "Set AI_JUPYTER_PORT to change the host Jupyter port (default: 8888)."
    exit
fi

if ! docker image inspect "${image_name}:latest" >/dev/null 2>&1; then
    echo "Building missing Docker image ${image_name}:latest..."
    build_image
fi

docker_args=(
    run --rm
)
if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
else
    docker_args+=(-i)
fi
docker_args+=(
    --user "$(id -u):$(id -g)"
    --env HOME=/tmp
    --env HF_HOME=/ai-cache
    --env MODEL_DIR=/models
    --env AI_MODEL_DEV_ROOT=/workspace/ai-model-dev
    --env PATH=/workspace/ai-model-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
    --mount "type=bind,source=${ai_root}/models,target=/models,readonly"
    --mount "type=bind,source=${ai_root}/cache/huggingface,target=/ai-cache"
    --mount "type=bind,source=${ai_root},target=/workspace/ai-model-dev"
    --workdir /workspace/ai-model-dev/char-completion-transformer
)

case "${action}" in
    shell)
        port="${AI_JUPYTER_PORT:-8888}"
        docker "${docker_args[@]}" --publish "127.0.0.1:${port}:8888" "${image_name}" bash "$@"
        ;;
    verify)
        docker "${docker_args[@]}" "${image_name}" \
            python /workspace/ai-model-dev/.devcontainer/verify_environment.py "$@"
        ;;
    jupyter|jupiter)
        port="${AI_JUPYTER_PORT:-8888}"
        echo "Open the tokenized URL at http://127.0.0.1:${port}"
        docker "${docker_args[@]}" --publish "127.0.0.1:${port}:8888" "${image_name}" \
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "$@"
        ;;
    *)
        echo "Unknown action: ${action}" >&2
        echo "Run: ./ai-dev-container.sh help" >&2
        exit 2
        ;;
esac
