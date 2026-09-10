# Reconstructed from ai-model-dev:latest and DEVCONTAINER.md (2026-09-10).
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY .devcontainer/requirements-ai.txt /tmp/requirements-ai.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install -r /tmp/requirements-ai.txt \
    && rm /tmp/requirements-ai.txt

USER vscode
RUN python -m ipykernel install --user --name ai-container --display-name "Python (AI container)"
