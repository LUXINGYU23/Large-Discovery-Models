FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl file git jq ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY lock/requirements.lock /tmp/requirements.lock
RUN python -m pip install --retries 10 --timeout 120 -r /tmp/requirements.lock
