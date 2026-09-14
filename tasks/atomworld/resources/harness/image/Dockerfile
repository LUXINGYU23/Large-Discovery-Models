FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl jq ripgrep && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes -r /tmp/requirements.lock
# OCI filesystem import does not propagate Docker ENV into Gondolin execs.
# Install into Python's default import path so every guest tool can import it.
COPY atomworld_tools /usr/local/lib/python3.11/site-packages/atomworld_tools
