FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:ca34639fbbd2411ad3b9518c340327e2e1e75674e93b2bb6897f35a0a0ba4626

WORKDIR /src
COPY . /src
COPY .clusterfuzzlite/requirements.lock /tmp/clusterfuzzlite-requirements.lock

RUN python3 -m pip install --require-hashes -r /tmp/clusterfuzzlite-requirements.lock
