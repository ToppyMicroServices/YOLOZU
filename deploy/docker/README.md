# Docker images (CPU) + GHCR

This folder contains **CPU-friendly** Dockerfiles that install the `yolozu` CLI via `pip`.

Published images (if enabled in CI) live on **GitHub Container Registry (GHCR)**.

## GHCR images

On a release tag `vX.Y.Z`, the workflow `.github/workflows/container.yml` publishes:

- Minimal (no torch): `ghcr.io/toppymicroservices/yolozu:X.Y.Z`
- Demo (includes torch extra): `ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z`

Example pulls:

```bash
docker pull ghcr.io/toppymicroservices/yolozu:1.0.0
docker pull ghcr.io/toppymicroservices/yolozu-demo:1.0.0
```

Notes:
- If you created the tag *before* adding the workflow, it won’t auto-run for that historical tag.
  Use **Actions → container → Run workflow** and select the tag, or cut a new tag.
- After the first push, you may need to set the package visibility to **Public** in GitHub UI.

## Run examples

Minimal image:

```bash
docker run --rm ghcr.io/toppymicroservices/yolozu:1.0.0 doctor --output -
docker run --rm ghcr.io/toppymicroservices/yolozu:1.0.0 demo instance-seg
```

Demo image:

```bash
docker run --rm ghcr.io/toppymicroservices/yolozu-demo:1.0.0 demo instance-seg
docker run --rm ghcr.io/toppymicroservices/yolozu-demo:1.0.0 demo continual --method ewc_replay
```

## Local build

Minimal:

```bash
docker build --pull -f deploy/docker/Dockerfile -t yolozu:local .
docker run --rm yolozu:local --help
```

Demo:

```bash
docker build --pull -f deploy/docker/Dockerfile.demo -t yolozu-demo:local .
docker run --rm yolozu-demo:local demo instance-seg
docker run --rm yolozu-demo:local demo continual --method ewc_replay
```

## Container update

Refresh base layers and rebuild locally:

```bash
docker build --pull -f deploy/docker/Dockerfile -t yolozu:local .
docker build --pull -f deploy/docker/Dockerfile.demo -t yolozu-demo:local .
```

Optional Python base override:

```bash
docker build --pull --build-arg PYTHON_IMAGE=python:3.12-slim-bookworm -f deploy/docker/Dockerfile -t yolozu:local .
```
