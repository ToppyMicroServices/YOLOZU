# Docker images (CPU) + GHCR / NGC

This folder contains **CPU-friendly** Dockerfiles that install the `yolozu` CLI via `pip`.

Published images (if enabled in CI) live on **GitHub Container Registry (GHCR)** and can also be mirrored to **NVIDIA NGC** under `nvcr.io/yolozu/...` when `NGC_API_KEY` is configured.

## Published images

For container-related pull requests, `.github/workflows/container.yml` now performs build validation without publishing images.
On a release tag `vX.Y.Z` (or a manual workflow run with `release_tag=vX.Y.Z`), the same workflow publishes:

- Minimal (no torch):
  - `ghcr.io/toppymicroservices/yolozu:X.Y.Z`
  - `nvcr.io/yolozu/yolozu:X.Y.Z`
- Demo (includes torch extra):
  - `ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z`
  - `nvcr.io/yolozu/yolozu-demo:X.Y.Z`
- Both images also receive the `latest` tag on release-tag/manual-release builds.

Example pulls:

```bash
docker pull ghcr.io/toppymicroservices/yolozu:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu:latest
docker pull ghcr.io/toppymicroservices/yolozu-demo:latest
docker pull nvcr.io/yolozu/yolozu:X.Y.Z
docker pull nvcr.io/yolozu/yolozu-demo:X.Y.Z
docker pull nvcr.io/yolozu/yolozu:latest
docker pull nvcr.io/yolozu/yolozu-demo:latest
```

Notes:
- Pull requests that touch container workflow/deploy/packaging inputs now build these Dockerfiles before merge, but they do not publish to GHCR/NGC.
- If you created the tag *before* adding the workflow, it won’t auto-run for that historical tag.
  Use **Actions → container → Run workflow** and provide `release_tag=vX.Y.Z`, or cut a new tag.
- After the first push, you may need to set the package visibility to **Public** in GitHub UI.
- NGC mirrors require repository secret `NGC_API_KEY`.

## Run examples

Minimal image:

```bash
docker run --rm ghcr.io/toppymicroservices/yolozu:X.Y.Z doctor --output -
docker run --rm ghcr.io/toppymicroservices/yolozu:X.Y.Z demo instance-seg
```

Demo image:

```bash
docker run --rm ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z demo instance-seg
docker run --rm ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z demo continual --method ewc_replay
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

The CPU images are intentionally bootstrapped from exact-version lockfiles via
\path{tools/ci/install_with_hashes.py}, so container breakage should be debugged by inspecting
\path{requirements-locks/requirements-runtime.lock} and
\path{requirements-locks/requirements-demo-extra.lock} first.
