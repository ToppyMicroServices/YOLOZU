# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub Issue for potential security vulnerabilities.

Primary reporting channel (official):
- Open a **GitHub Security Advisory** for this repository (private by default):
- GitHub → this repo → **Security** → **Advisories** → **Report a vulnerability**

Fallback channel:
- Open a minimal public issue (no exploit details) and request a private channel with maintainers.

## Response and disclosure policy

- Initial maintainer response target: within **5 business days**.
- Triage/update cadence target: at least **weekly** until disposition.
- Coordinated disclosure is preferred; public disclosure should wait until a fix or mitigation exists.

## Supported versions

Security fixes are provided for:
- the latest published major/minor release line on PyPI
- release branches explicitly marked as supported in release notes

## Supported runtime scope (security maintenance target)

- Python: `3.10`, `3.11`, `3.12`
- OS: Linux and macOS (CPU path)
- GPU stack (CUDA/TensorRT/OpenCV CUDA): best-effort; environment-specific and validated via dedicated GPU workflows when available

## Dependency and third-party policy

- Python dependency/license policy: `docs/license_policy.md`
- Release-time contract boundary: `docs/release_1_0_stability.md`
- Required release gates include compatibility checks (`tools/check_golden_compatibility.py`) and packaging integrity gates (wheel/sdist content checks in CI)

## Supported versions

Only the latest released version on PyPI is supported for security fixes.
