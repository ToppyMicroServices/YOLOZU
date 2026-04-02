Hash-locked requirement sets used by CI, container builds, and reproducibility-sensitive workflows live here.

- `requirements-*.lock` files are intentionally grouped out of the repository root.
- Human-oriented install entrypoints such as `requirements.txt`, `requirements-dev.txt`, and `requirements-test.txt` stay at the repository root.
- When updating workflow or docs references, prefer `requirements-locks/<file>.lock`.
