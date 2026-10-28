Hash-locked requirement sets used by CI, container builds, and reproducibility-sensitive workflows live here.

- `requirements-*.lock` files are intentionally grouped out of the repository root.
- Human-oriented install entrypoints such as `requirements.txt`, `requirements-dev.txt`, and `requirements-test.txt` stay at the repository root.
- When updating workflow or docs references, prefer `requirements-locks/<file>.lock`.
- `requirements-scheduled.lock` includes the complete runtime/build dependency set and pre-recorded PyPI wheel hashes for the two scheduled monitors (CPython 3.12, Linux x86_64). Install it directly with `pip --require-hashes --only-binary=:all:`; it is not a cross-platform lock.
