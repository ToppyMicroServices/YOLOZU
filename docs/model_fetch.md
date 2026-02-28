# Model fetch

YOLOZU provides a registry-driven model download flow so teams can standardize weight intake before export/eval.

## Commands

List available model IDs:

```bash
yolozu list models
```

The table includes license and sha256 status to help you avoid unsafe downloads.

Download a model artifact:

```bash
yolozu fetch <model_id> --out models --accept-license
```

Optional flags:

- `--cache-dir ~/.cache/yolozu/models` (default cache root)
- `--registry /path/to/model_registry.json` (override packaged registry)
- `--force` (re-download and overwrite output)
- `--retries 3` and `--timeout 60` (download robustness)
- `--allow-unsafe` (allow missing sha256 in registry; not recommended)
- `--allow-non-apache` (allow copyleft/unknown licenses; not recommended)

## Safety and reproducibility

- `--accept-license` is required; otherwise `yolozu fetch` exits non-zero.
- Registry entries should include `sha256` for reproducibility. If `sha256` is missing, `yolozu fetch` requires `--allow-unsafe`.
- Licenses are treated as part of the safety boundary. If a model license is not Apache-friendly, `yolozu fetch` requires `--allow-non-apache`.
- Cached artifacts are stored under `~/.cache/yolozu/models` by default.
- Each download writes `models/<model_id>/meta.json` containing:
  - `source` (`hf_hub`, `github_release`, `official_url`, `mirror_urls`)
  - `source_url_used` (actual URL used when multiple mirrors are configured)
  - `version`
  - `license`
  - `sha256`
  - `created_at`

## Mirror fallback (`mirror_urls`)

When a registry entry uses `source.type = "mirror_urls"`, `yolozu fetch` tries URLs in order and
stops on the first successful download that passes SHA-256 verification.

Minimal registry snippet:

```json
{
  "id": "example-model",
  "source": {
    "type": "mirror_urls",
    "urls": [
      "https://mirror-a.example.com/model.bin",
      "https://mirror-b.example.com/model.bin"
    ]
  }
}
```

## Registry priority policy

For new model sources, prefer:

1. Hugging Face Hub (`hf_hub`)
2. GitHub Releases (`github_release`)
3. Direct official URLs (`official_url`)

This keeps revisions pin-able and improves long-term reproducibility.
