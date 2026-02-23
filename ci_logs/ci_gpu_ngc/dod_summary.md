# gpu-ngc DoD summary

- status: **skip**
- run_id: `22293141309`
- sha: `5839e89fd57c09f14189b43680c47dbf828742fa`
- has_runner: `false`
- probe_status: `http_403`
- gpu_job_result: `skipped`

## Findings
- No idle self-hosted GPU runner was available.
- Runner discovery API returned 403.

## Guidance
- Set repository secret RUNNER_DISCOVERY_TOKEN (PAT with self-hosted runner visibility).
