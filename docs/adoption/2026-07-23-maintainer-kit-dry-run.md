# Design-Partner Kit Maintainer Dry-Run - 2026-07-23

This was a repository maintainer's procedural dry-run of the
[observation kit](design_partner_observation_kit.md). It used only
repository-owned smoke data and did not involve an external participant. It is
not a design-partner session, not adoption evidence, and does not count toward
the three consented external observations.

## Execution evidence

The maintainer ran the checked stable-lane task from a clean worktree and wrote
the generated report under `/private/tmp`, outside the repository.

| Step | Repository-owned input | Result | Measured command time |
|---|---|---|---:|
| Strict dataset validation | `data/smoke` | exit `0` | 0.17 s |
| Strict predictions validation | `data/smoke/predictions/predictions_dummy.json` | exit `0`; ten legacy-entry migration notices | 0.06 s |
| COCO dry-run report | same smoke dataset and predictions | exit `0`; report written with `dry_run: true` and null metrics | 0.08 s |

The total measured command execution time was 0.31 seconds. This excludes
reading, discussion, and report-discovery time and therefore is not an
onboarding-time claim. The generated report was inspected only to confirm its
dry-run status and was not committed.

The participant-local preflight command
`python3 -c 'import pycocotools'` exited `1` because the optional dependency was
not installed in this rehearsal environment. A non-dry evaluation was therefore
not attempted. This confirms that smoke success alone cannot qualify a
comparable report; it is not an external blocker observation.

## Completed privacy-safe worksheet

| Field | Dry-run value |
|---|---|
| Session ID | `MAINTAINER-DRY-RUN-01` |
| Session date | `2026-07-23` |
| Participant type | `maintainer dry-run` |
| Consent to participate / minimal notes | not applicable; no external participant |
| Recording | `off` |
| Persona class | repository maintainer |
| Current stack | repository Python CLI |
| Job to be done | verify that the observation task and worksheet can be completed with repository-owned data |
| Artifact route | `repository smoke` |
| First install | `not attempted` |
| First proof/demo | `not attempted` |
| First real evaluation | `not attempted` |
| Time to checked report | command execution 0.31 s; onboarding time not measured |
| Time to first comparable report | `not reached`; the smoke run was dry-run only |
| Command/step outcomes | three exits `0`; migration notices were non-blocking |
| Blocker categories | environment prerequisite: optional COCOeval dependency absent; external repetition threshold not applicable |
| Abandoned steps/workarounds | non-dry evaluation was outside this rehearsal and was not attempted |
| Moderator help | not applicable |
| Follow-up intent | not applicable |
| Public attribution permission | `not requested` |
| Beads follow-up | `threshold not met` |

## Dry-run conclusion

The worksheet represented consent and follow-up as not applicable for an
internal rehearsal, kept the checked smoke route separate from a real external
evaluation, and required no personal, employer, customer, artifact, or security
data. The migration notices did not block the checked task. The missing optional
COCOeval dependency correctly kept the non-dry result at `not attempted`. These
maintainer-only observations do not meet the repeated external-blocker
threshold.

Real design-partner outcomes remain unknown until consented external sessions
are completed.
