# Consented Design-Partner Observation Kit

Use this kit for a short, consented observation of the YOLOZU stable lane. The
goal is to learn whether a practitioner who already has compatible
object-detection predictions and permitted ground truth can reach a validated
report, not to obtain an endorsement or collect their data.

The canonical task starts with the checked
[precomputed-predictions quickstart](../README.md#a-evaluate-from-precomputed-predictions-no-inference-deps)
and the stable
[predictions interface contract](../predictions_schema.md). The repository
smoke path is a proof of mechanics only: because `eval-coco --dry-run` does not
calculate metrics, do not count it as a real external evaluation or a
comparable metric report.

## Invitation

> YOLOZU maintainers are looking for practitioners who already have compatible
> object-detection predictions and permitted YOLO-format ground truth for a
> 30-45 minute onboarding observation. We want to see whether the stable
> `predictions.json` validation and evaluation path fits your workflow. This is
> a product-learning session, not a request for an endorsement. Participation
> is voluntary. We take minimal, de-identified workflow notes and do not record
> audio, video, or the screen by default. You may use the repository smoke data,
> or keep a sanitized artifact entirely in your own environment. Please do not
> send datasets, model files, predictions, logs, credentials, personal data, or
> vulnerability details. Report security findings privately to
> `security@toppymicros.com`; policy:
> <https://toppymicros.com/security-policy.html>. If interested, reply with only
> a broad stack category, such as Ultralytics, Detectron2, MMDetection, or
> custom.

## Consent and recording boundary

Before any observation, obtain an explicit `yes` to participation and minimal
note-taking. Read this boundary aloud:

> I will record only a pseudonymous session ID, broad persona and stack
> categories, the job to be done, elapsed times, completion states, blocker
> categories, abandoned steps, help given, and follow-up intent. You can skip a
> question, pause, or stop at any time. Please do not show or send secrets,
> personal data, proprietary artifacts, raw logs, or vulnerability details.
> Audio, video, and screen recording are off. Session consent does not grant
> permission to publish your name, organization, logo, quotation, artifact, or
> case study. Private session notes and maintainer-controlled correspondence
> are kept for a 14-day correction or deletion window, then the de-identified
> counts and categories are aggregated and the individual material is deleted.
> If you consent to one follow-up, the minimum contact information may remain
> until that follow-up or for at most 90 days. Provider-managed backups may
> follow the provider's separate retention. After irreversible aggregation
> without an identity mapping, an individual record cannot be located.

- If participation or note-taking consent is not explicit, do not run the
  session or retain notes.
- Screen sharing is optional and is not recorded. The participant may stop it
  before opening private files or tools.
- Any audio, video, or screen recording requires separate prior written consent
  that states the purpose, access boundary, retention period, and deletion date.
  A recording must never be committed to this repository.
- Public use of a participant name, organization, logo, quotation, artifact, or
  case study requires separate explicit written permission. Participation,
  note-taking, recording, and public attribution are distinct permissions.
- A participant may ask for uncommitted session notes to be deleted. Once notes
  are irreversibly aggregated without an identity mapping, explain that an
  individual record can no longer be located.

## Pre-session checklist

- [ ] Assign a pseudonymous ID such as `DP-2026-01`; keep any contact-to-ID
      mapping outside the repository.
- [ ] Confirm the participant's real-evaluation task is object detection and
      that permitted YOLO-format ground truth is available under
      `images/<split>` and `labels/<split>`.
- [ ] Choose the repository smoke fixture or a participant-owned sanitized
      artifact that stays in their environment.
- [ ] Confirm the participant's wrapped predictions use the canonical
      normalized `cxcywh` bbox shape, contain contiguous zero-based `class_id`
      values plus `score`, and pass strict validation.
- [ ] Confirm that every prediction `image` key exactly matches a dataset key
      or has a unique matching basename.
- [ ] Confirm contiguous class IDs mean the same thing in predictions and
      ground truth. Alternate raw bbox formats and category-ID mapping lanes are
      outside this observation; normalize them before the session.
- [ ] Confirm COCOeval support is installed in the participant environment with
      `yolozu[coco]` and that `python3 -c 'import pycocotools'` succeeds. The
      smoke dry run does not check this dependency.
- [ ] Ask the participant to close unrelated windows and remove credentials,
      tokens, customer names, private paths, and notifications from view.
- [ ] Confirm the participant will not upload or send model weights, datasets,
      predictions, raw logs, or environment dumps.
- [ ] Share the stable quickstart and predictions interface contract links.
- [ ] Confirm whether the checked smoke route and the participant-data
      evaluation will both be attempted.
- [ ] Define two timing points: start the activation timer when the participant
      begins the first stable-lane command; mark when they locate the checked
      dry-run report; stop only when they locate a real comparable report.
      Record pauses and moderator help separately.
- [ ] Read the consent boundary and record `yes` or stop.
- [ ] Keep the private security route in
      [SECURITY.md](../../SECURITY.md) ready; do not put vulnerability details
      in session notes, public issues, or Beads.

## Checked observation task

Start with the repository-owned smoke fixture:

```bash
python3 -m yolozu validate dataset data/smoke --strict
python3 -m yolozu validate predictions \
	data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_coco_eval_dry_run.json
```

This is the current checked stable-lane quickstart. Treat warnings, moderator
help, abandoned commands, and time spent finding the report as observations.
Do not treat the dry-run report's null metrics as evaluation results.

### Participant-local real-evaluation preflight

Use this lane only for compatible object-detection predictions and permitted
YOLO-format ground truth. The participant keeps all files local. Before timing
the real evaluation, confirm:

- `python3 -c 'import pycocotools'` succeeds;
- strict dataset and predictions validation both exit `0`;
- `image` join keys match exactly or by unique basename;
- bboxes use canonical normalized `cxcywh`; and
- contiguous zero-based class IDs match ground truth.

For canonical normalized `cxcywh` bboxes and already contiguous `class_id`
values, the participant-local command template is:

```bash
python3 -m yolozu validate dataset \
	/absolute/path/to/yolo-dataset \
	--split val \
	--strict
python3 -m yolozu validate predictions \
	/absolute/path/to/wrapped_predictions.json --strict
python3 -m yolozu eval-coco \
	--dataset /absolute/path/to/yolo-dataset \
	--split val \
	--predictions /absolute/path/to/wrapped_predictions.json \
	--bbox-format cxcywh_norm \
	--output /absolute/path/to/coco_eval.json
```

Do not substitute an absolute bbox format or a category-ID mapping lane in this
observation: those artifacts do not satisfy the canonical strict-validation
preflight above. Normalize them to canonical `cxcywh_norm` bboxes and
contiguous zero-based class IDs before the session, or record the real
evaluation as `not attempted`.

The observer records only completion state, elapsed time, and de-identified
blocker categories. The participant does not send the artifact or report. Count
a `first comparable report` only when the non-dry command exits `0`, the report
contains non-null metrics, and all preflight checks above passed against the
participant's permitted ground truth. Otherwise record `not reached`; never
infer completion from the smoke dry run.

## Moderator script

1. **Open:** explain that the session tests the workflow, not the participant.
   Confirm the time boundary, voluntary participation, note-taking consent, and
   recording-off state.
2. **Context:** ask for a broad persona class, broad current stack, and one
   sentence describing the evaluation job. Do not ask for an employer, project,
   dataset, model, customer, or repository name.
3. **Observe the checked smoke route:** ask the participant to think aloud while using
   the checked task. Let them choose the documentation route. Record each
   command outcome, warnings, report-discovery time, abandoned steps, and any
   help given.
4. **Observe the real handoff when permitted:** complete the participant-local
   preflight, then ask the participant to validate a sanitized wrapped
   `predictions.json` and evaluate it locally. They retain the files. Stop if
   private or sensitive content appears.
5. **Debrief:** ask:
   - What did you expect to happen?
   - Which step was unclear, blocked, or unnecessary?
   - Did you abandon or work around any step?
   - Would you use this for another evaluation? Why or why not?
   - May the maintainers contact you for one follow-up? Record only
     `yes`, `no`, or `unsure` in the worksheet.
6. **Close:** summarize only the categories you plan to retain, invite
   corrections, repeat the private security route, and state that attribution
   would require a separate permission request.

## Privacy-safe worksheet

Copy this template into private working notes. Do not commit an individual
external-session worksheet or the contact-to-ID mapping. Transfer only reviewed
aggregate counts and de-identified blocker categories into an adoption report.

| Field | Allowed value |
|---|---|
| Session ID | Pseudonymous ID with no contact information |
| Session date | Date or week; use a week when an exact date could identify someone |
| Participant type | `external design partner` or `maintainer dry-run` |
| Consent to participate / minimal notes | `yes`; otherwise stop. `not applicable` is allowed only for a maintainer rehearsal with no external participant |
| Recording | `off`, unless separately consented outside this worksheet |
| Persona class | Broad role, for example practitioner, ML engineer, or evaluator |
| Current stack | Broad framework/runtime category only |
| Job to be done | One de-identified sentence |
| Artifact route | `repository smoke` or `participant-local sanitized artifact` |
| First install | `completed`, `not attempted`, or `blocked` |
| First proof/demo | `completed`, `not attempted`, or `blocked` |
| First real evaluation | `completed`, `not attempted`, or `blocked` |
| Time to checked report | Elapsed minutes or `not reached` |
| Time to first comparable report | Elapsed minutes or `not reached`; never use the smoke dry run |
| Command/step outcomes | Category and exit state only; no raw logs |
| Blocker categories | Sanitized categories, severity, and affected step |
| Abandoned steps/workarounds | Sanitized description |
| Moderator help | What kind of help and when, without verbatim private content; `not applicable` only for a maintainer rehearsal |
| Follow-up intent | `yes`, `no`, or `unsure`; `not applicable` only for a maintainer rehearsal |
| Public attribution permission | `not requested` or separately documented; never implied |
| Beads follow-up | Sanitized Bead ID or `threshold not met` |

Do not put names, organizations, logos, email addresses, usernames, IP
addresses, customer or project names, dataset or model names, file paths,
predictions, labels, metrics tied to a private project, raw logs, screenshots,
recordings, verbatim private quotations, credentials, or vulnerability details
in the worksheet.

## Anonymization and retention

1. Keep session-specific correspondence, the individual worksheet, and any ID
   mapping outside the repository with access limited to maintainers who need
   it for the session.
2. Replace identifying details with the allowed worksheet categories. Generalize
   rare category combinations when they could identify a participant.
3. Paraphrase workflow observations. Do not retain a quotation unless separate
   written publication permission covers the exact text.
4. Send the de-identified follow-up, allow 14 calendar days for correction or
   deletion requests, and do not aggregate the session before that window
   closes. Then delete the individual worksheet, raw notes, contact-to-ID
   mapping, and maintainer-controlled session correspondence from the active
   mailbox and trash. If the participant consented to one follow-up, the
   minimum contact address and ID mapping may remain only until that follow-up
   occurs or 90 days after the session, whichever comes first. Do not retain
   the earlier correspondence body, attachments, or session summary under this
   exception; delete new follow-up correspondence and attachments when the
   follow-up is complete.
5. A deletion request received before aggregation removes the session from the
   tally and deletes its uncommitted notes, mapping, and maintainer-controlled
   correspondence. Provider-managed backups may follow a separate provider
   retention policy; do not describe active-mailbox deletion as a backup purge.
   Recordings follow the earlier, separately consented deletion date.
6. Publish only aggregate counts or de-identified blocker categories. A
   session's participation does not make its identity public.
7. Keep public-name and logo permission separate from session, note-taking, and
   recording consent. At that later permission request, disclose how long its
   record will be retained and how approved public use can be withdrawn. Keep
   the permission record outside the repository only while the approved public
   use remains active, and delete it when that use ends. A public document
   should include only the scope the participant approved.

## Blocker-to-Beads triage

- Normalize each observation to a blocker category and affected stable-lane
  step.
- Search open Beads before creating a new item.
- Create or update a sanitized Beads issue when the same blocker category
  appears in at least two consented external sessions. Include the count,
  impact, affected step, reproducible repository-owned evidence when available,
  and testable acceptance criteria. Do not include participant IDs or private
  artifacts.
- A single reproducible data-loss, incorrect-result, or stable-path
  availability failure may be filed immediately; label it as single-session
  evidence rather than calling it repeated.
- Increment a private aggregate blocker tally with only category and count.
  Do not retain a one-off observation in a repository or public session
  summary. Delete its individual notes under the retention rule; create or
  update a Beads issue if the category count later reaches the threshold.
- Never file vulnerability details in Beads. Stop the observation and direct
  the participant to `security@toppymicros.com` through
  [SECURITY.md](../../SECURITY.md). A sanitized engineering follow-up can be
  considered only after coordinated disclosure makes that safe.

## Post-session follow-up

> Thank you for the YOLOZU workflow observation. The de-identified summary we
> retained is: broad job category `[category]`; checked smoke route
> `[completed/blocked/not attempted]`; participant-data evaluation
> `[completed/blocked/not attempted]`; elapsed time `[minutes/not reached]`;
> blocker categories `[categories/none]`; follow-up intent
> `[yes/no/unsure]`. We did not retain your artifacts, raw logs, credentials, or
> private project details. Please reply within 14 calendar days if this summary
> needs correction or if you want uncommitted session notes deleted. We will
> not publish your name, organization, logo, quotation, artifact, or case study
> without a separate explicit written permission. Report any security finding
> privately to `security@toppymicros.com`; policy:
> <https://toppymicros.com/security-policy.html>. Do not include vulnerability
> details in a public reply or issue.
