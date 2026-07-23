# Monthly Aggregate Feedback Review

Copy this file to `docs/adoption/YYYY-MM-feedback-review.md` after the calendar
month closes. Record aggregates and consented public links only. Do not paste
email contents, personal data, organization names, private artifact names, or
security-report details. Link an individual public report only when its intake
explicitly permits public citation; otherwise record `aggregate-only`.

Review month:

Review date:

Owner: YOLOZU maintainers (`develop@toppymicros.com`)

Linked Beads monthly-review item:

## Source coverage

| Source | Reviewed through | Included items | Exclusions or unavailable evidence |
|---|---|---:|---|
| GitHub Issues |  |  | Exclude pull requests, maintainer tests, and security reports |
| GitHub Discussions |  |  | Exclude maintainer announcements and polls |
| General support email |  |  | Count only feedback with anonymous aggregate-use consent |

Use the public APIs already listed in [`README.md`](README.md) to collect Issues
and Discussions. Review the full month, not only items that remain open. Do not
copy support-mailbox content into the repository.

## Response target

The public target is initial triage or a request for missing information within
5 business days. It is not a resolution guarantee. Apply the auditable clock
and qualifying-human-response definitions in [`../support.md`](../support.md).

| Qualifying consented non-security requests | Initial response within target | Outside target | Unknown |
|---:|---:|---:|---:|
|  |  |  |  |

Reason for any aggregate outside-target count:

## Aggregate demand

Frequency is the number of independent, qualifying requests in the review
month. Do not count maintainer-authored test reports, duplicates from the same
workflow, inferred downloads, or feedback that lacks aggregate-use consent.
In the `Public evidence` column, use an individual URL only when public-citation
consent was selected; otherwise write `aggregate-only`.

| Category | Frequency | Highest non-security impact | Resolved or answered | Recurring blocker summary | Public evidence | Beads action |
|---|---:|---|---:|---|---|---|
| First-run failure |  |  |  |  |  |  |
| Other product failure |  |  |  |  |  |  |
| Integration request |  |  |  |  |  |  |
| Evaluation question |  |  |  |  |  |  |
| Feature demand |  |  |  |  |  |  |

Use one non-security impact level:

- `high`: the stable validation/evaluation lane is unusable and no safe
  workaround is known;
- `medium`: a documented workflow is blocked or unreliable, but a safe
  workaround exists; or
- `low`: a question, documentation gap, convenience request, or unverified
  preference does not block the stable lane.

Security severity is never assessed here. Route possible vulnerabilities
privately through [`SECURITY.md`](../../SECURITY.md) and exclude their details
and links from this review.

## Confirmed signals and unknowns

Confirmed:

<!-- Add confirmed signals as list items. -->

Unknown:

<!-- Add unknowns as list items. -->

Do not infer people, successful installs, or demand from download, clone, or
view totals.

## Feedback-to-Beads decisions

After the month closes, run `bd list --all --limit 0 --json` and search for the exact
title `Review YYYY-MM aggregate YOLOZU feedback`. Reuse the single matching
item. If there is no match, create it with:

```bash
bd create "Review YYYY-MM aggregate YOLOZU feedback" \
  --type chore \
  --priority 2 \
  --parent YOLOZU-ll2 \
  --labels adoption,feedback,support \
  --description "Record category frequency and highest non-security impact from consented feedback. Review: docs/adoption/YYYY-MM-feedback-review.md" \
  --acceptance "The dated review records source coverage, response-target results, category frequency, impact, evidence boundaries, and linked follow-up Beads."
```

Do not create a second item when the exact title already exists. If duplicates
already exist, select one canonical item and record the other IDs in its notes
before closing or superseding them.

For each actionable recurring blocker, create or update a separate Bead. Record
the aggregate frequency, impact, consented public evidence links, scope
boundary, and a testable acceptance criterion. Do not add an individual report
URL without public-citation consent. Link the Bead ID in the table above.

Prioritization is a maintainer decision, not an automatic promise. A single
`high`-impact stable-lane failure justifies immediate triage. Repeated
`medium`-impact reports justify a scoped follow-up when evidence is
reproducible. `low`-impact requests may be grouped or left documented when
evidence does not justify implementation.

After saving the dated review, attach the aggregate result to the canonical
monthly Bead and close it:

```bash
bd update <id> --append-notes "Review: docs/adoption/YYYY-MM-feedback-review.md; frequencies: <category=count>; highest impact: <level>; follow-ups: <Bead IDs or none>"
bd close <id> --reason "Recorded the monthly aggregate review, evidence boundaries, and follow-up decisions."
```

## Actions

| Bead ID | Action | Evidence and scope | Owner | Target review |
|---|---|---|---|---|
|  |  |  |  |  |

## Next review

Next review month:

Expected review date:
