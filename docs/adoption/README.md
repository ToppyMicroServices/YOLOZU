# Adoption Measurement

YOLOZU does not include product telemetry or phone-home behavior. Adoption is
therefore measured from aggregate public-surface signals and consented user
feedback, not from individual usage tracking.

## Weekly owner and cadence

- Owner: YOLOZU maintainers (`develop@toppymicros.com`)
- Collection day: Thursday
- Report path: `docs/adoption/YYYY-MM-DD-baseline.md`
- Review path: [`quarterly_review_template.md`](quarterly_review_template.md)

## Weekly snapshots

- [`2026-07-23-baseline.md`](2026-07-23-baseline.md)
- [`2026-07-30-baseline.md`](2026-07-30-baseline.md)
- [`2026-08-06-baseline.md`](2026-08-06-baseline.md)
- [`2026-08-13-baseline.md`](2026-08-13-baseline.md)
- [`2026-08-20-baseline.md`](2026-08-20-baseline.md)
- [`2026-08-27-baseline.md`](2026-08-27-baseline.md)
- [`2026-09-03-baseline.md`](2026-09-03-baseline.md)

## Monthly feedback review

- Owner: YOLOZU maintainers (`develop@toppymicros.com`)
- Review time: after each calendar month closes
- Template:
  [`monthly_feedback_review_template.md`](monthly_feedback_review_template.md)
- Output path: `docs/adoption/YYYY-MM-feedback-review.md`
- Beads record: one monthly review item plus separate scoped work items for
  actionable recurring blockers

Reviews:

- [`2026-07-feedback-review.md`](2026-07-feedback-review.md)

The monthly review records category frequency, highest non-security impact,
response-target results, public evidence, unknowns, and linked Beads actions.
It includes only public reports and support feedback with anonymous
aggregate-use consent. Security reports remain on the private path in
[`SECURITY.md`](../../SECURITY.md) and are excluded from the review.

## Consented onboarding observations

For consented onboarding observations, use the
[`design_partner_observation_kit.md`](design_partner_observation_kit.md).
Its maintainer-only procedural rehearsal is recorded in
[`2026-07-23-maintainer-kit-dry-run.md`](2026-07-23-maintainer-kit-dry-run.md);
the rehearsal is not an external design-partner session or adoption evidence.

## Funnel definitions

| Stage | Aggregate signal | Interpretation |
|---|---|---|
| Product-page discovery | Plausible page views and tagged outbound clicks, when available | Directional discovery only; no person-level profile |
| Repository discovery | GitHub unique views, referrers, stars, and forks | Views are a short-window signal; stars and forks are cumulative |
| Package discovery | PyPI Stats downloads without known mirrors | Includes CI, scanners, and repeat installs; not a user count |
| First install | Consented session or support report that confirms an install | Unknown unless a user supplies evidence |
| First proof/demo | Consented session or support report with `doctor --proof` or demo evidence | A confirmed completion, not an inferred download conversion |
| First evaluation | Consented session or issue with a validated comparable report | Primary activation event |
| Support | External GitHub issue/discussion or support contact | Count only real external requests; keep security reports private |
| Repeat use | Consented follow-up confirming a second evaluation | Primary retention signal |

## Privacy boundary

- Collect only repository- or package-level aggregates and consented notes.
- Do not add telemetry to the YOLOZU package, CLI, demos, or generated reports.
- Do not store IP addresses, user-agent strings, raw access logs, email contents,
  prediction artifacts, dataset names, or employer names in these reports.
- Record a public name or organization only with explicit permission.
- Keep security reports on the private path in [`SECURITY.md`](../../SECURITY.md).
- Treat GitHub clone and PyPI download totals as automation-sensitive. They are
  monitoring signals, not evidence of people, installs, or successful use.

## Weekly collection

Run from an authenticated maintainer shell. GitHub traffic endpoints retain only
the latest 14 days, so weekly collection is required to preserve a trend.

```bash
gh api 'repos/ToppyMicroServices/YOLOZU/traffic/views?per=day'
gh api 'repos/ToppyMicroServices/YOLOZU/traffic/clones?per=day'
gh api repos/ToppyMicroServices/YOLOZU/traffic/popular/referrers
gh api repos/ToppyMicroServices/YOLOZU/traffic/popular/paths
gh api repos/ToppyMicroServices/YOLOZU
gh api repos/ToppyMicroServices/YOLOZU/stargazers \
  -H 'Accept: application/vnd.github.star+json' --paginate
```

Retrieve aggregate PyPI Stats data at most once per day:

```bash
curl -L --compressed --fail --silent --show-error \
  -A 'YOLOZU adoption baseline' \
  https://pypistats.org/api/packages/yolozu/recent
```

For support signals, use the GraphQL issue and discussion connections so the
response requests association and outcome fields without requesting author
identities. The issue connection excludes pull requests. Exclude bots,
maintainers, members, and collaborators, then retain only aggregate counts in
the adoption report. Confirm that neither connection requires another page;
if it does, paginate with the same identity-free field selection.

```bash
gh api graphql -f query='query {
  repository(owner: "ToppyMicroServices", name: "YOLOZU") {
    issues(first: 100) {
      nodes { authorAssociation state author { __typename } }
      pageInfo { hasNextPage }
    }
    discussions(first: 100) {
      nodes { authorAssociation answerChosenAt author { __typename } }
      pageInfo { hasNextPage }
    }
  }
}' --jq '{
  issues_has_next_page: .data.repository.issues.pageInfo.hasNextPage,
  discussions_has_next_page: .data.repository.discussions.pageInfo.hasNextPage,
  external_issues: ([.data.repository.issues.nodes[] |
    select(.author.__typename != "Bot" and
      .authorAssociation != "OWNER" and .authorAssociation != "MEMBER" and
      .authorAssociation != "COLLABORATOR")] |
    length),
  closed_external_issues: ([.data.repository.issues.nodes[] |
    select(.author.__typename != "Bot" and .state == "CLOSED" and
      .authorAssociation != "OWNER" and .authorAssociation != "MEMBER" and
      .authorAssociation != "COLLABORATOR")] |
    length),
  external_discussions: ([.data.repository.discussions.nodes[] |
    select(.author.__typename != "Bot" and
      .authorAssociation != "OWNER" and .authorAssociation != "MEMBER" and
      .authorAssociation != "COLLABORATOR")] |
    length),
  answered_external_discussions: ([.data.repository.discussions.nodes[] |
    select(.author.__typename != "Bot" and .answerChosenAt != null and
      .authorAssociation != "OWNER" and .authorAssociation != "MEMBER" and
      .authorAssociation != "COLLABORATOR")] |
    length)
}'
```

## Clone interpretation rule

Do not estimate people by dividing clone totals. Mark the clone signal
`automation-sensitive` when any of the following is true:

- daily clones are more than five times daily unique cloners;
- clone volume is inconsistent with repository views;
- a repeated near-constant daily pattern is visible; or
- maintainer CI, release, security, mirror, or scanner activity is known.

When marked, report raw totals for auditability but set the human-adoption
interpretation to `unknown`. A GitHub unique cloner is not necessarily a unique
person, and repeated days cannot be summed into a user count.

## Confirmed activation ledger

Keep only consented, minimal records. An internal ledger may use:

| Session ID | Date | Persona class | First install | First proof/demo | First evaluation | Repeat use | Public-name permission |
|---|---|---|---|---|---|---|---|

Use pseudonymous session IDs. Link only to public GitHub discussions/issues or
non-sensitive evidence. Do not commit private contact details or user artifacts.
