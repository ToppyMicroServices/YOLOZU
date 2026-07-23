# Support and feedback

Choose the narrowest route that fits the request. Public routes are preferred
for reusable, non-sensitive answers.

| Need | Structured route | Exploratory route |
|---|---|---|
| Install, `doctor`, demo, validation, or first-evaluation failure | [First-run failure form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=first_run_failure.yml) | [Q&A Discussion](https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/q-a) |
| Reproducible failure after first-run setup | [Bug report form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=bug_report.yml) | [Q&A Discussion](https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/q-a) |
| Framework, exporter, runtime, or predictions interface contract integration | [Integration request form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=integration_request.yml) | [Ideas Discussion](https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/ideas) |
| Protocol, metric, comparability, or report interpretation | [Evaluation question form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=evaluation_question.yml) | [Q&A Discussion](https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/q-a) |
| Evidence-backed capability demand | [Feature demand form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=feature_demand.yml) | [Ideas Discussion](https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/ideas) |

Non-security support is also available at `develop@toppymicros.com`. Do not send
credentials, proprietary datasets, model weights, prediction files, or personal
data. For a possible vulnerability, use the private process in
[`SECURITY.md`](../SECURITY.md), not an Issue, Discussion, or the general
support mailbox.

## What to include

Issue forms capture the same minimum fields used during triage:

- goal and the decision or workflow being attempted;
- source framework and public version or commit, or `Not applicable`;
- sanitized environment details, including YOLOZU and Python versions;
- minimal commands, settings, and observed output;
- a public minimal reproduction, sanitized artifact metadata, or `None`;
- non-security impact and any safe workaround, plus frequency for integration
  or feature demand; and
- confirmation that public content is safe to publish, consent to follow-up in
  the public thread, and an explicit choice to allow aggregate use with a public
  Issue citation, aggregate-only use, or neither.

When starting in a Discussion or email, copy the following block. For email,
set both consent fields explicitly; feedback without aggregate-use consent is
not included in adoption summaries.

```text
Goal:
Source framework and version:
Environment:
Commands/settings and observed output:
Sanitized public artifact or metadata (or None):
Frequency and impact:
Safe for public posting: yes/no
Consent to follow-up: yes/no
Consent to aggregate use: yes/no
Consent to cite this public thread (public routes only): yes/no
```

Do not include names of people, customers, employers, private datasets, private
models, access tokens, credentials, private URLs, or raw private artifacts.
Sanitize local paths and logs. A checksum can establish artifact identity
without publishing the artifact.

## Maintainer response target

For non-security requests, maintainers target an initial triage response or
request for missing information within **5 business days**. This is a target,
not a service-level guarantee or a fix timeline. Resolution depends on
reproducibility, impact, maintainer capacity, and the maturity boundary of the
affected lane. Security reports use the separate response targets in
[`SECURITY.md`](../SECURITY.md).

For monthly measurement, the clock starts on the first Monday-to-Friday date
after the Issue, Discussion, or email was received, using `Europe/Tallinn`
calendar dates. The fifth Monday-to-Friday date ends at 23:59 in that timezone;
public holidays are not subtracted. A qualifying initial response is a human
maintainer reply that answers the question, states the triage category and next
step, or asks for specific missing intake fields. Automated acknowledgements do
not count, and the clock does not restart after follow-up.

## Public triage labels

The repository uses the current GitHub label set as follows:

| Label | Triage meaning |
|---|---|
| `bug` | A failure report under investigation; the label alone does not confirm a product defect |
| `question` | Evaluation, usage, protocol, or report-interpretation question |
| `enhancement` | Integration request or evidence-backed feature demand |
| `documentation` | The accepted next action is primarily a documentation change |
| `duplicate` | The same request is tracked in another public item |
| `invalid` | The report is outside the documented scope or cannot be acted on with the supplied evidence |
| `wontfix` | Maintainers reviewed the request and do not plan to implement it |
| `help wanted` | Maintainers welcome a community contribution |
| `good first issue` | The scoped follow-up is suitable for a first contribution |

The issue-form title prefixes distinguish `First run`, `Bug`, `Integration`,
`Evaluation`, and `Feature` during monthly aggregation. `Bug` maps to the
`Other product failure` row. A label or title prefix classifies the request; it
is not evidence that the requested capability is supported or scheduled.

## Feedback loop

Maintainers triage public reports, request only the missing minimum evidence,
and link accepted engineering work to Beads. Aggregate frequency and
non-security impact are reviewed monthly using
[`adoption/monthly_feedback_review_template.md`](adoption/monthly_feedback_review_template.md).
No user-level profile or private artifact is copied into the review. An
individual public report is linked only when its reporter selected the
public-citation option.

## Legal

- © 2026 ToppyMicroServices OÜ
- Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia
- Registry code: 16551297
