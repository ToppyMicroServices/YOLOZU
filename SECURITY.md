# Security Policy

This repository is maintained by ToppyMicroServices OÜ.

For the full coordinated disclosure policy, see:
https://toppymicros.com/security-policy.html

Machine-readable policy:
https://toppymicros.com/.well-known/security.txt

## Scope

In scope:

- Public assets under `toppymicros.com`
- Public repositories maintained by ToppyMicroServices OÜ, including this repository

Out of scope (non-exhaustive):

- Best-practice suggestions without a demonstrable exploit path
- Self-XSS and browser or devtools-only issues
- Volumetric denial of service

## Reporting a vulnerability

Please report vulnerabilities to:
`security@toppymicros.com`

Use the subject line:
`[SECURITY] <short summary>`

This mailbox is used for coordinated vulnerability disclosure.

Include:

- Affected asset and vulnerability summary
- Reproduction steps or proof of concept
- Impact assessment
- Optional remediation guidance

## Response Targets

- Acknowledgement target: within 5 business days
- Remediation target: generally 30 days; complex issues may require up to 60 days

## Safe Harbor

If you act in good faith and follow this policy, we will not pursue legal action for your research activities.

## Supported runtime scope

Security maintenance is focused on the latest released version of this repository and its documented public surfaces.

- Python: supported according to the current package metadata and release documentation
- OS/runtime coverage: Linux and macOS public paths; GPU-specific validation remains environment-dependent
- Repository scope: this repository is in scope under the coordinated disclosure policy above

## Dependency and third-party policy

- Follow the coordinated disclosure path above for vulnerabilities affecting bundled code, dependency usage, or third-party integrations in this repository
- When relevant, include dependency/package names, affected versions, and any upstream advisories in the report
- Public best-practice suggestions without a demonstrable exploit path remain out of scope
