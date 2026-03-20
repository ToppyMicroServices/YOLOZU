# Security and Cryptography Scope

YOLOZU is not a cryptography product. Its primary purpose is evaluation,
benchmarking, training/inference orchestration, and interface-contract-first
artifact validation for vision models.

This note makes the repository's security and cryptography boundaries explicit
for contributors, operators, and badge-style audits.

## What YOLOZU does by default

- uses standard Python and ecosystem libraries for transport, hashing, and file integrity tasks
- prefers publicly reviewed cryptographic primitives provided by dedicated libraries instead of custom implementations
- uses artifact hashing and fingerprints for reproducibility, provenance, and cache identity
- keeps external runtime integrations such as CUDA, TensorRT, and vendor SDKs outside the repository's default trusted base

## What YOLOZU does not do by default

- implement its own cryptographic protocol
- implement password storage for external-user authentication
- ship a custom key-management or session-encryption subsystem
- require broken cryptographic algorithms as a default security mechanism

Because of that scope, many badge-style cryptography requirements are `N/A`
for the repository itself. When security-sensitive cryptographic functionality
is needed, contributors should use established FLOSS libraries that are
specifically designed for that purpose.

## Project policy

When code in this repository needs cryptographic behavior:

- use dedicated, publicly reviewed libraries instead of custom crypto code
- keep secure defaults enabled unless an explicitly unsafe compatibility mode is requested
- do not introduce broken algorithms such as MD4, MD5, single DES, or RC4 as default security mechanisms
- do not rely on SHA-1 or similarly weak primitives for security decisions
- use cryptographically secure randomness for secrets, keys, tokens, and nonces

## Integrity hashing vs security mechanisms

This repository may use hashes for:

- artifact identity
- dataset fingerprints
- cache keys
- deduplication
- report stability checks

Those uses are not automatically security mechanisms. If a hash is used for a
security-sensitive purpose, contributors must use an appropriate modern
primitive and document the decision.

## TLS and insecure compatibility paths

Any opt-in insecure transport path must remain non-default, clearly documented,
and justified by interoperability constraints. Insecure compatibility paths must
not silently replace secure defaults.

## Export-control note

YOLOZU does not primarily exist to provide encryption features. If future work
adds software that includes, activates, or enables encryption functionality for
distribution outside the United States or to non-US persons, maintainers should
review the relevant export-control obligations before release.
