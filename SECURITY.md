# Security Policy

## Supported versions

Security fixes are applied to the latest published release on PyPI
(`insufficient-effort`). Older minor versions are not patched.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Email **cameron.lyons2@gmail.com** with:

- a description of the issue and its impact
- steps to reproduce, or a proof of concept if available
- affected package versions / commit SHAs if known

You should receive an acknowledgement within a few business days. Once a fix is
available, a patched release will be published and credit given if you want it.

## Scope

IER is a scientific library that processes local survey matrices. Typical risks
are dependency vulnerabilities and misuse of untrusted input files via the CLI.
Reports about statistical methodology disagreements are out of scope for this
policy — please open a regular issue instead.
