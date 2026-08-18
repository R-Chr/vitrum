# Security Policy

## Supported versions

Security fixes are applied to the latest released version only. Users on older versions
should upgrade before reporting.

| Version | Supported |
| ------- | --------- |
| 1.1.x   | ✅        |
| < 1.1   | ❌        |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/R-Chr/vitrum/security/advisories/new),
or by email to rasmus.christensen.a1@tohoku.ac.jp. Include what an attacker can do, the
version affected, and a reproduction if you have one.

You can expect an acknowledgement within a week. If the report is confirmed, a fix and an
advisory follow; you will be credited unless you ask otherwise.

## Scope

`vitrum` is a scientific analysis library: it parses simulation output and structure files,
and issues network requests only through the optional `volume_estimation` extra (Materials
Project API). Reports involving malformed input files, path handling, or deserialisation
are in scope.
