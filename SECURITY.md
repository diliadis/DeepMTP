# Security Policy

## Supported versions

DeepMTP is pre-1.0 software. Security fixes are made on the active development
line and included in its next release; backports to 0.0.22 and earlier releases
are not currently planned.

| Version | Security fixes |
| --- | --- |
| Active development line | Supported |
| 0.0.22 and earlier | No planned backports |

## Reporting a vulnerability

Please report a suspected vulnerability privately by emailing
[dimitrios.iliadis@ugent.be](mailto:dimitrios.iliadis@ugent.be) with the subject
`DeepMTP security report`. Do not open a public issue until disclosure has been
coordinated.

Include as much of the following as possible:

- The affected DeepMTP version or commit and relevant dependency versions.
- The operating system and Python version.
- A description of the impact and the conditions required to trigger it.
- Minimal reproduction steps using synthetic, non-sensitive data.
- Any proposed mitigation or patch, if available.
- Whether the issue is already public or has been shared with anyone else.

Do not send private datasets, credentials, API tokens, proprietary model
weights, or other sensitive artifacts. The maintainer will aim to acknowledge a
report within seven days, investigate the impact, and coordinate a fix and
disclosure timeline with the reporter. This is a best-effort response target,
not a guaranteed service-level agreement.

Use the public
[issue tracker](https://github.com/diliadis/DeepMTP/issues) for ordinary bugs,
incorrect scientific results without a security impact, feature requests, and
questions.

## Trusted artifacts

DeepMTP checkpoints can contain pickled Python configuration objects. Load only
checkpoints and serialized configuration files from sources you trust. Loading
an untrusted `.pt` or `.pkl` file can execute malicious code through the
underlying Python/PyTorch deserialization mechanism.

Reports about DeepMTP bypassing an expected trust boundary, unsafe archive or
path handling, credential exposure, dependency compromise, or unintended
disclosure of training data are in scope. Resource exhaustion from an
intentionally extreme training configuration and normal CPU/GPU floating-point
variation are generally ordinary robustness issues unless they cross a
security boundary.

## Dependency security

Every pull request is checked for dependency changes. The dependency-review
workflow rejects newly introduced vulnerabilities with moderate, high, or
critical severity across runtime, development, and unknown dependency scopes.
It reports the first patched version when GitHub's advisory data provides one.

Repository administrators should keep the dependency graph, Dependabot alerts,
and Dependabot security updates enabled. Require the `Review dependency
changes` status check on the default branch so a failed review cannot be
bypassed accidentally. Existing dependency alerts should be triaged separately;
the pull-request check is designed to prevent adding new known vulnerabilities,
not to certify the complete historical dependency graph.

GitHub Actions are pinned to immutable full commit SHAs, with the corresponding
release tag retained as a comment for review. Dependabot should update both the
commit and readable release annotation through focused pull requests.

PyPI releases use Trusted Publishing instead of a long-lived repository secret.
The release workflow builds and verifies distributions in an unprivileged job,
then passes only those artifacts to a protected `pypi` environment. The
publication job has the OIDC `id-token: write` permission but cannot check out
the repository or run project build scripts.

## Disclosure

Once a fix is ready, the maintainer and reporter should agree on a coordinated
disclosure date. The public advisory should describe affected versions, impact,
mitigation, fixed versions, and credit for the reporter if desired. Security
releases are recorded in `CHANGELOG.md`.
