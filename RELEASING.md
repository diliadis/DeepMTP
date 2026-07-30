# Releasing DeepMTP

This checklist describes the repository's tag-driven release process. The
package version is derived from Git by `setuptools-scm`; do not add or edit a
version constant manually.

The placeholders `X.Y.Z` and `vX.Y.Z` below must be replaced with the intended
PEP 440-compatible version and Git tag, for example `0.1.0` and `v0.1.0`.

## One-time repository setup

- Confirm the releaser can push tags and publish GitHub Releases.
- Create a GitHub Actions environment named `pypi`. Restrict deployments to
  tags matching `v*` and consider requiring maintainer approval.
- In the DeepMTP project's PyPI publishing settings, add a trusted publisher
  with owner `diliadis`, repository `DeepMTP`, workflow `publish.yml`, and
  environment `pypi`. These values, including capitalization, must match the
  GitHub repository and workflow exactly.
- Protect the PyPI account with multi-factor authentication.
- Keep the `publish.yml` workflow restricted to GitHub's `release: published`
  event.

The publishing workflow uses PyPI Trusted Publishing. GitHub exchanges an OIDC
identity for a short-lived, release-specific PyPI credential; the workflow does
not read a repository API-token secret. Configure the GitHub environment and
PyPI publisher before publishing a release. After the first successful trusted
release, remove any historical `PYPI_API_TOKEN` GitHub secret and revoke the
corresponding PyPI token.

## 1. Prepare the release

- Choose the version according to the compatibility impact of the changes.
- Move relevant entries from `[Unreleased]` in `CHANGELOG.md` into a dated
  `## [X.Y.Z] - YYYY-MM-DD` section.
- Update the comparison links at the bottom of `CHANGELOG.md` so `[Unreleased]`
  compares `vX.Y.Z...HEAD` and `[X.Y.Z]` compares the previous tag to
  `vX.Y.Z`.
- Confirm user-visible breaking changes are explained in `MIGRATION.rst`.
- Confirm installation commands, supported Python versions, optional extras,
  notebooks, and citations are current.
- Confirm the release commit contains no credentials, private data, generated
  result directories, or large unintended artifacts.
- Open a release pull request and require the complete CI workflow to pass.

## 2. Validate the release commit

From a clean, up-to-date checkout of the release branch:

The release-candidate marker runs five small, offline, one-epoch workloads:

| Workload | Mode | Validation setting | Input coverage |
| --- | --- | --- | --- |
| Multi-label classification (MLC) | Classification | B | Instance features and label IDs |
| Multivariate regression (MTR) | Regression | B | Instance features and response IDs |
| Dyadic prediction (DP) | Classification | D | Instance and target features |
| Multi-task learning (MTL) | Classification | B | Instance features and task IDs |
| Matrix completion (MC) | Regression | A | Instance and target IDs only |

```bash
git status --short
python -m pytest
python -m pytest -m release_candidate
pre-commit run --all-files --show-diff-on-failure
sphinx-build -W --keep-going -b html docs/source docs/_build/html
python -m build --outdir dist/preflight
python -m twine check --strict dist/preflight/*
python scripts/verify_release_artifacts.py --contract-only dist/preflight/*
```

The status command must produce no output before and after the checks. A
development-version suffix is expected before the release tag exists. This
preflight build catches packaging failures before tagging; it is not uploaded.
The release-candidate run must report all five workloads as passed.

The distribution-contract verifier checks the preflight wheel and source
distribution automatically:

- Wheel metadata has the expected name, version, Python requirement,
  dependencies, extras, license, and project URLs.
- The wheel contains `DeepMTP` runtime modules but not `DeepMTP/tests`.
- The source distribution contains the license, README, changelog, migration
  guide, contributor guide, security policy, release checklist, documentation,
  demos, and build metadata.
- `python -m twine check --strict` reports both distributions as valid.
- Package name, Python requirement, license, project URLs, runtime extras, and
  wheel/sdist versions agree with the supported public contract.

## 3. Tag and validate the release artifacts

Create an annotated tag locally only after the release pull request is merged
and the target branch is up to date:

```bash
git switch main
git pull --ff-only
git status --short
git tag -a vX.Y.Z -m "DeepMTP X.Y.Z"
git show --no-patch vX.Y.Z
python -m build --outdir dist/vX.Y.Z
python -m twine check --strict dist/vX.Y.Z/*
python scripts/verify_release_artifacts.py vX.Y.Z dist/vX.Y.Z/*
```

Use a new version-specific output directory so stale distributions cannot be
uploaded by mistake. The verifier reads package metadata from exactly one wheel
and one source distribution, rejects development or local versions, and requires
both versions to match the release tag.

Test the tagged wheel, rather than the source checkout, in a new environment:

```bash
python -m venv dist/release-smoke
dist/release-smoke/bin/python -m pip install --upgrade pip
for wheel in dist/vX.Y.Z/*.whl; do
  dist/release-smoke/bin/python -m pip install "${wheel}[all]"
done
dist/release-smoke/bin/python -m pip check
dist/release-smoke/bin/python scripts/smoke_test_installation.py \
  --require-wheel-install \
  --output-dir dist/release-smoke-results
dist/release-smoke/bin/python scripts/smoke_test_optional_extras.py \
  --require-wheel-install
```

On Windows, use `dist\release-smoke\Scripts\python.exe`. The script refuses a
source-checkout import, runs a small offline CPU experiment, and prints the
installed version. Confirm that version is exactly `X.Y.Z`.

Push the tag only after the tagged artifacts pass these checks:

```bash
git push origin vX.Y.Z
```

Do not move or reuse a published tag. Wait for the complete CI workflow on the
tagged commit to pass before publishing a GitHub Release.

## 4. Publish

- Draft a GitHub Release from the existing `vX.Y.Z` tag.
- Copy the matching changelog section into the release notes and link the
  migration guide when behavior changed.
- Mark a pre-release appropriately for alpha, beta, or release-candidate tags.
- Publish the GitHub Release. This triggers `.github/workflows/publish.yml`.
- Confirm the `Build and verify distributions` job built fresh distributions,
  passed strict metadata validation, verified both artifact versions against
  the GitHub Release tag, installed every runtime extra, passed both installed
  smoke tests, and uploaded the verified workflow artifact.
- Approve the protected `pypi` environment deployment if approval is required.
- Confirm the separate `Publish verified distributions to PyPI` job downloaded
  that artifact, authenticated through Trusted Publishing, uploaded both
  distributions, and generated publish attestations.

Never run a second manual upload while the publishing workflow is active.

## 5. Verify the published release

- Confirm PyPI shows the expected version, files, metadata, and SHA256 hashes.
- Install the exact version from PyPI in another clean environment:

  ```bash
  python -m pip install "DeepMTP[all]==X.Y.Z"
  python -m pip check
  python -c "import DeepMTP; print(DeepMTP.__version__)"
  python scripts/smoke_test_optional_extras.py --require-wheel-install
  ```

- Repeat the offline CPU smoke test against the PyPI installation.
- Check that the Read the Docs build completed and displays the release
  version.
- Confirm README badges and the GitHub Release point to the published version.
- Announce the release only after these checks pass.

## Failed releases

PyPI files and versions are immutable. Do not delete and recreate a tag, reuse a
version, or overwrite release artifacts.

- If publishing fails before PyPI accepts files, fix the workflow in a new
  commit, keep the original tag unchanged, and decide whether to retry the
  GitHub workflow or issue a new version.
- If a broken version reaches PyPI, mark it as yanked, document the reason, fix
  the problem on a new commit, and publish a new patch version.
- For a security issue, follow `SECURITY.md`, avoid public details before the
  coordinated disclosure, and release the smallest safe fix as a new version.
- Record the outcome and any user action in `CHANGELOG.md`.
