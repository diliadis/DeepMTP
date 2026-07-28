# Releasing DeepMTP

This checklist describes the repository's tag-driven release process. The
package version is derived from Git by `setuptools-scm`; do not add or edit a
version constant manually.

The placeholders `X.Y.Z` and `vX.Y.Z` below must be replaced with the intended
PEP 440-compatible version and Git tag, for example `0.1.0` and `v0.1.0`.

## One-time repository setup

- Confirm the releaser can push tags and publish GitHub Releases.
- Add a scoped PyPI API token to the GitHub Actions repository secret
  `PYPI_API_TOKEN`. The token should be limited to the DeepMTP PyPI project.
- Protect the PyPI account with multi-factor authentication.
- Keep the `publish.yml` workflow restricted to GitHub's `release: published`
  event.

The current publishing workflow uses `PYPI_API_TOKEN`. Migrating it to PyPI
Trusted Publishing should be handled as a separate infrastructure change and
tested before removing the token.

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
```

The status command must produce no output before and after the checks. A
development-version suffix is expected before the release tag exists. This
preflight build catches packaging failures before tagging; it is not uploaded.
The release-candidate run must report all five workloads as passed.

Inspect the preflight wheel and source distribution:

- Wheel metadata has the expected name, version, Python requirement,
  dependencies, extras, license, and project URLs.
- The wheel contains `DeepMTP` runtime modules but not `DeepMTP/tests`.
- The source distribution contains the license, README, changelog, migration
  guide, contributor guide, security policy, release checklist, documentation,
  demos, and build metadata.
- `python -m twine check --strict` reports both distributions as valid.

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
dist/release-smoke/bin/python -m pip install dist/vX.Y.Z/*.whl
dist/release-smoke/bin/python -c "import DeepMTP; print(DeepMTP.__version__)"
```

On Windows, use `dist\release-smoke\Scripts\python.exe`. Confirm the printed
version is exactly `X.Y.Z`, then run a small offline CPU experiment from outside
the repository so the installed wheel, not the source tree, is imported.

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
- Confirm the publish job built fresh distributions, passed strict metadata
  validation, verified both artifact versions against the GitHub Release tag,
  and uploaded both artifacts to PyPI.

Never run a second manual upload while the publishing workflow is active.

## 5. Verify the published release

- Confirm PyPI shows the expected version, files, metadata, and SHA256 hashes.
- Install the exact version from PyPI in another clean environment:

  ```bash
  python -m pip install "DeepMTP==X.Y.Z"
  python -c "import DeepMTP; print(DeepMTP.__version__)"
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
