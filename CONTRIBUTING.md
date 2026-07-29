# Contributing to DeepMTP

DeepMTP is scientific software. Changes should preserve scientifically
meaningful behavior unless the behavior change is intentional, documented, and
covered by a regression test.

## Development setup

DeepMTP supports Python 3.10 and newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
pre-commit install
```

Conda users can bootstrap the package and a compatible pip version, then
install the same development group:

```bash
conda env create -f environment.yml
conda activate deepmtp
python -m pip install --group dev
pre-commit install
```

Install additional runtime integrations with `.[all]`, or select only the
needed extras from `datasets`, `hpo`, `image`, `streamlit`, and `tracking`.
Install the appropriate PyTorch build separately when GPU acceleration is
required.

## Local checks

Run the same default checks enforced by CI:

```bash
python -m pytest
pre-commit run --all-files
sphinx-build -W --keep-going -b html docs/source docs/_build/html
python -m build
python -m twine check --strict dist/*
```

Reproduce the lower-bound compatibility job in a fresh Python 3.10
environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e . --group test --constraint constraints/minimum.txt
python -m pip check
python -m pytest
```

The default test configuration excludes tests marked `network` and `slow`.
Run a marked suite explicitly when its requirements are available:

```bash
python -m pytest -m network
python -m pytest -m slow
python -m pytest -m integration
```

## Making changes

- Work on a topic branch and keep commits focused.
- Add a regression test before fixing a confirmed bug.
- Add user-visible changes to the `Unreleased` section of
  [CHANGELOG.md](CHANGELOG.md).
- Keep the core usable without Streamlit, W&B, TensorBoard, Torchvision, or
  downloadable-dataset dependencies.
- Preserve the public API during modernization. Introduce a warning and a
  documented migration path before removing public behavior.
- Update user-facing documentation when installation, configuration, or output
  changes.

## Pull requests

Describe the problem, the chosen approach, scientific or compatibility impact,
and the checks that were run. Keep unrelated formatting and refactoring out of
the same pull request. Tests should use small synthetic or local fixtures by
default; network, slow, and GPU requirements must use the corresponding pytest
markers.

Do not include credentials, private research data, generated experiment
results, downloaded datasets, or model checkpoints. Follow
[SECURITY.md](SECURITY.md) instead of opening a public issue for a suspected
vulnerability.

The `Review dependency changes` CI check rejects pull requests that introduce
known moderate-or-higher vulnerabilities. Review its summary and update or
remove the affected dependency instead of disabling the check. License policy
is not enforced by this workflow; a future license gate must first define and
document the project's accepted licenses.

Maintainers should use [RELEASING.md](RELEASING.md) for the tag-driven release
process. Release history and pending user-visible changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

Dependabot checks Python requirements, GitHub Actions, and pre-commit hooks
weekly. Minor and patch version updates are grouped; major and security updates
remain separate for focused review. The tested lower-bound series in
`constraints/minimum.txt` is intentionally excluded and must be updated
together with `pyproject.toml` and its compatibility tests.

The staged technical direction and remaining work are tracked in
[MODERNIZATION.md](MODERNIZATION.md).
