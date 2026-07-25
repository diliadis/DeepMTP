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
python -m pip install -e ".[dev]"
```

Install additional runtime integrations with `.[all]`, or select only the
needed extras from `datasets`, `hpo`, `image`, `streamlit`, and `tracking`.
Install the appropriate PyTorch build separately when GPU acceleration is
required.

## Local checks

Run the same default checks enforced by CI:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy
python -m build
python -m twine check --strict dist/*
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
- Keep the core usable without Streamlit, W&B, TensorBoard, Torchvision, or
  downloadable-dataset dependencies.
- Preserve the public API during modernization. Introduce a warning and a
  documented migration path before removing public behavior.
- Update user-facing documentation when installation, configuration, or output
  changes.

The staged technical direction and remaining work are tracked in
[MODERNIZATION.md](MODERNIZATION.md).
