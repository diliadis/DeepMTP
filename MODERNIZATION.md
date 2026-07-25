# DeepMTP modernization roadmap

This document tracks the staged modernization of DeepMTP. The guiding rule is
to preserve scientifically meaningful behaviour while adding tests around each
change. Behaviour changes must be explicit, documented, and covered by a
regression test.

## Compatibility policy

- Support Python 3.10 and newer, with CI initially covering Python 3.10–3.12.
- Keep CPU execution fully supported; GPU acceleration is optional.
- Preserve the public API during the stabilization phases. Use warnings and a
  documented deprecation window before removing public names or arguments.
- Keep unit tests offline, deterministic, and small. Network downloads, GPU
  tests, and long-running model training belong in separately marked test
  suites.
- Treat Streamlit, experiment tracking, dataset downloads, and image models as
  optional integrations around a usable core package.

## Phase 0 — Baseline and safeguards

- [x] Inventory the repository and trace the main runtime paths.
- [x] Record confirmed correctness and maintainability risks.
- [x] Create a dedicated modernization branch.
- [x] Capture a tiny CPU classification and regression baseline.
- [x] Add test markers for unit, integration, network, GPU, and slow tests.

Exit criterion: a developer can run a fast, offline test suite and reproduce
two small end-to-end training runs.

## Phase 1 — Correctness fixes

- [x] Stop mutating default split ratios between calls.
- [x] Fix default training when `evaluate_train=False`.
- [x] Evaluate classification training metrics on probabilities.
- [x] Make `predict(return_predictions=...)` honour its argument.
- [x] Correct macro and instance undersampling.
- [x] Make checkpoint/config serialization reliable for image transforms.
- [x] Cover single-class and singleton evaluation groups.
- [x] Validate identifiers used for generated one-hot features.

Exit criterion: every confirmed bug has a regression test and the CPU smoke
tests pass.

## Phase 2 — Configuration and public API

- [x] Introduce a typed, validated configuration object at the trainer
  boundary while preserving the legacy dictionary surface.
- [x] Keep dictionary input as a compatibility adapter.
- [x] Either implement or deprecate currently inert options.
- [x] Validate architecture names, metrics, split ratios, paths, and devices at
  the API boundary.
- [x] Replace mutable argument defaults throughout the package.
- [x] Define stable public imports and a package version in `DeepMTP/__init__.py`.

Exit criterion: invalid configurations fail early with actionable messages and
all supported options have tests.

## Phase 3 — Architecture and separation of concerns

- [x] Extract experiment directories, summaries, configuration artifacts, and
  checkpoints into a tested persistence component.
- [x] Extract forward, backward, loss, and prediction collection into a tested
  numerical training engine.
- [x] Extract metric scheduling and evaluation into a typed, tested component.
- [x] Route trainer lifecycle events and summary-table formatting through an
  injectable progress observer.
- [x] Extract typed model and dataloader factories from the core trainer.
- [x] Replace wildcard imports and import-time RNG mutation in the core
  trainer, using isolated model and dataloader generators instead.
- [ ] Separate data preparation, datasets, models, training, evaluation,
  persistence, and experiment reporting.
- [ ] Migrate internal dictionary lookups to typed configuration access as the
  trainer is split, while preserving the public dictionary adapter.
- [ ] Replace wildcard imports and module-level global state.
- [ ] Replace direct printing with standard logging and optional progress
  callbacks.
- [x] Inject W&B and TensorBoard reporting integrations through a shared,
  lazily imported reporter interface.
- [x] Make checkpoint loading device-safe and explicit.

Exit criterion: the core trainer has no dependency on Streamlit, W&B, or
TensorBoard, and integrations can be tested independently.

## Phase 4 — Streamlit and optional integrations

- [x] Replace `main_streamlit.py` with a thin progress/reporting adapter over
  the core trainer.
- [x] Reuse the same random-search and Hyperband implementations in Streamlit.
- [ ] Create a canonical `streamlit_app.py` only if a maintained application is
  still a project goal.
- [ ] Move W&B, TensorBoard, Streamlit, image models, and downloadable datasets
  behind optional dependency groups and lazy imports.

Exit criterion: there is only one implementation of training and HPO logic.

## Phase 5 — Packaging, automation, and quality gates

- [ ] Move package metadata to the PEP 621 `[project]` table in
  `pyproject.toml`.
- [ ] Define reproducible dependency groups for development, testing, docs, and
  optional integrations.
- [ ] Add Ruff formatting/linting and a pragmatic type-checking baseline.
- [ ] Replace the legacy Travis configuration and old GitHub Actions workflow.
- [ ] Test supported Python versions on Linux, macOS, and Windows where useful.
- [ ] Build and inspect wheel/sdist artifacts in CI.
- [ ] Add pre-commit hooks after the initial formatting migration.

Exit criterion: a clean checkout can be installed, tested, linted, built, and
documented using repository commands.

## Phase 6 — Documentation and release readiness

- [ ] Repair README and notebook examples against the tested public API.
- [ ] Replace network-dependent demos with small local examples where possible.
- [ ] Document migration from the last published release.
- [ ] Add changelog, contributor guide, security policy, and release checklist.
- [ ] Run a release-candidate validation on representative MLC, MTR, DP, MTL,
  and matrix-completion tasks.

Exit criterion: documentation examples execute in CI and a release can be
produced from a tagged commit without manual repository edits.

## Definition of done for each change

1. A failing regression test demonstrates the previous problem.
2. The smallest coherent implementation change fixes it.
3. Relevant unit and integration tests pass.
4. Public behaviour and migration impact are documented.
5. Formatting, linting, typing, and build checks pass at the configured level.
