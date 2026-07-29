# DeepMTP modernization roadmap

This document tracks the staged modernization of DeepMTP. The guiding rule is
to preserve scientifically meaningful behaviour while adding tests around each
change. Behaviour changes must be explicit, documented, and covered by a
regression test.

## Compatibility policy

- Support Python 3.10 and newer, with CI covering Python 3.10–3.14 on Linux and
  Python 3.12 on macOS and Windows.
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
- [x] Always calculate the metric used for best-epoch model selection.
- [x] Make trainer run state explicit and checkpoint saves side-effect-free.
- [x] Weight epoch losses by observations when batch sizes differ.
- [x] Make evaluation metric results complete and repair ranking NDCG.
- [x] Enforce micro-only metric averaging for validation setting A.
- [x] Make RRMSE and inverse score-scaling requirements explicit.
- [x] Make per-target evaluation details an explicit opt-in.
- [x] Reject top-k evaluation when no grouped averaging is configured.
- [x] Validate combined-MLP layer settings before model construction.
- [x] Validate convolutional architecture settings before model construction.
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
- [x] Move model-ready PyTorch datasets and samplers out of the legacy data
  preparation utility while preserving compatibility imports.
- [x] Extract interaction normalization, schema checks, novelty detection, and
  validation-setting inference behind a focused compatibility boundary.
- [x] Extract feature normalization and cross-input consistency validation
  while preserving the legacy data utility surface.
- [x] Extract split orchestration, defensive copying, and post-split
  validation while preserving legacy entry points.
- [x] Extract feature scaling and end-to-end preparation orchestration,
  leaving the historical data utility module as a compatibility façade.
- [x] Replace wildcard imports and import-time RNG mutation in the core
  trainer, using isolated model and dataloader generators instead.
- [x] Use typed configuration attributes for core trainer read paths.
- [x] Use typed configuration at built-in model and reporting boundaries.
- [x] Make ResNet fine-tuning cutoffs explicit and cover them with offline
  trainability tests.
- [x] Separate data preparation, datasets, models, training, evaluation,
  persistence, and experiment reporting.
- [x] Migrate internal dictionary lookups to typed configuration access as the
  trainer is split, while preserving the public dictionary adapter.
- [x] Replace wildcard imports and import-time process-wide state mutations.
- [x] Replace configuration-boundary prints with structured normalization
  warnings and silent documented defaults.
- [x] Route trainer setup and checkpoint lifecycle messages through progress
  observers.
- [x] Route HPO schedule diagnostics and worker configuration messages through
  HPO progress observers.
- [x] Route early-stopping patience diagnostics through trainer progress
  observers.
- [x] Route detailed metric results and evaluation diagnostics through
  injectable progress observers.
- [x] Establish a reusable data progress observer and route split-validation
  messages through it.
- [x] Route interaction parsing and cross-split validation messages through
  data progress observers.
- [x] Route validation-setting inference messages through data progress
  observers.
- [x] Route instance and target feature-ingestion messages through data
  progress observers.
- [x] Route instance and target cross-input consistency messages through data
  progress observers.
- [x] Route feature-scaling lifecycle and skip warnings through data progress
  observers.
- [x] Route validation-setting A-D split lifecycle messages through data
  progress observers.
- [x] Route top-level data-processing orchestration and nested helper messages
  through one shared data progress observer.
- [x] Route public dataset loader and download lifecycle messages through
  injectable data progress observers.
- [x] Expose reusable MTR catalog formatting and route its legacy presentation
  helper through a data progress observer.
- [x] Confine direct printing to explicit console adapters and use injectable
  progress callbacks in core workflows, enforced by an AST boundary test.
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
- [x] Move W&B, TensorBoard, Streamlit, image models, and downloadable datasets
  behind optional dependency groups and lazy imports.
- [x] Use torchvision's explicit weights API and support offline image-model
  construction.

Exit criterion: there is only one implementation of training and HPO logic.

## Phase 5 — Packaging, automation, and quality gates

- [x] Move package metadata to the PEP 621 `[project]` table in
  `pyproject.toml`.
- [x] Define runtime extras for optional integrations and standardized PEP 735
  dependency groups for development, testing, quality checks, and docs.
- [x] Add Ruff formatting/linting and a pragmatic type-checking baseline.
- [x] Replace the legacy Travis configuration and old GitHub Actions workflow.
- [x] Test supported Python versions on Linux, macOS, and Windows where useful.
- [x] Verify the declared minimum dependency series in a dedicated Python 3.10
  job.
- [x] Build and inspect wheel/sdist artifacts in CI.
- [x] Add pre-commit hooks after the initial formatting migration.

Exit criterion: a clean checkout can be installed, tested, linted, built, and
documented using repository commands.

## Phase 6 — Documentation and release readiness

- [x] Replace the README quick start with a tested, typed, offline example.
- [x] Modernize demo notebook imports and checkpoint restoration calls, with
  static API regression coverage.
- [x] Repair README and notebook examples against the tested public API.
- [x] Replace network-dependent demos with small local examples where possible.
- [x] Document migration from the last published release.
- [x] Add changelog, contributor guide, security policy, and release checklist.
- [x] Run a release-candidate validation on representative MLC, MTR, DP, MTL,
  and matrix-completion tasks.

Exit criterion: documentation examples execute in CI and a release can be
produced from a tagged commit without manual repository edits.

## Phase 7 — Package layout

- [x] Group HPO workers, optimizers, progress events, and structural types
  under `DeepMTP.hpo` while preserving historical import and pickle paths.
- [x] Group optional Streamlit adapters under a focused integration package
  while preserving historical import and pickle paths.
- [x] Group branch models and model construction under `DeepMTP.models` while
  preserving historical import and pickle paths.
- [x] Complete the `DeepMTP.runtime` package:
  - [x] Group persistence, presentation, reporting, and evaluation-progress
    services while preserving historical import and pickle paths.
  - [x] Group numerical evaluation and training services while preserving
    historical import and pickle paths.
- [x] Complete the `DeepMTP.data` package:
  - [x] Group model-ready loading and data-progress services while preserving
    historical import and pickle paths.
  - [x] Group interaction, feature, splitting, and preparation services while
    preserving historical import and pickle paths.
  - [x] Group built-in and downloadable dataset helpers while preserving
    historical imports, pickle paths, and lazy optional dependencies.

Exit criterion: package paths communicate ownership without breaking supported
legacy imports or persisted artifacts.

## Definition of done for each change

1. A failing regression test demonstrates the previous problem.
2. The smallest coherent implementation change fixes it.
3. Relevant unit and integration tests pass.
4. Public behaviour and migration impact are documented.
5. Formatting, linting, typing, and build checks pass at the configured level.
