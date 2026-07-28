# Changelog

All notable changes to DeepMTP are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with entries grouped
by their effect on users and maintainers.

## [Unreleased]

This development line contains substantial changes from 0.0.22. Existing users
should read the [migration guide](MIGRATION.rst) before moving experiments or
checkpoints.

### Added

- A typed `DeepMTPConfig` API with centralized validation, structured
  normalization warnings, and compatibility conversion for existing
  dictionaries.
- Stable top-level imports for the trainer, configuration, data preparation,
  model factories, progress observers, reporters, and runtime services.
- Explicit progress and reporting interfaces for console output, Streamlit,
  TensorBoard, and Weights & Biases.
- Per-branch dropout settings, top-k grouped metrics, isolated random seeds,
  and deterministic class undersampling.
- Optional dependency groups for datasets, HPO, image models, Streamlit,
  tracking, testing, quality checks, and documentation.
- Offline, executable quick-start, training, dataset, and HPO examples.
- A repeatable release-candidate matrix covering representative MLC, MTR, DP,
  MTL, and matrix-completion workloads.
- Ruff, mypy, pre-commit, strict documentation builds, package artifact
  validation, and a cross-platform GitHub Actions test matrix.

### Changed

- Python 3.10 and PyTorch 2.0 are now the minimum supported versions.
- CI verifies Python 3.10 through 3.14 and runs an offline CPU experiment against
  the built wheel in an isolated environment.
- Packaging uses the PEP 621 project table and derives versions from Git tags
  with `setuptools-scm`.
- Dataset downloaders, Torchvision, Streamlit, TensorBoard, Weights & Biases,
  and ConfigSpace are loaded only when their integrations are used.
- Configuration, data, model, training, evaluation, HPO, and persistence
  boundaries validate inputs earlier and report focused errors.
- Data preparation copies caller-owned input, isolates split randomness, fits
  feature scalers on training data only, and enforces split/entity invariants.
- `DeepMTP.predict()` honors `return_predictions`; callers receive only metrics
  unless they explicitly request the prediction frame.
- A trainer instance supports one training run. Create a new `DeepMTP` object
  for each additional run.
- Checkpoints load through the CPU, use atomic persistence, and retain their
  historical payload keys.
- Convolutional branches use Torchvision's explicit weights API and can disable
  pretrained weights for offline construction.
- Streamlit training and HPO adapters reuse the same core implementations as
  non-Streamlit runs.
- HPO workers, optimizers, progress events, and structural types now live in
  the focused `DeepMTP.hpo` package. Historical module paths remain
  compatibility facades for existing imports and persisted objects.
- Optional Streamlit trainers, HPO wrappers, and progress renderers now live
  under `DeepMTP.integrations.streamlit`, with historical root modules retained
  as compatibility facades.

### Fixed

- Mean training loss is weighted correctly when the final batch has a
  different size.
- Data splitting, feature alignment, entity novelty detection, samplers, and
  generated datasets reject invalid or ambiguous inputs instead of silently
  producing inconsistent data.
- Evaluation rejects unsupported metric combinations and avoids silent `NaN`
  results when required baselines are unavailable.
- Checkpoint configuration overrides no longer mutate the stored checkpoint
  configuration.
- Importing DeepMTP no longer changes NumPy or PyTorch's process-wide random
  state.
- Dataset caching and extraction use validated paths and safer filesystem
  boundaries.
- Release automation rejects wheels and source distributions whose metadata does
  not exactly match the GitHub Release tag.
- Sphinx documentation displays the package version derived from Git metadata.

### Deprecated

- Legacy configuration options that have no runtime effect now emit
  `ConfigDeprecationWarning`: `momentum`, `weighted_loss`,
  `use_instance_features`, `use_target_features`, `load_pretrained_model`,
  `pretrained_model_path`, and non-default
  `comb_mlp_nodes_reducing_factor`.

### Removed

- Trainer configuration no longer accepts `RRMSE`, because the trainer does not
  retain the per-target training baselines required to calculate it correctly.
  The low-level evaluation utility still supports `RRMSE` when callers provide
  those baselines.
- Legacy Travis and duplicate GitHub Actions test configurations.

## [0.0.22] - 2022-12-08

### Changed

- Added `tqdm` as a runtime dependency.
- Expanded the test workflow's operating-system and Python coverage.
- Refreshed the Hyperband example and related worker configuration.

[Unreleased]: https://github.com/diliadis/DeepMTP/compare/v0.0.22...HEAD
[0.0.22]: https://github.com/diliadis/DeepMTP/releases/tag/v0.0.22
