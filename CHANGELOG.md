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
- Optional runtime extras for datasets, HPO, image models, SciPy sparse inputs,
  Streamlit, and experiment tracking.
- Standardized PEP 735 dependency groups for testing, quality checks,
  documentation, and the complete development toolchain.
- Offline, executable quick-start, training, dataset, and HPO examples.
- A repeatable release-candidate matrix covering representative MLC, MTR, DP,
  MTL, and matrix-completion workloads.
- A dedicated Python 3.10 CI job that verifies the advertised minimum
  dependency series remains installable and passes the offline test suite.
- Scoped weekly Dependabot updates for Python requirements, GitHub Actions, and
  pre-commit hooks, with minor and patch releases grouped for focused review.
- Pull-request dependency review that rejects newly introduced
  moderate-or-higher known vulnerabilities and reports patched versions.
- Immutable full-commit pins for GitHub Actions, with readable release
  annotations and automated pinning-policy coverage.
- PyPI Trusted Publishing through an isolated OIDC-enabled publication job,
  with verified artifact handoff and publish attestations.
- Trainable ID-embedding branches for known entity axes, including direct
  integer-ID batching and validation-setting safeguards for novel entities.
- Typed model-ready batches and a shared branch-encoder contract for dense,
  ID, image, and custom inputs.
- Mixed tabular branches with training-only numeric preprocessing, learned
  categorical embeddings, optional feature gating, structured batches, and
  checkpointed preprocessing state.
- Sparse high-dimensional branches for SciPy matrices and PyTorch COO/CSR
  tensors, including sparse-preserving splitting and batching, sparse first
  projections, early modality/width validation, and checkpoint restoration.
- Token-sequence branches with explicit vocabulary and padding configuration,
  dynamically padded token IDs, attention masks, sequence lengths, packed GRU
  encoding, early validation, and checkpoint restoration.
- Mask-aware 1-D convolutional token-sequence encoders with configurable odd
  kernel widths, stacked temporal layers, and global max pooling.
- Position-aware Transformer token-sequence encoders with padding-aware
  self-attention, configurable heads and feed-forward widths, and masked mean
  pooling.
- Composite entity branches that combine named dense, sparse, token-sequence,
  and known-ID encoders through validated concatenation, including structured
  batching, validation-setting safeguards, and checkpoint restoration.
- Opt-in missing-modality support for composite branches with explicit
  per-component presence masks, present-only encoding, and checkpointed learned
  missing representations.
- Opt-in gated composite fusion with sample-specific component weights,
  presence-aware missing-modality handling, unequal-width component support,
  and unchanged concatenation defaults.
- Opt-in composite modality self-attention with unequal-width projections,
  missing-key masking, an all-missing fusion-token fallback, residual
  contextualization, and checkpoint restoration.
- Typed registries for fusion models, task losses, and output heads, with
  deterministic discovery and focused resolution errors.
- Validated loss selection with mean squared error, mean absolute error, and
  Huber objectives for regression, plus an explicit legacy binary
  cross-entropy compatibility option.
- Opt-in mutually exclusive multiclass classification using zero-based class
  IDs, architecture-wide multi-logit output layers, cross-entropy training,
  softmax predictions, class-averaged metrics, and per-class prediction
  probabilities.
- Ruff, mypy, pre-commit, strict documentation builds, package artifact
  validation, and a cross-platform GitHub Actions test matrix.

### Changed

- Python 3.10 and PyTorch 2.0 are now the minimum supported versions.
- CI verifies Python 3.10 through 3.14 and runs an offline CPU experiment against
  the built wheel in an isolated environment.
- Packaging uses the PEP 621 project table and derives versions from Git tags
  with `setuptools-scm`.
- CI, contributor setup, Conda, and Read the Docs install unpublished
  developer tooling from the standardized dependency groups.
- Dataset downloaders, Torchvision, Streamlit, TensorBoard, Weights & Biases,
  and ConfigSpace are loaded only when their integrations are used.
- Multi-label dataset dumps are read directly with the standard library,
  removing the unnecessary `scikit-multilearn` runtime dependency while
  preserving the existing cache format.
- Epilepsy metric grouping and false-alarm windows use NumPy and the standard
  library instead of the single-purpose `more-itertools` dependency.
- Configuration, data, model, training, evaluation, HPO, and persistence
  boundaries validate inputs earlier and report focused errors.
- Dataloaders expose modality-specific branch inputs through a mapping-compatible
  `MTPBatch`, and fusion models validate standardized encoder outputs without
  inspecting internal module layouts.
- Binary classification optimizes raw logits with `BCEWithLogitsLoss` for
  stable extreme-score gradients while metrics and predictions still receive
  sigmoid probabilities.
- Integer-valued targets retain their historical regression interpretation
  unless multiclass classification is declared explicitly during data
  preparation.
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
- Branch networks, combination architectures, and model construction now live
  under `DeepMTP.models`; historical model modules remain compatibility
  facades.
- Built-in dataset helpers, interaction normalization, feature validation,
  splitting, preparation, model-ready loading, and data-progress services now
  live under `DeepMTP.data`; historical modules remain compatibility facades.
- Numerical training and evaluation, persistence, progress presentation,
  reporting, and evaluation-progress services now live under
  `DeepMTP.runtime`; historical modules remain compatibility facades.

### Fixed

- The Conda environment now installs the development toolchain instead of
  unrelated optional runtime integrations.
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
