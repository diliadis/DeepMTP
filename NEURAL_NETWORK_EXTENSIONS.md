# DeepMTP neural-network and input roadmap

This roadmap proposes incremental extensions to DeepMTP's two-branch model.
The order is intentional: each new input type should use a common branch
contract instead of adding special cases to the trainer.

## Design principles

- Preserve the meaning of validation settings A–D.
- Keep existing configurations and checkpoints working unless a change is
  explicitly deprecated.
- Make new behavior opt-in and validate incompatible combinations before
  training.
- Keep the core installation lightweight; specialized encoders should use
  optional dependencies.
- Add small unit tests and at least one CPU end-to-end test for every encoder,
  fusion strategy, loss, and output head.

## Prioritized input extensions

### 1. Learned ID embeddings

Status: implemented and merged.

- [x] Add `EMBEDDING` as a built-in branch architecture.
- [x] Pass integer IDs to `torch.nn.Embedding` without constructing dense
  one-hot vectors.
- [x] Support dot-product, MLP, and Kronecker combination architectures.
- [x] Allow feature and ID branches to be mixed when the ID side is known at
  inference time.
- [x] Reject ID embeddings for an entity axis that is novel in the configured
  validation setting.
- [x] Keep the legacy one-hot behavior unchanged for existing MLP
  configurations.
- [x] Cover configuration, batching, model construction, training, prediction,
  and public imports with tests.

The public configuration is:

```python
from DeepMTP import DeepMTPConfig

config = DeepMTPConfig(
    validation_setting="A",
    problem_mode="classification",
    metrics_average=["micro"],
    instance_branch_architecture="EMBEDDING",
    instance_branch_input_dim=number_of_instances,
    target_branch_architecture="EMBEDDING",
    target_branch_input_dim=number_of_targets,
    embedding_size=64,
)
```

`instance_branch_input_dim` and `target_branch_input_dim` are vocabulary sizes
for embedding branches. Interaction IDs must be zero-based integers in
`[0, input_dim)`. `embedding_size` is the learned vector width.

An ID embedding cannot infer a representation for an unseen entity. Therefore:

| Validation setting | Instance ID embedding | Target ID embedding |
|---|---:|---:|
| A: known instances and targets | supported | supported |
| B: novel instances | rejected | supported |
| C: novel targets | supported | rejected |
| D: novel instances and targets | rejected | rejected |

Side features are rejected on an `EMBEDDING` branch to avoid silently ignoring
data. A later hybrid encoder can combine ID and side-feature representations
explicitly.

### 2. Mixed tabular inputs

Status: implemented and merged.

- [x] Add categorical-column embeddings alongside continuous features.
- [x] Define numeric normalization and missing-value policies in the saved
  preprocessing state.
- [x] Support an explicit tabular schema rather than requiring each row to contain
  one already-flattened vector.
- [x] Add optional feature gating for combining numeric and categorical
  representations.
- [x] Treat precomputed embeddings as ordinary dense MLP features without
  introducing a redundant input type.

The public schema declares numeric columns and fixed categorical vocabularies:

```python
from DeepMTP import DeepMTPConfig

config = DeepMTPConfig(
    validation_setting="B",
    problem_mode="classification",
    instance_branch_architecture="TABULAR",
    instance_branch_tabular_schema={
        "numeric_columns": ["age", "dose"],
        "categorical_columns": {
            "site": {
                "categories": ["Ghent", "Paris", "London"],
                "embedding_dim": 3,
            }
        },
        "numeric_normalization": "standard",
        "numeric_missing": "mean",
        "categorical_missing": "unknown",
        "categorical_unknown": "unknown",
        "feature_gating": True,
    },
    target_branch_architecture="EMBEDDING",
    target_branch_input_dim=number_of_targets,
    embedding_size=32,
)
```

Numeric statistics are fitted only on training entities and stored as
`instance_tabular_preprocessing_state` or
`target_tabular_preprocessing_state`. Validation, test, and restored-model
prediction reuse that state. Missing and out-of-vocabulary categories use the
reserved index zero unless their policy is configured as `error`.

### 3. Sparse high-dimensional inputs (completed)

- [x] Accept SciPy sparse matrices and PyTorch sparse tensors without densifying
  the complete dataset.
- [x] Add sparse linear projection branches for fingerprints, bag-of-words, and
  large indicator matrices.
- [x] Benchmark memory use and throughput against the dense MLP path with
  `scripts/benchmark_sparse_inputs.py`.

### 4. Sequence and pretrained-embedding inputs

- Add padded or ragged token-sequence batches with attention masks.
- Provide lightweight recurrent/1-D convolutional encoders and an optional
  Transformer encoder.
- Accept precomputed embeddings from domain models as ordinary dense features.
- Add optional adapters for protein, chemical, and text foundation models
  without making them core dependencies.

### 5. Graph inputs

- Define a graph batch adapter and optional graph neural-network branch.
- Initially support precomputed molecular graph objects, then general
  heterogeneous graphs if a concrete use case requires them.
- Keep graph libraries in a separate optional dependency group.

### 6. Time-series inputs

- Represent values, timestamps, masks, and sequence lengths explicitly.
- Add 1-D convolutional and Transformer-style temporal encoders.
- Define leakage-safe temporal splitting and test it separately from settings
  A–D.

### 7. Multimodal and partially missing inputs

- Introduce a composite branch that combines multiple encoders for one entity.
- Support explicit masks for missing modalities.
- Compare concatenation, gated fusion, and attention-based fusion.
- Add modality dropout so models do not become unusable when an optional input
  is absent.

## Neural-network functionality

### Common contracts

- [x] Define typed tensor batch objects for dense vectors, IDs, images, and
  custom branches.
- [x] Preserve historical dataloader mapping keys through the typed
  `MTPBatch` compatibility interface.
- [x] Give every branch an explicit input kind, output dimension, and shared
  encoder protocol.
- [x] Validate encoder output rank, batch size, width, and dtype before fusion.
- [x] Remove module-layout inspection from fusion construction.
- [x] Add structured numeric/categorical batches for tabular encoders.
- [ ] Add structured batch contracts for sequences and graphs when those
  encoders are introduced.
- [x] Separate fusion, output-head, and loss selection into registries.
- [ ] Store all preprocessing metadata with checkpoints (tabular preprocessing
  is complete; legacy dense-scaler metadata remains).

### Losses and output heads

- [x] Use `BCEWithLogitsLoss` for numerically stable binary classification while
  applying sigmoid only for metrics and returned predictions.
- [x] Add documented configuration and registry support for binary
  cross-entropy, mean squared error, mean absolute error, and Huber losses.
- [x] Add mutually exclusive multiclass classification with zero-based class
  IDs, architecture-wide multi-logit heads, cross-entropy training, softmax
  predictions, class-aware metrics, and checkpoint coverage.
- [ ] Add optional ranking losses after their sampling and validation-setting
  semantics are defined.
- Add explicit heads for ordinal, count, and bounded regression tasks.
- Implement class/example weighting only with a tested definition for each
  validation setting.

### Fusion

- Keep dot product, concatenation plus MLP, and Kronecker product as baselines.
- Add bilinear fusion as a lower-cost interaction model.
- Add gated fusion and cross-attention only after common branch contracts are
  established.
- Make dimensional requirements and parameter growth visible during
  configuration validation.

### Modern branch blocks

- Add configurable activations, residual MLP blocks, layer normalization, and
  initialization policies.
- Replace architecture-specific dimension inspection with the common branch
  output contract.
- Add parameter-count and shape summaries for reproducibility.

### Training performance

- Add automatic mixed precision for supported CUDA devices.
- Evaluate `torch.compile` as an opt-in optimization with an eager fallback.
- Add gradient clipping and gradient accumulation.
- Record device, precision, dependency, seed, and preprocessing metadata with
  each experiment.

## Suggested delivery order

1. Learned ID embeddings.
2. Formalize batch and encoder contracts.
3. Mixed tabular inputs and precomputed embeddings.
4. Formalize fusion, loss, and output-head registries.
5. Numerically stable logits/loss handling and richer output heads.
6. Sparse inputs.
7. Sequence encoders.
8. Hybrid and missing-modality support.
9. Graph and time-series encoders, driven by concrete datasets.
10. Performance features after correctness baselines are established.

Each slice is complete when its invalid configurations fail early, its public
behavior is documented, focused and end-to-end tests pass offline, type and
format checks pass, and a saved model can be restored for prediction.
