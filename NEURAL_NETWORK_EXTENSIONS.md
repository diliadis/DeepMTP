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

### 4. Sequence, pretrained-embedding, and foundation-model inputs

Status: core sequence encoders and the generic foundation-model adapter
baseline are implemented. Parameter-efficient training, caching, and
production lifecycle support remain planned.

- [x] Add padded token-sequence batches with attention masks and original
  lengths, including dynamic collation of variable-length rows.
- [x] Provide a lightweight packed-GRU encoder.
- [x] Add a mask-aware 1-D convolutional encoder.
- [x] Add an optional position-aware Transformer encoder.
- [x] Accept precomputed embeddings from domain models as ordinary dense
  features.
- [x] Add optional adapters for protein, chemical, and text foundation models
  without making them core dependencies.

#### Foundation-model roadmap

Deliver the remaining work in the following order so each stage has stable
checkpoint and experiment-tracking semantics before the next optimization is
introduced:

- [x] Provide a generic Hugging Face `AutoTokenizer`/`AutoModel` adapter
  with frozen and full fine-tuning, explicit pooling and length policies,
  trainable output projection, composite-branch support, lazy dependencies,
  and resolved model/tokenizer revision capture.
- [ ] Add parameter-efficient fine-tuning with Hugging Face PEFT and LoRA
  rather than implementing adapter layers from scratch. Keep it opt-in,
  validate target modules and hyperparameters before training, and preserve
  the existing ``frozen`` and ``full`` modes.
- [ ] Support adapter-only checkpoint export and restoration. Tie every
  adapter checkpoint and W&B Artifact to the exact base-model and tokenizer
  revisions required to restore it.
- [ ] Add an optional entity-level cache for pooled frozen-backbone outputs
  before the trainable projection. Include model/tokenizer revisions,
  tokenization settings, pooling, hidden size, and dtype in cache identity and
  reject stale or incompatible entries.
- [ ] Add foundation-model gradient checkpointing after automatic mixed
  precision has a stable trainer-level contract. Cover both full and
  parameter-efficient fine-tuning without changing CPU behavior.
- [ ] Define documented support tiers: offline contract tests for the generic
  adapter, a local tiny-model integration test in regular CI, and opt-in
  compatibility checks for the documented ESM-2, ChemBERTa, BERT, and
  BiomedBERT reference checkpoints.
- [ ] Consider quantized model loading only after device placement, dtype,
  optimizer, checkpoint-restoration, and platform-support semantics are
  explicit. Keep quantization dependencies separate and optional.

### 5. Graph inputs (homogeneous molecular graphs completed)

- [x] Define a typed graph batch adapter and optional graph neural-network
  branch.
- [x] Support precomputed homogeneous molecular
  ``torch_geometric.data.Data`` objects.
- [x] Provide GIN for node-only graphs and GINE for molecular edge features,
  with configurable graph-level pooling.
- [x] Support graph encoders as required or optional composite components.
- [x] Keep PyTorch Geometric in a separate optional dependency group.
- [ ] Add heterogeneous graphs only if a concrete dataset requires them.

### 6. Time-series inputs

- Represent values, timestamps, masks, and sequence lengths explicitly.
- Add 1-D convolutional and Transformer-style temporal encoders.
- Define leakage-safe temporal splitting and test it separately from settings
  A–D.

### 7. Multimodal and partially missing inputs (complete)

- [x] Introduce a composite branch that concatenates multiple named dense,
  sparse, sequence, or known-ID encoders for one entity.
- [x] Support explicit masks and learned representations for missing modalities.
- [x] Add opt-in, presence-aware gated concatenation while preserving unequal
  component widths.
- [x] Add attention-based fusion and compare it with concatenation and gated
  fusion.
- [x] Add modality dropout so models do not become unusable when an optional input
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
- [x] Add a structured batch contract for sequences.
- [x] Add a structured batch contract for graphs.
- [x] Separate fusion, output-head, and loss selection into registries.
- [x] Store tabular and legacy dense-scaler preprocessing metadata with
  checkpoints, including split provenance and validated replay helpers.

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
- Gated and modality self-attention fusion are available inside composite
  branches.
- Add cross-attention between entity branches only after its dimensional and
  validation-setting semantics are defined.
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
- [x] Add observation-weighted gradient accumulation with correct final partial
  groups and optional global norm clipping at optimizer-step boundaries.
- [x] Record device, dtype, dependency, seed, source, training-result, and
  preprocessing metadata with each saved experiment checkpoint.

### Reproducibility and experiment tracking

Status: reproducible checkpoints and W&B integration implemented.

- [x] Add an optional versioned checkpoint manifest without changing the
  historical three required payload keys.
- [x] Capture fitted legacy dense scaling, tabular preprocessing references,
  split provenance, environment versions, Git revision, parameter counts,
  selected epoch, and final metrics.
- [x] Replay saved dense transformations and reject incompatible split or
  feature-width state before training.
- [x] Organize W&B Runs with names, groups, job types, tags, notes, modes,
  pinned configuration keys, epoch-aware metrics, summaries, and failure
  status.
- [x] Publish versioned W&B model Artifacts with aliases, input Artifact
  lineage, optional Registry linking, and direct Artifact restoration.
- [x] Add bounded prediction Tables plus classification confusion, ROC, and
  precision-recall charts.
- [ ] Add exact interrupted-training resume only after epoch-local RNG,
  scheduler, scaler, and dataloader state have a stable contract.
- [ ] Add a first-class sweep adapter only when it can share configuration
  validation with the existing HPO APIs.

## Suggested delivery order

1. Learned ID embeddings.
2. Formalize batch and encoder contracts.
3. Mixed tabular inputs and precomputed embeddings.
4. Formalize fusion, loss, and output-head registries.
5. Numerically stable logits/loss handling and richer output heads.
6. Sparse inputs.
7. Sequence encoders (GRU, 1-D convolution, Transformer, and the generic
   Hugging Face foundation-model adapter baseline complete). Continue
   foundation-model work with PEFT/LoRA, adapter lifecycle support, frozen
   embedding caches, memory optimizations, and compatibility tiers;
   investigate quantization last.
8. Hybrid input concatenation, learned missing-modality representations, gated
   fusion, attention fusion, and modality dropout (complete).
9. Homogeneous molecular graph encoders (complete), followed by time-series
   encoders and heterogeneous graphs only when driven by concrete datasets.
10. Reproducible checkpoints and W&B experiment lineage (complete).
11. Performance features after correctness baselines are established.

Each slice is complete when its invalid configurations fail early, its public
behavior is documented, focused and end-to-end tests pass offline, type and
format checks pass, and a saved model can be restored for prediction.
