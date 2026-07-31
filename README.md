<p align="center">
  <img src="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/logo_transparent_cropped.png" alt="DeepMTP" height="240">
</p>

<h1 align="center">DeepMTP</h1>

<p align="center">
  A PyTorch framework for multi-target prediction with flexible two-branch neural networks.
</p>

<p align="center">
  <a href="https://github.com/diliadis/DeepMTP/actions/workflows/ci.yml"><img src="https://github.com/diliadis/DeepMTP/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/DeepMTP/"><img src="https://img.shields.io/pypi/v/DeepMTP.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/DeepMTP/"><img src="https://img.shields.io/pypi/pyversions/DeepMTP.svg" alt="Supported Python versions"></a>
  <a href="https://deepmtp.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/deepmtp/badge/?version=latest" alt="Documentation"></a>
  <a href="https://github.com/diliadis/DeepMTP/blob/main/LICENSE.md"><img src="https://img.shields.io/github/license/diliadis/DeepMTP" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://deepmtp.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#choose-your-input-representations">Input guide</a> ·
  <a href="#demos">Colab demos</a> ·
  <a href="https://github.com/diliadis/DeepMTP/issues">Issues</a>
</p>

---

DeepMTP learns from observations of the form `(instance, target, score)`. It
encodes instances and targets independently, fuses their representations, and
predicts the score for each pair. The same interface covers:

- multi-label classification (MLC);
- multivariate regression (MTR);
- multi-task learning (MTL);
- dyadic prediction (DP); and
- matrix completion (MC).

Use dense, sparse, tabular, sequence, foundation-model, molecular-graph,
image, learned-ID, composite, or custom inputs on either branch. DeepMTP also
provides reproducible checkpoints, PEFT/LoRA support, experiment tracking, and
random-search or Hyperband optimization without making those integrations
mandatory.

> Upgrading from DeepMTP 0.0.22? Read the
> [0.0.22 → 0.0.23 migration guide](MIGRATION.rst) and
> [changelog](CHANGELOG.md) before reusing an existing experiment.

## Why DeepMTP?

| Need | DeepMTP support |
|---|---|
| One model across MTP problem families | A shared two-branch abstraction for MLC, MTR, MTL, DP, and MC |
| Generalize to new entities | Explicit validation settings for known or novel instances and targets |
| Mix representation types | Built-in dense, sparse, tabular, sequence, foundation, graph, image, ID, and composite encoders |
| Choose how branches interact | Dot product, concatenation plus MLP, or Kronecker-product fusion |
| Train larger models safely | Gradient accumulation, gradient clipping, early stopping, and deterministic data handling |
| Track and restore experiments | Versioned checkpoints, preprocessing provenance, TensorBoard, and Weights & Biases Artifacts |
| Adapt pretrained encoders efficiently | Hugging Face Transformers with frozen, full fine-tuning, or PEFT/LoRA modes |

## Installation

DeepMTP supports Python 3.10–3.14. CPU execution is fully supported; a CUDA
GPU is optional.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install DeepMTP
```

Install only the integrations you use:

| Extra | Install command | Adds |
|---|---|---|
| Datasets | `pip install "DeepMTP[datasets]"` | Downloadable benchmark datasets |
| Foundation | `pip install "DeepMTP[foundation]"` | Transformers and PEFT/LoRA |
| Graph | `pip install "DeepMTP[graph]"` | PyTorch Geometric encoders |
| HPO | `pip install "DeepMTP[hpo]"` | ConfigSpace, random search, and Hyperband |
| Image | `pip install "DeepMTP[image]"` | Torchvision models and transforms |
| Sparse | `pip install "DeepMTP[sparse]"` | SciPy sparse matrices |
| Streamlit | `pip install "DeepMTP[streamlit]"` | Streamlit progress adapters |
| Tracking | `pip install "DeepMTP[tracking]"` | TensorBoard and Weights & Biases |
| Visualization | `pip install "DeepMTP[visualization]"` | Rich terminal architecture diagrams and SVG export |
| Everything | `pip install "DeepMTP[all]"` | All runtime integrations |

For GPU acceleration, install the appropriate PyTorch build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) first.
See the [installation guide](https://deepmtp.readthedocs.io/en/latest/tutorial/installation.html)
for source and Conda workflows.

## Quick start

This complete CPU example trains on a synthetic multi-label dataset, evaluates
it, and returns pair-level predictions. It requires no downloads or optional
dependencies.

<!-- quickstart-example:start -->
```python
import numpy as np

from DeepMTP import DeepMTP, DeepMTPConfig, data_process

rng = np.random.default_rng(42)
scores = (
    np.arange(24)[:, np.newaxis] + np.arange(4)[np.newaxis, :]
) % 2
data = {
    "train": {
        "y": scores,
        "X_instance": rng.normal(size=(24, 3)),
        "X_target": None,
    }
}
train, validation, test, data_info = data_process(
    data,
    validation_setting="B",
)

config = DeepMTPConfig(
    validation_setting=data_info["detected_validation_setting"],
    problem_mode=data_info["detected_problem_mode"],
    general_architecture_version="dot_product",
    compute_mode="cpu",
    num_workers=0,
    train_batchsize=16,
    val_batchsize=16,
    num_epochs=1,
    metrics=["accuracy"],
    metrics_average=["macro"],
    evaluate_val=True,
    use_early_stopping=False,
    save_model=False,
    results_path="results",
    experiment_name="quickstart",
    instance_branch_architecture="MLP",
    instance_branch_input_dim=data_info["instance_branch_input_dim"],
    instance_branch_nodes_per_layer=[8],
    target_branch_architecture="MLP",
    target_branch_input_dim=data_info["target_branch_input_dim"],
    target_branch_nodes_per_layer=[8],
    embedding_size=4,
)

model = DeepMTP(config)
validation_results = model.train(train, validation, test)
test_results, predictions = model.predict(test, return_predictions=True)
```
<!-- quickstart-example:end -->

DeepMTP writes `config.json`, `summary.txt`, and any enabled logs or checkpoint
below `results/quickstart`. The four core steps remain the same for every input
type:

1. Describe observed scores and optional side information.
2. Call `data_process` with the intended validation setting and input kinds.
3. Build a validated `DeepMTPConfig`.
4. Train with `DeepMTP.train` and infer with `DeepMTP.predict`.

## Mental model

Let `x` denote an instance, `t` a target, and `y` their observed relationship.
DeepMTP learns two encoders:

```text
instance x ──> instance encoder ──> pₓ ──┐
                                         ├──> fusion ──> score ŷ(x, t)
target t   ──> target encoder   ──> qₜ ──┘
```

The score can be binary, multiclass, or continuous. Dot-product fusion requires
equal branch widths. MLP and Kronecker fusion can combine unequal widths.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp_overview_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp_overview_light.svg">
    <img src="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp_overview_light.svg" alt="DeepMTP independently encodes instances and targets, fuses their representations, and produces binary, multiclass, or regression predictions." width="100%">
  </picture>
</p>

Inspect the same architecture directly from Python or a saved configuration:

```python
from DeepMTP import print_architecture

print_architecture(config)
```

```bash
deepmtp architecture results/quickstart/config.json
```

Install `DeepMTP[visualization]` for responsive Rich panels, parameter
statistics, and SVG export. Without it, DeepMTP falls back to a portable plain
text representation. See the [architecture visualization guide](https://deepmtp.readthedocs.io/en/latest/tutorial/architecture_visualization.html).

### Generalization settings

The validation setting states which entity types may be unseen at evaluation
time. Novel entities require side information; a learned ID lookup cannot
represent an entity absent from training.

| Setting | Evaluation pairs | Instance side information | Target side information |
|---|---|---:|---:|
| **A** | Known instances × known targets | Optional | Optional |
| **B** | Novel instances × known targets | Required | Optional |
| **C** | Known instances × novel targets | Optional | Required |
| **D** | Novel instances × novel targets | Required | Required |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/validation-settings-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/validation-settings-light.svg">
    <img src="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/validation-settings-light.svg" alt="Validation settings A through D distinguish whether evaluation contains known or novel instances, targets, or both." width="100%" loading="lazy">
  </picture>
</p>

See the [configuration guide](https://deepmtp.readthedocs.io/en/latest/tutorial/configuration.html)
for split semantics and validation rules.

## Choose your input representations

Each branch selects an encoder independently. A compound–protein model can,
for example, combine a molecular graph branch with a protein foundation-model
branch.

| Architecture | Input | Good fit | Dependency |
|---|---|---|---|
| `MLP` | Dense numeric vectors | Descriptors and precomputed embeddings | Core |
| `SPARSE` | COO/CSR or SciPy matrices | Fingerprints, bag-of-words, indicators | Core for PyTorch; `sparse` for SciPy |
| `TABULAR` | Numeric and categorical columns | Mixed entity metadata | Core |
| `SEQUENCE` | Token IDs and masks | Proteins, SMILES, text | Core; `foundation` for Hugging Face |
| `GRAPH` | Homogeneous PyG graphs | Molecular graphs | `graph` |
| `CONV` | Images | Visual entity representations | `image` |
| `EMBEDDING` | Zero-based entity IDs | Known entities without side features | Core |
| `COMPOSITE` | Named component mapping | Multimodal or partially missing inputs | Depends on components |
| `CUSTOM` | User-defined batch type | Domain-specific encoders | Core |

### Learned entity IDs

Use `EMBEDDING` rather than a dense one-hot MLP when entities are known during
training. Settings B, C, and D reject an ID branch on any novel entity axis.

<!-- id-embedding-example:start -->
```python
from DeepMTP import DeepMTPConfig

number_of_instances = 24
number_of_targets = 4

config = DeepMTPConfig(
    validation_setting="A",
    problem_mode="classification",
    general_architecture_version="dot_product",
    metrics_average=["micro"],
    instance_branch_architecture="EMBEDDING",
    instance_branch_input_dim=number_of_instances,
    target_branch_architecture="EMBEDDING",
    target_branch_input_dim=number_of_targets,
    embedding_size=32,
)
```
<!-- id-embedding-example:end -->

### Mixed tabular data

`TABULAR` keeps numeric and categorical preprocessing explicit and stores the
fitted training-only state for compatible restoration.

<!-- tabular-config-example:start -->
```python
import pandas as pd

from DeepMTP import DeepMTPConfig

instance_features = pd.DataFrame(
    {
        "id": [0, 1, 2],
        "age": [22.0, None, 41.0],
        "site": ["Ghent", "Paris", "Ghent"],
    }
)

tabular_schema = {
    "numeric_columns": ["age"],
    "categorical_columns": {
        "site": {
            "categories": ["Ghent", "Paris"],
            "embedding_dim": 2,
        }
    },
    "numeric_normalization": "standard",
    "numeric_missing": "mean",
    "categorical_unknown": "unknown",
    "feature_gating": True,
}

number_of_targets = 4

config = DeepMTPConfig(
    validation_setting="B",
    problem_mode="classification",
    general_architecture_version="dot_product",
    instance_branch_architecture="TABULAR",
    instance_branch_tabular_schema=tabular_schema,
    instance_branch_nodes_per_layer=[16],
    target_branch_architecture="EMBEDDING",
    target_branch_input_dim=number_of_targets,
    embedding_size=8,
)
```
<!-- tabular-config-example:end -->

Read the [tabular guide](https://deepmtp.readthedocs.io/en/latest/tutorial/tabular_inputs.html)
for missing-value, normalization, and unknown-category policies.

### Sparse high-dimensional data

`SPARSE` preserves sparsity through splitting, batching, and the first
projection. Apply sparse-safe preprocessing upstream; legacy dense scaling is
rejected rather than silently densifying the data.

```python
config = DeepMTPConfig(
    validation_setting="B",
    problem_mode="classification",
    instance_branch_architecture="SPARSE",
    instance_branch_input_dim=fingerprint_width,
    instance_branch_nodes_per_layer=[128, 32],
    target_branch_architecture="MLP",
    target_branch_input_dim=number_of_targets,
    target_branch_nodes_per_layer=[32],
    embedding_size=32,
)
```

See the [sparse-input guide](https://deepmtp.readthedocs.io/en/latest/tutorial/sparse_inputs.html)
for SciPy support, explicit IDs, and benchmark tooling.

### Token sequences

`SEQUENCE` batches contain dynamically padded token IDs, attention masks, and
original lengths. Choose `gru`, `conv1d`, or `transformer`; all ignore padded
positions.

<!-- sequence-config-example:start -->
```python
from DeepMTP import DeepMTPConfig

vocabulary_size = 32
number_of_targets = 4

config = DeepMTPConfig(
    validation_setting="B",
    problem_mode="classification",
    general_architecture_version="dot_product",
    instance_branch_architecture="SEQUENCE",
    instance_branch_input_dim=vocabulary_size,
    instance_branch_sequence_encoder="transformer",
    instance_branch_sequence_embedding_dim=16,
    instance_branch_sequence_transformer_num_heads=4,
    instance_branch_sequence_transformer_feedforward_dim=64,
    instance_branch_sequence_num_layers=1,
    instance_branch_sequence_padding_idx=0,
    target_branch_architecture="MLP",
    target_branch_input_dim=number_of_targets,
    target_branch_nodes_per_layer=[8],
    embedding_size=8,
)
```
<!-- sequence-config-example:end -->

Raw token rows must be unpadded and cannot contain the configured padding ID.
The [sequence guide](https://deepmtp.readthedocs.io/en/latest/tutorial/sequence_inputs.html)
documents collation and each encoder.

### Foundation models and LoRA

The optional Hugging Face adapter uses `AutoTokenizer` and `AutoModel` for
protein, chemical, biomedical, or general text encoders. Training modes are
`frozen`, `full`, and `lora`.

```python
from DeepMTP import FoundationLoRASpec, FoundationModelSpec

protein_model = FoundationModelSpec(
    model_name="facebook/esm2_t6_8M_UR50D",
    revision="<pinned-model-commit>",
    pooling="masked_mean",
    train_mode="lora",
    lora=FoundationLoRASpec(
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=("query", "value"),
    ),
    max_length=1024,
)
```

Tokenize raw values with `tokenize_foundation_sequences`, then pass the
normalized specification to a `SEQUENCE` branch with
`sequence_encoder="foundation"`. Hub revision capture, pooling, length policy,
composite use, compact adapter checkpoints, and supported reference models are
covered in the [foundation-model guide](https://deepmtp.readthedocs.io/en/latest/tutorial/foundation_models.html).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/foundation-model-lifecycle-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/foundation-model-lifecycle-light.svg">
    <img src="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/foundation-model-lifecycle-light.svg" alt="LoRA trains compact adapters around a pinned foundation model, and adapter checkpoints restore those adapters and task weights with the recorded base identity." width="100%" loading="lazy">
  </picture>
</p>

### Molecular graphs

`GRAPH` consumes homogeneous `torch_geometric.data.Data` objects and uses GIN
or edge-aware GINE layers with mean, sum, or max graph pooling. Each graph
needs floating node features, COO `edge_index`, and optional floating edge
features. See the [graph guide](https://deepmtp.readthedocs.io/en/latest/tutorial/graph_inputs.html).

### Composite and partially missing inputs

`COMPOSITE` combines named dense, sparse, graph, sequence, ID, or tabular
components. Fusion can use concatenation, sample-specific gates, or modality
self-attention. Optional components carry presence masks and learned missing
representations; training-only modality dropout can improve robustness.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp-composite-branch-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp-composite-branch-light.svg">
    <img src="https://raw.githubusercontent.com/diliadis/DeepMTP/main/images/claude_design_assets/deepmtp-composite-branch-light.svg" alt="A composite branch independently encodes multiple modalities, substitutes learned representations for missing optional inputs, and combines components with presence-aware fusion." width="100%" loading="lazy">
  </picture>
</p>

```python
config = DeepMTPConfig(
    validation_setting="B",
    problem_mode="classification",
    general_architecture_version="mlp",
    instance_branch_architecture="COMPOSITE",
    instance_branch_composite_fusion="attention",
    instance_branch_composite_attention_dim=32,
    instance_branch_composite_attention_num_heads=4,
    instance_branch_composite_components={
        "descriptors": {
            "architecture": "MLP",
            "input_dim": descriptor_width,
            "output_dim": 16,
        },
        "tokens": {
            "architecture": "SEQUENCE",
            "input_dim": vocabulary_size,
            "output_dim": 16,
            "embedding_dim": 16,
            "optional": True,
        },
    },
    instance_branch_composite_modality_dropout=0.2,
    target_branch_architecture="MLP",
    target_branch_input_dim=number_of_targets,
    target_branch_nodes_per_layer=[32],
    comb_mlp_nodes_per_layer=[32],
)
```

The [composite-input guide](https://deepmtp.readthedocs.io/en/latest/tutorial/composite_inputs.html)
explains component schemas, missingness, and fusion behavior.

## Tasks, losses, and metrics

Binary classification is the backward-compatible classification default and
uses logits with binary cross entropy. Regression defaults to mean squared
error and also supports MAE and Huber losses.

Multiclass mode predicts exactly one of at least three classes. Labels must be
zero-based integer IDs, and every declared class must occur in training data.

<!-- multiclass-config-example:start -->
```python
import numpy as np

from DeepMTP import DeepMTPConfig, data_process

scores = (
    np.arange(18)[:, np.newaxis] + np.arange(3)[np.newaxis, :]
) % 3
train, validation, test, data_info = data_process(
    {"train": {"y": scores}},
    validation_setting="A",
    classification_mode="multiclass",
)

config = DeepMTPConfig(
    validation_setting=data_info["detected_validation_setting"],
    problem_mode="classification",
    classification_mode=data_info["detected_classification_mode"],
    num_classes=data_info["detected_num_classes"],
    metrics=["accuracy", "f1_score", "auroc"],
    metrics_average=["micro"],
    multiclass_average="macro",
    compute_mode="cpu",
    instance_branch_architecture="EMBEDDING",
    instance_branch_input_dim=data_info["instance_branch_input_dim"],
    target_branch_architecture="EMBEDDING",
    target_branch_input_dim=data_info["target_branch_input_dim"],
    embedding_size=8,
)
```
<!-- multiclass-config-example:end -->

| Problem | Default loss | Metrics |
|---|---|---|
| Binary classification | Binary cross entropy with logits | Hamming loss, AUROC, AUPR, accuracy, recall, precision, F1 |
| Multiclass classification | Cross entropy | Accuracy, recall, precision, F1, AUROC, AUPR |
| Regression | Mean squared error | RMSE, MSE, MAE, R² |

`metrics_average` groups observations globally, by target, or by instance.
`multiclass_average` combines classes within applicable multiclass metrics.
See the [multiclass guide](https://deepmtp.readthedocs.io/en/latest/tutorial/multiclass_classification.html)
for prediction columns and averaging rules.

## Training and reproducibility

### Larger effective batches and stable gradients

```python
config = DeepMTPConfig(
    # architecture and task settings...
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0,
)
```

Accumulation combines observation-weighted microbatches before an optimizer
step, including an uneven final group. Clipping applies a global norm limit at
optimizer-step boundaries. Their inert defaults (`1` and `None`) preserve
historical behavior.

### Checkpoints

| Format | File | Contains | Use it for |
|---|---|---|---|
| `full` (default) | `model.pt` | Model, optimizer, configuration, provenance | Restoration, optimizer warm starts, and fully fine-tuned backbones |
| `foundation_adapter` | `foundation_adapter.zip` | PEFT adapters, DeepMTP task weights, identities, hashes | Compact LoRA inference or warm starts |

```python
restored = DeepMTP(
    {
        "results_path": "./results",
        "experiment_name": "restored",
    },
    checkpoint_dir="./results/training-run/model.pt",
)
```

Compact adapter checkpoints exclude base-model and optimizer weights. Fully
fine-tuned foundation models therefore require `full`. Both formats record
preprocessing and environment metadata; load checkpoints only from trusted
sources. Read the [checkpoint guide](https://deepmtp.readthedocs.io/en/latest/tutorial/reproducible_checkpoints.html)
for replay and compatibility details.

### TensorBoard and Weights & Biases

Set `use_tensorboard_logger=True` for local TensorBoard events. For W&B, supply
the project and entity; richer uploads remain explicitly opt-in.

```python
config = DeepMTPConfig(
    # architecture and task settings...
    wandb_project_entity="my-team",
    wandb_project_name="deepmtp",
    wandb_run_name="protein-lora",
    wandb_group="ablations",
    wandb_tags=["protein", "lora"],
    wandb_watch="gradients",
    wandb_log_model_artifact=True,
    wandb_model_artifact_aliases=["latest", "best"],
    wandb_log_predictions=True,
)
```

DeepMTP records epoch-aware metrics, summaries, failures, parameter counts,
bounded prediction tables, model Artifacts, input lineage, and optional W&B
Registry links. Restore either checkpoint format directly:

```python
model = DeepMTP.from_wandb_artifact(
    "my-team/deepmtp/deepmtp-model:best",
    {
        "results_path": "./results",
        "experiment_name": "restored",
    },
)
```

See the [W&B guide](https://deepmtp.readthedocs.io/en/latest/tutorial/wandb_tracking.html)
for offline mode, privacy-conscious defaults, lineage, and Registry setup.

## Hyperparameter optimization

DeepMTP includes random search and Hyperband through the optional `hpo` extra.
Hyperband allocates increasing epoch budgets to promising configurations and
discards weaker candidates early. This small offline example uses one epoch as
an API check; real studies should expand both the search space and budget.

<details>
<summary>Show the complete Hyperband example</summary>

<!-- hpo-example:start -->
```python
import ConfigSpace as CS
import numpy as np

from DeepMTP import DeepMTP, data_process
from DeepMTP.hpo import BaseWorker, HyperBand

rng = np.random.default_rng(42)
scores = (
    np.arange(16)[:, np.newaxis] + np.arange(3)[np.newaxis, :]
) % 2
data = {
    "train": {
        "y": scores,
        "X_instance": rng.normal(size=(16, 3)),
        "X_target": None,
    }
}
train, validation, test, data_info = data_process(
    data,
    validation_setting="B",
)

config_space = CS.ConfigurationSpace(seed=42)
config_space.add(
    [
        CS.Float(
            "learning_rate",
            (1e-4, 1e-2),
            default=1e-3,
            log=True,
        ),
        CS.Integer("embedding_size", (2, 4), default=3),
    ]
)

base_config = {
    "hpo_results_path": "hpo_results",
    "validation_setting": data_info["detected_validation_setting"],
    "problem_mode": data_info["detected_problem_mode"],
    "general_architecture_version": "dot_product",
    "compute_mode": "cpu",
    "num_workers": 0,
    "train_batchsize": 16,
    "val_batchsize": 16,
    "metrics": [],
    "metrics_average": ["macro"],
    "evaluate_train": False,
    "evaluate_val": False,
    "use_early_stopping": False,
    "save_model": True,
    "verbose": False,
    "instance_branch_architecture": "MLP",
    "instance_branch_input_dim": data_info["instance_branch_input_dim"],
    "target_branch_architecture": "MLP",
    "target_branch_input_dim": data_info["target_branch_input_dim"],
}

worker = BaseWorker(
    train,
    validation,
    test,
    data_info,
    base_config,
    metric_to_optimize="loss",
)
optimizer = HyperBand(
    base_worker=worker,
    configspace=config_space,
    eta=2,
    max_budget=1,
    direction="min",
)
best_experiment = optimizer.run_optimizer()

best_model = DeepMTP(
    best_experiment.info["config"],
    checkpoint_dir=best_experiment.info["model_dir"],
)
best_model_results = best_model.predict(test, verbose=True)
```
<!-- hpo-example:end -->

</details>

The [HPO guide](https://deepmtp.readthedocs.io/en/latest/tutorial/hpo.html)
documents search-space routing, budgets, and restoration.

## Demos

The version-controlled notebooks use small datasets and are executed by the
test suite. Their first cell installs the matching source when opened in a
fresh Colab runtime; local execution uses the active environment.

| Workflow | Open in Colab |
|---|---|
| Dataset preparation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/load_datasets.ipynb) |
| Multi-label classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MLC.ipynb) |
| Multivariate regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MTR.ipynb) |
| Multi-task learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MTL.ipynb) |
| Matrix completion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MC.ipynb) |
| Dyadic prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_DP.ipynb) |
| Hyperband | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/hyperband_example.ipynb) |
| Random search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/random_search_example.ipynb) |

The links are checked weekly. Verify their mapping and live endpoints manually
with:

```bash
python scripts/check_colab_links.py
```

## Documentation and project links

- [Documentation](https://deepmtp.readthedocs.io/en/latest/): tutorials and API reference
- [CHANGELOG.md](CHANGELOG.md): user-visible changes by release
- [MIGRATION.rst](MIGRATION.rst): compatibility guidance for existing experiments
- [NEURAL_NETWORK_EXTENSIONS.md](NEURAL_NETWORK_EXTENSIONS.md): implemented and planned model extensions
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup and pull-request checks
- [SECURITY.md](SECURITY.md): supported versions and private vulnerability reporting
- [RELEASING.md](RELEASING.md): maintainer release process
- [Issue tracker](https://github.com/diliadis/DeepMTP/issues): bugs and feature requests

## Development

```bash
git clone https://github.com/diliadis/DeepMTP.git
cd DeepMTP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
pre-commit run --all-files
python -m pytest
```

Conda users can start with `conda env create -f environment.yml`. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Citation

If DeepMTP supports your research, please cite the SoftwareX paper:

```bibtex
@article{iliadis2023deepmtp,
  title = {DeepMTP: A Python-based deep learning framework for multi-target prediction},
  author = {Iliadis, Dimitrios and De Baets, Bernard and Waegeman, Willem},
  journal = {SoftwareX},
  volume = {23},
  pages = {101516},
  year = {2023},
  publisher = {Elsevier},
  doi = {10.1016/j.softx.2023.101516}
}
```

Related work:

- [Two-branch architectures across multi-target prediction settings](https://link.springer.com/article/10.1007/s10994-021-06104-5)
- [Hyperparameter optimization for two-branch neural networks](https://arxiv.org/abs/2211.04362)
- [Embedding aggregation for drug–target interaction prediction](https://www.biorxiv.org/content/10.1101/2023.09.25.559265v2.abstract)

## License

DeepMTP is released under the [MIT License](LICENSE.md).
