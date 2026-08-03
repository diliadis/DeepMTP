Foundation-model sequence encoders
==================================

DeepMTP can use a Hugging Face encoder as an opt-in ``SEQUENCE`` branch for
protein sequences, SMILES strings, or text. The integration loads an existing
foundation model; it does not train a language model from scratch.

Install the optional dependency:

.. code-block:: console

   pip install "DeepMTP[foundation]"

The core ``DeepMTP`` installation does not import or require Transformers or
PEFT. Both are installed by the ``foundation`` extra, and imports remain lazy.

Supported model contract
------------------------

The generic adapter uses ``AutoTokenizer`` and ``AutoModel``. A compatible
model must:

* accept ``input_ids`` and ``attention_mask``;
* declare a padding token and vocabulary size;
* expose a positive ``hidden_size``, ``d_model``, or ``dim``;
* return ``last_hidden_state`` for ``cls`` or ``masked_mean`` pooling, or
  ``pooler_output`` for ``pooler`` pooling.

DeepMTP tests the generic adapter contract and a locally saved BERT model
without requiring network access in CI. The following checkpoints are
documented reference choices for the corresponding domains. Other compatible
encoder checkpoints may work through the same adapter, but individual Hub
checkpoints are not downloaded as part of the automated test suite.

.. list-table::
   :header-rows: 1

   * - Domain
     - Reference checkpoint
   * - Protein
     - ``facebook/esm2_t6_8M_UR50D``
   * - Chemical/SMILES
     - ``DeepChem/ChemBERTa-77M-MLM``
   * - General text
     - ``google-bert/bert-base-uncased``
   * - Biomedical text
     - ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext``

Tokenizing raw values
---------------------

Foundation models require their own tokenizer. ``tokenize_foundation_sequences``
converts raw strings to unpadded integer rows, which lets the existing DeepMTP
collator pad only to the longest sequence in each batch.

.. code-block:: python

   from DeepMTP import FoundationModelSpec, tokenize_foundation_sequences

   protein_model = FoundationModelSpec(
       model_name="facebook/esm2_t6_8M_UR50D",
       revision="<pinned-model-commit>",
       pooling="masked_mean",
       train_mode="frozen",
       max_length=1024,
       truncation=False,
   )
   proteins = tokenize_foundation_sequences(
       ["MKTAYIAKQRQISFVKSHFSRQ", "GILGFVFTLTVPSER"],
       protein_model,
   )

The returned ``TokenizedFoundationSequences`` behaves as a sequence of NumPy
integer arrays and also records:

* the complete tokenizer vocabulary size;
* its padding ID;
* requested model and tokenizer revisions;
* the resolved tokenizer revision when the Hub provides one;
* tokenizer-declared boundary-token IDs used by ``masked_mean`` pooling;
* maximum-length and truncation settings.

Pass it to ``data_process`` with the normal sequence feature declaration:

.. code-block:: python

   train, validation, test, data_info = data_process(
       {
           "train": {
               "y": interaction_scores,
               "X_instance": proteins,
               "X_target": None,
           }
       },
       validation_setting="B",
       instance_feature_kind="sequence",
   )

Use ``proteins.vocabulary_size`` rather than the vocabulary inferred from only
the observed rows. The foundation model embedding table covers the tokenizer's
complete vocabulary.

Length policy
-------------

Truncation is never implicit. With ``max_length`` configured and
``truncation=False``, tokenization raises an error if any encoded row exceeds
the limit. Set ``truncation=True`` only when removing trailing tokens is valid
for the domain and task. Enabling truncation requires an explicit
``max_length``.

Configuring the branch
----------------------

Use ``instance_branch_sequence_encoder="foundation"`` or the corresponding
target field. Reuse the normalized specification returned by tokenization so
the resolved tokenizer revision is retained:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       instance_branch_architecture="SEQUENCE",
       instance_branch_input_dim=proteins.vocabulary_size,
       instance_branch_sequence_encoder="foundation",
       instance_branch_sequence_padding_idx=proteins.padding_idx,
       instance_branch_foundation_model=proteins.spec.to_dict(),
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[64],
       embedding_size=256,
   )

The adapter pools the foundation-model representation and always applies a
trainable linear projection to the DeepMTP branch output size. Keeping this
projection even when the widths already match ensures that frozen backbones
still have a small task-specific adapter to optimize. Dot-product models
project both branches to ``embedding_size``. MLP and Kronecker fusion use
``instance_branch_sequence_output_dim`` or
``target_branch_sequence_output_dim``.

Pooling
-------

``masked_mean``
   Average token states selected by the attention mask. Padding never
   contributes. When DeepMTP's tokenizer helper is used, tokenizer-declared
   CLS, SEP, BOS, and EOS tokens are also excluded by default. Set
   ``exclude_special_tokens_from_pooling=False`` to include them, or pass
   ``pooling_excluded_token_ids`` when supplying pre-tokenized rows. This is
   the portable default.

``cls``
   Use the first token state. Select this only for models whose tokenizer and
   pretraining convention assign a sequence representation to the first token.

``pooler``
   Use the model's explicit ``pooler_output``. DeepMTP reports an error if the
   selected model does not provide it.

Training modes
--------------

``frozen``
   Freeze every backbone parameter and keep the backbone in evaluation mode,
   including while the surrounding DeepMTP model trains. Only the output
   projection and downstream DeepMTP parameters are optimized. This is the
   default and is usually the best first experiment.

``full``
   Fine-tune the complete foundation model with the DeepMTP objective. This
   consumes substantially more accelerator memory and creates much larger
   optimizer states.

``lora``
   Freeze the pretrained backbone weights and train low-rank adapters injected
   by Hugging Face PEFT, together with the DeepMTP output projection and
   downstream parameters. This adapts the foundation model while usually using
   far fewer trainable parameters and optimizer states than ``full`` mode.

LoRA configuration
------------------

LoRA is explicit and opt-in. Supply ``FoundationLoRASpec`` with a positive
rank and alpha, dropout in ``[0, 1)``, and at least one target-module name:

.. code-block:: python

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

``rank`` controls the dimension of the low-rank update. Larger values add
trainable capacity and parameters. ``alpha`` scales that update, while
``dropout`` regularizes the LoRA branch during training. PEFT initializes the
default LoRA update as a no-op, so the wrapped model initially retains the
pretrained behavior.

Target modules are architecture-specific. PEFT treats each listed value as an
exact module name or module-name suffix. For example, BERT-family and ESM-2
attention projections expose ``query`` and ``value`` suffixes. Do not copy
these names to a different architecture without inspecting that model.
DeepMTP checks every requested suffix against the loaded, unwrapped model and
reports available leaf-module suffixes when a target does not match. PEFT then
performs its own compatibility check for the matched module type.

Only the LoRA parameters inside the foundation backbone are trainable in this
mode. The ordinary DeepMTP projection remains trainable because it belongs to
the task model rather than the pretrained backbone. ``model.state_dict()`` and
normal DeepMTP checkpoints continue to contain the complete restorable model
state.

Caching frozen entity representations
--------------------------------------

A frozen foundation backbone returns the same pooled representation every time
it sees the same tokenized entity. DeepMTP can cache that representation before
the trainable output projection, avoiding repeated backbone work when one
protein, molecule, or text entity participates in many interaction pairs.

Caching is explicit and is valid only with ``train_mode="frozen"``. A
memory-only cache is useful within one process:

.. code-block:: python

   from DeepMTP import FoundationCacheSpec, FoundationModelSpec

   protein_model = FoundationModelSpec(
       model_name="facebook/esm2_t6_8M_UR50D",
       revision="<pinned-model-commit>",
       pooling="masked_mean",
       train_mode="frozen",
       cache=FoundationCacheSpec(memory_capacity=4096),
   )

``memory_capacity`` is the maximum number of pooled CPU tensors retained by
the in-process least-recently-used cache. The default is 1024. Set a smaller
limit when entity embeddings are wide or host memory is constrained.

Add ``directory`` to persist entries across runs:

.. code-block:: python

   protein_model = FoundationModelSpec(
       model_name="facebook/esm2_t6_8M_UR50D",
       revision="<pinned-model-commit>",
       pooling="masked_mean",
       train_mode="frozen",
       cache=FoundationCacheSpec(
           directory="./cache/esm2-pooled",
           memory_capacity=4096,
       ),
   )

   # Use the tokenizer-normalized spec so the resolved tokenizer revision is
   # available before ModelFactory constructs the persistent cache.
   proteins = tokenize_foundation_sequences(raw_proteins, protein_model)
   config = DeepMTPConfig(
       # ...
       instance_branch_foundation_model=proteins.spec.to_dict(),
   )

Set ``memory_capacity=0`` with a directory for disk-only operation. Persistent
caching requires both resolved model and tokenizer revisions. This prevents a
moving Hub branch or an unidentified local snapshot from silently reusing
outputs generated by different weights or tokenization rules.

Cache entries are keyed by each entity's unpadded token IDs. The namespace also
includes the resolved model and tokenizer revisions, maximum-length and
truncation policy, pooling settings, excluded special-token IDs, vocabulary and
padding metadata, hidden width, and backbone dtype. Each disk tensor is checked
against that identity, shape, and dtype when loaded. An unreadable, stale, or
incompatible manifest or entry raises an error rather than falling back
silently. New files are written atomically.

The cache stores pooled tensors on CPU. Only the compact pooled batch is moved
to the model device, and the trainable projection is still evaluated normally,
so its gradients and downstream optimization are unchanged. Duplicate misses
within a batch are encoded once. Cache tensors are derived data: they are not
copied into early-stopping snapshots, full checkpoints, compact adapter
archives, or W&B Artifacts. The normalized cache settings remain present in
the saved configuration and W&B run configuration.

Inspect or clear a built encoder through:

.. code-block:: python

   bundle = ModelFactory(config).build()
   encoder = bundle.instance_branch

   print(encoder.cache_stats)
   encoder.clear_cache()                 # memory only
   encoder.clear_cache(disk=True)        # this identity's disk entries too

For a cached foundation component inside ``COMPOSITE``, select the named
component first, for example
``bundle.instance_branch.component_encoders["protein"]``.

Do not enable this cache for ``full`` or ``lora`` training. Their backbone
representations change as parameters are optimized, so DeepMTP rejects the
combination during configuration validation. Cache directories should be
treated as disposable derived artifacts and excluded from source control.

To measure the effect with a checkpoint and workload representative of your
project, run the bundled benchmark. Repeat ``--value`` to supply domain-specific
protein, SMILES, or text entities:

.. code-block:: console

   python scripts/benchmark_foundation_cache.py \
       --model-name google-bert/bert-base-uncased \
       --revision <pinned-model-commit> \
       --repeats 32 \
       --device cuda

The report separates an uncached median, the first cache pass, and the warm
cache median. Model and tokenizer loading are intentionally outside the timed
region. If ``--cache-directory`` points at existing compatible entries, the
first cache pass may already use persistent hits.

Compact adapter checkpoints
---------------------------

For a smaller LoRA deployment or warm-start artifact, opt into the compact
checkpoint format:

.. code-block:: python

   config = DeepMTPConfig(
       # foundation-model and training settings...
       checkpoint_format="foundation_adapter",
   )

After training, ``save_model=True`` writes ``foundation_adapter.zip`` instead
of ``model.pt``. The archive contains:

* PEFT-native ``adapter_config.json`` and ``adapter_model.safetensors`` files
  for every LoRA foundation component;
* DeepMTP projections, fusion layers, output layers, and other task-owned
  state in ``task_state.pt``;
* the normalized configuration and checkpoint provenance metadata;
* exact base-model and tokenizer identities plus SHA-256 hashes for every
  required state file.

It deliberately excludes all foundation-backbone tensors and optimizer state.
Consequently, it is an inference or warm-start format, not an exact interrupted
training resume. Use the default ``checkpoint_format="full"`` when optimizer
restoration is required or any foundation branch uses ``train_mode="full"``.
Compact export requires at least one LoRA branch; additional frozen foundation
branches are recorded and reconstructed without adapter files.

Restore the compact archive through the same constructor argument used for
full checkpoints:

.. code-block:: python

   restored = DeepMTP(
       {
           "results_path": "./results",
           "experiment_name": "restored",
       },
       checkpoint_dir="./results/training-run/foundation_adapter.zip",
   )

DeepMTP first rebuilds the recorded base models, restores task-owned weights,
and then applies each PEFT adapter. Remote Hub identities must include resolved
model and tokenizer revisions so restoration cannot silently move to newer
weights. A local model directory is treated as a user-managed immutable
snapshot and must exist at restoration time. Archive hashes detect accidental
or malicious modification, but checkpoints and model repositories should still
be loaded only from trusted sources.

When ``wandb_log_model_artifact=True``, the same archive is uploaded as the
model Artifact and its foundation identities are attached as Artifact
metadata. ``DeepMTP.from_wandb_artifact`` discovers and restores either one
``model.pt`` or one ``foundation_adapter.zip`` without a separate API.

Composite inputs
----------------

A foundation encoder can be one component of ``COMPOSITE``:

.. code-block:: python

   instance_branch_composite_components = {
       "descriptors": {
           "architecture": "MLP",
           "input_dim": descriptor_width,
           "output_dim": 64,
           "nodes_per_layer": [128, 64],
       },
       "smiles": {
           "architecture": "SEQUENCE",
           "input_dim": smiles_tokens.vocabulary_size,
           "output_dim": 128,
           "encoder": "foundation",
           "padding_idx": smiles_tokens.padding_idx,
           "foundation_model": smiles_tokens.spec.to_dict(),
       },
   }

The component works with concatenation, gated fusion, attention fusion,
optional modalities, and modality dropout through the existing composite
contracts.

Reproducibility and checkpoints
-------------------------------

Use an immutable model revision, preferably a Hub commit ID. Transformers
accepts a branch, tag, or commit through ``revision``; a moving branch such as
``main`` is less reproducible.

When the Hub exposes a resolved commit, DeepMTP records it in
``resolved_model_revision`` or ``resolved_tokenizer_revision``. These values
are included in the normalized configuration, local checkpoints, and W&B run
configuration. Restoration prefers the resolved revision.

The checkpoint state dictionary contains both the foundation backbone and the
DeepMTP projection, including PEFT adapter parameters in ``lora`` mode. Model
construction still needs the Transformers package, PEFT when applicable, and
either the pinned model metadata in the local Hugging Face cache, a local model
directory, or network access to the recorded revision.

Security and offline use
------------------------

``trust_remote_code`` defaults to ``False``. Set it to ``True`` only after
reviewing and pinning a model repository whose custom Python implementation is
required. This setting may execute repository code on the local machine.

Set ``local_files_only=True`` to prohibit Hub downloads. The requested model,
tokenizer, and revision must already exist in the local Hugging Face cache or
be supplied through a local directory.

Model and dataset licenses, gated-repository authentication, and acceptable
use remain the responsibility of the user.
