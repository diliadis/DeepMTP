Foundation-model sequence encoders
==================================

DeepMTP can use a Hugging Face encoder as an opt-in ``SEQUENCE`` branch for
protein sequences, SMILES strings, or text. The integration loads an existing
foundation model; it does not train a language model from scratch.

Install the optional dependency:

.. code-block:: console

   pip install "DeepMTP[foundation]"

The core ``DeepMTP`` installation does not import or require Transformers.

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

Parameter-efficient fine-tuning such as LoRA is not part of the initial
adapter. It can be layered on after frozen and full fine-tuning behavior is
stable.

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
DeepMTP projection. Model construction still needs the Transformers package
and either the pinned model metadata in the local Hugging Face cache, a local
model directory, or network access to the recorded revision.

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
