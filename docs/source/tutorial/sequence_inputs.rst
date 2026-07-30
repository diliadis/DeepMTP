Token-sequence inputs
=====================

A ``SEQUENCE`` branch accepts already-tokenized variable-length entity
features. It is intended for inputs such as protein residues, chemical string
tokens, or text tokens when each entity has one ordered integer sequence.

DeepMTP does not tokenize raw strings. Tokenization and vocabulary construction
remain domain-specific preprocessing steps outside the core package.

Input format
------------

Pass a collection containing one unpadded integer sequence per entity:

.. code-block:: python

   instance_sequences = [
       [5, 12, 8],
       [7],
       [3, 14, 9, 6],
       [11, 2],
   ]

Rows receive zero-based entity IDs in collection order. To preserve explicit
IDs, pass a pandas ``DataFrame`` with unique ``id`` and ``features`` columns:

.. code-block:: python

   import pandas as pd

   instance_sequences = pd.DataFrame(
       {
           "id": [10, 30, 50],
           "features": [
               [5, 12, 8],
               [7],
               [3, 14, 9, 6],
           ],
       }
   )

Declare the representation while preparing data so variable-length rows are
validated as token sequences:

.. code-block:: python

   from DeepMTP import data_process

   train, validation, test, data_info = data_process(
       {
           "train": {
               "y": interaction_scores,
               "X_instance": instance_sequences,
               "X_target": None,
           }
       },
       validation_setting="B",
       instance_feature_kind="sequence",
   )

Token IDs must be non-negative integers. Every sequence must contain at least
one token. Raw rows must not include the configured padding ID because DeepMTP
adds padding during collation and derives masks and lengths from the original
rows.

Configuration
-------------

``instance_branch_input_dim`` or ``target_branch_input_dim`` is the complete
token vocabulary size, including reserved padding and unknown-token entries.
It is not the maximum sequence length.

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       compute_mode="cpu",
       metrics=["accuracy"],
       metrics_average=["macro"],
       instance_branch_architecture="SEQUENCE",
       instance_branch_input_dim=vocabulary_size,
       instance_branch_sequence_encoder="transformer",
       instance_branch_sequence_embedding_dim=64,
       instance_branch_sequence_transformer_num_heads=4,
       instance_branch_sequence_transformer_feedforward_dim=256,
       instance_branch_sequence_num_layers=1,
       instance_branch_sequence_padding_idx=0,
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[32],
       embedding_size=32,
   )

Three lightweight sequence encoders are available:

* ``"gru"`` packs valid tokens before applying a recurrent encoder.
* ``"conv1d"`` applies same-width temporal convolutions, masks padded
  positions after every layer, and uses masked global max pooling.
* ``"transformer"`` adds deterministic sinusoidal positions, applies
  padding-aware self-attention, and uses masked mean pooling.

``instance_branch_sequence_num_layers`` and
``target_branch_sequence_num_layers`` control the number of stacked encoder
layers. Conv1D kernel widths are configured separately for each branch with
``*_branch_sequence_conv_kernel_size`` and must be positive odd integers so
the temporal width remains unchanged.

Transformer attention heads and feed-forward widths are configured with
``*_branch_sequence_transformer_num_heads`` and
``*_branch_sequence_transformer_feedforward_dim``. The embedding width must be
divisible by the head count. Sinusoidal positions are generated for the
current batch width, so no maximum sequence length is stored in configuration.
The existing per-branch dropout setting applies between stacked GRU layers
and to the GRU output, after each Conv1D activation, or within Transformer
attention and feed-forward sublayers.

For dot-product fusion, the sequence output width is ``embedding_size`` so
both branches match. For MLP and Kronecker fusion, configure
``instance_branch_sequence_output_dim`` or
``target_branch_sequence_output_dim``; the default is 64.

Structured batches
------------------

The collator pads only to the longest row in each batch. For sequences
``[12, 7, 31]`` and ``[4]`` with padding ID zero, the model receives:

.. code-block:: text

   token_ids       = [[12, 7, 31],
                      [ 4, 0,  0]]
   attention_mask  = [[ 1, 1,  1],
                      [ 1, 0,  0]]
   lengths         = [3, 1]

``SequenceInput`` carries those three tensors together. It validates that the
mask is left-aligned, each length equals the number of valid positions, valid
positions do not contain the padding ID, and padded positions do.

The GRU uses packed sequences. Conv1D explicitly resets padded positions after
each temporal layer and excludes them from global pooling. The Transformer
passes the inverse attention mask to every self-attention layer and excludes
padded outputs from mean pooling. Consequently, dynamic padding does not
affect any encoder's entity representation. All three emit one fixed-width
floating-point vector per entity and work with the existing dot-product, MLP,
and Kronecker fusion models.

Compatibility and checkpoints
-----------------------------

``SEQUENCE`` is opt-in. Existing dense, sparse, tabular, image, ID-embedding,
and custom branches retain their historical behavior. Legacy numeric feature
scaling rejects token sequences because token IDs are categorical rather than
continuous values.

Sequence configuration and encoder weights are stored in the existing
checkpoint payload. Restore with the same encoder name, vocabulary size,
padding ID, embedding width, layer count, and fusion settings. Conv1D
checkpoints also retain their kernel width. Older GRU checkpoints omit that
new field and load with its inert default. Transformer checkpoints additionally
retain their head count and feed-forward width; older GRU and Conv1D
checkpoints load with inactive defaults for those settings.

Precomputed domain-model embeddings are already fixed-width dense vectors.
Pass those through an ``MLP`` branch rather than treating them as token
sequences.
