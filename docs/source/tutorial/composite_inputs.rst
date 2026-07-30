Composite branch inputs
=======================

A ``COMPOSITE`` branch encodes multiple named representations of the same
entity and concatenates their standardized outputs. For example, one instance
can have a dense descriptor vector and a variable-length token sequence:

.. code-block:: python

   instance_features = {
       "descriptors": descriptor_matrix,
       "tokens": token_sequences,
   }

The component names are part of the data and model contract. Declare how each
side-feature collection should be interpreted during preparation:

.. code-block:: python

   from DeepMTP import data_process

   train, validation, test, data_info = data_process(
       {
           "train": {
               "y": interaction_scores,
               "X_instance": instance_features,
               "X_target": None,
           }
       },
       validation_setting="B",
       instance_feature_kind={
           "descriptors": None,
           "tokens": "sequence",
       },
   )

``None`` keeps dense and sparse auto-detection. ``"sequence"`` identifies
already-tokenized integer rows. Every side-feature component must contain the
same entity IDs. Array inputs receive zero-based IDs by row position; a
DataFrame can preserve explicit IDs with ``id`` and ``features`` columns.

Configuration
-------------

Each component declares its own architecture, input dimension, and output
dimension. The component order in the configuration determines concatenation
order:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   component_dims = data_info["instance_branch_component_input_dims"]

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       compute_mode="cpu",
       metrics=["accuracy"],
       metrics_average=["macro"],
       instance_branch_architecture="COMPOSITE",
       instance_branch_composite_components={
           "descriptors": {
               "architecture": "MLP",
               "input_dim": component_dims["descriptors"],
               "output_dim": 32,
               "nodes_per_layer": [64],
           },
           "tokens": {
               "architecture": "SEQUENCE",
               "input_dim": component_dims["tokens"],
               "output_dim": 32,
               "encoder": "gru",
               "embedding_dim": 32,
               "num_layers": 1,
               "padding_idx": 0,
           },
       },
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[64],
       embedding_size=64,
   )

For dot-product fusion, the sum of the component ``output_dim`` values must
equal ``embedding_size``. MLP and Kronecker fusion use the concatenated width
directly.

Supported components
--------------------

The first composite slice supports:

* ``MLP`` for fixed-width dense vectors;
* ``SPARSE`` for SciPy or PyTorch sparse rows;
* ``SEQUENCE`` with the GRU, Conv1D, or Transformer encoders; and
* ``EMBEDDING`` for a known entity ID.

MLP and sparse components accept ``nodes_per_layer``, ``num_layers``,
``dropout_rate``, and ``batch_norm``. Sequence components accept ``encoder``,
``embedding_dim``, ``num_layers``, ``padding_idx``, ``dropout_rate``,
``conv_kernel_size``, ``transformer_num_heads``, and
``transformer_feedforward_dim``.

An ID component is added by configuration and should not appear in the
side-feature mapping. It is rejected when the corresponding entity axis is
novel in the validation setting, just like a standalone ``EMBEDDING`` branch.
For example, an instance ID plus descriptor hybrid is valid in settings A and
C, but not B or D.

Batch and encoder contract
--------------------------

``CompositeInput`` is a mapping from component name to an existing model input
such as a tensor or ``SequenceInput``. It validates that every component has
the same batch size and moves or pins all component tensors together.

``CompositeEncoder`` runs each configured child encoder, validates its
``[batch_size, output_dim]`` result, and concatenates the results in declared
order. The outer trainer still receives one instance input and one target
input, so training, prediction, fusion, and checkpoint code do not need
composite-specific branches.

Compatibility and current limits
--------------------------------

``COMPOSITE`` is opt-in. Existing dense, sparse, sequence, tabular, image,
embedding, and custom configurations keep their previous batch and checkpoint
formats. Composite component configuration and weights are stored in the
normal checkpoint payload and can be restored for prediction.

Legacy feature scaling accepts only one dense representation and is therefore
rejected for composite inputs. Scale dense components upstream; sparse and
token components must remain unscaled.

Every configured side-feature component is currently required for every
entity. Explicit missing-modality masks, learned missing representations,
gated or attention fusion, and modality dropout remain planned follow-up
work. Nested composites and tabular, image, or custom components are not yet
supported.
