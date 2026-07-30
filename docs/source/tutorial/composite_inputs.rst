Composite branch inputs
=======================

A ``COMPOSITE`` branch encodes multiple named representations of the same
entity and fuses their standardized outputs. For example, one instance can
have a dense descriptor vector and a variable-length token sequence:

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
already-tokenized integer rows. Required side-feature components must contain
the same entity IDs. Array inputs receive zero-based IDs by row position; a
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

Component fusion
----------------

``instance_branch_composite_fusion`` and
``target_branch_composite_fusion`` select how the components within one branch
are combined. ``"concat"`` is the default and concatenates the component
encodings without adding parameters, preserving existing configuration and
checkpoint behavior.

Set the option to ``"gated"`` to learn one sample-specific sigmoid gate per
component:

.. code-block:: python

   config = DeepMTPConfig(
       # ...
       instance_branch_architecture="COMPOSITE",
       instance_branch_composite_fusion="gated",
       instance_branch_composite_components={
           "descriptors": descriptor_component,
           "tokens": token_component,
       },
   )

The gate network receives every encoded component followed by one presence
indicator per component. Each scalar gate multiplies its complete component
encoding before the gated encodings are concatenated. Components may therefore
have different output dimensions, and the branch output width remains the sum
of those dimensions. The normal task loss trains the component encoders and
gates jointly; no gate labels are required.

Set the option to ``"attention"`` when component representations should
interact before concatenation:

.. code-block:: python

   config = DeepMTPConfig(
       # ...
       instance_branch_architecture="COMPOSITE",
       instance_branch_composite_fusion="attention",
       instance_branch_composite_attention_dim=32,
       instance_branch_composite_attention_num_heads=4,
       instance_branch_composite_components={
           "descriptors": descriptor_component,
           "tokens": token_component,
       },
   )

Each component has its own projection from its declared output width to
``composite_attention_dim``. The projected components and an always-present
learned fusion token form a short modality-token sequence. Multi-head
self-attention contextualizes those tokens, after which component-specific
projections return them to their original widths. A residual connection adds
each contextualized update to its original encoding before concatenation.
Consequently, attention supports unequal component widths and preserves the
same branch output width as concatenation and gated fusion.

``composite_attention_dim`` and ``composite_attention_num_heads`` must be
positive integers, and the dimension must be divisible by the head count.
These requirements are validated before model construction.

Let ``M`` be the component count, ``D`` the sum of component output widths,
and ``A`` the attention dimension. The current strategies compare as follows:

.. list-table::
   :header-rows: 1
   :widths: 16 37 25 22

   * - Strategy
     - Behavior
     - Added parameters
     - Fusion cost
   * - ``concat``
     - Joins component encodings directly.
     - ``0``
     - Linear in ``D``
   * - ``gated``
     - Learns one sample-specific sigmoid scale per component.
     - ``M * (D + M + 1)``
     - Linear in ``D`` and ``M``
   * - ``attention``
     - Learns contextual relationships between modality tokens.
     - ``2*A*D + M*A + D + 4*A^2 + 7*A``
     - Quadratic in ``M + 1`` and linear in ``A``

The attention parameter expression includes biased input/output projections,
PyTorch multi-head attention, layer normalization, and the learned fusion
token. Encoder and downstream interaction-model parameters are excluded from
all three rows.

Optional components
-------------------

Set ``optional=True`` during both data preparation and model configuration
when a component is unavailable for some entities. For example, descriptors
can cover instances 0, 1, and 2 while token sequences cover only 0 and 2:

.. code-block:: python

   import pandas as pd

   instance_features = {
       "descriptors": pd.DataFrame(
           {
               "id": [0, 1, 2],
               "features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
           }
       ),
       "tokens": pd.DataFrame(
           {
               "id": [0, 2],
               "features": [[1, 4, 2], [3, 2]],
           }
       ),
   }

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
           "tokens": {
               "kind": "sequence",
               "optional": True,
           },
       },
   )

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
               "input_dim": 2,
               "output_dim": 32,
           },
           "tokens": {
               "architecture": "SEQUENCE",
               "input_dim": data_info[
                   "instance_branch_component_input_dims"
               ]["tokens"],
               "output_dim": 32,
               "optional": True,
           },
       },
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[64],
       embedding_size=64,
   )

The descriptors establish the required entity IDs. The missing token value for
instance 1 is preserved through splitting and loading. Its batch mask is
``[True, False, True]`` when those three instances appear in order.

If every side-feature component is optional, preparation uses the union of
their declared IDs. An entity must therefore appear in at least one component
source, or be included explicitly with a ``None`` feature value. Each supplied
feature source must contain at least one present value for every optional
component so DeepMTP can infer its input dimension.

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

Optional MLP and sparse components cannot enable batch normalization because a
training batch may contain only one present row. ID components cannot be
optional because an entity ID is always available once its interaction row is
known.

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
``MaskedComponentInput`` stores an optional component's present-only values
plus a Boolean ``presence_mask`` for the full batch. An entirely missing
component in one batch has ``values=None`` and an all-false mask.

``CompositeEncoder`` runs each configured child encoder, validates its
``[present_count, output_dim]`` result, scatters it back into the full batch,
and fills missing rows with a zero-initialized learned representation. It then
applies the configured component fusion in declared order. Gated fusion
receives the explicit presence indicators, so it can suppress a missing
component or preserve a useful missingness signal.

Attention fusion excludes a missing component from the key/value evidence seen
by every modality. Its learned missing representation remains a query and can
therefore receive context from present modalities without influencing them.
The fusion token is never masked, so a row where every optional component is
missing still has a finite attention source and produces a stable learned
fallback. The outer trainer still receives one instance input and one target
input, so training, prediction, interaction fusion, and checkpoint code do not
need missing-modality branches.

Compatibility and current limits
--------------------------------

``COMPOSITE`` is opt-in. Existing dense, sparse, sequence, tabular, image,
embedding, and custom configurations keep their previous batch and checkpoint
formats. Composite component configuration and weights are stored in the
normal checkpoint payload and can be restored for prediction.

``optional`` defaults to ``False``. Existing composite configurations create
no missing-representation parameters, so checkpoints written before this
feature retain the same model state keys. Opt-in learned representations are
saved and restored with the other branch parameters.

``instance_branch_composite_fusion`` and
``target_branch_composite_fusion`` default to ``"concat"``. Concatenating
branches create no gate or attention parameters, so existing composite state
dictionaries remain unchanged. A ``"gated"`` checkpoint stores its gate weights
and fusion setting in the normal checkpoint payload. An ``"attention"``
checkpoint also stores its component projections, self-attention block, layer
normalization, fusion token, attention dimension, and head count.

Legacy feature scaling accepts only one dense representation and is therefore
rejected for composite inputs. Scale dense components upstream; sparse and
token components must remain unscaled.

Modality dropout remains planned follow-up work. Nested composites and tabular,
image, or custom components are not yet supported.
