Molecular graph inputs
======================

DeepMTP's optional ``GRAPH`` branch represents each entity as one homogeneous
graph and produces one fixed-width vector per graph. This is suitable for
precomputed molecular graphs and other data with one node and edge type.

Installation
------------

PyTorch Geometric is not a core dependency. Install the graph extra::

   python -m pip install "DeepMTP[graph]"

Input contract
--------------

Each entity is a ``torch_geometric.data.Data`` object with:

* ``x``: finite floating node features shaped
  ``[num_nodes, node_features]``;
* ``edge_index``: integer COO indices shaped ``[2, num_edges]``;
* ``edge_attr``: optional finite floating edge features shaped
  ``[num_edges, edge_features]``.

Every graph must contain at least one node. All graphs on one entity axis must
use the same node width and must either all omit edge features or all use the
same edge width. DeepMTP validates and clones caller-owned graphs during data
preparation.

.. code-block:: python

   import torch
   from torch_geometric.data import Data

   graph = Data(
       x=torch.tensor(
           [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]],
           dtype=torch.float32,
       ),
       edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
       edge_attr=torch.tensor(
           [[1.0, 0.0], [1.0, 0.0]],
           dtype=torch.float32,
       ),
   )

Data preparation
----------------

Declare graph features explicitly. A sequence assigns zero-based IDs in row
order. A DataFrame with exactly ``id`` and ``features`` columns preserves
explicit IDs.

.. code-block:: python

   from DeepMTP import data_process

   train, validation, test, data_info = data_process(
       {
           "train": {
               "y": interaction_scores,
               "X_instance": molecular_graphs,
               "X_target": None,
           }
       },
       validation_setting="B",
       instance_feature_kind="graph",
   )

``data_info["instance_branch_input_dim"]`` is the inferred node-feature width.
``data_info["instance_branch_graph_edge_dim"]`` is the inferred edge-feature
width, or ``None`` when ``edge_attr`` is absent. Legacy branch-level feature
scaling is rejected because node and edge preprocessing must be applied
explicitly upstream.

Branch configuration
--------------------

Use the inferred widths when configuring the branch:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       instance_branch_architecture="GRAPH",
       instance_branch_input_dim=data_info["instance_branch_input_dim"],
       instance_branch_graph_edge_dim=data_info[
           "instance_branch_graph_edge_dim"
       ],
       instance_branch_graph_hidden_dim=64,
       instance_branch_graph_output_dim=32,
       instance_branch_graph_num_layers=3,
       instance_branch_graph_pooling="mean",
       instance_branch_graph_use_edge_features=True,
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[32],
       embedding_size=32,
   )

With edge features enabled, DeepMTP uses GINE message passing and requires a
positive edge width. Set ``branch_graph_use_edge_features=False`` and
``branch_graph_edge_dim=None`` for node-only GIN. Pooling can be ``"mean"``,
``"sum"``, or ``"max"``. ``branch_graph_output_dim`` controls output width for
MLP and Kronecker fusion; dot-product fusion uses ``embedding_size``.

Batch contract
--------------

PyTorch Geometric offsets node indices while collating. DeepMTP immediately
converts that result to :class:`DeepMTP.data.GraphInput`, which contains:

* ``node_features``;
* ``edge_index``;
* optional ``edge_features``;
* ``graph_index``, mapping every node to a graph in the batch;
* ``num_graphs``.

This keeps PyG objects at the data boundary. Device transfer, pinned memory,
encoder validation, fusion, training, prediction, and checkpoint restoration
use the same typed contracts as other DeepMTP modalities.

Composite graph components
--------------------------

A graph can be a required or optional component of a ``COMPOSITE`` branch.
Declare ``kind="graph"`` during preparation and ``architecture="GRAPH"`` in
the matching component configuration:

.. code-block:: python

   instance_feature_kind = {
       "molecule": {"kind": "graph", "optional": True},
       "descriptors": None,
   }

   instance_branch_composite_components = {
       "molecule": {
           "architecture": "GRAPH",
           "input_dim": component_dims["molecule"],
           "edge_dim": component_edge_dims["molecule"],
           "hidden_dim": 64,
           "output_dim": 16,
           "num_layers": 3,
           "pooling": "mean",
           "use_edge_features": True,
           "optional": True,
       },
       "descriptors": {
           "architecture": "MLP",
           "input_dim": component_dims["descriptors"],
           "output_dim": 16,
       },
   }

The component widths are available from
``instance_branch_component_input_dims`` and
``instance_branch_component_graph_edge_dims`` in ``data_info``. Optional graph
components use the same presence masks, learned missing representations,
fusion strategies, and modality dropout as other optional components.

Scope
-----

This adapter deliberately supports homogeneous precomputed graphs. It does not
generate molecular graphs from SMILES or require RDKit, and it does not yet
accept heterogeneous PyG graphs. Those integrations can be added separately
without changing the current batch or branch defaults.
