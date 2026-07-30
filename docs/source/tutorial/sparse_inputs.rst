Sparse high-dimensional inputs
==============================

A ``SPARSE`` branch accepts high-dimensional entity features without
materializing dense dataset rows or dense model batches. This is useful for
fingerprints, bag-of-words vectors, one-hot indicators, and other inputs whose
stored values are mostly zero.

Supported inputs
----------------

Pass either of these as ``X_instance`` or ``X_target``:

* a two-dimensional PyTorch COO or CSR sparse tensor; or
* a SciPy sparse matrix. Install SciPy support with
  ``python -m pip install "DeepMTP[sparse]"``.

Matrix inputs receive zero-based entity IDs in row order, matching the
existing NumPy convention. To preserve explicit IDs, pass a pandas
``DataFrame`` with unique ``id`` and ``features`` columns, where every feature
value is one sparse row. All rows must use the same backend and width.

.. code-block:: python

   import numpy as np
   from scipy import sparse

   instance_features = sparse.csr_matrix(
       (
           np.array([1.0, 0.5, 2.0], dtype=np.float32),
           (
               np.array([0, 1, 2]),
               np.array([17, 120_004, 999_999]),
           ),
       ),
       shape=(3, 1_000_000),
   )

Configuration
-------------

Select ``SPARSE`` for the branch and declare the full input width. Layer
options have the same meaning as for ``MLP``:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       compute_mode="cpu",
       metrics=["accuracy"],
       metrics_average=["macro"],
       instance_branch_architecture="SPARSE",
       instance_branch_input_dim=1_000_000,
       instance_branch_nodes_per_layer=[128, 32],
       target_branch_architecture="MLP",
       target_branch_input_dim=number_of_targets,
       target_branch_nodes_per_layer=[32],
       embedding_size=32,
   )

The dataloader converts each sparse batch to a floating PyTorch COO tensor.
``SparseMLP`` applies the first dense projection with ``torch.sparse.mm``;
activation, batch normalization, dropout, and later layers operate on the
resulting low-dimensional dense representation. Dot-product, MLP, and
Kronecker fusion all accept sparse branches.

Memory and preprocessing
------------------------

Sparse inputs remain sparse through normalization, splitting, row selection,
collation, device transfer, and the first projection. The projection weights
remain dense, so their memory cost is
``input_dim * first_output_width`` parameters. Choose a modest first width for
extremely wide inputs.

Legacy ``scale_instance_features`` and ``scale_target_features`` options reject
sparse inputs because the existing scikit-learn path stacks dense rows.
Apply sparse-safe preprocessing before calling ``data_process`` instead.
Avoid transformations such as mean centering that turn implicit zeros into
stored nonzero values.

Configuration and checkpoint payloads retain their existing structure.
``SPARSE`` is opt-in, so existing ``MLP`` configurations and checkpoints are
unchanged. A sparse checkpoint must be restored with the same architecture,
input width, and layer settings.

Benchmarking
------------

The repository includes a reproducible CPU comparison of input storage and
first-projection throughput:

.. code-block:: console

   python scripts/benchmark_sparse_inputs.py \
       --batch-size 512 --input-dim 50000 --density 0.001

Results depend on shape, density, hardware, and PyTorch version. Use dimensions
representative of the intended experiment rather than treating the defaults as
a universal performance claim.
