Mixed tabular inputs
====================

A ``TABULAR`` branch keeps continuous measurements and categorical values as
different inputs. Continuous columns are normalized with statistics fitted on
training entities. Each categorical column is converted to integer indices and
passed through its own learned embedding table.

Input format
------------

Pass entity features as a pandas ``DataFrame`` with one unique ``id`` column
and the columns declared by the branch schema:

.. code-block:: python

   import numpy as np
   import pandas as pd

   instance_features = pd.DataFrame(
       {
           "id": np.arange(6),
           "age": [22.0, 35.0, None, 51.0, 47.0, 29.0],
           "dose": [1.2, 0.8, 1.5, 1.1, 0.7, 1.0],
           "site": ["Ghent", "Paris", "Ghent", "London", "Paris", None],
       }
   )

The schema fixes column order and categorical vocabularies before model
construction:

.. code-block:: python

   tabular_schema = {
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
   }

Category index zero is reserved for missing or unknown values when the
corresponding policy is ``"unknown"``. Set either policy to ``"error"`` when
such values should stop data loading instead.

Configuration
-------------

Select ``TABULAR`` independently for either branch. It can be paired with an
``MLP``, ``EMBEDDING``, ``CONV``, ``CUSTOM``, or another ``TABULAR`` branch:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       general_architecture_version="dot_product",
       compute_mode="cpu",
       metrics=["accuracy"],
       metrics_average=["macro"],
       instance_branch_architecture="TABULAR",
       instance_branch_tabular_schema=tabular_schema,
       instance_branch_nodes_per_layer=[16],
       target_branch_architecture="EMBEDDING",
       target_branch_input_dim=number_of_targets,
       embedding_size=8,
   )

For a dot-product model, both branch encoders output ``embedding_size``
features. For MLP and Kronecker fusion, the last branch width comes from
``instance_branch_nodes_per_layer`` or ``target_branch_nodes_per_layer``.

Preprocessing and checkpoints
-----------------------------

``DataLoaderFactory.for_training`` fits imputation and normalization only on
the training entity frame. The same state is reused for validation and test
data. DeepMTP records this state in
``instance_tabular_preprocessing_state`` or
``target_tabular_preprocessing_state`` and includes it in saved configurations
and checkpoints.

Restored models therefore apply the original training statistics during
prediction. Calling prediction on a new, untrained ``TABULAR`` model is
rejected because fitting preprocessing on prediction data would leak
information.

Numeric options are:

``numeric_normalization``
   ``"standard"`` for zero mean and unit population standard deviation,
   ``"minmax"`` for training-range scaling, or ``"none"``.

``numeric_missing``
   ``"mean"`` for training-mean imputation, ``"zero"`` for raw zero
   imputation, or ``"error"``.

Constant numeric columns use a scale of one, avoiding division by zero.

Precomputed embeddings
----------------------

Vectors produced by another model do not need a separate branch type. Pass
them as the existing ``features`` vectors and use an ``MLP`` branch. This
keeps precomputed embeddings on the efficient dense path while ``TABULAR`` is
reserved for schema-shaped numeric and categorical columns.
