DeepMTP.data package
====================

The data namespace owns built-in dataset helpers, interaction normalization,
feature validation, splitting, preparation, model-ready loading, and structured
progress events:

.. code-block:: python

   from DeepMTP.data import (
       DataLoaderFactory,
       DataProgressObserver,
       TrainingDataLoaders,
       data_process,
   )

Batches
-------

Model-ready dataloaders return :class:`DeepMTP.data.batches.MTPBatch`. It is a
mapping compatible with the historical tensor keys and also exposes typed
dense, ID, image, tabular, or custom branch inputs. Tabular inputs keep numeric
and categorical tensors separate. Calling ``batch.to(device)`` moves every
component together.

.. automodule:: DeepMTP.data.batches
   :members:
   :undoc-members:
   :show-inheritance:

Tabular preprocessing
---------------------

Tabular schemas declare numeric columns, category vocabularies, missing-value
policies, and normalization. Fitted training statistics are serializable and
are reused for validation, testing, and restored-model prediction.

.. automodule:: DeepMTP.data.tabular
   :members:
   :undoc-members:
   :show-inheritance:

Dataset helpers
---------------

Dataset-specific dependencies remain lazy until this module or one of its
package-level exports is used.

.. automodule:: DeepMTP.data.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Interactions
------------

.. automodule:: DeepMTP.data.interactions
   :members:
   :undoc-members:
   :show-inheritance:

Features
--------

.. automodule:: DeepMTP.data.features
   :members:
   :undoc-members:
   :show-inheritance:

Splitting
---------

.. automodule:: DeepMTP.data.splitting
   :members:
   :undoc-members:
   :show-inheritance:

Preparation
-----------

.. automodule:: DeepMTP.data.preparation
   :members:
   :undoc-members:
   :show-inheritance:

Loading
-------

.. automodule:: DeepMTP.data.loading
   :members:
   :undoc-members:
   :show-inheritance:

Progress
--------

.. automodule:: DeepMTP.data.progress
   :members:
   :undoc-members:
   :show-inheritance:

Package contents
----------------

.. automodule:: DeepMTP.data
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:
