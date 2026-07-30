DeepMTP.runtime package
=======================

The runtime namespace owns numerical training and evaluation, experiment
persistence, lifecycle presentation, reporting integrations, and structured
evaluation progress:

.. code-block:: python

   from DeepMTP.runtime import (
       Evaluator,
       ExperimentStore,
       LOSS_REGISTRY,
       OUTPUT_HEAD_REGISTRY,
       ProgressObserver,
       TensorBoardReporter,
       TrainingEngine,
   )

Task components
---------------

``LOSS_REGISTRY`` and ``OUTPUT_HEAD_REGISTRY`` expose the components selected
for classification and regression. Binary classification defaults to raw
model scores and ``BCEWithLogitsLoss``; sigmoid is applied only to values
collected for metrics and prediction. The explicit
``binary_cross_entropy`` compatibility loss applies sigmoid before
``BCELoss``.

Regression supports ``mean_squared_error``, ``mean_absolute_error``, and
``huber``. Every regression loss uses raw model output and an identity output
head. ``build_task_components`` rejects task/loss combinations that do not
share a defined contract, and every component builder returns a fresh PyTorch
module for each trainer.

.. automodule:: DeepMTP.runtime.components
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
----------

.. automodule:: DeepMTP.runtime.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Training
--------

.. automodule:: DeepMTP.runtime.training
   :members:
   :undoc-members:
   :show-inheritance:

Persistence
-----------

.. automodule:: DeepMTP.runtime.persistence
   :members:
   :undoc-members:
   :show-inheritance:

Checkpoint metadata
-------------------

The versioned manifest records preprocessing, training, environment, source,
and tracking provenance while preserving legacy checkpoints without a
``metadata`` entry.

.. automodule:: DeepMTP.runtime.metadata
   :members:
   :undoc-members:
   :show-inheritance:

Presentation
------------

.. automodule:: DeepMTP.runtime.presentation
   :members:
   :undoc-members:
   :show-inheritance:

Reporting
---------

.. automodule:: DeepMTP.runtime.reporting
   :members:
   :undoc-members:
   :show-inheritance:

W&B Artifacts
-------------

.. automodule:: DeepMTP.runtime.wandb
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation progress
-------------------

.. automodule:: DeepMTP.runtime.evaluation_progress
   :members:
   :undoc-members:
   :show-inheritance:

Package contents
----------------

.. automodule:: DeepMTP.runtime
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:
