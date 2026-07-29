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

``LOSS_REGISTRY`` and ``OUTPUT_HEAD_REGISTRY`` expose the behavior-preserving
defaults selected for classification and regression. The component builders
return fresh PyTorch modules for each trainer.

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
