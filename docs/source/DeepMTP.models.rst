DeepMTP.models package
======================

The canonical model API groups reusable branch networks, combination
architectures, and construction:

.. code-block:: python

   from DeepMTP.models import (
       BranchEncoder,
       ConvNet,
       MLP,
       ModelFactory,
       TabularEncoder,
   )

Encoder contracts
-----------------

All built-in branch encoders declare an input kind and output dimension. Their
shared runtime contract requires a floating output shaped
``[batch_size, output_dim]``.

Most encoders consume one tensor. ``TabularEncoder`` consumes a structured
``TabularInput`` containing a floating numeric tensor and an integer
categorical tensor while producing the same standardized encoder output.

Custom branch factories should declare ``input_kind`` and ``output_dim`` on
their returned module. For compatibility, the model factory can infer a
missing non-dot-product ``output_dim`` from a final ``Linear`` layer, but this
structural inference is deprecated.

.. automodule:: DeepMTP.models.contracts
   :members:
   :undoc-members:
   :show-inheritance:

Branch models
-------------

.. automodule:: DeepMTP.models.branches
   :members:
   :undoc-members:
   :show-inheritance:

Combination models and construction
-----------------------------------

.. automodule:: DeepMTP.models.factory
   :members:
   :undoc-members:
   :show-inheritance:

Package contents
----------------

.. automodule:: DeepMTP.models
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:
