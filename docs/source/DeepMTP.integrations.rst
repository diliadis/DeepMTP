DeepMTP.integrations package
============================

Optional integrations remain separate from the core training and HPO
implementations. Importing ``DeepMTP.integrations`` does not load Streamlit.

Streamlit
---------

Install the optional dependency before constructing a Streamlit adapter:

.. code-block:: console

   python -m pip install "DeepMTP[streamlit]"

The canonical Streamlit API is available from one namespace:

.. code-block:: python

   from DeepMTP.integrations.streamlit import (
       DeepMTP as StreamlitDeepMTP,
       HyperBand,
       RandomSearch,
       StreamlitHPOProgressObserver,
       StreamlitProgressObserver,
   )

Trainer
~~~~~~~

.. automodule:: DeepMTP.integrations.streamlit.trainer
   :members:
   :undoc-members:
   :show-inheritance:

HPO adapters
~~~~~~~~~~~~

.. automodule:: DeepMTP.integrations.streamlit.hpo
   :members:
   :undoc-members:
   :show-inheritance:

Progress renderers
~~~~~~~~~~~~~~~~~~

.. automodule:: DeepMTP.integrations.streamlit.progress
   :members:
   :undoc-members:
   :show-inheritance:

Package contents
~~~~~~~~~~~~~~~~

.. automodule:: DeepMTP.integrations.streamlit
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:
