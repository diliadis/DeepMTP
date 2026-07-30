Demo notebooks
##############

The notebooks use small local datasets and are executed by the test suite.
Each Colab link opens the version-controlled notebook rather than a separate
copy. The first code cell detects a fresh Colab runtime and installs the
matching DeepMTP source from the repository's ``main`` branch. Hyperband and
random search install the ``hpo`` extra as part of that bootstrap. Local
execution skips the bootstrap and uses the active environment.

The live endpoints are checked weekly. Maintainers can run the same check
manually from the repository root:

.. code-block:: console

   python scripts/check_colab_links.py

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Name
     - Link
   * - **Local dataset preparation**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/load_datasets.ipynb>`__
   * - **Multi-label classification (MLC)** 
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MLC.ipynb>`__
   * - **Multivariate regression (MTR)**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MTR.ipynb>`__
   * - **Multi-task learning (MTL)**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MTL.ipynb>`__
   * - **Matrix Completion (MC)**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_MC.ipynb>`__
   * - **Dyadic Prediction (DP)**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/main_example_DP.ipynb>`__
   * - **Hyperband**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/hyperband_example.ipynb>`__
   * - **Random search**
     - `Open in Colab <https://colab.research.google.com/github/diliadis/DeepMTP/blob/main/DEMO/random_search_example.ipynb>`__
