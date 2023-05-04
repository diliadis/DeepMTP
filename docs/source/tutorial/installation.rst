Installation
============================

Dependencies
------------

**DeepMTP** works with **Python 3.7 or later**.

Installation
------------

**DeepMTP** is
`available on PyPI <https://pypi.org/project/DeepMTP/>`_
and can be installed using **pip**::

  # create and activate a conda environment
  conda create -n DeepMTP_env python=3.8
  conda activate DeepMTP_env

  # if a gpu is available
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

  # if a gpu is NOT available
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

  # install DeepMTP
  pip install DeepMTP

Alternatively, you can install directly from the **source code**. Simply clone the **Git**
repository of the project and run the following commands::

  git clone https://github.com/diliadis/DeepMTP.git
  cd DeepMTP
  conda env create -f environment.yml
  conda activate DeepMTP_env


.. Development
.. -----------


.. Testing
.. -------


.. Generating the documentation
.. ----------------------------