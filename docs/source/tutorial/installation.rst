Installation
============

Requirements
------------

DeepMTP supports Python 3.10 and newer. CPU execution is fully supported. A
CUDA-capable GPU is optional and may significantly reduce training time for
larger experiments.

Install from PyPI
-----------------

Create an isolated environment and install the core package::

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install DeepMTP

On Windows, activate the environment with
``.venv\Scripts\activate``.

To use a GPU, install the appropriate PyTorch distribution from the
`official PyTorch installation guide
<https://pytorch.org/get-started/locally/>`_ first.

Optional integrations
---------------------

Install only the integrations required by an experiment::

  python -m pip install "DeepMTP[datasets]"
  python -m pip install "DeepMTP[hpo]"
  python -m pip install "DeepMTP[graph]"
  python -m pip install "DeepMTP[image]"
  python -m pip install "DeepMTP[sparse]"
  python -m pip install "DeepMTP[streamlit]"
  python -m pip install "DeepMTP[tracking]"

The ``all`` extra installs every runtime integration::

  python -m pip install "DeepMTP[all]"

Install from source
-------------------

For development, clone the repository and install the editable development
environment::

  git clone https://github.com/diliadis/DeepMTP.git
  cd DeepMTP
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e . --group dev

Run the offline suite and quality checks::

  python -m pytest
  ruff check .
  ruff format --check .
  mypy

Conda users may bootstrap an equivalent environment from the repository root::

  conda env create -f environment.yml
  conda activate deepmtp
  python -m pip install --group dev
