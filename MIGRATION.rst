Migrating from DeepMTP 0.0.22
================================

This guide covers migration from `DeepMTP 0.0.22 on PyPI
<https://pypi.org/project/DeepMTP/0.0.22/>`_, the latest published release and
the version represented by the `v0.0.22 Git tag
<https://github.com/diliadis/DeepMTP/tree/v0.0.22>`_, to the current
development line. It focuses on changes that can affect existing experiments
rather than listing every internal refactor.

Before upgrading
----------------

* Create a new environment instead of updating the environment used to produce
  published results.
* Keep the original environment lock file, configuration, checkpoint, and
  result directory with the experiment.
* Re-run a small, fixed-seed experiment and compare its split sizes, selected
  epoch, metrics, and predictions before moving a larger workload.
* Treat checkpoints as trusted files. DeepMTP checkpoints contain Python
  configuration metadata and are loaded with PyTorch's pickle-based loader.

Python and dependencies
-----------------------

DeepMTP 0.0.22 supported Python 3.7 and installed dataset downloaders,
Torchvision, Streamlit, TensorBoard, and Weights & Biases as mandatory
dependencies. The current package requires Python 3.10 or newer and keeps the
core installation smaller.

To test the current source checkout in an isolated environment:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Install only the runtime integrations used by an experiment:

.. code-block:: console

   python -m pip install -e ".[datasets]"   # downloadable benchmark datasets
   python -m pip install -e ".[hpo]"        # ConfigSpace
   python -m pip install -e ".[image]"      # Torchvision models and transforms
   python -m pip install -e ".[streamlit]"  # Streamlit adapters
   python -m pip install -e ".[tracking]"   # TensorBoard and Weights & Biases
   python -m pip install -e ".[all]"        # all runtime integrations

Imports
-------

The 0.0.22 module-level imports remain available where practical, but new code
should use the tested top-level API.

0.0.22 style:

.. code-block:: python

   from DeepMTP.main import DeepMTP
   from DeepMTP.utils.data_utils import data_process
   from DeepMTP.utils.utils import generate_config

Current style:

.. code-block:: python

   from DeepMTP import DeepMTP, DeepMTPConfig, data_process

Dataset loader functions such as ``load_process_MLC`` remain in
``DeepMTP.dataset``. New HPO code should use the focused ``DeepMTP.hpo``
package:

.. code-block:: python

   from DeepMTP.hpo import BaseWorker, HyperBand, RandomSearch

The historical ``DeepMTP.hpo_worker``, ``DeepMTP.random_search``,
``DeepMTP.simple_hyperband``, ``DeepMTP.hpo_progress``, and
``DeepMTP.hpo_types`` module paths remain compatibility facades. Existing
imports and persisted objects that refer to those paths continue to resolve.

Configuration
-------------

``DeepMTP`` still accepts a dictionary, and ``generate_config`` remains a
compatibility adapter that returns one. New code should prefer
``DeepMTPConfig`` because invalid combinations then fail at the configuration
boundary instead of during model construction or training.

For example, an old MLP configuration placed branch options in nested
dictionaries:

.. code-block:: python

   from DeepMTP.utils.utils import generate_config

   config = generate_config(
       validation_setting=data_info["detected_validation_setting"],
       problem_mode=data_info["detected_problem_mode"],
       compute_mode="cpu",
       instance_branch_architecture="MLP",
       instance_branch_input_dim=data_info["instance_branch_input_dim"],
       instance_branch_params={"instance_branch_nodes_per_layer": [8]},
       target_branch_architecture="MLP",
       target_branch_input_dim=data_info["target_branch_input_dim"],
       target_branch_params={"target_branch_nodes_per_layer": [8]},
   )

The typed equivalent uses explicit fields:

.. code-block:: python

   from DeepMTP import DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting=data_info["detected_validation_setting"],
       problem_mode=data_info["detected_problem_mode"],
       compute_mode="cpu",
       instance_branch_architecture="MLP",
       instance_branch_input_dim=data_info["instance_branch_input_dim"],
       instance_branch_nodes_per_layer=[8],
       target_branch_architecture="MLP",
       target_branch_input_dim=data_info["target_branch_input_dim"],
       target_branch_nodes_per_layer=[8],
   )

Existing dictionaries can be validated incrementally. Unknown extension keys
are retained when a mapping is converted:

.. code-block:: python

   from DeepMTP import DeepMTP, DeepMTPConfig

   validated_config = DeepMTPConfig.from_mapping(existing_config)
   model = DeepMTP(validated_config)

Review these validation changes:

* Names for validation settings, problem modes, branch architectures, metrics,
  and averaging schemes are normalized before validation.
* Split, architecture, dimensions, layer widths, device, metrics, and
  early-stopping selections are checked eagerly.
* Validation setting A accepts only ``["micro"]`` metric averaging. The
  compatibility adapter emits ``ConfigNormalizationWarning`` when it adjusts
  an averaging policy.
* A selected metric must be ``"loss"`` or a configured
  ``"<metric>_<averaging>"`` combination.
* ``RRMSE`` is rejected by trainer configuration because training does not
  retain the per-target baselines it needs. The low-level evaluation utility
  still supports it when those baselines are supplied.
* ``momentum``, ``weighted_loss``, ``use_instance_features``,
  ``use_target_features``, ``load_pretrained_model``,
  ``pretrained_model_path``, and a non-default
  ``comb_mlp_nodes_reducing_factor`` are legacy options with no runtime effect.
  Non-default use emits ``ConfigDeprecationWarning`` instead of being silently
  ignored.
* ``random_seed`` now controls isolated model initialization and training-data
  shuffling. Importing DeepMTP no longer changes NumPy or PyTorch's process-wide
  random state. Set ``random_seed=None`` for nondeterministic initialization.

Data preparation
----------------

The input and four-value return contract of ``data_process`` is retained:

.. code-block:: python

   train, validation, test, data_info = data_process(
       data,
       validation_setting="B",
       seed=42,
   )

The current pipeline deep-copies the input mapping, validates split ratios and
entity consistency more strictly, and uses local random generators. Feature
scalers are fit on the training split only and then applied to validation and
test data. These corrections can change splits or scaled values from a 0.0.22
experiment, so compare processed data before comparing model results.

Prediction and training lifecycle
---------------------------------

In 0.0.22, ``predict`` always returned a ``(metrics, predictions)`` tuple even
when ``return_predictions`` was false. It now honors the flag:

.. code-block:: python

   metrics = model.predict(test)
   metrics, predictions = model.predict(test, return_predictions=True)

A ``DeepMTP`` instance now supports one call to ``train``. This makes reporter
cleanup, early-stopping state, and saved artifacts unambiguous. Construct a new
instance for another run:

.. code-block:: python

   first_model = DeepMTP(config)
   first_results = first_model.train(train, validation, test)

   second_model = DeepMTP(config)
   second_results = second_model.train(train, validation, test)

Checkpoints
-----------

The historical checkpoint payload keys (``model_state_dict``,
``optimizer_state_dict``, and ``config``) are retained. Loading is explicitly
mapped through the CPU before the model and optimizer are moved to the selected
device.

Always pass the path by keyword:

.. code-block:: python

   restored = DeepMTP(
       {"compute_mode": "cpu"},
       checkpoint_dir=checkpoint_path,
   )
   restored_metrics = restored.predict(test)

Do not use ``DeepMTP(config, checkpoint_path)``. The second positional argument
is the custom instance-branch factory, not the checkpoint path. Configuration
values passed during restoration override the values stored in the checkpoint.

A 0.0.22 checkpoint can still require intervention if its configuration is no
longer valid or its model structure differs from the current implementation.
Validate the stored configuration first, preserve the original checkpoint, and
retrain when PyTorch reports incompatible state-dictionary keys or shapes.

Optional integrations and offline runs
--------------------------------------

Tracking, Streamlit, image models, HPO, and downloadable datasets are imported
only when used. Missing integrations now produce a focused installation error
instead of preventing ``import DeepMTP``.

Torchvision's explicit weights API may download pretrained weights on first
use. For an offline convolutional experiment, disable pretrained weights on
each convolutional branch:

.. code-block:: python

   config = DeepMTPConfig(
       validation_setting="B",
       problem_mode="classification",
       instance_branch_architecture="CONV",
       instance_branch_input_dim=3,
       instance_branch_conv_pretrained=False,
       target_branch_architecture="MLP",
       target_branch_input_dim=12,
       target_branch_nodes_per_layer=[8],
   )

Verification checklist
----------------------

Before accepting migrated results:

#. Confirm that configuration validation completes without unexpected
   deprecation or normalization warnings.
#. Compare train, validation, and test entity IDs and row counts with the
   original run.
#. Compare feature scaling and the selected validation metric.
#. Run one CPU epoch with tracking and downloads disabled.
#. Test both prediction return forms if downstream code consumes prediction
   frames.
#. Restore a copied checkpoint and compare predictions on a small fixed input.
#. Record the DeepMTP version, Python version, dependency versions, and random
   seed with the migrated experiment.
