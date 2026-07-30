Reproducible checkpoints
########################

DeepMTP checkpoints keep the historical model, optimizer, and configuration
entries and may now include a versioned ``metadata`` manifest:

.. code-block:: text

   model_state_dict
   optimizer_state_dict
   config
   metadata                  # optional in legacy checkpoints
     format_version
     created_at
     preprocessing
     training
     environment
     tracking

The manifest records the model seed, validation setting, selected epoch,
validation and test results, parameter counts, effective device, default
floating-point dtype, Python and relevant package versions, and the current Git
commit and dirty-state when the source checkout is available. Tracking
identifiers such as the W&B run ID are included when a capable reporter is
active.

Dense preprocessing replay
==========================

When ``data_process`` performs legacy ``Standard`` or ``MinMax`` dense-feature
scaling, it fits only on training entities. The fitted offsets and scales are
serialized explicitly rather than pickling a scikit-learn object:

.. code-block:: python

   from DeepMTP import data_process

   train, validation, test, data_info = data_process(
       raw_data,
       validation_setting="B",
       scale_instance_features="Standard",
       shuffle=False,
   )

   preparation_state = data_info["data_preparation_state"]

The same state is attached to all three processed splits. Passing those splits
to ``DeepMTP.train`` captures it in the runtime configuration and checkpoint.
DeepMTP rejects a mixture of splits carrying different states.

A restored model exposes a copy through ``preprocessing_state`` and can apply
one saved branch scaler directly:

.. code-block:: python

   from DeepMTP import DeepMTP

   restored = DeepMTP(
       {
           "results_path": "./results",
           "experiment_name": "restored",
       },
       checkpoint_dir="./results/original/model.pt",
   )

   scaled_instances = restored.transform_dense_features(
       raw_instance_features,
       branch="instance",
   )

To prepare another compatible dataset with exactly the saved statistics, pass
the full state back to ``data_process``. Saved scaler methods take effect even
when the corresponding ``scale_*_features`` argument is omitted:

.. code-block:: python

   train, validation, test, data_info = data_process(
       new_raw_data,
       validation_setting="B",
       dense_preprocessing_state=restored.preprocessing_state,
   )

The state validates its format version, feature width, finite numeric values,
split ratios, seed, and scaling method before transforming data.

Tabular preprocessing
=====================

Explicit ``TABULAR`` branches continue to store their schemas, category
vocabularies, imputation values, and numeric statistics in
``instance_tabular_preprocessing_state`` and
``target_tabular_preprocessing_state``. The checkpoint manifest references
those normalized configuration values alongside the legacy dense state, so the
two preprocessing paths remain distinct.

Loading and compatibility
=========================

Checkpoints written before the manifest was introduced remain loadable. Their
three required keys are unchanged, ``checkpoint_metadata`` is ``None``, and
configuration overrides supplied to the constructor retain their historical
precedence. A direct call to ``ExperimentStore.save_checkpoint`` without
``metadata`` also writes the exact legacy three-key payload.

New metadata formats are rejected when their version is unknown instead of
being interpreted incorrectly. The manifest improves provenance and inference
replay; it is not an exact interrupted-training resume mechanism. It does not
capture epoch-local random-number-generator states or dataloader cursor state.

PyTorch checkpoint files can contain Python objects. Load only checkpoints and
W&B Artifacts from sources you trust.
