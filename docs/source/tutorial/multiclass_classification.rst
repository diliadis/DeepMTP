Multiclass classification
=========================

Multiclass mode predicts exactly one class for every observed
instance-target interaction. It is intended for mutually exclusive outcomes
such as ``inactive``, ``weak``, and ``strong``. It is different from
multi-label classification, where several independent binary targets may be
positive at the same time.

Label contract
--------------

Encode the classes as consecutive integer IDs ``0`` through ``C - 1``, where
``C`` is at least three. Do not one-hot encode the labels. Every declared
class must occur in the training interactions, while validation and test
splits may contain any subset of the declared classes.

Multiclass interpretation is explicit:

.. code-block:: python

   import numpy as np

   from DeepMTP import data_process

   scores = (
       np.arange(18)[:, np.newaxis] + np.arange(3)[np.newaxis, :]
   ) % 3
   train, validation, test, data_info = data_process(
       {"train": {"y": scores}},
       validation_setting="A",
       classification_mode="multiclass",
   )

``data_process`` infers ``detected_num_classes`` when ``num_classes`` is
omitted. You can instead declare it explicitly. The complete supplied data
must then use every ID in the declared range:

.. code-block:: python

   train, validation, test, data_info = data_process(
       data,
       validation_setting="B",
       classification_mode="multiclass",
       num_classes=4,
   )

The opt-in is deliberate. A regression target containing the integers 0, 1,
and 2 is numerically indistinguishable from three class IDs. Without
``classification_mode="multiclass"``, DeepMTP preserves the historical
interpretation of non-binary numeric scores as regression.

Model configuration
-------------------

Pass the detected contract to ``DeepMTPConfig``:

.. code-block:: python

   from DeepMTP import DeepMTP, DeepMTPConfig

   config = DeepMTPConfig(
       validation_setting=data_info["detected_validation_setting"],
       problem_mode="classification",
       classification_mode=data_info["detected_classification_mode"],
       num_classes=data_info["detected_num_classes"],
       metrics=["accuracy", "precision", "recall", "f1_score", "auroc", "aupr"],
       metrics_average=["micro"],
       multiclass_average="macro",
       compute_mode="cpu",
       instance_branch_architecture="EMBEDDING",
       instance_branch_input_dim=data_info["instance_branch_input_dim"],
       target_branch_architecture="EMBEDDING",
       target_branch_input_dim=data_info["target_branch_input_dim"],
       embedding_size=16,
   )

   model = DeepMTP(config)
   validation_results = model.train(train, validation, test)
   test_results, predictions = model.predict(
       test,
       return_predictions=True,
   )

Multiclass mode defaults to ``cross_entropy`` loss and
``multiclass_average="macro"``. All three fusion architectures produce
``num_classes`` raw logits:

* Dot-product fusion projects the elementwise branch interaction to the class
  logits.
* MLP fusion uses ``num_classes`` units in its final linear layer.
* Kronecker fusion uses ``num_classes`` outputs in its final projection.

``CrossEntropyLoss`` consumes the raw logits during training. Softmax is
applied only for evaluation and returned predictions. This keeps the loss
numerically stable and ensures that each probability row sums to one.

Metric averaging
----------------

DeepMTP has two separate averaging concepts:

``metrics_average``
   Controls the multi-target grouping. ``micro`` evaluates all interactions
   together, ``macro`` evaluates each target and averages the target results,
   and ``instance`` evaluates each instance and averages the instance
   results. Validation setting A supports only ``micro``.

``multiclass_average``
   Controls how classes are combined inside precision, recall, F1, one-vs-rest
   AUROC, and one-vs-rest AUPR. Choose ``micro``, ``macro``, or ``weighted``.
   Accuracy and hamming loss do not use class averaging.

For example, ``metrics_average=["macro"]`` with
``multiclass_average="weighted"`` first computes a support-weighted
multiclass metric for each target and then averages those target-level
results.

Prediction output
-----------------

The prediction frame retains the common identifier and value columns and adds
multiclass probabilities:

.. code-block:: text

   instance_id
   target_id
   true_values
   predicted_values
   predicted_probability
   probability_class_0
   probability_class_1
   ...

``predicted_values`` contains the winning class ID.
``predicted_probability`` contains that class's probability. The
``probability_class_<id>`` columns retain the complete distribution for
calibration analysis, threshold-independent metrics, and downstream
decisions.

Compatibility and current limits
--------------------------------

Configurations and checkpoints without ``classification_mode`` continue to
use binary classification. Binary and regression model parameter names and
output shapes are unchanged.

This initial multiclass contract does not yet support string labels, one-hot
labels, ordinal losses, class-weighted losses, or multilabel rows.
``top_k`` is also rejected because its existing DeepMTP meaning ranks
instance-target interactions rather than classes. Validation settings A
through D otherwise retain their existing entity-novelty semantics.
