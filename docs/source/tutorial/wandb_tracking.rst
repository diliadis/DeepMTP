Weights & Biases tracking
##########################

Install the optional tracking dependencies before enabling W&B:

.. code-block:: console

   python -m pip install "DeepMTP[tracking]"

DeepMTP uses the current W&B Run and Artifact APIs. It declares ``epoch`` as
the step metric, assigns min/max summaries to known training and validation
metrics, records best/test results and parameter counts in the run summary, and
finishes failed runs with a non-zero exit status.

Run organization
================

Set both the project and entity to enable the integration:

.. code-block:: python

   from DeepMTP import DeepMTP, data_process
   from DeepMTP.utils import generate_config

   config = generate_config(
       # Data-derived dimensions and model options are omitted here.
       wandb_project_entity="my-team",
       wandb_project_name="deepmtp",
       wandb_run_name="fingerprint-baseline",
       wandb_group="ablations",
       wandb_job_type="train",
       wandb_tags=["sparse", "baseline"],
       wandb_notes="Training-only preprocessing comparison",
       wandb_mode="online",
   )

``wandb_mode`` accepts ``online``, ``offline``, or ``disabled``. Leaving it as
``None`` lets W&B environment variables and SDK defaults choose the mode.
Related experiments can use a common ``wandb_group`` while ``wandb_job_type``,
tags, and notes make their role searchable in the W&B workspace.

Model monitoring is controlled separately:

.. code-block:: python

   config["wandb_watch"] = "gradients"  # "parameters", "all", or None
   config["wandb_watch_log_freq"] = 1000
   config["wandb_log_graph"] = False

Source upload is deliberately opt-in with ``wandb_log_code=True``. DeepMTP
logs supported source and project-text files while excluding hidden paths.

Artifacts and lineage
=====================

Enable model Artifact publication to upload the locally saved checkpoint,
configuration, and summary as one versioned unit:

.. code-block:: python

   config["wandb_log_model_artifact"] = True
   config["wandb_model_artifact_name"] = "deepmtp-fingerprint-model"
   config["wandb_model_artifact_aliases"] = ["latest", "best"]

The checkpoint's reproducibility manifest is also attached as Artifact
metadata. Artifact logging happens whenever the trained model checkpoint is
saved; it therefore also works after an explicit ``model.save_model()`` call.

Declare input Artifact references to record dataset or feature lineage:

.. code-block:: python

   config["wandb_input_artifacts"] = [
       "my-team/deepmtp/fingerprint-dataset:v2",
   ]

Input references require online mode because W&B must resolve them. Model
Artifacts can be recorded in offline runs and synchronized later.

To link the logged model version into an existing W&B Registry collection,
configure both names:

.. code-block:: python

   config["wandb_registry_name"] = "Models"
   config["wandb_registry_collection"] = "DeepMTP"

Registry linking uses ``Run.link_artifact`` and the current
``wandb-registry-{registry}/{collection}`` target path. It requires online mode
and ``wandb_log_model_artifact=True``. Create the Registry and a compatible
collection in W&B before the run.

Prediction Tables and charts
============================

Test predictions can be logged as an immutable W&B Table:

.. code-block:: python

   config["wandb_log_predictions"] = True
   config["wandb_prediction_table_max_rows"] = 1000

The row limit is enforced with deterministic sampling based on
``random_seed``. Classification runs also log a confusion matrix, ROC curve,
and precision-recall curve. Prediction upload is off by default because rows
can contain sensitive entity identifiers and consume workspace storage.

Restoring a model Artifact
==========================

Use a W&B Artifact reference or alias without a separate download step:

.. code-block:: python

   restored = DeepMTP.from_wandb_artifact(
       "my-team/deepmtp/deepmtp-fingerprint-model:best",
       {
           "results_path": "./results",
           "experiment_name": "restored",
       },
       download_dir="./artifact-cache",
   )

The helper uses ``wandb.Api().artifact``, requires exactly one ``model.pt`` in
the downloaded Artifact, and then follows the normal trusted-checkpoint loading
path. Artifact name, version, digest, download root, and original reference are
available on ``restored.wandb_artifact``.

Privacy-conscious defaults
==========================

Metric and configuration tracking begins only when both project fields are
set. Source, model Artifact, and prediction uploads are each disabled by
default. Use ``wandb_mode="offline"`` for network-free local capture or
``wandb_mode="disabled"`` to preserve a shared configuration while suppressing
the W&B run.
