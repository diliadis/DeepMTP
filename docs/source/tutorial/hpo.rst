Hyperparameter Optimization
###########################

DeepMTP includes random-search and Hyperband optimizers for automating model
selection. Hyperband is a practical option for many of the MTP problem settings
supported by the project.

Hyperband
*********

One of the core steps in any standard HPO method is the performance evaluation of a given configuration. This can be manageable for simple models that are relatively cheap to train and test, but can be a significant bottleneck for more complex models that need hours or even days to train. This is particularly evident in deep learning, as big neural networks with millions of parameters trained on increasingly larger datasets can deem traditional black-box HPO methods impractical.

Addressing this issue, multi-fidelity HPO methods have been devised to discard unpromising hyperparameter configurations already at an early stage. To this end, the evaluation procedure is adapted to support cheaper evaluations of hyperparameter configurations, such as evaluating on sub-samples (feature-wise or instance-wise) of the provided data set or executing the training procedure only for a certain number of epochs in the case of iterative learners. The more promising candidates are subsequently evaluated on increasing budgets until a maximum assignable budget is reached.

A popular representative of such methods is Hyperband. Hyperband builds upon Successive Halving (SH), where a set of n candidates is first evaluated on a small budget. 

Combining Hyperband with DeepMTP
********************************

Install the HPO dependency with ``pip install "DeepMTP[hpo]"``. The example
below uses synthetic data and one epoch so it can also serve as a quick API
check; increase ``max_budget`` and expand the configuration space for real
experiments. Sampled branch-specific parameters must retain their ``instance_``
or ``target_`` prefixes so ``BaseWorker`` can route them to the correct branch.

.. hpo-example-start

.. code-block:: python

    import ConfigSpace as CS
    import numpy as np

    from DeepMTP import DeepMTP, data_process
    from DeepMTP.hpo import BaseWorker, HyperBand

    rng = np.random.default_rng(42)
    scores = (
        np.arange(16)[:, np.newaxis] + np.arange(3)[np.newaxis, :]
    ) % 2
    data = {
        "train": {
            "y": scores,
            "X_instance": rng.normal(size=(16, 3)),
            "X_target": None,
        }
    }
    train, validation, test, data_info = data_process(
        data,
        validation_setting="B",
    )

    config_space = CS.ConfigurationSpace(seed=42)
    config_space.add(
        [
            CS.Float(
                "learning_rate",
                (1e-4, 1e-2),
                default=1e-3,
                log=True,
            ),
            CS.Integer("embedding_size", (2, 4), default=3),
        ]
    )

    base_config = {
        "hpo_results_path": "hpo_results",
        "validation_setting": data_info["detected_validation_setting"],
        "problem_mode": data_info["detected_problem_mode"],
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "num_workers": 0,
        "train_batchsize": 16,
        "val_batchsize": 16,
        "metrics": [],
        "metrics_average": ["macro"],
        "evaluate_train": False,
        "evaluate_val": False,
        "use_early_stopping": False,
        "save_model": True,
        "verbose": False,
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": data_info["instance_branch_input_dim"],
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": data_info["target_branch_input_dim"],
    }

    worker = BaseWorker(
        train,
        validation,
        test,
        data_info,
        base_config,
        metric_to_optimize="loss",
    )
    optimizer = HyperBand(
        base_worker=worker,
        configspace=config_space,
        eta=2,
        max_budget=1,
        direction="min",
    )
    best_experiment = optimizer.run_optimizer()

    best_model = DeepMTP(
        best_experiment.info["config"],
        checkpoint_dir=best_experiment.info["model_dir"],
    )
    best_model_results = best_model.predict(test, verbose=True)

.. hpo-example-end
