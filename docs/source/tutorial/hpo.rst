Hyperparameter Optimization
###########################

To further automate DeepMTP we decided to benchmark different popular hyperparameter optimization (HPO) methods (The resulting paper will be pubished in the near future). Based on the results, we concluded that Hyperband is a viable option for the majority of the MTP problem settings DeepMTP considers.

Hyperband
*********

One of the core steps in any standard HPO method is the performance evaluation of a given configuration. This can be manageable for simple models that are relatively cheap to train and test, but can be a significant bottleneck for more complex models that need hours or even days to train. This is particularly evident in deep learning, as big neural networks with millions of parameters trained on increasingly larger datasets can deem traditional black-box HPO methods impractical.

Addressing this issue, multi-fidelity HPO methods have been devised to discard unpromising hyperparameter configurations already at an early stage. To this end, the evaluation procedure is adapted to support cheaper evaluations of hyperparameter configurations, such as evaluating on sub-samples (feature-wise or instance-wise) of the provided data set or executing the training procedure only for a certain number of epochs in the case of iterative learners. The more promising candidates are subsequently evaluated on increasing budgets until a maximum assignable budget is reached.

A popular representative of such methods is Hyperband. Hyperband builds upon Successive Halving (SH), where a set of n candidates is first evaluated on a small budget. 

Combining Hyperband with DeepMTP
********************************

DeepMTP offers a basic Hyperband implementation natively, so the code needs only modification::

    from DeepMTP.dataset import load_process_MLC
    from DeepMTP.main import DeepMTP
    from DeepMTP.utils.utils import generate_config
    from DeepMTP.simple_hyperband import BaseWorker
    from DeepMTP.simple_hyperband import HyperBand
    import ConfigSpace.hyperparameters as CSH


    # define the configuration space
    cs= CS.ConfigurationSpace()
    # REALLY IMPORTANT: all hyperparameters for the instance or target branch should have the 'instance_' or 'target_' prefix
    lr= CSH.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-3, default_value="1e-3", log=True)
    embedding_size= CSH.UniformIntegerHyperparameter('embedding_size', lower=8, upper=2048, default_value=64, log=False)
    instance_branch_layers= CSH.UniformIntegerHyperparameter('instance_branch_layers', lower=1, upper=2, default_value=1, log=False)
    instance_branch_nodes_per_layer= CSH.UniformIntegerHyperparameter('instance_branch_nodes_per_layer', lower=8, upper=2048, default_value=64, log=False)
    target_branch_layers = CSH.UniformIntegerHyperparameter('target_branch_layers', lower=1, upper=2, default_value=1, log=False)
    target_branch_nodes_per_layer = CSH.UniformIntegerHyperparameter('target_branch_nodes_per_layer', lower=8, upper=2048, default_value=64, log=False)
    dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.4, log=False)
    batch_norm = CSH.CategoricalHyperparameter('batch_norm', [True, False])
    cs.add_hyperparameters(
        [
            lr,
            embedding_size,
            instance_branch_layers,
            instance_branch_nodes_per_layer,
            target_branch_layers,
            target_branch_nodes_per_layer,
            dropout_rate,
            batch_norm,
        ]
    )

    #adding condition on the hyperparameter values
    cond = CS.GreaterThanCondition(dropout_rate, instance_branch_layers, 1)
    cond2 = CS.GreaterThanCondition(batch_norm, instance_branch_layers, 1)
    cond3 = CS.GreaterThanCondition(dropout_rate, target_branch_layers, 1)
    cond4 = CS.GreaterThanCondition(batch_norm, target_branch_layers, 1)
    cs.add_condition(CS.OrConjunction(cond, cond3))
    cs.add_condition(CS.OrConjunction(cond2, cond4))

    # load dataset
    data = load_process_MLC(dataset_name='yeast', variant='undivided', features_type='numpy')
    # process and split
    train, val, test, data_info = data_process(data, validation_setting='B', verbose=True)

    # initialize the minimal configuration
    config = {    
        'hpo_results_path': './hyperband/',
        'instance_branch_input_dim': data_info['instance_branch_input_dim'],
        'target_branch_input_dim': data_info['target_branch_input_dim'],
        'validation_setting': data_info['detected_validation_setting'],
        'general_architecture_version': 'dot_product',
        'problem_mode': data_info['detected_problem_mode'],
        'compute_mode': 'cuda:1',
        'train_batchsize': 512,
        'val_batchsize': 512,
        'num_epochs': 6,
        'num_workers': 8,
        'metrics': ['hamming_loss', 'auroc'],
        'metrics_average': ['macro'],
        'patience': 10,
        'evaluate_train': True,
        'evaluate_val': True,
        'verbose': True,
        'results_verbose': False,
        'use_early_stopping': True,
        'use_tensorboard_logger': True,
        'wandb_project_name': 'DeepMTP_v2',
        'wandb_project_entity': 'diliadis',
        'metric_to_optimize_early_stopping': 'loss',
        'metric_to_optimize_best_epoch_selection': 'loss',
        'instance_branch_architecture': 'MLP',
        'target_branch_architecture': 'MLP',
        'save_model': True,
        'eval_every_n_epochs': 10,
        'additional_info': {'eta': 3, 'max_budget': 9}
        }

    # initialize the BaseWorker that will be used by Hyperband's optimizer
    worker = BaseWorker(train, val, test, data_info, config, 'loss')
    # initialize the optimizers
    hb = HyperBand(
        base_worker=worker,
        configspace=cs,
        eta=config['additional_info']['eta'],
        max_budget=config['additional_info']['max_budget'],
        direction='min',
        verbose=True
    )
    # start-up the optimizer
    best_overall_config = hb.run_optimizer()

    # load the best model and generate predictions on the test set
    best_model = DeepMTP(best_overall_config.info['config'], best_overall_config.info['model_dir'])
    best_model_results = best_model.predict(test, verbose=True)