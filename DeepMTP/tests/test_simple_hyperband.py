from DeepMTP.simple_hyperband import HyperBand
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def test_HyperBand():
    # define the configuration space
    cs= CS.ConfigurationSpace()
    
    lr= CSH.UniformFloatHyperparameter(
        "learning_rate", lower=1e-6, upper=1e-3, default_value="1e-3", log=True
    )
    cs.add_hyperparameters([lr])

    embedding_size= CSH.UniformIntegerHyperparameter(
        "embedding_size", lower=8, upper=2048, default_value=64, log=False
    )

    instance_branch_layers= CSH.UniformIntegerHyperparameter(
        "instance_branch_layers", lower=1, upper=2, default_value=1, log=False
    )

    instance_branch_nodes_per_layer= CSH.UniformIntegerHyperparameter(
        "instance_branch_nodes_per_layer", lower=8, upper=2048, default_value=64, log=False
    )

    target_branch_layers = CSH.UniformIntegerHyperparameter(
        "target_branch_layers", lower=1, upper=2, default_value=1, log=False
    )

    target_branch_nodes_per_layer = CSH.UniformIntegerHyperparameter(
        "target_branch_nodes_per_layer", lower=8, upper=2048, default_value=64, log=False
    )

    dropout_rate = CSH.UniformFloatHyperparameter(
        "dropout_rate", lower=0.0, upper=0.9, default_value=0.4, log=False
    )

    batch_norm = CSH.CategoricalHyperparameter("batch_norm", [True, False])

    cs.add_hyperparameters(
        [
            embedding_size,
            instance_branch_layers,
            instance_branch_nodes_per_layer,
            target_branch_layers,
            target_branch_nodes_per_layer,
            dropout_rate,
            batch_norm,
        ]
    )
    
    hb = HyperBand(
        base_worker=None,
        configspace=cs,
        eta=3,
        max_budget=81,
        direction="min",
        verbose=True
    )
    
    assert hb.eta == 3
    assert hb.max_budget == 81
    
    precalc_hb = {81: {'n_i': [81, 27, 9, 3, 1], 'r_i': [1, 3, 9, 27, 81], 'num_iters': 5},
        27: {'n_i': [27, 9, 3, 1], 'r_i': [3, 9, 27, 81], 'num_iters': 4},
        9: {'n_i': [9, 3, 1], 'r_i': [9, 27, 81], 'num_iters': 3},
        6: {'n_i': [6, 2], 'r_i': [27, 81], 'num_iters': 2},
        5: {'n_i': [5], 'r_i': [81], 'num_iters': 1}}
    
    assert hb.budgets_per_bracket == precalc_hb