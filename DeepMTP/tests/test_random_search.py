from DeepMTP.random_search import RandomSearch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def test_RandomSearch():
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
    
    hb = RandomSearch(
        base_worker=None,
        configspace=cs,
        budget=10,
        max_num_epochs=200, 
        direction="min",
        verbose=True
    )
    
    assert hb.budget == 10
    assert hb.max_num_epochs == 200
