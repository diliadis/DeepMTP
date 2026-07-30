from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from DeepMTP.config import ConfigNormalizationWarning, DeepMTPConfig


def _load_torchvision_transforms() -> Any:
    try:
        return importlib.import_module("torchvision.transforms")
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Image transforms require the optional 'image' dependencies. "
            "Install them with `pip install 'DeepMTP[image]'`."
        ) from exc


def get_optimization_direction(metric_name: str) -> Literal["max", "min"]:
    """Determines if the goal is to maximize or minimize based on the name of the metric

    Args:
        metric_name (sting): the name of the metric

    Returns:
        string: max if the goal is go maximize or min if the goal is to mimize
    """
    metrics_to_max = (
        "sensitivity",
        "f1_score",
        "recall",
        "positive_predictive_value",
        "R2",
        "accuracy",
        "precision",
        "auroc",
        "aupr",
    )
    metrics_to_min = (
        "false_alarm_rate_per_hour",
        "hamming_loss",
        "RRMSE",
        "RMSE",
        "MSE",
        "MAE",
        "loss",
    )
    if any(name in metric_name for name in metrics_to_max):
        return "max"
    if any(name in metric_name for name in metrics_to_min):
        return "min"
    raise ValueError(f"Unknown optimization metric: {metric_name!r}.")


class BaseExperimentInfo:  # pragma: no cover
    """A class used to keep track of all relevant info of a given experiment. This is mainly used by the HPO methods."""

    def __init__(self, config: Any, budget: int | float) -> None:
        self.config = config
        self.score = 0.0
        self.budget = budget
        self.info: dict[str, Any] = {}

    def update_score(self, score: float) -> None:
        self.score = score

    def get_config(self) -> Any:
        return self.config

    def get_budget(self) -> int | float:
        return self.budget

    def __lt__(self, other: BaseExperimentInfo) -> bool:
        return self.score < other.score

    def __le__(self, other: BaseExperimentInfo) -> bool:
        return self.score <= other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseExperimentInfo):
            return NotImplemented
        return self.score == other.score

    def __gt__(self, other: BaseExperimentInfo) -> bool:
        return self.score > other.score

    def __ge__(self, other: BaseExperimentInfo) -> bool:
        return self.score >= other.score

    def __repr__(self) -> str:
        return (
            "config: "
            + str(self.config)
            + "  |  budget: "
            + str(self.budget)
            + "  |  score: "
            + str(self.score)
            + "\n"
            + str(self.info)
            + "\n"
            + "\n"
        )


def get_default_dropout_rate() -> float:  # pragma: no cover
    """To return the default dropout rate

    Returns:
        int: The value 0
    """
    return 0.0


def get_default_batch_norm() -> bool:  # pragma: no cover
    """To return the default batch_norm value

    Returns:
        float: The value False
    """
    return False


# dafault transofmations applied for resnet inputs
def get_default_train_transform() -> Any:  # pragma: no cover
    """To return the default transformation pipeline for a resnet during training

    Returns:
       torchvision.transforms : A transformation pipeline
    """
    transforms = _load_torchvision_transforms()
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(pretrained_size, padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
        ]
    )


def get_default_inference_transform() -> Any:  # pragma: no cover
    """To return the default transformation pipeline for a resnet during inference

    Returns:
       torchvision.transforms : A transformation pipeline
    """
    transforms = _load_torchvision_transforms()
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
        ]
    )


def generate_config(
    validation_setting: str | None = None,
    general_architecture_version: str = "dot_product",
    problem_mode: str | None = None,
    learning_rate: float = 0.001,
    decay: float = 0,
    batch_norm: bool | str = False,
    dropout_rate: float = 0,
    dropout_rate_instance_branch: float | None = None,
    dropout_rate_target_branch: float | None = None,
    momentum: float = 0.9,
    weighted_loss: bool = False,
    compute_mode: str = "cuda:0",
    num_workers: int = 1,
    train_batchsize: int = 512,
    val_batchsize: int = 512,
    num_epochs: int = 100,
    random_seed: int | None = 2,
    metrics: Sequence[str] | None = None,
    metrics_average: Sequence[str] | None = None,
    top_k: int | None = None,
    patience: int = 10,
    delta: float = 0,
    evaluate_train: bool = False,
    evaluate_val: bool = False,
    verbose: bool = False,
    results_verbose: bool = False,
    eval_instance_verbose: bool = False,
    eval_target_verbose: bool = False,
    return_results_per_target: bool = False,
    use_early_stopping: bool = True,
    use_tensorboard_logger: bool = False,
    wandb_project_name: str | None = None,
    wandb_project_entity: str | None = None,
    results_path: str = "./results/",
    experiment_name: str | None = None,
    save_model: bool = True,
    metric_to_optimize_early_stopping: str = "loss",
    metric_to_optimize_best_epoch_selection: str = "loss",
    instance_branch_architecture: str | None = None,
    use_instance_features: bool = False,
    instance_branch_input_dim: int | None = None,
    instance_branch_tabular_schema: Mapping[str, Any] | None = None,
    instance_train_transforms: Any = None,
    instance_inference_transforms: Any = None,
    target_branch_architecture: str | None = None,
    use_target_features: bool = False,
    target_branch_input_dim: int | None = None,
    target_branch_tabular_schema: Mapping[str, Any] | None = None,
    target_train_transforms: Any = None,
    target_inference_transforms: Any = None,
    comb_mlp_nodes_reducing_factor: int = 2,
    comb_mlp_nodes_per_layer: list[int] | int | None = None,
    comb_mlp_layers: int | None = None,
    embedding_size: int = 100,
    eval_every_n_epochs: int = 10,
    load_pretrained_model: bool = False,
    pretrained_model_path: str = "",
    running_hpo: bool = False,
    additional_info: Mapping[str, Any] | None = None,
    instance_branch_params: Mapping[str, Any] | None = None,
    target_branch_params: Mapping[str, Any] | None = None,
    hpo_results_path: str = "./",
    loss: str | None = None,
    classification_mode: str | None = None,
    num_classes: int | None = None,
    multiclass_average: str | None = None,
) -> dict[str, Any]:
    """Creates a dictionary that is used to configure the neural network. Contains some base logic that checks if some of the parameters make sense. It has to be updated each time a new feature is added.

    Args:
        validation_setting (str, optional): The validation setting of the given problem. The possible values are A, B, C, D. Defaults to None.
        general_architecture_version (str, optional): Enables a specific version of the general neural network architecture. Available options are: "mlp" for the mlp version, "dot_product" for the dot product version, "kronecker": for the kronecker product version. Default value if "dot_product"
        problem_mode (str, optional): The type of task for the given problem. The possible values are classification or regression. Defaults to None.
        classification_mode (str, optional): Use "binary" (default for classification) or "multiclass". Multiclass mode requires ``num_classes``.
        num_classes (int, optional): Number of mutually exclusive classes. Required and at least three for multiclass classification.
        multiclass_average (str, optional): Class averaging for multiclass precision, recall, F1, AUROC, and AUPR. Available values are "micro", "macro", and "weighted". Defaults to "macro".
        loss (str, optional): The training objective. Binary classification supports "binary_cross_entropy_with_logits" (default) and explicit legacy "binary_cross_entropy". Multiclass classification uses "cross_entropy". Regression supports "mean_squared_error" (default), "mean_absolute_error", and "huber". Defaults to the stable objective for the configured task.
        learning_rate (float, optional): The learning rate that will be used during training. Defaults to 0.001.
        decay (float, optional): The weight decay (L2 penalty) used by the Adam optimizer . Defaults to 0.
        batch_norm (bool, optional): The option to use batch normalization between the fully connected layers in the two branches. Defaults to False.
        dropout_rate (float, optional): The amount of dropout used in the layers of the two branches. Defaults to 0.
        dropout_rate_instance_branch (float, optional): The amount of dropout used in the layers of the instance branch, Can be used when assymetric overfitting between branches is observed. Defaults to 0.
        dropout_rate_target_branch (float, optional): The amount of dropout used in the layers of the target branch. Can be used when assymetric overfitting between branches is observed. Defaults to 0.
        momentum (float, optional): The momentum used by the optimizer. Defaults to 0.9.
        weighted_loss (bool, optional): Enables the use of class weights in the loss. Defaults to False.
        compute_mode (str, optional): The specific device that will be used during training. The possible values can be one the available gpus or the cpu(please dont). Defaults to 'cuda:0'.
        num_workers (int, optional): The number of sub-processes to use for data loading. Larger values usually improve performance but after a point training speed will become worse. Defaults to 1.
        train_batchsize (int, optional): The number of samples that comprise a batch from the training set. Defaults to 512.
        val_batchsize (int, optional): 	The number of samples that comprise a batch from the validation and test sets. Defaults to 512.
        num_epochs (int, optional): The max number of epochs allowed for training. Defaults to 100.
        random_seed (int, optional): The seed used for isolated model initialization and training-data shuffling. Use None for nondeterministic behavior. Defaults to 2.
        metrics (list, optional): The performance metrics that will be calculated. For classification tasks the available metrics are ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'] while for regression tasks the available metrics are ['RMSE', 'MSE', 'MAE', 'R2']. Defaults to ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'].
        metrics_average (list, optional): The averaging strategy that will be used to calculate the metric. The available options are ['macro', 'micro', 'instance']. Defaults to ['macro', 'micro'].
        top_k (int, optional): The number of top predictions used to calculate grouped top-k metric versions. Requires macro or instance averaging.
        patience (int, optional): The number of epochs that the network is allowed to continue training for while observing worse overall performance. Defaults to 10.
        delta (float, optional): The delta used during early stopping
        evaluate_train (bool, optional): Whether or not to calculate performance metrics over the training set. Defaults to False.
        evaluate_val (bool, optional): 	Whether or not to calculate performance metrics over the validation set. Defaults to False.
        verbose (bool, optional): Whether or not to print useful info about the training process in the terminal. Defaults to False.
        results_verbose (bool, optional): Whether or not to print useful info about the calculation of the performance metrics in the terminal. Defaults to False.
        return_results_per_target (bool, optional): Whether or not to return metrics per target. Defaults to False.
        use_early_stopping (bool, optional): Whether or not to use early stopping while training. Defaults to True.
        use_tensorboard_logger (bool, optional): Whether or not to log results in Tensorboard. Defaults to False.
        wandb_project_name (str, optional): The name of the wandb project that the results of an experiment will be logged. Defaults to None.
        wandb_project_entity (str, optional): The user name of the wandb account. Defaults to None.
        results_path (str, optional): The path the all relevant information will be saved to. Defaults to './results/'.
        experiment_name (str, optional): The name of the current experiment. This name will be used to local save and the wandb save. Defaults to None.
        save_model (bool, optional): Whether or not to save the model of the epoch with the best validation performance. Defaults to True.
        metric_to_optimize_early_stopping (str, optional): The metric that will be used for tracking by the early stopping routine. The value can be the loss or one of the available performance metrics.. Defaults to 'loss'.
        metric_to_optimize_best_epoch_selection (str, optional): The validation metric that will be used to determine the best configuration. The value can be the loss or one of the available performance metrics.. Defaults to 'loss'.
        instance_branch_architecture (str, optional): The type of architecture used in the instance branch: MLP for dense side features, SPARSE for high-dimensional SciPy or PyTorch sparse features, SEQUENCE for token IDs, TABULAR for explicit numeric and categorical columns, COMPOSITE for multiple named encoders, CONV for images, EMBEDDING for zero-based entity IDs, or CUSTOM for a user-supplied branch. Defaults to None.
        use_instance_features (bool, optional): Whether or not the instance features will be used. Defaults to False.
        instance_branch_input_dim (int, optional): The input dimension of the instance branch, or the number of instance IDs for an EMBEDDING branch. Defaults to None.
        instance_branch_tabular_schema (mapping, optional): Numeric columns, categorical vocabularies, missing-value policies, normalization, and optional feature gating for an instance TABULAR branch.
        instance_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the instance branch. Not currently implemented. Defaults to 2.
        instance_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the instance branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used instance_branch_layers times. Defaults to [10, 10, 10].
        instance_branch_layers (int, optional): The number of layers in the MLP version of the instance branch. (Only used if instance_branch_nodes_per_layer is int). Defaults to None.
        instance_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_branch_conv_architecture (str, optional): The convolutional architecture used in the instance branch: 'resnet' or 'VGG'. Defaults to 'resnet'.
        instance_branch_conv_architecture_version (str, optional): The ResNet version used in the instance branch: 'resnet18' or 'resnet101'. Defaults to 'resnet101'.
        instance_branch_conv_architecture_dense_layers (int, optional): The number of replacement dense layers for a ResNet instance branch: 1 or 2. Defaults to 1.
        instance_branch_conv_architecture_last_layer_trained (str, optional): The earliest trainable ResNet block in the instance branch: 'last' or 'layer4' through 'layer1'. Defaults to 'last'.
        instance_branch_conv_pretrained (bool, optional): Whether to initialize an instance convolutional branch with torchvision's default pretrained weights. Set to False to construct the model without a network download. Defaults to True.
        target_branch_architecture (str, optional): The type of architecture used in the target branch: MLP for dense side features, SPARSE for high-dimensional SciPy or PyTorch sparse features, SEQUENCE for token IDs, TABULAR for explicit numeric and categorical columns, COMPOSITE for multiple named encoders, CONV for images, EMBEDDING for zero-based entity IDs, or CUSTOM for a user-supplied branch. Defaults to None.
        use_target_features (bool, optional): Whether or not the target features will be used. Defaults to False.. Defaults to False.
        target_branch_input_dim (int, optional): The input dimension of the target branch, or the number of target IDs for an EMBEDDING branch. Defaults to None.
        target_branch_tabular_schema (mapping, optional): Numeric columns, categorical vocabularies, missing-value policies, normalization, and optional feature gating for a target TABULAR branch.
        target_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the target branch. Not currently implemented. Defaults to 2.
        target_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the target branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used target_branch_layers times. Defaults to [10, 10, 10].
        target_branch_layers (_type_, optional): The number of layers in the MLP version of the target branch. (Only used if target_branch_nodes_per_layer is int). Defaults to None.
        target_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        target_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        target_branch_conv_architecture (str, optional): The convolutional architecture used in the target branch: 'resnet' or 'VGG'. Defaults to 'resnet'.
        target_branch_conv_architecture_version (str, optional): The ResNet version used in the target branch: 'resnet18' or 'resnet101'. Defaults to 'resnet101'.
        target_branch_conv_architecture_dense_layers (int, optional): The number of replacement dense layers for a ResNet target branch: 1 or 2. Defaults to 1.
        target_branch_conv_architecture_last_layer_trained (str, optional): The earliest trainable ResNet block in the target branch: 'last' or 'layer4' through 'layer1'. Defaults to 'last'.
        target_branch_conv_pretrained (bool, optional): Whether to initialize a target convolutional branch with torchvision's default pretrained weights. Set to False to construct the model without a network download. Defaults to True.
        comb_mlp_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the combination MLP. (Only used if general_architecture_version in "mlp"). Not currently implemented. Defaults to 2.
        comb_mlp_nodes_per_layer (list or int, optional): Positive layer widths for the combination branch. If a list is provided, each element defines one layer. If an int is provided, the width is repeated ``comb_mlp_layers`` times. Used only when general_architecture_version is "mlp". Defaults to [10, 10, 10].
        comb_mlp_layers (int, optional): The positive number of repeated combination layers, required when comb_mlp_nodes_per_layer is an int. Used only when general_architecture_version is "mlp". Defaults to None.
        embedding_size (int, optional): The branch output size for dot-product models and for each EMBEDDING branch. Defaults to 100.
        eval_every_n_epochs (int, optional): The interval that indicates when the performance metrics are computed. Defaults to 10.
        load_pretrained_model (bool, optional): Whether or not a pretrained model will be loaded. Defaults to False.
        pretrained_model_path (str, optional): The path to the .pt file with the pretrained model (Only used if load_pretrained_model == True). Defaults to ''.
        running_hpo (bool, optional): Whether or not the base model will by used by an hpo method. This is used to adjust the prints. Defaults to False.
        additional_info (dict, optional): A dictionary that holds all other relevant info. Can be used as log adittional info for an experiment in wandb. Defaults to {}.
        eval_instance_verbose (str, optional): Printing of instance-wise warnings when evaluating performance metrics.
        eval_target_verbose (str, optional): Printing of target-wise warnings when evaluating performance metrics.
        hpo_results_path (str, optional): The directory the HPO results will be stored.
    Returns:
        dict: A dictionary with the config that will be used by the model to adjust the architecture and all other training-related information
    """
    metrics = (
        [
            "hamming_loss",
            "auroc",
            "f1_score",
            "aupr",
            "accuracy",
            "recall",
            "precision",
        ]
        if metrics is None
        else list(metrics)
    )
    metrics_average = (
        ["macro", "micro"] if metrics_average is None else list(metrics_average)
    )
    comb_mlp_nodes_per_layer = (
        [10, 10, 10] if comb_mlp_nodes_per_layer is None else comb_mlp_nodes_per_layer
    )
    additional_info = {} if additional_info is None else dict(additional_info)
    instance_branch_params = (
        {} if instance_branch_params is None else dict(instance_branch_params)
    )
    target_branch_params = (
        {} if target_branch_params is None else dict(target_branch_params)
    )

    base_config = {
        "validation_setting": validation_setting,
        "general_architecture_version": general_architecture_version,
        "problem_mode": problem_mode,
        "loss": loss,
        "learning_rate": learning_rate,
        "decay": decay,
        "batch_norm": batch_norm,
        "dropout_rate": dropout_rate,
        "dropout_rate_instance_branch": dropout_rate_instance_branch
        if dropout_rate_instance_branch is not None
        else dropout_rate,
        "dropout_rate_target_branch": dropout_rate_target_branch
        if dropout_rate_target_branch is not None
        else dropout_rate,
        "momentum": momentum,
        "weighted_loss": weighted_loss,
        "compute_mode": compute_mode,
        "num_workers": num_workers,
        "train_batchsize": train_batchsize,
        "val_batchsize": val_batchsize,
        "num_epochs": num_epochs,
        "random_seed": random_seed,
        "use_early_stopping": use_early_stopping,
        "patience": patience,
        "delta": delta,
        "evaluate_train": evaluate_train,
        "evaluate_val": evaluate_val,
        "verbose": verbose,
        "results_verbose": results_verbose,
        "eval_instance_verbose": eval_instance_verbose,
        "eval_target_verbose": eval_target_verbose,
        "return_results_per_target": return_results_per_target,
        "metric_to_optimize_early_stopping": metric_to_optimize_early_stopping,
        "metric_to_optimize_best_epoch_selection": metric_to_optimize_best_epoch_selection,
        "instance_branch_architecture": instance_branch_architecture,
        "target_branch_architecture": target_branch_architecture,
        "use_instance_features": use_instance_features,
        "use_target_features": use_target_features,
        "use_tensorboard_logger": use_tensorboard_logger,
        "wandb_project_name": wandb_project_name,
        "wandb_project_entity": wandb_project_entity,
        "results_path": results_path,
        "experiment_name": experiment_name,
        "save_model": save_model,
        "instance_branch_input_dim": instance_branch_input_dim,
        "target_branch_input_dim": target_branch_input_dim,
        "eval_every_n_epochs": eval_every_n_epochs,
        "load_pretrained_model": load_pretrained_model,
        "pretrained_model_path": pretrained_model_path,
        "instance_train_transforms": instance_train_transforms,
        "instance_inference_transforms": instance_inference_transforms,
        "target_train_transforms": target_train_transforms,
        "target_inference_transforms": target_inference_transforms,
        "running_hpo": running_hpo,
        "hpo_results_path": hpo_results_path,
        "additional_info": additional_info,
        "top_k": top_k,
    }
    normalized_classification_mode = (
        classification_mode.lower()
        if isinstance(classification_mode, str)
        else classification_mode
    )
    if classification_mode is not None:
        base_config["classification_mode"] = classification_mode
    if num_classes is not None or normalized_classification_mode == "multiclass":
        base_config["num_classes"] = num_classes
    if multiclass_average is not None or normalized_classification_mode == "multiclass":
        base_config["multiclass_average"] = multiclass_average

    # Normalize user-facing names before centralized typed validation.
    if problem_mode == "classification":
        metrics = [m.lower() for m in metrics]
    elif problem_mode == "regression":
        metrics = [m.upper() for m in metrics]

    metrics_average = [m.lower() for m in metrics_average]

    if validation_setting == "A":
        if metrics_average != ["micro"]:
            warnings.warn(
                "Validation setting A supports only micro-averaged metrics; "
                "metrics_average has been normalized to ['micro'].",
                ConfigNormalizationWarning,
                stacklevel=2,
            )
            metrics_average = ["micro"]
    elif validation_setting in ["B", "C"]:
        if "macro" not in metrics_average:
            warnings.warn(
                "Macro averaging is recommended for validation setting "
                f"{validation_setting}; 'macro' has been added to metrics_average.",
                ConfigNormalizationWarning,
                stacklevel=2,
            )
            metrics_average.append("macro")
    elif validation_setting == "D":
        if "micro" not in metrics_average:
            warnings.warn(
                "Micro averaging is recommended for validation setting D; "
                "'micro' has been added to metrics_average.",
                ConfigNormalizationWarning,
                stacklevel=2,
            )
            metrics_average.append("micro")
    else:
        raise ValueError(
            f"Validation setting {validation_setting!r} is not recognized."
        )

    base_config["metrics"] = metrics
    base_config["metrics_average"] = metrics_average

    if batch_norm == "True":
        base_config["batch_norm"] = True
    elif batch_norm == "False":
        base_config["batch_norm"] = False

    general_architecture_version = general_architecture_version.lower()
    if general_architecture_version == "dot_product":
        base_config["embedding_size"] = embedding_size
    elif general_architecture_version == "mlp":
        base_config.update(
            {
                "comb_mlp_nodes_per_layer": comb_mlp_nodes_per_layer,
                "comb_mlp_layers": comb_mlp_layers,
                "comb_mlp_nodes_reducing_factor": comb_mlp_nodes_reducing_factor,
            }
        )
    elif general_architecture_version == "kronecker":
        pass
    else:
        raise ValueError(
            "Value: " + str(general_architecture_version) + " is not a valid option"
        )
    if {"EMBEDDING", "COMPOSITE"}.intersection(
        {
            instance_branch_architecture,
            target_branch_architecture,
        }
    ):
        base_config["embedding_size"] = embedding_size

    if instance_branch_architecture in {"MLP", "SPARSE", "TABULAR"}:
        base_config.update(
            {
                "instance_branch_nodes_per_layer": [10, 10, 10],
                "instance_branch_layers": None,
                "instance_branch_nodes_reducing_factor": 2,
            }
        )
        for parameter_name in [
            "instance_branch_nodes_per_layer",
            "instance_branch_layers",
            "instance_branch_nodes_reducing_factor",
        ]:
            if parameter_name in instance_branch_params:
                base_config[parameter_name] = instance_branch_params[parameter_name]
        if instance_branch_architecture == "TABULAR":
            base_config["instance_branch_tabular_schema"] = instance_branch_params.get(
                "instance_branch_tabular_schema",
                instance_branch_tabular_schema,
            )

    elif instance_branch_architecture == "COMPOSITE":
        base_config["instance_branch_composite_components"] = (
            instance_branch_params.get("instance_branch_composite_components")
        )
        base_config["instance_branch_composite_fusion"] = instance_branch_params.get(
            "instance_branch_composite_fusion",
            "concat",
        )
        base_config["instance_branch_composite_attention_dim"] = (
            instance_branch_params.get(
                "instance_branch_composite_attention_dim",
                64,
            )
        )
        base_config["instance_branch_composite_attention_num_heads"] = (
            instance_branch_params.get(
                "instance_branch_composite_attention_num_heads",
                4,
            )
        )
        base_config["instance_branch_composite_modality_dropout"] = (
            instance_branch_params.get(
                "instance_branch_composite_modality_dropout",
                0.0,
            )
        )
    elif instance_branch_architecture == "SEQUENCE":
        sequence_defaults = {
            "instance_branch_sequence_encoder": "gru",
            "instance_branch_sequence_embedding_dim": 64,
            "instance_branch_sequence_output_dim": 64,
            "instance_branch_sequence_conv_kernel_size": 3,
            "instance_branch_sequence_transformer_num_heads": 4,
            "instance_branch_sequence_transformer_feedforward_dim": 256,
            "instance_branch_sequence_num_layers": 1,
            "instance_branch_sequence_padding_idx": 0,
        }
        base_config.update(sequence_defaults)
        for parameter_name in sequence_defaults:
            if parameter_name in instance_branch_params:
                base_config[parameter_name] = instance_branch_params[parameter_name]

    elif instance_branch_architecture == "CONV":
        base_config.update(
            {
                "instance_branch_conv_architecture": "resnet",
                "instance_branch_conv_architecture_version": "resnet101",
                "instance_branch_conv_architecture_dense_layers": 1,
                "instance_branch_conv_architecture_last_layer_trained": "last",
                "instance_branch_conv_pretrained": True,
                "instance_train_transforms": get_default_train_transform()
                if instance_train_transforms is None
                else instance_train_transforms,
                "instance_inference_transforms": get_default_inference_transform()
                if instance_inference_transforms is None
                else instance_inference_transforms,
            }
        )
        for parameter_name in [
            "instance_branch_conv_architecture",
            "instance_branch_conv_architecture_version",
            "instance_branch_conv_architecture_dense_layers",
            "instance_branch_conv_architecture_last_layer_trained",
            "instance_branch_conv_pretrained",
            "instance_train_transforms",
            "instance_inference_transforms",
        ]:
            if parameter_name in instance_branch_params:
                base_config[parameter_name] = instance_branch_params[parameter_name]
    elif instance_branch_architecture == "EMBEDDING":
        pass
    elif (
        instance_branch_architecture == "CUSTOM"
    ):  # this is still a work in progress but should do the trick, for now...
        base_config.update(instance_branch_params)
    elif instance_branch_architecture is None:
        raise ValueError("The instance branch has to be explicitly defined")
    else:
        raise ValueError(
            str(instance_branch_architecture)
            + " is not a valid name for the architecture in the instance branch"
        )

    if target_branch_architecture in {"MLP", "SPARSE", "TABULAR"}:
        base_config.update(
            {
                "target_branch_nodes_per_layer": [10, 10, 10],
                "target_branch_layers": None,
                "target_branch_nodes_reducing_factor": 2,
            }
        )
        for parameter_name in [
            "target_branch_nodes_per_layer",
            "target_branch_layers",
            "target_branch_nodes_reducing_factor",
        ]:
            if parameter_name in target_branch_params:
                base_config[parameter_name] = target_branch_params[parameter_name]
        if target_branch_architecture == "TABULAR":
            base_config["target_branch_tabular_schema"] = target_branch_params.get(
                "target_branch_tabular_schema",
                target_branch_tabular_schema,
            )

    elif target_branch_architecture == "COMPOSITE":
        base_config["target_branch_composite_components"] = target_branch_params.get(
            "target_branch_composite_components"
        )
        base_config["target_branch_composite_fusion"] = target_branch_params.get(
            "target_branch_composite_fusion",
            "concat",
        )
        base_config["target_branch_composite_attention_dim"] = target_branch_params.get(
            "target_branch_composite_attention_dim",
            64,
        )
        base_config["target_branch_composite_attention_num_heads"] = (
            target_branch_params.get(
                "target_branch_composite_attention_num_heads",
                4,
            )
        )
        base_config["target_branch_composite_modality_dropout"] = (
            target_branch_params.get(
                "target_branch_composite_modality_dropout",
                0.0,
            )
        )
    elif target_branch_architecture == "SEQUENCE":
        sequence_defaults = {
            "target_branch_sequence_encoder": "gru",
            "target_branch_sequence_embedding_dim": 64,
            "target_branch_sequence_output_dim": 64,
            "target_branch_sequence_conv_kernel_size": 3,
            "target_branch_sequence_transformer_num_heads": 4,
            "target_branch_sequence_transformer_feedforward_dim": 256,
            "target_branch_sequence_num_layers": 1,
            "target_branch_sequence_padding_idx": 0,
        }
        base_config.update(sequence_defaults)
        for parameter_name in sequence_defaults:
            if parameter_name in target_branch_params:
                base_config[parameter_name] = target_branch_params[parameter_name]

    elif target_branch_architecture == "CONV":
        base_config.update(
            {
                "target_branch_conv_architecture": "resnet",
                "target_branch_conv_architecture_version": "resnet101",
                "target_branch_conv_architecture_dense_layers": 1,
                "target_branch_conv_architecture_last_layer_trained": "last",
                "target_branch_conv_pretrained": True,
                "target_train_transforms": get_default_train_transform()
                if target_train_transforms is None
                else target_train_transforms,
                "target_inference_transforms": get_default_inference_transform()
                if target_inference_transforms is None
                else target_inference_transforms,
            }
        )
        for parameter_name in [
            "target_branch_conv_architecture",
            "target_branch_conv_architecture_version",
            "target_branch_conv_architecture_dense_layers",
            "target_branch_conv_architecture_last_layer_trained",
            "target_branch_conv_pretrained",
            "target_train_transforms",
            "target_inference_transforms",
        ]:
            if parameter_name in target_branch_params:
                base_config[parameter_name] = target_branch_params[parameter_name]
    elif target_branch_architecture == "EMBEDDING":
        pass
    elif (
        target_branch_architecture == "CUSTOM"
    ):  # this is still a work in progress but should do the trick, for now...
        base_config.update(target_branch_params)
    elif target_branch_architecture is None:
        raise ValueError("The target branch has to be explicitly defined")
    else:
        raise ValueError(
            str(target_branch_architecture)
            + " is not a valid name for the architecture in the target branch"
        )

    validated_config: dict[str, Any] = DeepMTPConfig.from_mapping(base_config).to_dict()
    return validated_config
