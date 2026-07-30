Configuration options
#####################

General configuration
**************************

In the code snippet above the function generate_config is shown without any specific parameters. In practive, the function offers many parameters that define multiple characteristics of the architecture of the two-branch neural network, aspects of training, validating, testing etc. The following section can be used as a cheatsheet for users, explaining the meaning and rationale of every parameter.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter name
     - Description
   * - **training**
     -
   * - **num_epochs** 
     - The max number of epochs allowed for training
   * - **learning_rate**
     - The learning rate used to determine the step size at each iteration of the optimization process
   * - **loss**
     - Training objective. Binary classification defaults to
       ``binary_cross_entropy_with_logits``; multiclass classification uses
       ``cross_entropy``. Regression defaults to ``mean_squared_error`` and
       also supports ``mean_absolute_error`` and ``huber``.
   * - **classification_mode**
     - ``binary`` (the backward-compatible classification default) or
       ``multiclass``
   * - **num_classes**
     - Number of mutually exclusive classes; required and at least 3 for
       multiclass classification
   * - **decay**
     - The weight decay (L2 penalty) used by the Adam optimizer
   * - **compute_mode**
     - The device that is going to be used to actually train the neural network. The valid options are **cpu** if the user wants to train slowly or **cuda:id** if the user wants to train on the **id** gpu of the system
   * - **num_workers**
     - The number of sub-processes to use for data loading. Larger values usually improve performance but after a point training speed will become worse
   * - **train_batchsize**
     - The number of samples that comprise a batch from the training set
   * - **val_batchsize**
     - The number of samples that comprise a batch from the validation and test sets
   * - **patience**
     - The number of epochs that the network is allowed to continue training for while observing worse overall performance
   * - **delta**
     - Minimum change in the monitored quantity to qualify as an improvement
   * - **return_results_per_target**
     - Whether to include per-target metric values in the results; requires
       ``macro`` averaging
   * - **evaluate_train**
     - Whether or not to calculate performance metrics over the training set
   * - **evaluate_val**
     - Whether or not to calculate performance metrics over the validation set
   * - **eval_every_n_epochs**
     - The interval that indicates when the performance metrics are computed
   * - **use_early_stopping**
     - Whether or not to use early stopping while training
   * - **Metrics**
     -
   * - **metrics**
     - The performance metrics that will be calculated. For classification tasks the available metrics are ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'] while for regression tasks the available metrics are ['RMSE', 'MSE', 'MAE', 'R2']
   * - **metrics_average**
     - The averaging strategy used to calculate metrics. The available options
       are ['macro', 'micro', 'instance']; validation setting A supports only
       ['micro'].
   * - **multiclass_average**
     - Class averaging for multiclass precision, recall, F1, AUROC, and AUPR:
       ``micro``, ``macro`` (default), or ``weighted``
   * - **top_k**
     - Number of top predictions used to calculate grouped metric variants;
       requires ``macro`` or ``instance`` averaging and is unavailable for
       multiclass classification
   * - **metric_to_optimize_early_stopping**
     - The metric that will be used for tracking by the early stopping routine. The value can be the **loss** or one of the available performance metrics.
   * - **metric_to_optimize_best_epoch_selection**
     - The validation metric that will be used to determine the best configuration. The value can be the **loss** or one of the available performance metrics.
   * - **Printing - Saving - Logging**
     -
   * - **verbose**
     - Whether or not to print useful in the terminal
   * - **use_tensorboard_logger**
     - Whether or not to log results in files that Tensoboard can read and visualize
   * - **wandb_project_name**
     - Defines the name of the wandb project that the results of an experiment will be logged
   * - **wandb_project_entity**
     - Defines the user name of the wandb account
   * - **results_path**
     - Defines the path the all relevant information will be saved to
   * - **experiment_name**
     - Defines the name of the current experiment. This name will be used to local save and the wandb save
   * - **save_model**
     - Whether or not to save the model of the epoch with the best validation performance
   * - **General architecture architecture**
     -
   * - **general_architecture_version**
     - Enables a specific version of the general neural network architecture. Available options are **mlp** for the mlp version, **dot_product** for the dot product version, **kronecker**: for the kronecker product version. Default value is **dot_product**
   * - **batch_norm**
     - The option to use batch normalization between the fully connected layers in the two branches
   * - **dropout_rate**
     - The amount of dropout used in the layers of the two branches
   * - **Instance branch architecture**
     -
   * - **instance_branch_architecture**
     - The instance encoder: **MLP** for dense vectors, **SPARSE** for sparse
       vectors, **GRAPH** for PyTorch Geometric graphs, **SEQUENCE** for token
       IDs, **TABULAR** for explicit numeric and categorical columns, **CONV**
       for images, **EMBEDDING** for zero-based IDs, or **CUSTOM** for a
       user-supplied branch
   * - **instance_branch_input_dim**
     - The input feature dimension, graph node-feature width, sequence
       vocabulary size, or number of instance IDs for an **EMBEDDING** branch
   * - **instance_branch_tabular_schema**
     - Column names, category vocabularies, normalization, missing-value
       policies, and optional feature gating for an instance **TABULAR** branch
   * - **instance_train_transforms**
     - The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures
   * - **instance_inference_transforms**
     - The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures
   * - **instance_branch_params**
     - A dictionary that holds all the hyperparameters needed to configure the architecture present in the instance branch. The include key-value pairs like the following:
   * - **Target branch architecture**
     -
   * - **target_branch_architecture**
     - The target encoder: **MLP** for dense vectors, **SPARSE** for sparse
       vectors, **GRAPH** for PyTorch Geometric graphs, **SEQUENCE** for token
       IDs, **TABULAR** for explicit numeric and categorical columns, **CONV**
       for images, **EMBEDDING** for zero-based IDs, or **CUSTOM** for a
       user-supplied branch
   * - **target_branch_input_dim**
     - The input feature dimension, graph node-feature width, sequence
       vocabulary size, or number of target IDs for an **EMBEDDING** branch
   * - **target_branch_tabular_schema**
     - Column names, category vocabularies, normalization, missing-value
       policies, and optional feature gating for a target **TABULAR** branch
   * - **target_train_transforms**
     - The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures
   * - **target_inference_transforms**
     - The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures
   * - **target_branch_params**
     - A dictionary that holds all the hyperparameters needed to configure the architecture present in the target branch.
   * - **Combination branch architecture**
     -
   * - **comb_mlp_nodes_per_layer**
     - Positive layer widths for the combination branch. A list defines each
       layer; an integer repeats that width ``comb_mlp_layers`` times.
       (**Only used if general_architecture_version == mlp**)
   * - **comb_mlp_layers**
     - Positive number of repeated combination layers, required when
       ``comb_mlp_nodes_per_layer`` is an integer.
       (**Only used if general_architecture_version == mlp**)
   * - **embedding_size**
     - The output width of both branches for a dot-product model and of every
       **EMBEDDING** branch
   * - **Pretrained models**
     -
   * - **load_pretrained_model**
     - Whether or not a pretrained model will be loaded
   * - **pretrained_model_path**
     - The path to the .pt file with the pretrained model (**Only used if load_pretrained_model == True**)
   * - **Other**
     -
   * - **additional_info**
     - A dictionary that holds all other relevant info. Can be used as log adittional info for an experiment in wandb
   * - **validation_setting**
     - The validation setting of the specific example

An **EMBEDDING** branch consumes entity IDs directly and does not accept side
features. It is supported only when that entity axis is known at inference
time: both branches in setting A, the target branch in setting B, and the
instance branch in setting C. Setting D requires side-feature encoders for both
axes.

A **TABULAR** branch consumes a DataFrame with an ``id`` column plus the
numeric and categorical columns in its explicit schema. Training-only
normalization and imputation statistics are saved with the model. See
:doc:`tabular_inputs` for the complete schema and an example.

A **SEQUENCE** branch consumes unpadded integer token rows. ``data_process``
must receive ``instance_feature_kind="sequence"`` or
``target_feature_kind="sequence"`` for that axis. Batches carry dynamically
padded IDs, attention masks, and original lengths. See :doc:`sequence_inputs`
for vocabulary, padding, packed-GRU, masked-Conv1D, Transformer, and
checkpoint configuration.

A **GRAPH** branch consumes one homogeneous
``torch_geometric.data.Data`` object per entity. ``data_process`` must receive
``instance_feature_kind="graph"`` or ``target_feature_kind="graph"`` for that
axis. Node width is inferred as ``branch_input_dim`` and edge width as
``branch_graph_edge_dim``. The encoder uses GINE with edge features or GIN
when ``branch_graph_use_edge_features=False``, followed by configurable mean,
sum, or max pooling. See :doc:`graph_inputs` for the data, batch,
configuration, composite, and checkpoint contracts.

A **COMPOSITE** branch consumes a named mapping of side-feature collections
and fuses the outputs of independently configured dense, graph, sparse,
sequence, or known-ID component encoders. ``data_process`` receives a matching feature-kind
mapping, and each component declares its own input and output dimensions.
``instance_branch_composite_fusion`` and
``target_branch_composite_fusion`` select default ``"concat"``, opt-in
sample-specific ``"gated"`` concatenation, or modality self-``"attention"``.
Attention fusion uses the corresponding ``composite_attention_dim`` and
``composite_attention_num_heads`` branch settings. Components marked optional
use explicit presence masks and learned missing representations. The
``instance_branch_composite_modality_dropout`` and
``target_branch_composite_modality_dropout`` probabilities default to zero and
temporarily hide present optional components during training only. See
:doc:`composite_inputs` for configuration, fusion, validation-setting, batch,
checkpoint, and missing-modality behavior.

For mutually exclusive outcomes, set
``classification_mode="multiclass"`` during both ``data_process`` and
configuration, and pass the detected ``num_classes`` value. Multiclass labels
are consecutive integer IDs beginning at zero. See
:doc:`multiclass_classification` for the data, loss, metric, prediction, and
checkpoint contract.

RRMSE remains available through the low-level ``get_performance_results``
utility for macro evaluation when a training mean is provided for every
target. The trainer does not yet retain those baselines, so RRMSE is rejected
in trainer configurations instead of returning silent ``NaN`` results.

``generate_config`` applies documented branch defaults silently. If it changes
``metrics_average`` to satisfy or recommend a validation-setting policy, it
emits ``ConfigNormalizationWarning`` so applications can display, filter, or
escalate that adjustment using Python's standard warnings controls.


Instance and target branch hyperparameters
******************************************

As mentioned before, all hyperparameters needed to define the architecture of the instance or target branch are passed as key-value pairs in the **instance_branch_params** and **target_branch_params**.


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - Possible key names currently supported in the **instance_branch_params** dictionary
     -
   * - **instance_branch_nodes_per_layer**
     - Defines the number of nodes in the **MLP** version of the instance branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used **instance_branch_layers** times
   * - **instance_branch_layers**
     - The number of layers in the MLP version of the instance branch. (Only used if **instance_branch_nodes_per_layer** is int)
   * - **instance_branch_conv_architecture**
     - Instance convolutional architecture: ``resnet`` or ``VGG``
   * - **instance_branch_conv_architecture_version**
     - Instance ResNet version: ``resnet18`` or ``resnet101``
   * - **instance_branch_conv_architecture_dense_layers**
     - Number of replacement instance ResNet dense layers: 1 or 2
   * - **instance_branch_conv_architecture_last_layer_trained**
     - Earliest trainable instance ResNet block: ``last`` or ``layer4``
       through ``layer1``
   * - **instance_branch_conv_pretrained**
     - Use torchvision's default pretrained weights. Defaults to ``True``;
       set to ``False`` to construct the model without downloading weights
   * - Possible key names currently supported in the **instance_branch_params** dictionary
     -
   * - **target_branch_nodes_per_layer**
     - Defines the number of nodes in the **MLP** version of the target branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used **target_branch_layers** times
   * - **target_branch_layers**
     - The number of layers in the MLP version of the target branch. (Only used if **target_branch_nodes_per_layer** is int)
   * - **target_branch_conv_architecture**
     - Target convolutional architecture: ``resnet`` or ``VGG``
   * - **target_branch_conv_architecture_version**
     - Target ResNet version: ``resnet18`` or ``resnet101``
   * - **target_branch_conv_architecture_dense_layers**
     - Number of replacement target ResNet dense layers: 1 or 2
   * - **target_branch_conv_architecture_last_layer_trained**
     - Earliest trainable target ResNet block: ``last`` or ``layer4``
       through ``layer1``
   * - **target_branch_conv_pretrained**
     - Use torchvision's default pretrained weights. Defaults to ``True``;
       set to ``False`` to construct the model without downloading weights

Pretrained convolutional branches use torchvision's current ``DEFAULT``
weight enum. Torchvision may download those weights into its local cache the
first time a model is created. Setting the corresponding
``*_branch_conv_pretrained`` option to ``False`` passes ``weights=None`` for
offline construction.


Example of a generating a configuration::

    config = generate_config(    
        instance_branch_input_dim = data_info['instance_branch_input_dim'],
        target_branch_input_dim = data_info['target_branch_input_dim'],
        validation_setting = data_info['detected_validation_setting'],
        general_architecture_version = 'dot_product',
        problem_mode = data_info['detected_problem_mode'],
        loss = 'mean_squared_error',
        learning_rate = 0.001,
        decay = 0,
        batch_norm = False,
        dropout_rate = 0,
        momentum = 0.9,
        weighted_loss = False,
        compute_mode = 'cuda:0',
        train_batchsize = 1024,
        val_batchsize = 1024,
        num_epochs = 200,
        num_workers = 8,
        metrics = ['RMSE', 'MSE'],
        metrics_average = ['macro', 'micro'],
        patience = 10,

        evaluate_train = True,
        evaluate_val = True,

        verbose = False,
        results_verbose = False,
        use_early_stopping = True,
        use_tensorboard_logger = True,
        wandb_project_name = 'Dummy_project_1',
        wandb_project_entity = None,
        metric_to_optimize_early_stopping = 'loss',
        delta=0.01,
        metric_to_optimize_best_epoch_selection = 'loss',

        instance_branch_architecture = 'MLP',
        use_instance_features = True,
        instance_branch_params = {
            'instance_branch_nodes_reducing_factor': 2,
            'instance_branch_nodes_per_layer': [100, 100],
            'instance_branch_layers': None,
            # 'instance_branch_conv_architecture': 'resnet',
            # 'instance_branch_conv_architecture_version': 'resnet101',
            # 'instance_branch_conv_architecture_dense_layers': 1,
            # 'instance_branch_conv_architecture_last_layer_trained': 'last',
        },


        target_branch_architecture = 'MLP',
        use_target_features = True,
        target_branch_params = {
            'target_branch_nodes_reducing_factor': 2,
            'target_branch_nodes_per_layer': [100, 100],
            'target_branch_layers': None,
            # 'target_branch_conv_architecture': 'resnet',
            # 'target_branch_conv_architecture_version': 'resnet101',
            # 'target_branch_conv_architecture_dense_layers': 1,
            # 'target_branch_conv_architecture_last_layer_trained': 'last',
        },
        
        embedding_size = 100,
        comb_mlp_nodes_reducing_factor = 2,
        comb_mlp_nodes_per_layer = [2048, 2048, 2048],
        comb_mlp_layers = None, 

        save_model = True,

        eval_every_n_epochs = 1,

        additional_info = {})
