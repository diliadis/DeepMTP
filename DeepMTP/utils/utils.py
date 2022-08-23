import torchvision.transforms as transforms

def get_optimization_direction(metric_name: str) -> str:
    '''Determines if the goal is to maximize or minimize based on the name of the metric

    Args:
        metric_name (sting): the name of the metric

    Returns:
        string: max if the goal is go maximize or min if the goal is to mimize  
    '''
    metrics_to_max = ['sensitivity', 'f1_score', 'recall', 'positive_predictive_value']
    if True in [n in metric_name for n in metrics_to_max]:
        return 'max'
    return 'min'

class BaseExperimentInfo:
    """A class used to keep track of all relevant info of a given experiment. This is mainly used by the HPO methods.
    """    
    def __init__(self, config, budget):
        self.config = config
        self.score = 0
        self.budget = budget
        self.info = {}

    def update_score(self, score):
        self.score = score

    def get_config(self):
        return self.config

    def get_budget(self):
        return self.budget

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return self.score != other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __repr__(self):
        return (
            'config: '
            + str(self.config)
            + '  |  budget: '
            + str(self.budget)
            + '  |  score: '
            + str(self.score)
            + '\n'
            + str(self.info)
            + '\n'
            + '\n'
        )

def generate_config_v1(
    validation_setting = None,
    enable_dot_product_version = True,
    problem_mode = None,
    learning_rate = 0.001,
    decay = 0,
    batch_norm = False,
    dropout_rate = 0,
    momentum = 0.9,
    weighted_loss = False,
    compute_mode = 'cuda:0',
    num_workers = 1,
    train_batchsize = 512,
    val_batchsize = 512,
    num_epochs = 100,
    metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
    metrics_average = ['macro', 'micro'],
    patience = 10,

    evaluate_train = False,
    evaluate_val = False,

    verbose = False,
    results_verbose = False,
    return_results_per_target=False,
    use_early_stopping = True,
    use_tensorboard_logger = False,
    wandb_project_name = None,
    wandb_project_entity = None,
    results_path = './results/',
    experiment_name = None,
    save_model = True,
    metric_to_optimize_early_stopping = 'loss',
    metric_to_optimize_best_epoch_selection = 'loss',

    instance_branch_architecture = None,
    use_instance_features = False,
    instance_branch_input_dim = None,
    instance_branch_nodes_reducing_factor = 2,
    instance_branch_nodes_per_layer = [10, 10, 10],
    instance_branch_layers = None,
    instance_train_transforms = None,
    instance_inference_transforms = None,
    instance_branch_conv_architecture = 'resnet',
    instance_branch_conv_architecture_version = 'resnet101',
    instance_branch_conv_architecture_dense_layers = 1,
    instance_branch_conv_architecture_last_layer_trained = 'last',

    target_branch_architecture = None,
    use_target_features = False,
    target_branch_input_dim = None,
    target_branch_nodes_reducing_factor = 2,
    target_branch_nodes_per_layer = [10, 10, 10],
    target_branch_layers = None,
    target_train_transforms = None,
    target_inference_transforms = None,
    target_branch_conv_architecture = 'resnet',
    target_branch_conv_architecture_version = 'resnet101',
    target_branch_conv_architecture_dense_layers = 1,
    target_branch_conv_architecture_last_layer_trained = 'last',

    comb_mlp_nodes_reducing_factor = 2,
    comb_mlp_nodes_per_layer = [10, 10, 10],
    comb_mlp_branch_layers = None,

    embedding_size = 100,
    eval_every_n_epochs = 10,

    load_pretrained_model = False,
    pretrained_model_path = '',
    running_hpo = False,
    additional_info = {},

    instance_branch_custom_params = {},
    target_branch_custom_params = {},

):
    ''' Creates a dictionary that is used to configure the neural network. Contains some base logic that checks if some of the parameters make sense. It has to be updated each time a new feature is added.

    Args:
        validation_setting (str, optional): The validation setting of the given problem. The possible values are A, B, C, D. Defaults to None.
        enable_dot_product_version (bool, optional): Enables the version of the neural network that just computes the dot product of the two embedding vectors. Otherwise, the MLP version is used. Defaults to True.
        problem_mode (str, optional): The type of task for the given problem. The possible values are classification or regression. Defaults to None.
        learning_rate (float, optional): The learning rate that will be used during training. Defaults to 0.001.
        decay (float, optional): The weight decay (L2 penalty) used by the Adam optimizer . Defaults to 0.
        batch_norm (bool, optional): The option to use batch normalization between the fully connected layers in the two branches. Defaults to False.
        dropout_rate (float, optional): The amount of dropout used in the layers of the two branches. Defaults to 0.
        momentum (float, optional): The momentum used by the optimizer. Defaults to 0.9.
        weighted_loss (bool, optional): Enables the use of class weights in the loss. Defaults to False.
        compute_mode (str, optional): The specific device that will be used during training. The possible values can be one the available gpus or the cpu(please dont). Defaults to 'cuda:0'.
        num_workers (int, optional): The number of sub-processes to use for data loading. Larger values usually improve performance but after a point training speed will become worse. Defaults to 1.
        train_batchsize (int, optional): The number of samples that comprise a batch from the training set. Defaults to 512.
        val_batchsize (int, optional): 	The number of samples that comprise a batch from the validation and test sets. Defaults to 512.
        num_epochs (int, optional): The max number of epochs allowed for training. Defaults to 100.
        metrics (list, optional): The performance metrics that will be calculated. For classification tasks the available metrics are ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'] while for regression tasks the available metrics are ['RMSE', 'MSE', 'MAE', 'R2', 'RRMSE']. Defaults to ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'].
        metrics_average (list, optional): The averaging strategy that will be used to calculate the metric. The available options are ['macro', 'micro', 'instance']. Defaults to ['macro', 'micro'].
        patience (int, optional): The number of epochs that the network is allowed to continue training for while observing worse overall performance. Defaults to 10.
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
        instance_branch_architecture (str, optional): The type of architecture that will be used in the instance branch. Currently, there are two available options, MLP: a basic fully connected feed-forward neural network is used, CONV a convolutional neural network is used. Defaults to None.
        use_instance_features (bool, optional): Whether or not the instance features will be used. Defaults to False.
        instance_branch_input_dim (int, optional): The input dimension of the instance branch. Defaults to None.
        instance_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the instance branch. Not currently implemented. Defaults to 2.
        instance_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the instance branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used instance_branch_layers times. Defaults to [10, 10, 10].
        instance_branch_layers (int, optional): The number of layers in the MLP version of the instance branch. (Only used if instance_branch_nodes_per_layer is int). Defaults to None.
        instance_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_branch_conv_architecture (str, optional): The type of the convolutional architecture that is used in the instance branch. Defaults to 'resnet'.
        instance_branch_conv_architecture_version (str, optional): The version of the specific type of convolutional architecture that is used in the instance branch. Defaults to 'resnet101'.
        instance_branch_conv_architecture_dense_layers (int, optional): The number of dense layers that are used at the end of the convolutional architecture of the instance branch. Defaults to 1.
        instance_branch_conv_architecture_last_layer_trained (str, optional): When using pre-trained architectures, the user can define that last layer that will be frozen during training. Defaults to 'last'.
        target_branch_architecture (str, optional): The type of architecture that will be used in the target branch. Currently, there are two available options, MLP: a basic fully connected feed-forward neural network is used, CONV a convolutional neural network is used. Defaults to None.
        use_target_features (bool, optional): Whether or not the target features will be used. Defaults to False.. Defaults to False.
        target_branch_input_dim (int, optional): The input dimension of the target branch. Defaults to None.
        target_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the target branch. Not currently implemented. Defaults to 2.
        target_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the target branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used target_branch_layers times. Defaults to [10, 10, 10].
        target_branch_layers (_type_, optional): The number of layers in the MLP version of the target branch. (Only used if target_branch_nodes_per_layer is int). Defaults to None.
        target_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        target_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        target_branch_conv_architecture (str, optional): The type of the convolutional architecture that is used in the target branch. Defaults to 'resnet'.
        target_branch_conv_architecture_version (str, optional): The version of the specific type of convolutional architecture that is used in the target branch. Defaults to 'resnet101'.
        target_branch_conv_architecture_dense_layers (int, optional): The number of dense layers that are used at the end of the convolutional architecture of the target branch. Defaults to 1.
        target_branch_conv_architecture_last_layer_trained (str, optional): When using pre-trained architectures, the user can define that last layer that will be frozen during training. Defaults to 'last'.
        comb_mlp_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the combination MLP. (Only used if enable_dot_product_version == False). Not currently implemented. Defaults to 2.
        comb_mlp_nodes_per_layer (list, optional): The number of nodes in the combination branch. If list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used 'comb_mlp_branch_layers' times. (Only used if enable_dot_product_version == False). Defaults to [10, 10, 10].
        comb_mlp_branch_layers (int, optional): The number of layers in the combination branch. (Only used if enable_dot_product_version == False). Defaults to None.
        embedding_size (int, optional): The size of the embeddings outputted by the two branches. (Only used if enable_dot_product_version == True). Defaults to 100.
        eval_every_n_epochs (int, optional): The interval that indicates when the performance metrics are computed. Defaults to 10.
        load_pretrained_model (bool, optional): Whether or not a pretrained model will be loaded. Defaults to False.
        pretrained_model_path (str, optional): The path to the .pt file with the pretrained model (Only used if load_pretrained_model == True). Defaults to ''.
        running_hpo (bool, optional): Whether or not the base model will by used by an hpo method. This is used to adjust the prints. Defaults to False.
        additional_info (dict, optional): A dictionary that holds all other relevant info. Can be used as log adittional info for an experiment in wandb. Defaults to {}.


    Returns:
        dict: A dictionary with the config that will be used by the model to adjust the architecture and all other training-related information
    '''

    if metric_to_optimize_early_stopping not in [m+'_'+m_a for m in metrics for m_a in metrics_average]+['loss']:
        raise AttributeError('The metric requested to track during early stopping ('+metric_to_optimize_early_stopping+') is should be also defined in the metrics field of the configuration object '+str([m+'_'+m_a for m in metrics+['loss'] for m_a in metrics_average]))

    base_config = {
        'validation_setting': validation_setting,
        'enable_dot_product_version': enable_dot_product_version,
        'problem_mode': problem_mode,
        'learning_rate': learning_rate,
        'decay': decay, 
        'batch_norm': batch_norm,
        'dropout_rate': dropout_rate,
        'momentum': momentum,
        'weighted_loss': weighted_loss,
        'compute_mode': compute_mode,
        'num_workers': num_workers,
        'train_batchsize': train_batchsize,
        'val_batchsize': val_batchsize,
        'num_epochs': num_epochs,
        'use_early_stopping': use_early_stopping,
        'patience': patience,
        'evaluate_train': evaluate_train,
        'evaluate_val': evaluate_val,
        'verbose': verbose,
        'results_verbose': results_verbose,
        'return_results_per_target': return_results_per_target,
        'metric_to_optimize_early_stopping': metric_to_optimize_early_stopping,
        'metric_to_optimize_best_epoch_selection': metric_to_optimize_best_epoch_selection,
        'instance_branch_architecture': instance_branch_architecture,
        'target_branch_architecture': target_branch_architecture,
        'use_instance_features': use_instance_features,
        'use_target_features': use_target_features,
        'use_tensorboard_logger': use_tensorboard_logger,
        'wandb_project_name': wandb_project_name,
        'wandb_project_entity': wandb_project_entity,
        'results_path': results_path,
        'experiment_name': experiment_name,
        'save_model': save_model,
        'instance_branch_input_dim': instance_branch_input_dim,
        'target_branch_input_dim': target_branch_input_dim,
        'eval_every_n_epochs': eval_every_n_epochs,
        'load_pretrained_model': load_pretrained_model,
        'pretrained_model_path': pretrained_model_path,
        'instance_train_transforms': instance_train_transforms,
        'instance_inference_transforms': instance_inference_transforms,
        'target_train_transforms': target_train_transforms,
        'target_inference_transforms': target_inference_transforms,
        'running_hpo': running_hpo,
    }

    # various sanity checks for the metrics and averaging options that are provided by the user
    classification_metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision']
    regression_metrics = ['RMSE', 'MSE', 'MAE', 'R2', 'RRMSE']

    if problem_mode == 'classification':
        metrics = [m.lower() for m in metrics]
        unknown_metrics = set(metrics).difference(set(classification_metrics))
        if unknown_metrics != set():
            raise Exception('Detected unknown metrics for the current classification task: '+str(unknown_metrics))
    else:
        metrics = [m.upper() for m in metrics]
        unknown_metrics = set(metrics).difference(set(regression_metrics))
        if unknown_metrics != set():
            raise Exception('Detected unknown metrics for the current regression task: '+str(unknown_metrics))
    base_config['metrics'] = metrics

    metric_averaging_schemes = ['macro', 'micro', 'instance']
    metrics_average = [m.lower() for m in metrics_average]
    unknown_metric_averaging_schemes = set(metrics_average).difference(set(metric_averaging_schemes))
    if unknown_metric_averaging_schemes != set():
        raise Exception('Detected unknown metric averaging scheme for the current '+problem_mode+' task: '+str(unknown_metric_averaging_schemes))    
        
    if validation_setting == 'A':
        if 'instance' in metrics or 'macro' in metrics:
            print('The macro and instance-wise averaging schemes are not recommended while on validation setting A. The experiments will calculate the micro averaged version of the selected metrics')
            metrics_average = ['micro']
    elif validation_setting in ['B', 'C']:
        if 'macro' not in metrics_average:
            print('Macro-averaging is the adviced averaging option for validation setting '+validation_setting+'. The macro option will be included in the results')
            metrics_average.append('macro')
    elif validation_setting == 'D':
        if 'micro' not in metrics_average:
            print('Micro-averaging is the adviced averaging option for validation setting '+validation_setting+'. The micro option will be included in the results')
            metrics_average.append('micro')
    else:
        raise Exception('Validation setting '+validation_setting+' is not recognized.')
    
    base_config['metrics_average'] = metrics_average

    if batch_norm == 'True':
        base_config['batch_norm'] = True  
    elif batch_norm == 'False':
        base_config['batch_norm'] = False
      
    if enable_dot_product_version:
        base_config['embedding_size'] = embedding_size
    else:
        base_config.update(
            {
                'comb_mlp_nodes_per_layer': comb_mlp_nodes_per_layer,
                'comb_mlp_branch_layers': comb_mlp_branch_layers,
                'comb_mlp_nodes_reducing_factor': comb_mlp_nodes_reducing_factor,
            }
        )

    if instance_branch_architecture == 'MLP':
        base_config.update(
            {
                'instance_branch_nodes_per_layer': instance_branch_nodes_per_layer,
                'instance_branch_layers': instance_branch_layers,
                'instance_branch_nodes_reducing_factor': instance_branch_nodes_reducing_factor,
            }
        )
    elif instance_branch_architecture == 'CONV':
        base_config.update(
            {
                'instance_branch_conv_architecture': instance_branch_conv_architecture,
                'instance_branch_conv_architecture_version': instance_branch_conv_architecture_version,
                'instance_branch_conv_architecture_dense_layers': instance_branch_conv_architecture_dense_layers,
                'instance_branch_conv_architecture_last_layer_trained': instance_branch_conv_architecture_last_layer_trained,
                'instance_train_transforms': instance_train_transforms if instance_train_transforms is not None else get_default_train_transform(),
                'instance_inference_transforms': instance_inference_transforms if instance_inference_transforms is not None else get_default_inference_transform(),
            }
        )
    elif instance_branch_architecture == 'CUSTOM': # this is still a work in progress but should do the trick, for now...
        base_config.update(instance_branch_custom_params)
    elif instance_branch_architecture is None:
        raise AttributeError('The instance branch has to be explicitly defined')
    else:
        raise AttributeError(str(instance_branch_architecture)+' is not a valid name for the architecture in the instance branch')


    if target_branch_architecture == 'MLP':
        base_config.update(
            {
                'target_branch_nodes_per_layer': target_branch_nodes_per_layer,
                'target_branch_layers': target_branch_layers,
                'target_branch_nodes_reducing_factor': target_branch_nodes_reducing_factor,
            }
        )
    elif target_branch_architecture == 'CONV':
        base_config.update(
            {
                'target_branch_conv_architecture': target_branch_conv_architecture,
                'target_branch_conv_architecture_version': target_branch_conv_architecture_version,
                'target_branch_conv_architecture_dense_layers': target_branch_conv_architecture_dense_layers,
                'target_branch_conv_architecture_last_layer_trained': target_branch_conv_architecture_last_layer_trained,
                'target_train_transforms': target_train_transforms if target_train_transforms is not None else get_default_train_transform(),
                'target_inference_transforms': target_inference_transforms if target_inference_transforms is not None else get_default_inference_transform(),
            }
        )
    elif target_branch_architecture == 'CUSTOM': # this is still a work in progress but should do the trick, for now...
        base_config.update(target_branch_custom_params)
    elif target_branch_architecture is None:
        raise AttributeError('The target branch has to be explicitly defined')
    else:
        raise AttributeError(str(target_branch_architecture)+' is not a valid name for the architecture in the target branch')

    return base_config

def get_default_dropout_rate():
    ''' To return the default dropout rate

    Returns:
        int: The value 0
    '''    
    return 0

def get_default_batch_norm():
    ''' To return the default batch_norm value

    Returns:
        float: The value False 
    '''   
    return False

# dafault transofmations applied for resnet inputs 
def get_default_train_transform():
    ''' To return the default transformation pipeline for a resnet during training

    Returns:
       torchvision.transforms : A transformation pipeline 
    '''  
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    return transforms.Compose([
                           transforms.Resize((pretrained_size, pretrained_size)),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])
def get_default_inference_transform():
    ''' To return the default transformation pipeline for a resnet during inference

    Returns:
       torchvision.transforms : A transformation pipeline 
    ''' 
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    return transforms.Compose([
                           transforms.Resize((pretrained_size, pretrained_size)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

def generate_config(
    validation_setting = None,
    enable_dot_product_version = True,
    problem_mode = None,
    learning_rate = 0.001,
    decay = 0,
    batch_norm = False,
    dropout_rate = 0,
    momentum = 0.9,
    weighted_loss = False,
    compute_mode = 'cuda:0',
    num_workers = 1,
    train_batchsize = 512,
    val_batchsize = 512,
    num_epochs = 100,
    metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
    metrics_average = ['macro', 'micro'],
    patience = 10,

    evaluate_train = False,
    evaluate_val = False,

    verbose = False,
    results_verbose = False,
    return_results_per_target=False,
    use_early_stopping = True,
    use_tensorboard_logger = False,
    wandb_project_name = None,
    wandb_project_entity = None,
    results_path = './results/',
    experiment_name = None,
    save_model = True,
    metric_to_optimize_early_stopping = 'loss',
    metric_to_optimize_best_epoch_selection = 'loss',

    instance_branch_architecture = None,
    use_instance_features = False,
    instance_branch_input_dim = None,
    instance_train_transforms = None,
    instance_inference_transforms = None,

    target_branch_architecture = None,
    use_target_features = False,
    target_branch_input_dim = None,
    target_train_transforms = None,
    target_inference_transforms = None,

    comb_mlp_nodes_reducing_factor = 2,
    comb_mlp_nodes_per_layer = [10, 10, 10],
    comb_mlp_branch_layers = None,

    embedding_size = 100,
    eval_every_n_epochs = 10,

    load_pretrained_model = False,
    pretrained_model_path = '',
    running_hpo = False,
    additional_info = {},

    instance_branch_params = {},
    target_branch_params = {},

):
    ''' Creates a dictionary that is used to configure the neural network. Contains some base logic that checks if some of the parameters make sense. It has to be updated each time a new feature is added.

    Args:
        validation_setting (str, optional): The validation setting of the given problem. The possible values are A, B, C, D. Defaults to None.
        enable_dot_product_version (bool, optional): Enables the version of the neural network that just computes the dot product of the two embedding vectors. Otherwise, the MLP version is used. Defaults to True.
        problem_mode (str, optional): The type of task for the given problem. The possible values are classification or regression. Defaults to None.
        learning_rate (float, optional): The learning rate that will be used during training. Defaults to 0.001.
        decay (float, optional): The weight decay (L2 penalty) used by the Adam optimizer . Defaults to 0.
        batch_norm (bool, optional): The option to use batch normalization between the fully connected layers in the two branches. Defaults to False.
        dropout_rate (float, optional): The amount of dropout used in the layers of the two branches. Defaults to 0.
        momentum (float, optional): The momentum used by the optimizer. Defaults to 0.9.
        weighted_loss (bool, optional): Enables the use of class weights in the loss. Defaults to False.
        compute_mode (str, optional): The specific device that will be used during training. The possible values can be one the available gpus or the cpu(please dont). Defaults to 'cuda:0'.
        num_workers (int, optional): The number of sub-processes to use for data loading. Larger values usually improve performance but after a point training speed will become worse. Defaults to 1.
        train_batchsize (int, optional): The number of samples that comprise a batch from the training set. Defaults to 512.
        val_batchsize (int, optional): 	The number of samples that comprise a batch from the validation and test sets. Defaults to 512.
        num_epochs (int, optional): The max number of epochs allowed for training. Defaults to 100.
        metrics (list, optional): The performance metrics that will be calculated. For classification tasks the available metrics are ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'] while for regression tasks the available metrics are ['RMSE', 'MSE', 'MAE', 'R2', 'RRMSE']. Defaults to ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'].
        metrics_average (list, optional): The averaging strategy that will be used to calculate the metric. The available options are ['macro', 'micro', 'instance']. Defaults to ['macro', 'micro'].
        patience (int, optional): The number of epochs that the network is allowed to continue training for while observing worse overall performance. Defaults to 10.
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
        instance_branch_architecture (str, optional): The type of architecture that will be used in the instance branch. Currently, there are two available options, MLP: a basic fully connected feed-forward neural network is used, CONV a convolutional neural network is used. Defaults to None.
        use_instance_features (bool, optional): Whether or not the instance features will be used. Defaults to False.
        instance_branch_input_dim (int, optional): The input dimension of the instance branch. Defaults to None.
        instance_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the instance branch. Not currently implemented. Defaults to 2.
        instance_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the instance branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used instance_branch_layers times. Defaults to [10, 10, 10].
        instance_branch_layers (int, optional): The number of layers in the MLP version of the instance branch. (Only used if instance_branch_nodes_per_layer is int). Defaults to None.
        instance_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        instance_branch_conv_architecture (str, optional): The type of the convolutional architecture that is used in the instance branch. Defaults to 'resnet'.
        instance_branch_conv_architecture_version (str, optional): The version of the specific type of convolutional architecture that is used in the instance branch. Defaults to 'resnet101'.
        instance_branch_conv_architecture_dense_layers (int, optional): The number of dense layers that are used at the end of the convolutional architecture of the instance branch. Defaults to 1.
        instance_branch_conv_architecture_last_layer_trained (str, optional): When using pre-trained architectures, the user can define that last layer that will be frozen during training. Defaults to 'last'.
        target_branch_architecture (str, optional): The type of architecture that will be used in the target branch. Currently, there are two available options, MLP: a basic fully connected feed-forward neural network is used, CONV a convolutional neural network is used. Defaults to None.
        use_target_features (bool, optional): Whether or not the target features will be used. Defaults to False.. Defaults to False.
        target_branch_input_dim (int, optional): The input dimension of the target branch. Defaults to None.
        target_branch_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the target branch. Not currently implemented. Defaults to 2.
        target_branch_nodes_per_layer (list, optional): Defines the number of nodes in the MLP version of the target branch. if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used target_branch_layers times. Defaults to [10, 10, 10].
        target_branch_layers (_type_, optional): The number of layers in the MLP version of the target branch. (Only used if target_branch_nodes_per_layer is int). Defaults to None.
        target_train_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures. Defaults to None.
        target_inference_transforms (_type_, optional): The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures. Defaults to None.
        target_branch_conv_architecture (str, optional): The type of the convolutional architecture that is used in the target branch. Defaults to 'resnet'.
        target_branch_conv_architecture_version (str, optional): The version of the specific type of convolutional architecture that is used in the target branch. Defaults to 'resnet101'.
        target_branch_conv_architecture_dense_layers (int, optional): The number of dense layers that are used at the end of the convolutional architecture of the target branch. Defaults to 1.
        target_branch_conv_architecture_last_layer_trained (str, optional): When using pre-trained architectures, the user can define that last layer that will be frozen during training. Defaults to 'last'.
        comb_mlp_nodes_reducing_factor (int, optional): The factor that will be used to create a smooth bottleneck in the combination MLP. (Only used if enable_dot_product_version == False). Not currently implemented. Defaults to 2.
        comb_mlp_nodes_per_layer (list, optional): The number of nodes in the combination branch. If list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used 'comb_mlp_branch_layers' times. (Only used if enable_dot_product_version == False). Defaults to [10, 10, 10].
        comb_mlp_branch_layers (int, optional): The number of layers in the combination branch. (Only used if enable_dot_product_version == False). Defaults to None.
        embedding_size (int, optional): The size of the embeddings outputted by the two branches. (Only used if enable_dot_product_version == True). Defaults to 100.
        eval_every_n_epochs (int, optional): The interval that indicates when the performance metrics are computed. Defaults to 10.
        load_pretrained_model (bool, optional): Whether or not a pretrained model will be loaded. Defaults to False.
        pretrained_model_path (str, optional): The path to the .pt file with the pretrained model (Only used if load_pretrained_model == True). Defaults to ''.
        running_hpo (bool, optional): Whether or not the base model will by used by an hpo method. This is used to adjust the prints. Defaults to False.
        additional_info (dict, optional): A dictionary that holds all other relevant info. Can be used as log adittional info for an experiment in wandb. Defaults to {}.


    Returns:
        dict: A dictionary with the config that will be used by the model to adjust the architecture and all other training-related information
    '''

    if metric_to_optimize_early_stopping not in [m+'_'+m_a for m in metrics for m_a in metrics_average]+['loss']:
        raise AttributeError('The metric requested to track during early stopping ('+metric_to_optimize_early_stopping+') is should be also defined in the metrics field of the configuration object '+str([m+'_'+m_a for m in metrics+['loss'] for m_a in metrics_average]))

    base_config = {
        'validation_setting': validation_setting,
        'enable_dot_product_version': enable_dot_product_version,
        'problem_mode': problem_mode,
        'learning_rate': learning_rate,
        'decay': decay, 
        'batch_norm': batch_norm,
        'dropout_rate': dropout_rate,
        'momentum': momentum,
        'weighted_loss': weighted_loss,
        'compute_mode': compute_mode,
        'num_workers': num_workers,
        'train_batchsize': train_batchsize,
        'val_batchsize': val_batchsize,
        'num_epochs': num_epochs,
        'use_early_stopping': use_early_stopping,
        'patience': patience,
        'evaluate_train': evaluate_train,
        'evaluate_val': evaluate_val,
        'verbose': verbose,
        'results_verbose': results_verbose,
        'return_results_per_target': return_results_per_target,
        'metric_to_optimize_early_stopping': metric_to_optimize_early_stopping,
        'metric_to_optimize_best_epoch_selection': metric_to_optimize_best_epoch_selection,
        'instance_branch_architecture': instance_branch_architecture,
        'target_branch_architecture': target_branch_architecture,
        'use_instance_features': use_instance_features,
        'use_target_features': use_target_features,
        'use_tensorboard_logger': use_tensorboard_logger,
        'wandb_project_name': wandb_project_name,
        'wandb_project_entity': wandb_project_entity,
        'results_path': results_path,
        'experiment_name': experiment_name,
        'save_model': save_model,
        'instance_branch_input_dim': instance_branch_input_dim,
        'target_branch_input_dim': target_branch_input_dim,
        'eval_every_n_epochs': eval_every_n_epochs,
        'load_pretrained_model': load_pretrained_model,
        'pretrained_model_path': pretrained_model_path,
        'instance_train_transforms': instance_train_transforms,
        'instance_inference_transforms': instance_inference_transforms,
        'target_train_transforms': target_train_transforms,
        'target_inference_transforms': target_inference_transforms,
        'running_hpo': running_hpo,
    }

    # various sanity checks for the metrics and averaging options that are provided by the user
    classification_metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision']
    regression_metrics = ['RMSE', 'MSE', 'MAE', 'R2', 'RRMSE']

    if problem_mode == 'classification':
        metrics = [m.lower() for m in metrics]
        unknown_metrics = set(metrics).difference(set(classification_metrics))
        if unknown_metrics != set():
            raise Exception('Detected unknown metrics for the current classification task: '+str(unknown_metrics))
    else:
        metrics = [m.upper() for m in metrics]
        unknown_metrics = set(metrics).difference(set(regression_metrics))
        if unknown_metrics != set():
            raise Exception('Detected unknown metrics for the current regression task: '+str(unknown_metrics))
    base_config['metrics'] = metrics

    metric_averaging_schemes = ['macro', 'micro', 'instance']
    metrics_average = [m.lower() for m in metrics_average]
    unknown_metric_averaging_schemes = set(metrics_average).difference(set(metric_averaging_schemes))
    if unknown_metric_averaging_schemes != set():
        raise Exception('Detected unknown metric averaging scheme for the current '+problem_mode+' task: '+str(unknown_metric_averaging_schemes))    
        
    if validation_setting == 'A':
        if 'instance' in metrics or 'macro' in metrics:
            print('The macro and instance-wise averaging schemes are not recommended while on validation setting A. The experiments will calculate the micro averaged version of the selected metrics')
            metrics_average = ['micro']
    elif validation_setting in ['B', 'C']:
        if 'macro' not in metrics_average:
            print('Macro-averaging is the adviced averaging option for validation setting '+validation_setting+'. The macro option will be included in the results')
            metrics_average.append('macro')
    elif validation_setting == 'D':
        if 'micro' not in metrics_average:
            print('Micro-averaging is the adviced averaging option for validation setting '+validation_setting+'. The micro option will be included in the results')
            metrics_average.append('micro')
    else:
        raise Exception('Validation setting '+validation_setting+' is not recognized.')
    
    base_config['metrics_average'] = metrics_average

    if batch_norm == 'True':
        base_config['batch_norm'] = True  
    elif batch_norm == 'False':
        base_config['batch_norm'] = False
      
    if enable_dot_product_version:
        base_config['embedding_size'] = embedding_size
    else:
        base_config.update(
            {
                'comb_mlp_nodes_per_layer': comb_mlp_nodes_per_layer,
                'comb_mlp_branch_layers': comb_mlp_branch_layers,
                'comb_mlp_nodes_reducing_factor': comb_mlp_nodes_reducing_factor,
            }
        )


    if instance_branch_architecture == 'MLP':
        base_config.update(
            {
                'instance_branch_nodes_per_layer': [10, 10, 10],
                'instance_branch_layers': None,
                'instance_branch_nodes_reducing_factor': 2,
            }
        )
        for p in ['instance_branch_nodes_per_layer', 'instance_branch_layers', 'instance_branch_nodes_reducing_factor']:
            if p not in instance_branch_params:
                print('Warning: '+p+' is a necessary hyperparameter to define the instance branch. Using '+str(base_config[p])+' as the default')
            else:
                base_config[p] = instance_branch_params[p]

    elif instance_branch_architecture == 'CONV':

        base_config.update(
            {
                'instance_branch_conv_architecture': 'resnet',
                'instance_branch_conv_architecture_version': 'resnet101',
                'instance_branch_conv_architecture_dense_layers': 1,
                'instance_branch_conv_architecture_last_layer_trained': 'last',
                'instance_train_transforms': get_default_train_transform() if instance_train_transforms is None else instance_train_transforms,
                'instance_inference_transforms': get_default_inference_transform() if instance_inference_transforms is None else instance_inference_transforms,
            }
        )
        for p in ['instance_branch_conv_architecture', 'instance_branch_conv_architecture_version', 'instance_branch_conv_architecture_dense_layers', 'instance_branch_conv_architecture_last_layer_trained', 'instance_train_transforms', 'instance_inference_transforms']:
            if p not in instance_branch_params:
                print('Warning: '+p+' is a necessary hyperparameter to define the instance branch. Using '+str(base_config[p])+' as the default')
            else:
                base_config[p] = instance_branch_params[p]
    elif instance_branch_architecture == 'CUSTOM': # this is still a work in progress but should do the trick, for now...
        base_config.update(instance_branch_params)
    elif instance_branch_architecture is None:
        raise AttributeError('The instance branch has to be explicitly defined')
    else:
        raise AttributeError(str(instance_branch_architecture)+' is not a valid name for the architecture in the instance branch')


    if target_branch_architecture == 'MLP':
        base_config.update(
            {
                'target_branch_nodes_per_layer': [10, 10, 10],
                'target_branch_layers': None,
                'target_branch_nodes_reducing_factor': 2,
            }
        )
        for p in ['target_branch_nodes_per_layer', 'target_branch_layers', 'target_branch_nodes_reducing_factor']:
            if p not in target_branch_params:
                print('Warning: '+p+' is a necessary hyperparameter to define the target branch. Using '+str(base_config[p])+' as the default')
            else:
                base_config[p] = target_branch_params[p]

    elif target_branch_architecture == 'CONV':

        base_config.update(
            {
                'target_branch_conv_architecture': 'resnet',
                'target_branch_conv_architecture_version': 'resnet101',
                'target_branch_conv_architecture_dense_layers': 1,
                'target_branch_conv_architecture_last_layer_trained': 'last',
                'target_train_transforms': get_default_train_transform() if target_train_transforms is None else target_train_transforms,
                'target_inference_transforms': get_default_inference_transform() if target_inference_transforms is None else target_inference_transforms,
            }
        )
        for p in ['target_branch_conv_architecture', 'target_branch_conv_architecture_version', 'target_branch_conv_architecture_dense_layers', 'target_branch_conv_architecture_last_layer_trained', 'target_train_transforms', 'target_inference_transforms']:
            if p not in target_branch_params:
                print('Warning: '+p+' is a necessary hyperparameter to define the target branch. Using '+str(base_config[p])+' as the default')
            else:
                base_config[p] = target_branch_params[p]
    elif target_branch_architecture == 'CUSTOM': # this is still a work in progress but should do the trick, for now...
        base_config.update(target_branch_params)
    elif target_branch_architecture is None:
        raise AttributeError('The target branch has to be explicitly defined')
    else:
        raise AttributeError(str(target_branch_architecture)+' is not a valid name for the architecture in the target branch')

    return base_config