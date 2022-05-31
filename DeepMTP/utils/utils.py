import torchvision.transforms as transforms

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
    additional_info = {}

):

    if metric_to_optimize_early_stopping not in metrics+['loss']:
        raise AttributeError('The metric requested to track during early stopping is should be also defined in the metrics field of the configuration object')

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
        for k in ['instance', 'macro']:
            if k in metrics:
                print('The macro and instance-wise averaging schemes are not recommended while on validation setting A. The experiments will calculate the micro averaged version of the selected metrics')
                metric_averaging = ['micro']
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
    elif target_branch_architecture is None:
        raise AttributeError('The target branch has to be explicitly defined')
    else:
        raise AttributeError(str(target_branch_architecture)+' is not a valid name for the architecture in the target branch')

    return base_config


# dafault transofmations applied for resnet inputs 
def get_default_train_transform():
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
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    return transforms.Compose([
                           transforms.Resize((pretrained_size, pretrained_size)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])