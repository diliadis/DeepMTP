<p align="center"><img src="images/logo_transparent_cropped.png" alt="logo" height="300"/></p>

<h3 align="center">
<p> A Deep Learning Framework for Multi-target Prediction </h3>

This is the official repository of DeepMTP, a deep learning framework that can be used with multi-target prediction (MTP) problems. MTP can be seen as an umbrella term that cover many subareas of machine learning, which include multi-label classification (MLC), multivariate regression (MTR), multi-task learning (MTL), dyadic prediction (DP), and matrix completion (MC). The implementation is mainly written in Python and uses Pytorch for the implementation of the neural network. The goal is for any user to be able to train a model using only a few lines of code.

### Latest Updates
- [1/6/2022] The first implementation of DeepMTP is now live!!!


## Background

## What is MTP??
Multi-target prediction (MTP) serves as an umbrella term for machine learning tasks that concern the simultaneous prediction of multiple target variables. These include:
* Multi-label Classification
* Multivariate Regression
* Multitask Learning
* Hierarchical Multi-label Classification
* Dyadic Prediction
* Zero-shot Learning
* Matrix Completion
* (Hybrid) Matrix Completion
* Cold-start Collaborative Filtering

Despite the significant similarities, all these domains have evolved separately into distinct research areas over the last two decades. To better understand these similarities and differences it is important to get accustomed to the terminology and main concepts used in this field.

<p align="center"><img src="images/basic_MTP_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/basic_MTP.png#gh-light-mode-only" alt="logo" height="450"/></p>


A multi-target prediction problem is characterized by instances $x \in X$ and targets $t \in T$ with the following properties:

1. A training dataset $\mathcal{D}$ contains triplets $(x_i,t_j,y_{ij})$, where $x_i \in \mathcal{X}$ represents an instance, $t_j \in \mathcal{T}$ represents a target, and $y_{ij} \in \mathcal{Y}$ is the score that quantifies the relationship between an instance and a target, with $i\in\{1,\ldots,n\}$ and $j\in\{1,\ldots,m\}$. The scores can be arranged in an $n \times m$ matrix $\mathbf{Y}$ that is usually incomplete.

2. The score set $\mathcal{Y}$ consists of nominal, ordinal or real values.

3. During testing, the objective is to predict the score for any unobserved instance-target couple $(\mathbf{x},\mathbf{t}) \in \mathcal{X} \times \mathcal{T}$.

Using the formal definition above we can simplify the basic components that we need identify in a multi-target prediction problem:

1. instances, targets and the score the quantifies their relationship.
2. the type of the score value.
3. any side-information (features) that might be available for the instances.
4. any side-information (features) that might be available for the targets.
5. the test set contains novel instances, never before seen in the training set.
6. the test set contains novel targets, never before seen in the training set.

## How does DeepMTP work??
The DeepMTP framework is based on a flexible two branch neural network architecture that can be adapted to account for specific needs of the different MTP problem settings. The two branches are designed to take as input any available side information (features) for the instances and targets and then output two embedding vectors $\mathbf{p_x}$ and $\mathbf{q_t}$, respectively. The embedding can then be concatenated and passed through a series of fully-connected layers with a single output node (predicting the score of the instance-target pair). Alternatively, a more straightforward and less expensive approach replaces the series of fully-connected layers with a simple dot-product. In terms of the sizes allowed for the two embedding vectors $\mathbf{p_x}$ and $\mathbf{q_t}$, the MLP version allows for different sizes and the dot-product version requires the same size.  

<p align="center"><img src="images/mlp_plus_dot_product_white.png#gh-dark-mode-only" alt="logo" height="250"/></p>
<p align="center"><img src="images/mlp_plus_dot_product.png#gh-light-mode-only" alt="logo" height="250"/></p>

To better explain how the neural networks adapts to different cases, we will show different versions of the same general task, the prediction of interactions between chemical compounds and protein targets.

### Handling missing features for instances and/or targets
<details>
<summary>Click to expand!</summary>

1. In the first example, the user provides features for the proteins but not for the chemical compounds. In this case, the first branch uses the side information for the proteins and the second branch uses one-hot encoded features for the chemical compounds. The interaction matrix is populated with real values, so this is considered a regression task.

<p align="center"><img src="images/intro_instance_features_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/intro_instance_features.png#gh-light-mode-only" alt="logo" height="450"/></p>

2. In the second example, only the side information for the proteins is available. This can be seen as the reverse of the previous example, so following the same procedure, first branch uses one-hot encoded features and the second branch the actuall compound features.

<p align="center"><img src="images/intro_target_features_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/intro_target_features.png#gh-light-mode-only" alt="logo" height="450"/></p>

3. In the third example, side information is provided for both proteins and compounds, so both branches can utilize it.

<p align="center"><img src="images/intro_both_instance_and_target_features_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/intro_both_instance_and_target_features.png#gh-light-mode-only" alt="logo" height="450"/></p>

4. In the fourth and final example of this subsection, we are missing features for both instances and targets. This is not a realistic setting in our compound-protein interaction prediction task but has many applications in the area of recommender systems. In terms of the neural network, one-hot encoded vectors are used for both branches.

<p align="center"><img src="images/intro_no_instance_or_target_features_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/intro_no_instance_or_target_features.png#gh-light-mode-only" alt="logo" height="450"/></p>

</details>

### Handling different types of input features

<details>
<summary>Click to expand!</summary>

In the current state of machine learning, researchers try to extract useful information from different types of data. In the area of neural networks, when tabular data is available a series of fully-connected layers is common choise. The same can't be said for other types of inputs. In the area of image processing for example, convolutional neural networks are able to utilize images. The networks inside the two branches of the DeepMTP framework can use different types of sub-architectures to better handle different types of inputs. In the example below, we assume that protein features are in the form of standard vectors and the compounds are represented by their 2d images. DeepMTP adapts by using a fully connected neural network in the first branch and a convolutional neural network in the second branch.

<p align="center"><img src="images/intro_different_feature_types_white.png#gh-dark-mode-only" alt="logo" height="450"/></p>
<p align="center"><img src="images/intro_different_feature_types.png#gh-light-mode-only" alt="logo" height="450"/></p>
</details>

### Handling different validation settings

<details>
<summary>Click to expand!</summary>

All the previous examples and figures show only the training set. To show what happens while testing we will introduce 4 different cases (called validation settings) that are possible across the MTP problem settings. 

1. Setting A: Completing the missing values in the interaction matrix

In setting A the test set contains a subset of the instances and targets that we observe in the training set. This setting is usually selected when the interaction matrix contains missing values and becomes the only validation choice when instance and target features are not available.

<p align="center"><img src="images/intro_setting_A_white.png#gh-dark-mode-only" alt="logo" height="300"/></p>
<p align="center"><img src="images/intro_setting_A.png#gh-light-mode-only" alt="logo" height="300"/></p>

2. Setting B: predict for novel instances

In setting B the test set contains instances never before observed in the training set. This setting is the default option for popular MTP problem settings like multi-label classification and multivariate regression. ***In order to generalize to new instances, their side information has to be provided!***

<p align="center"><img src="images/intro_setting_B_white.png#gh-dark-mode-only" alt="logo" height="300"/></p>
<p align="center"><img src="images/intro_setting_B.png#gh-light-mode-only" alt="logo" height="300"/></p>

3. Setting C: predict for novel targets

In setting C the test set contains targets never before observed in the training set. This setting can be seen as the reverse of Setting B, as we can easily switch the instances and targets and arrive in Setting C. ***In order to generalize to new targets, their side information has to be provided!***

<p align="center"><img src="images/intro_setting_C_white.png#gh-dark-mode-only" alt="logo" height="300"/></p>
<p align="center"><img src="images/intro_setting_C.png#gh-light-mode-only" alt="logo" height="300"/></p>

2. Setting D: predict for pairs of novel instances and targets

Finally, in setting D the test set contains pairs of novel instances and targets never before observed in the training set. This is usually considered the most difficult generalization task compared to the others. ***In order to generalize to pairs of new instances and targets, the side information for both has to be provided!***

<p align="center"><img src="images/intro_setting_D_white.png#gh-dark-mode-only" alt="logo" height="300"/></p>
<p align="center"><img src="images/intro_setting_D.png#gh-light-mode-only" alt="logo" height="300"/></p>

</details>

## A few lines of code is all you need

```python
from DeepMTP.dataset import load_process_MLC
from DeepMTP.main import DeepMTP
from DeepMTP.utils.utils import generate_config

# load dataset
data = load_process_MLC(dataset_name='yeast', variant='undivided', features_type='numpy')
# process and split
train, val, test, data_info = data_process(data, validation_setting='B', verbose=True)

# generate a configuration for the experiment
config = generate_config(...)

# initialize model
model = DeepMTP(config)
# train, validate, test
validation_results = model.train(train, val, test)

# generate predictions from the trained model
results, preds = model.predict(train, return_predictions=True ,verbose=True)
```

### Configuration options
In the code snippet above the function generate_config is shown without any specific parameters. In practive, the function offers many parameters that define multiple characteristics of the architecture of the two-branch neural network, aspects of training, validating, testing etc. The following section can be used as a cheatsheet for users, explaining the meaning and rationale of every parameter.

| Parameter name  | Description |
| :--- | :--- |
| **Training** ||
| `num_epochs` | The max number of epochs allowed for training |
| `learning_rate` | The learning rate used to determine the step size at each iteration of the optimization process|
| `decay` | The weight decay (L2 penalty) used by the Adam optimizer|
| `compute_mode` | The device that is going to be used to actually train the neural network. The valid options are `cpu` if the user wants to train slowly or `cuda:id` if the user wants to train on the `id` gpu of the system|
| `num_workers` | The number of sub-processes to use for data loading. Larger values usually improve performance but after a point training speed will become worse|
| `train_batchsize` | The number of samples that comprise a batch from the training set |
| `val_batchsize` | The number of samples that comprise a batch from the validation and test sets |
| `patience` | The number of epochs that the network is allowed to continue training for while observing worse overall performance |
| `return_results_per_target` | Whether or not to returne the performance for every target separately |
| `evaluate_train` | Whether or not to calculate performance metrics over the training set |
| `evaluate_val` | Whether or not to calculate performance metrics over the validation set |
| `use_early_stopping` | Whether or not to use early stopping while training |
| **Metrics** ||
| `metrics` | The performance metrics that will be calculated. For classification tasks the available metrics are `['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision']` while for regression tasks the available metrics are `['RMSE', 'MSE', 'MAE', 'R2', 'RRMSE']` |
| `metrics_average` | The averaging strategy that will be used to calculate the metric. The available options are ['macro', 'micro', 'instance'] |
| `metric_to_optimize_early_stopping` | The metric that will be used for tracking by the early stopping routine. The value can be the `loss` or one of the available performance metrics. |
| `metric_to_optimize_best_epoch_selection` | The validation metric that will be used to determine the best configuration. The value can be the `loss` or one of the available performance metrics. |
| **Printing - Saving - Logging** ||
| `verbose` | Whether or not to print useful in the terminal |
| `use_tensorboard_logger` | Whether or not to log results in weights and biases |
| `wandb_project_name` | Defines the name of the wandb project that the results of an experiment will be logged (Will be used if `use_tensorboard_logger==True`) |
| `wandb_project_entity` | Defines the user name of the wandb account (Will be used if `use_tensorboard_logger==True`) |
| `results_path` | Defines the path the all relevant information will be saved to |
| `experiment_name` | Defines the name of the current experiment. This name will be used to local save and the wandb save |
| `save_model` | Whether or not to save the model of the epoch with the best validation performance |
| **General architecture architecture** ||
| `enable_dot_product_version` | Whether or not to use the dot-product version of the two branch architecture. In the dot product version, the two embeddings are used to calculate the dot product. If the value is False, the two embeddings are first concatenated and then passed to another series of fully connected layers |
| `batch_norm` | The option to use batch normalization between the fully connected layers in the two branches |
| `dropout_rate` | The amount of dropout used in the layers of the two branches |
| **Instance branch architecture** ||
| `instance_branch_architecture` | The type of architecture that will be used in the instance branch. Currently, there are two available options, `MLP`: a basic fully connected feed-forward neural network is used, `CONV` a convolutional neural network is used |
| `instance_branch_input_dim` | The input dimension of the instance branch |
| `instance_branch_nodes_per_layer` | Defines the number of nodes in the `MLP` version of the instance branch.  if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used `instance_branch_layers` times|
| `instance_branch_layers` | The number of layers in the MLP version of the instance branch. (Only used if `instance_branch_nodes_per_layer` is int) |
| `instance_train_transforms` | The Pytorch compatible transforms that can be used on the training samples. Useful when using images with convolutional architectures |
| `instance_inference_transforms` | The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures |
| `instance_branch_conv_architecture` | The type of the convolutional architecture that is used in the instance branch. |
| `instance_branch_conv_architecture_version` | The version of the specific type of convolutional architecture that is used in the instance branch. |
| `instance_branch_conv_architecture_dense_layers` | The number of dense layers that are used at the end of the convolutional architecture of the instance branch |
| `instance_branch_conv_architecture_last_layer_trained` | When using pre-trained architectures, the user can define that last layer that will be frozen during training |
|  **Target branch architecture**  ||
| `target_branch_architecture` | The type of architecture that will be used in the target branch. Currently, there are two available options, `MLP`: a basic fully connected feed-forward neural network is used, `CONV` a convolutional neural network is used |
| `target_branch_input_dim` | The input dimension of the target branch |
| `target_branch_nodes_per_layer` | Defines the number of nodes in the `MLP` version of the target branch.  if list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used `target_branch_layers` times|
| `target_branch_layers` | The number of layers in the MLP version of the target branch. (Only used if `target_branch_nodes_per_layer` is int) |
| `target_train_transforms` | The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures |
| `target_inference_transforms` | The Pytorch compatible transforms that can be used on the validation and test samples. Useful when using images with convolutional architectures |
| `target_branch_conv_architecture` | The type of the convolutional architecture that is used in the target branch. |
| `target_branch_conv_architecture_version` | The version of the specific type of convolutional architecture that is used in the target branch. |
| `target_branch_conv_architecture_dense_layers` | The number of dense layers that are used at the end of the convolutional architecture of the target branch |
| `target_branch_conv_architecture_last_layer_trained` | When using pre-trained architectures, the user can define that last layer that will be frozen during training |
|  **Combination branch architecture**  ||
| `comb_mlp_nodes_per_layer` |  Defines the number of nodes in the combination branch. If list, each element defines the number of nodes in the corresponding layer. If int, the same number of nodes is used 'comb_mlp_branch_layers' times. (Only used if `enable_dot_product_version == False`)|
| `comb_mlp_branch_layers` | The number of layers in the combination branch. (Only used if `enable_dot_product_version == False`) |
| `embedding_size` | The size of the embeddings outputted by the two branches. (Only used if `enable_dot_product_version == False`) |
|  **Pretrained models**  ||
| `eval_every_n_epochs` | The interval that indicates when the performance metrics are computed |
| `load_pretrained_model` | Whether or not a pretrained model will be loaded |
| `pretrained_model_path` | The path to the .pt file with the pretrained model (Only used if `load_pretrained_model == True`) |
|  **Other**  ||
| `additional_info` | A dictionary that holds all other relevant info. Can be used as log adittional info for an experiment in wandb |
| `validation_setting` | The validation setting of the specific example |



## Cite Us
If you use this package, please cite [our paper](https://link.springer.com/article/10.1007/s10994-021-06104-5):
```
@article{iliadis2022multi,
  title={Multi-target prediction for dummies using two-branch neural networks},
  author={Iliadis, Dimitrios and De Baets, Bernard and Waegeman, Willem},
  journal={Machine Learning},
  pages={1--34},
  year={2022},
  publisher={Springer}
}
```
