import pytest
import torch
from DeepMTP.hpo_worker import BaseWorker
from DeepMTP.dataset import load_process_MLC
from DeepMTP.utils.data_utils import data_process
'''
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
'''
def test_Baseworker():
    
    data = load_process_MLC(dataset_name='emotions', variant='divided', features_type='numpy', print_mode='basic')
    train, val, test, data_info = data_process(data, validation_setting=None, split_method='random', ratio={'train': 0.7, 'test': 0.2, 'val': 0.1}, shuffle=True, seed=42, verbose=False, print_mode='basic', scale_instance_features=None, scale_target_features=None)
    
    config = {    
    'hpo_results_path': './hyperband/',
    'instance_branch_input_dim': data_info['instance_branch_input_dim'],
    'target_branch_input_dim': data_info['target_branch_input_dim'],
    'validation_setting': data_info['detected_validation_setting'],
    'general_architecture_version': 'dot_product',
    'problem_mode': data_info['detected_problem_mode'],
    'compute_mode': 'cuda:0',
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
    'wandb_project_name': None,
    'wandb_project_entity': None,
    'metric_to_optimize_early_stopping': 'loss',
    'metric_to_optimize_best_epoch_selection': 'loss',
    'instance_branch_architecture': 'MLP',
    'target_branch_architecture': 'MLP',
    'save_model': True,
    'eval_every_n_epochs': 10,
    'additional_info': {'eta': 3, 'max_budget': 9}
    }
    
    worker = BaseWorker(train, val, test, data_info, config, 'loss', mode='standard')
    
    assert worker.train == train
    assert worker.val == val
    assert worker.test == test
    
    
