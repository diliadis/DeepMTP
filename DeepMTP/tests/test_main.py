import pytest
from torch import nn
import torch
from DeepMTP.branch_models import MLP
from DeepMTP.main import TwoBranchDotProductModel, TwoBranchMLPModel, TwoBranchKroneckerModel

test_TwoBranchDotProductModel_data = [
    {
        'pass_fail': True, 
        'instance_branch_input_size': 10, 
        'instance_branch_nodes_per_layer': [11, 12, 13],
        'target_branch_input_size': 20,
        'target_branch_nodes_per_layer': [21, 22, 23],
        'embedding_size': 100
    },
]


@pytest.mark.parametrize('test_TwoBranchDotProductModel_data', test_TwoBranchDotProductModel_data)
def test_TwoBranchDotProductModel(test_TwoBranchDotProductModel_data):

    instance_branch_model = MLP({},
                               input_dim=test_TwoBranchDotProductModel_data['instance_branch_input_size'], 
                               output_dim=test_TwoBranchDotProductModel_data['embedding_size'],
                               nodes_per_layer=test_TwoBranchDotProductModel_data['instance_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    target_branch_model = MLP({},
                               input_dim=test_TwoBranchDotProductModel_data['target_branch_input_size'], 
                               output_dim=test_TwoBranchDotProductModel_data['embedding_size'],
                               nodes_per_layer=test_TwoBranchDotProductModel_data['target_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    model = TwoBranchDotProductModel({}, instance_branch_model, target_branch_model)
    instance_features = torch.rand(1, test_TwoBranchDotProductModel_data['instance_branch_input_size'])
    target_features = torch.rand(1, test_TwoBranchDotProductModel_data['target_branch_input_size'])
    output = model.forward(instance_features, target_features)
    assert len(output) == 1


test_TwoBranchMLPModel_data = [
    {
        'pass_fail': True, 
        'instance_branch_input_size': 10, 
        'instance_branch_nodes_per_layer': [11, 12, 13],
        'target_branch_input_size': 20,
        'target_branch_nodes_per_layer': [21, 22, 23],
        'comb_mlp_nodes_per_layer': [10, 20, 30],
    },
]


@pytest.mark.parametrize('test_TwoBranchMLPModel_data', test_TwoBranchMLPModel_data)
def test_TwoBranchMLPModel(test_TwoBranchMLPModel_data):

    instance_branch_model = MLP({},
                               input_dim=test_TwoBranchMLPModel_data['instance_branch_input_size'], 
                               output_dim=test_TwoBranchMLPModel_data['instance_branch_nodes_per_layer'][-1],
                               nodes_per_layer=test_TwoBranchMLPModel_data['instance_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    target_branch_model = MLP({},
                               input_dim=test_TwoBranchMLPModel_data['target_branch_input_size'], 
                               output_dim=test_TwoBranchMLPModel_data['target_branch_nodes_per_layer'][-1],
                               nodes_per_layer=test_TwoBranchMLPModel_data['target_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    model = TwoBranchMLPModel(
        {
            'comb_mlp_nodes_per_layer': test_TwoBranchMLPModel_data['comb_mlp_nodes_per_layer'],
            'comb_mlp_layers': 2,
            'dropout_rate': 0,
            'batch_norm': False,
        }, 
        instance_branch_model,
        target_branch_model)
    
    instance_features = torch.rand(1, test_TwoBranchMLPModel_data['instance_branch_input_size'])
    target_features = torch.rand(1, test_TwoBranchMLPModel_data['target_branch_input_size'])
    output = model.forward(instance_features, target_features)
    assert len(output) == 1
    
    
test_TwoBranchKroneckerModel_data = [
    {
        'pass_fail': True, 
        'instance_branch_input_size': 10, 
        'instance_branch_nodes_per_layer': [11, 12, 13],
        'target_branch_input_size': 20,
        'target_branch_nodes_per_layer': [21, 22, 23],
        'embedding_size': 100
    },
]


@pytest.mark.parametrize('test_TwoBranchKroneckerModel_data', test_TwoBranchKroneckerModel_data)
def test_TwoBranchKroneckerModel(test_TwoBranchKroneckerModel_data):

    instance_branch_model = MLP({},
                               input_dim=test_TwoBranchKroneckerModel_data['instance_branch_input_size'], 
                               output_dim=test_TwoBranchKroneckerModel_data['embedding_size'],
                               nodes_per_layer=test_TwoBranchKroneckerModel_data['instance_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    target_branch_model = MLP({},
                               input_dim=test_TwoBranchKroneckerModel_data['target_branch_input_size'], 
                               output_dim=test_TwoBranchKroneckerModel_data['embedding_size'],
                               nodes_per_layer=test_TwoBranchKroneckerModel_data['target_branch_nodes_per_layer'],
                               num_layers=2,
                               dropout_rate=0,
                               batch_norm=False
    )
    
    model = TwoBranchKroneckerModel({}, instance_branch_model, target_branch_model)
    instance_features = torch.rand(1, test_TwoBranchKroneckerModel_data['instance_branch_input_size'])
    target_features = torch.rand(1, test_TwoBranchKroneckerModel_data['target_branch_input_size'])
    output = model.forward(instance_features, target_features)
    assert len(output) == 1