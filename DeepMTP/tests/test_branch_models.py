import pytest
from torch import nn
from DeepMTP.branch_models import MLP

test_MLP_data = [
    {'input_dim': 10, 'output_dim': 15, 'nodes_per_layer': 100, 'num_layers': 2, 'dropout': 0, 'batch_norm': False},
    {'input_dim': 10, 'output_dim': 15, 'nodes_per_layer': 123, 'num_layers': 2, 'dropout': 0, 'batch_norm': False},
    {'input_dim': 10, 'output_dim': 15, 'nodes_per_layer': [100, 200, 300], 'num_layers': 2, 'dropout': 0, 'batch_norm': False},
]


@pytest.mark.parametrize('test_MLP_data', test_MLP_data)
def test_MLP(test_MLP_data):
    calculated_nodes_per_layer = []
    
    model = MLP({}, input_dim=test_MLP_data['input_dim'], output_dim=test_MLP_data['output_dim'], nodes_per_layer=test_MLP_data['nodes_per_layer'], num_layers=test_MLP_data['num_layers'], dropout_rate=test_MLP_data['dropout'], batch_norm=test_MLP_data['batch_norm'])
    linear_layers = [layer for layer in model[0] if isinstance(layer, nn.Linear)]    
    
    calculated_nodes_per_layer.append(test_MLP_data['input_dim'])
    if not isinstance(test_MLP_data['nodes_per_layer'], list):
        calculated_nodes_per_layer.extend([test_MLP_data['nodes_per_layer']] * test_MLP_data['num_layers'])
    calculated_nodes_per_layer.append(test_MLP_data['output_dim'])
    
    for idx, layer in enumerate(linear_layers):
        assert layer.in_features == calculated_nodes_per_layer[idx]
        assert layer.out_features == calculated_nodes_per_layer[idx+1]
    
    
    