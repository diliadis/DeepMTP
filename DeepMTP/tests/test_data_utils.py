from DeepMTP.utils.data_utils import process_interaction_data
from DeepMTP.dataset import process_dummy_MLC
import pandas as pd
import numpy as np


data_format = ['numpy', 'dataframe']

@pytest.mark.parametrize('data_format', data_format)
def test_process_interaction_data(data_format):
	num_instances = 1000
	num_targets = 100
	num_instance_features = 2 

	data = process_dummy_MLC(num_features=num_instance_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format=data_format)
	info = process_interaction_data(data['train']['y'], verbose=False)
	
	assert info['original_format'] == data_format
	assert info['instance_id_type'] == 'int'
	assert info['target_id_type'] == 'int'
	
	if data_format == 'dataframe':
		assert data['train']['y'].equals(info['data'])
	else:
		triplets = [(i, j, data['train']['y'][i, j]) for i in range(data['train']['y'].shape[0]) for j in range(data['train']['y'].shape[1])]
		temp_df = pd.DataFrame(triplets, columns=['instance_id', 'target_id', 'value'])
		assert temp_df.equals(info['data'])