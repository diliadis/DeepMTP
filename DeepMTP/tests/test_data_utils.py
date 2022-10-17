from DeepMTP.utils.data_utils import process_interaction_data, check_interaction_files_format, check_interaction_files_column_type_format, check_variable_type
from DeepMTP.dataset import process_dummy_MLC, process_dummy_MTR
import pandas as pd
import numpy as np
import pytest


data_format = ['numpy', 'dataframe']

@pytest.mark.parametrize('data_format', data_format)
def test_process_interaction_data(data_format):
	num_instances = 1000
	num_targets = 100
	num_instance_features = 2 

	data = process_dummy_MLC(num_features=num_instance_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format=data_format)
	info = process_interaction_data(data['train']['y'], verbose=False)
	
	assert info['original_format'] == 'numpy' if data_format == 'numpy' else 'triplets'
	assert info['instance_id_type'] == 'int'
	assert info['target_id_type'] == 'int'
	
	if data_format == 'dataframe':
		assert data['train']['y'].equals(info['data'])
	else:
		triplets = [(i, j, data['train']['y'][i, j]) for i in range(data['train']['y'].shape[0]) for j in range(data['train']['y'].shape[1])]
		temp_df = pd.DataFrame(triplets, columns=['instance_id', 'target_id', 'value'])
		assert temp_df.equals(info['data'])


interaction_files_format_check_should_throw_error = [True, False]

@pytest.mark.parametrize('interaction_files_format_check_should_throw_error', interaction_files_format_check_should_throw_error)
def test_check_interaction_files_format(interaction_files_format_check_should_throw_error):
	data = {}
	if interaction_files_format_check_should_throw_error:
		data['train'] = {'y': {'original_format': 'triplets'}}
		data['val'] = {'y': {'original_format': 'numpy'}}
		data['test'] = {'y': {'original_format': 'triplets'}}
		with pytest.raises(Exception):
			check_interaction_files_format(data)
	else:
		data['train'] = {'y': {'original_format': 'triplets'}}
		data['val'] = {'y': {'original_format': 'triplets'}}
		data['test'] = {'y': {'original_format': 'triplets'}}
		try:
			check_interaction_files_format(data)
		except Exception as exc:
			assert False
   
check_interaction_files_column_type_format_data = [{
	'pass': {'train': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}, 'val': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}, 'test': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}},
	'pass': {'train': {'y': {'instance_id_type': 'str', 'target_id_type': 'int'}}, 'val': {'y': {'instance_id_type': 'str', 'target_id_type': 'int'}}, 'test': {'y': {'instance_id_type': 'str', 'target_id_type': 'int'}}},
	'fail': {'train': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}, 'val': {'y': {'instance_id_type': 'str', 'target_id_type': 'int'}}, 'test': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}},
	'fail': {'train': {'y': {'instance_id_type': 'int', 'target_id_type': 'str'}}, 'val': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}, 'test': {'y': {'instance_id_type': 'int', 'target_id_type': 'int'}}},
	}]

@pytest.mark.parametrize('check_interaction_files_column_type_format_data', check_interaction_files_column_type_format_data)
def test_check_interaction_files_column_type_format(check_interaction_files_column_type_format_data):
	for pass_fail, data in check_interaction_files_column_type_format_data.items():
		if pass_fail == 'pass':
			try:
				check_interaction_files_column_type_format(data)
			except Exception as exc:
				assert False
		else:
			with pytest.raises(Exception):
				check_interaction_files_format(data)

data_type = ['classification', 'regression']

@pytest.mark.parametrize('data_type', data_type)
def test_check_variable_type(data_type):
	num_instances = 1000
	num_targets = 100
	num_instance_features = 2 
 
	if data_type == 'classification':
		data = process_dummy_MLC(num_features=num_instance_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format='dataframe')
	elif data_type == 'regression':
		data = process_dummy_MTR(num_features=num_instance_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format='dataframe')
	else:
		raise Exception('Something went wrong. test_check_variable_type accepts only two values: '+str(['classification', 'regression']))
	
	if data_type == 'classification':
		assert check_variable_type(data['train']['y']) == 'binary'
	elif data_type == 'regression':
		assert check_variable_type(data['train']['y']) == 'real-valued'
