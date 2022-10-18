from DeepMTP.utils.data_utils import (
	process_interaction_data, 
	check_interaction_files_format, 
	check_interaction_files_column_type_format, 
	check_variable_type, 
 	check_target_variable_type, 
	check_novel_instances, 
	check_novel_targets, 
	get_estimated_validation_setting, 
	process_instance_features, 
	process_target_features, 
	cross_input_consistency_check_instances, 
	cross_input_consistency_check_targets
)
from DeepMTP.dataset import process_dummy_MLC, process_dummy_MTR, process_dummy_DP
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
				check_interaction_files_column_type_format(data)

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


classification_df = pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]})
regression_df = pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0.1, 0.2, 0.3, 0.4]})
check_target_variable_type_data = [{
	'pass': {'train': {'y': {'data': classification_df}}, 'val': {'y': {'data': classification_df}}, 'test': {'y': {'data': classification_df}}, 'mode': 'binary'},
	'pass': {'train': {'y': {'data': regression_df}}, 'val': {'y': {'data': regression_df}}, 'test': {'y': {'data': regression_df}}, 'mode': 'real-valued'},
	'fail': {'train': {'y': {'data': classification_df}}, 'val': {'y': {'data': regression_df}}, 'test': {'y': {'data': classification_df}}},
}]

@pytest.mark.parametrize('check_target_variable_type_data', check_target_variable_type_data)
def test_check_target_variable_type(check_target_variable_type_data):
	for pass_fail, data in check_target_variable_type_data.items():
		if pass_fail == 'pass':
			assert check_target_variable_type(data) == data['mode']
		else:
			with pytest.raises(Exception):
				check_target_variable_type(data)


check_novel_instances_data = [{
	'true': {'train': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}, 'test': {'data': pd.DataFrame({'instance_id': [4, 5, 6, 7], 'target_id': [0, 1, 2, 3], 'value': [0, 0, 0, 1]}), 'original_format': 'triplets'}},
	'false': {'train': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}, 'test': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}},

}]

@pytest.mark.parametrize('check_novel_instances_data', check_novel_instances_data)
def test_check_novel_instances(check_novel_instances_data):
	for true_false, data in check_novel_instances_data.items():
		if true_false == 'true':
			assert True == check_novel_instances(data['train'], data['test'])
		elif true_false == 'false':
			assert False == check_novel_instances(data['train'], data['test'])
   
   
check_novel_target_data = [{
	'true': {'train': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}, 'test': {'data': pd.DataFrame({'instance_id': [4, 5, 6, 7], 'target_id': [4, 5, 6, 7], 'value': [0, 0, 0, 1]}), 'original_format': 'triplets'}},
	'false': {'train': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}, 'test': {'data': pd.DataFrame({'instance_id': [0, 1, 2, 3], 'target_id': [0, 1, 2, 3], 'value': [0, 1, 0, 1]}), 'original_format': 'triplets'}},

}]

@pytest.mark.parametrize('check_novel_target_data', check_novel_target_data)
def test_check_novel_targets(check_novel_target_data):
	for true_false, data in check_novel_target_data.items():
		if true_false == 'true':
			assert True == check_novel_targets(data['train'], data['test'])
		elif true_false == 'false':
			assert False == check_novel_targets(data['train'], data['test'])
   

get_estimated_validation_setting_data = [{
	(True, True): 'D',
	(True, False): 'B',
	(False, True): 'C',
	(False, False): 'A',
	}]

@pytest.mark.parametrize('get_estimated_validation_setting_data', get_estimated_validation_setting_data)
def test_get_estimated_validation_setting(get_estimated_validation_setting_data):
	 for novel_instance_targets_tuple, true_validation_setting in get_estimated_validation_setting_data.items():
		 assert get_estimated_validation_setting(novel_instance_targets_tuple[0], novel_instance_targets_tuple[1], verbose=False) == true_validation_setting


@pytest.mark.parametrize('data_format', data_format)
def test_process_instance_features(data_format):
	num_instances = 1000
	num_targets = 100
	num_instance_features = 10

	data = process_dummy_MLC(num_features=num_instance_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format='numpy', features_format=data_format)
	if data_format == 'numpy':
		original_instance_features = pd.DataFrame(np.arange(len(data['train']['X_instance'])), columns=['id'])
		original_instance_features['features'] = [r for r in data['train']['X_instance']]
  
	instance_features = process_instance_features(data['train']['X_instance'], verbose=False)
 
	assert instance_features['num_features'] == num_instance_features
	assert instance_features['info'] == data_format
 
	if data_format == 'numpy':
		assert original_instance_features.equals(instance_features['data'])
	else:
		assert data['train']['X_instance'].equals(instance_features['data'])
  

@pytest.mark.parametrize('data_format', data_format)
def test_process_target_features(data_format):
	num_instances = 1000
	num_targets = 100
	num_instance_features = 20
	num_target_features = 10

	data = process_dummy_DP(num_instance_features=num_instance_features, num_target_features=num_target_features, num_instances=num_instances, num_targets=num_targets, interaction_matrix_format='numpy', instance_features_format='numpy', target_features_format=data_format)
	if data_format == 'numpy':
		original_target_features = pd.DataFrame(np.arange(len(data['train']['X_target'])), columns=['id'])
		original_target_features['features'] = [r for r in data['train']['X_target']]
  
	target_features = process_target_features(data['train']['X_target'], verbose=False)
 
	assert target_features['num_features'] == num_target_features
	assert target_features['info'] == data_format
 
	if data_format == 'numpy':
		assert original_target_features.equals(target_features['data'])
	else:
		assert data['train']['X_target'].equals(target_features['data'])


test_cross_input_consistency_check_instances_data = [{
	'pass': {
	 	'train': {
			'y': {'data': pd.DataFrame({'instance_id': [0, 0, 1, 1, 2, 2, 3, 3], 'target_id': [0, 1, 0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [0, 1, 2, 3], 'features': list(np.random.rand(4, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'val': {
			'y': {'data': pd.DataFrame({'instance_id': [4, 4, 5, 5], 'target_id': [0, 1, 0, 1], 'value': [0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [4, 5], 'features': list(np.random.rand(2, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'test': {
			'y': {'data': pd.DataFrame({'instance_id': [6, 6, 7, 7, 8, 8], 'target_id': [0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 0]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [6, 7, 8], 'features': list(np.random.rand(3, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
		'validation_setting': 'B'},
 
	'fail': { # the instance features in the validation set are less than what is required in the interaction matrix
	 	'train': {
			'y': {'data': pd.DataFrame({'instance_id': [0, 0, 1, 1, 2, 2, 3, 3], 'target_id': [0, 1, 0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [0, 1, 2, 3], 'features': list(np.random.rand(4, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'val': {
			'y': {'data': pd.DataFrame({'instance_id': [4, 4, 5, 5], 'target_id': [0, 1, 0, 1], 'value': [0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [4], 'features': list(np.random.rand(1, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'test': {
			'y': {'data': pd.DataFrame({'instance_id': [6, 6, 7, 7, 8, 8], 'target_id': [0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 0]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [6, 7, 8], 'features': list(np.random.rand(3, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
		'validation_setting': 'B'},
 
	'fail': { # 3 interaction files but 2 instance feature files (test instance features are missing)
	 	'train': {
			'y': {'data': pd.DataFrame({'instance_id': [0, 0, 1, 1, 2, 2, 3, 3], 'target_id': [0, 1, 0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [0, 1, 2, 3], 'features': list(np.random.rand(4, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'val': {
			'y': {'data': pd.DataFrame({'instance_id': [4, 4, 5, 5], 'target_id': [0, 1, 0, 1], 'value': [0, 1, 0, 1]}), 'original_format': 'numpy'},
			'X_instance': None,
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
	 	'test': {
			'y': {'data': pd.DataFrame({'instance_id': [6, 6, 7, 7, 8, 8], 'target_id': [0, 1, 0, 1, 0, 1], 'value': [0, 1, 0, 1, 0, 0]}), 'original_format': 'numpy'},
			'X_instance': {'data': pd.DataFrame({'id': [6, 7, 8], 'features': list(np.random.rand(3, 10))})},
		 	'X_target': {'data': pd.DataFrame({'id': [0, 1], 'features': list(np.random.rand(2, 10))})},
		 },
		'validation_setting': 'B'},
}]

@pytest.mark.parametrize('test_cross_input_consistency_check_instances_data', test_cross_input_consistency_check_instances_data)
def test_cross_input_consistency_check_instances(test_cross_input_consistency_check_instances_data):
	
	for pass_fail, data in test_cross_input_consistency_check_instances_data.items():
		if pass_fail  == 'pass':	
			try:
				cross_input_consistency_check_instances(data, data['validation_setting'], verbose=False)
			except Exception as exc:
				assert False
		else:
			with pytest.raises(Exception):
				cross_input_consistency_check_instances(data, data['validation_setting'], verbose=False)