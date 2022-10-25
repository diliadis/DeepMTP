from DeepMTP.dataset import load_process_MLC, load_process_MTR, load_process_DP, load_process_MC, load_process_MTL
import pytest

test_load_process_MLC_data = [
	{'pass_fail': 'pass', 'dataset_name': 'bibtex', 'variant': 'undivided','features_type': 'numpy'},
	{'pass_fail': 'pass', 'dataset_name': 'bibtex', 'variant': 'divided','features_type': 'dataframe'},
	{'pass_fail': 'pass', 'dataset_name': 'bibtex', 'variant': 'undivided','features_type': 'dataframe'},
	{'pass_fail': 'fail', 'dataset_name': 'bibtex', 'variant': 'divided','features_type': 'lalala'},
	{'pass_fail': 'fail', 'dataset_name': 'bibtex', 'variant': 'lalala','features_type': 'dataframe'},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala', 'variant': 'undivided','features_type': 'numpy'},
]

@pytest.mark.parametrize('test_load_process_MLC_data', test_load_process_MLC_data)
def test_load_process_MLC(test_load_process_MLC_data):
	pass_fail = test_load_process_MLC_data['pass_fail']
	variant = test_load_process_MLC_data['variant']
	features_type = test_load_process_MLC_data['features_type']
	dataset_name = test_load_process_MLC_data['dataset_name']
	
	if pass_fail == 'fail':
		with pytest.raises(Exception):
			data = load_process_MLC(path='./data', dataset_name=dataset_name, variant=variant, features_type=features_type, print_mode='basic')
	else:
		try:
			data = load_process_MLC(path='./data', dataset_name=dataset_name, variant=variant, features_type=features_type, print_mode='basic')
			if variant == 'undivided':
				assert isinstance(data['val']['y'] , type(None))
				assert isinstance(data['val']['X_instance'] , type(None))
				assert isinstance(data['val']['X_target'] , type(None))
				assert isinstance(data['test']['y'] , type(None))
				assert isinstance(data['test']['X_instance'] , type(None))
				assert isinstance(data['test']['X_target'] , type(None))
				assert isinstance(data['train']['X_target'] , type(None))
			
			else:
				assert isinstance(data['train']['X_target'] , type(None))
				assert isinstance(data['val']['y'] , type(None))
				assert isinstance(data['val']['X_instance'] , type(None))
				assert isinstance(data['val']['X_target'] , type(None))
				assert isinstance(data['test']['X_target'] , type(None))

			if features_type == 'numpy':
				assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
				if variant != 'undivided':
					assert data['test']['y'].shape[0] == data['test']['X_instance'].shape[0]
					
			else:
				assert data['train']['y'].shape[0] == len(data['train']['X_instance'])
				if variant != 'undivided':
					assert data['test']['y'].shape[0] == len(data['test']['X_instance'])
		except Exception as exc:
			assert False


test_load_process_MTR_data = [
	{'pass_fail': 'pass', 'dataset_name': 'enb','features_type': 'numpy'},
	{'pass_fail': 'pass', 'dataset_name': 'enb', 'features_type': 'dataframe'},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala','features_type': 'numpy'},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala','features_type': 'lalala'},
]

@pytest.mark.parametrize('test_load_process_MTR_data', test_load_process_MTR_data)    
def test_load_process_MTR(test_load_process_MTR_data):
	pass_fail = test_load_process_MTR_data['pass_fail']
	features_type = test_load_process_MTR_data['features_type']
	dataset_name = test_load_process_MTR_data['dataset_name']
	
	if pass_fail == 'fail':
		with pytest.raises(Exception):
			data = load_process_MTR(path='./data', dataset_name=dataset_name, features_type=features_type, print_mode='basic')
	else:
		try:
			data = load_process_MTR(path='./data', dataset_name=dataset_name, features_type=features_type, print_mode='basic')
			assert isinstance(data['val']['y'] , type(None))
			assert isinstance(data['val']['X_instance'] , type(None))
			assert isinstance(data['val']['X_target'] , type(None))
			assert isinstance(data['test']['y'] , type(None))
			assert isinstance(data['test']['X_instance'] , type(None))
			assert isinstance(data['test']['X_target'] , type(None))
			assert isinstance(data['train']['X_target'] , type(None))

			if features_type == 'numpy':
				assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
			else:
				assert data['train']['y'].shape[0] == len(data['train']['X_instance'])
		except Exception as exc:
			assert False
			

test_load_process_DP_data = [
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided', 'validation_setting': 'B', 'split_instance_features': True, 'split_target_features': False},
 	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided', 'validation_setting': 'B', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'undivided', 'validation_setting': 'B', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided','validation_setting': 'C', 'split_instance_features': False, 'split_target_features': True},
 	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided','validation_setting': 'C', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'undivided', 'validation_setting': 'C', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'undivided', 'validation_setting': 'C', 'split_instance_features': False, 'split_target_features': True},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided','validation_setting': 'D', 'split_instance_features': True, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'undivided', 'validation_setting': 'D', 'split_instance_features': True, 'split_target_features': True},
 	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'divided','validation_setting': 'D', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'pass', 'dataset_name': 'ern', 'variant': 'undivided', 'validation_setting': 'D', 'split_instance_features': False, 'split_target_features': True},
	{'pass_fail': 'pass', 'dataset_name': 'dpii', 'variant': 'divided', 'validation_setting': 'B', 'split_instance_features': True, 'split_target_features': False},
 
	{'pass_fail': 'fail', 'dataset_name': 'lalalala', 'variant': 'divided', 'validation_setting': 'B', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala', 'variant': 'divided', 'validation_setting': 'B', 'split_instance_features': False, 'split_target_features': False},
 	{'pass_fail': 'fail', 'dataset_name': 'lalalala', 'variant': 'divided', 'validation_setting': 'lalalala', 'split_instance_features': False, 'split_target_features': False},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala', 'variant': 'lalalala', 'validation_setting': 'B', 'split_instance_features': False, 'split_target_features': False},
]

@pytest.mark.parametrize('test_load_process_DP_data', test_load_process_DP_data)    
def test_load_process_DP(test_load_process_DP_data):
	pass_fail = test_load_process_DP_data['pass_fail']
	variant = test_load_process_DP_data['variant']
	validation_setting = test_load_process_DP_data['validation_setting']
	split_instance_features = test_load_process_DP_data['split_instance_features']
	split_target_features = test_load_process_DP_data['split_target_features']
 
	if pass_fail == 'fail':
		with pytest.raises(Exception): 
			data = load_process_DP(path='./data', dataset_name=test_load_process_DP_data['dataset_name'], variant=variant, random_state=42, split_ratio={'train': 0.7, 'val': 0.1, 'test': 0.2}, split_instance_features=split_instance_features, split_target_features=split_target_features, validation_setting=validation_setting, print_mode='basic')
	else:
		try:
			data = load_process_DP(path='./data', dataset_name=test_load_process_DP_data['dataset_name'], variant=variant, random_state=42, split_ratio={'train': 0.7, 'val': 0.1, 'test': 0.2}, split_instance_features=split_instance_features, split_target_features=split_target_features, validation_setting=validation_setting, print_mode='basic')
			if variant == 'undivided':
				assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
				assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
			else:
				if validation_setting == 'B':
					if split_instance_features:
						assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
						assert data['val']['y'].shape[0] == data['val']['X_instance'].shape[0]
						assert data['test']['y'].shape[0] == data['test']['X_instance'].shape[0]
						assert data['train']['y'].shape[1] == data['val']['y'].shape[1]
						assert data['train']['y'].shape[1] == data['test']['y'].shape[1]
					else:
						assert (data['train']['y'].shape[0] + data['val']['y'].shape[0] + data['test']['y'].shape[0]) == data['train']['X_instance'].shape[0]
					assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
					
				elif validation_setting == 'C':
					if split_target_features:
						assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
						assert data['val']['y'].shape[1] == data['val']['X_target'].shape[0]
						assert data['test']['y'].shape[1] == data['test']['X_target'].shape[0]
						assert data['train']['y'].shape[0] == data['val']['y'].shape[0]
						assert data['train']['y'].shape[0] == data['test']['y'].shape[0]
					else:
						assert (data['train']['y'].shape[1] + data['val']['y'].shape[1] + data['test']['y'].shape[1]) == data['train']['X_target'].shape[0]
					assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]

				elif validation_setting == 'D':
					if split_instance_features:
						assert data['train']['y'].shape[0] == data['train']['X_instance'].shape[0]
						assert data['val']['y'].shape[0] == data['val']['X_instance'].shape[0]
						assert data['test']['y'].shape[0] == data['test']['X_instance'].shape[0]
					else:
						assert (data['train']['y'].shape[0] + data['val']['y'].shape[0] + data['test']['y'].shape[0]) == data['train']['X_instance'].shape[0]
					
					if split_target_features:
						assert data['train']['y'].shape[1] == data['train']['X_target'].shape[0]
						assert data['val']['y'].shape[1] == data['val']['X_target'].shape[0]
						assert data['test']['y'].shape[1] == data['test']['X_target'].shape[0]
					else:
						assert (data['train']['y'].shape[1] + data['val']['y'].shape[1] + data['test']['y'].shape[1]) == data['train']['X_target'].shape[0]
		except Exception as exc:
			assert False

def test_load_process_MC():
	data = load_process_MC(path='./data', dataset_name='ml-100k', print_mode='basic')
	
	assert isinstance(data['train']['X_instance'] , type(None))
	assert isinstance(data['train']['X_target'] , type(None))
	assert isinstance(data['val']['y'] , type(None))
	assert isinstance(data['val']['X_instance'] , type(None))
	assert isinstance(data['val']['X_target'] , type(None))
	assert isinstance(data['test']['X_instance'] , type(None))
	assert isinstance(data['test']['X_target'] , type(None))
	assert isinstance(data['test']['y'] , type(None))


test_load_process_MTL_data = [
	{'pass_fail': 'pass', 'dataset_name': 'dog'},
	{'pass_fail': 'fail', 'dataset_name': 'lalalala'},
]

@pytest.mark.parametrize('test_load_process_MTL_data', test_load_process_MTL_data)   
def test_load_process_MTL(test_load_process_MTL_data):
	pass_fail = test_load_process_MTL_data['pass_fail']
	dataset_name = test_load_process_MTL_data['dataset_name']
	
	if pass_fail == 'fail':
		with pytest.raises(Exception):
			data = load_process_MTL(path='./data', dataset_name=dataset_name, print_mode='basic') 
	else:
		try:
			data = load_process_MTL(path='./data', dataset_name=dataset_name, print_mode='basic') 
			assert isinstance(data['train']['X_target'] , type(None))
			assert isinstance(data['val']['y'] , type(None))
			assert isinstance(data['val']['X_instance'] , type(None))
			assert isinstance(data['val']['X_target'] , type(None))
			assert isinstance(data['test']['X_target'] , type(None))
		except Exception as exc:
			assert False
	
