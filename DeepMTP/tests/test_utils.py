from DeepMTP.utils.utils import generate_config, get_optimization_direction
import pytest


def get_default_config_dict():
	return {
		'instance_branch_input_dim': 10,
		'target_branch_input_dim': 15,
		'validation_setting': 'B',
		'general_architecture_version': 'dot_product',
		'problem_mode': 'classification',
		'learning_rate': 0.001,
		'decay': 0,
		'batch_norm': False,
		'dropout_rate': 0,
		'momentum': 0.9,
		'weighted_loss': False,
		'compute_mode': 'cuda:7',
		'train_batchsize': 512,
		'val_batchsize': 512,
		'num_epochs': 50,
		'num_workers': 4,
		'metrics': ['hamming_loss', 'auroc'],
		'metrics_average': ['macro'],
		'patience': 10,

		'evaluate_train': True,
		'evaluate_val': True,

		'verbose': True,
		'results_verbose': False,
		'use_early_stopping': True,
		'use_tensorboard_logger': True,
		'wandb_project_name': 'Dummy_Project',
		'wandb_project_entity': 'username',
		'metric_to_optimize_early_stopping': 'loss',
		'metric_to_optimize_best_epoch_selection': 'loss',

		'instance_branch_architecture': 'MLP',
		'use_instance_features': False,

		'instance_branch_nodes_reducing_factor': 2,
		'instance_branch_nodes_per_layer': [123, 100],
		'instance_branch_layers': None,

		'target_branch_architecture': 'MLP',
		'use_target_features': False,

		'target_branch_nodes_reducing_factor': 2,
		'target_branch_nodes_per_layer': [132, 100],
		'target_branch_layers': None,

		
		'embedding_size': 30,
		'comb_mlp_nodes_reducing_factor': 2,
		'comb_mlp_nodes_per_layer': [2048, 2048, 2048],
		'comb_mlp_layers': None, 

		'save_model': True,

		'eval_every_n_epochs': 10,
		'delta': 0,
		'eval_instance_verbose': False,
		'eval_target_verbose': False,
		'return_results_per_target': False,
		'results_path': './results/',
		'experiment_name': None,
		'load_pretrained_model': False,
		'pretrained_model_path': '',
		'instance_train_transforms': None,
		'instance_inference_transforms': None,
		'target_train_transforms': None,
		'target_inference_transforms': None,
		'running_hpo': False,
		'hpo_results_path': './',
		'additional_info': {}
	}


get_optimization_direction_data = [('max', 'recall'), ('min', 'MSE')]

@pytest.mark.parametrize('get_optimization_direction_data', get_optimization_direction_data)    
def test_get_optimization_direction(get_optimization_direction_data):
	min_max = get_optimization_direction_data[0]
	metric_name = get_optimization_direction_data[1]
	assert get_optimization_direction(metric_name) == min_max


def test_generate_config():
	
	original_config = get_default_config_dict()
	
	config = generate_config(    
		instance_branch_input_dim = original_config['instance_branch_input_dim'],
		target_branch_input_dim = original_config['target_branch_input_dim'],
		validation_setting = original_config['validation_setting'],
		general_architecture_version = original_config['general_architecture_version'],
		problem_mode = original_config['problem_mode'],
		learning_rate = 0.001,
		decay = 0,
		batch_norm = False,
		dropout_rate = 0,
		momentum = 0.9,
		weighted_loss = False,
		compute_mode = 'cuda:7',
		train_batchsize = 512,
		val_batchsize = 512,
		num_epochs = 50,
		num_workers = 4,
		# metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
		# metrics_average = ['macro', 'micro'],
		metrics = original_config['metrics'],
		metrics_average = original_config['metrics_average'],
		patience = 10,

		evaluate_train = True,
		evaluate_val = True,

		verbose = True,
		results_verbose = False,
		use_early_stopping = True,
		use_tensorboard_logger = True,
		wandb_project_name = 'Dummy_Project',
		wandb_project_entity = 'username',
		metric_to_optimize_early_stopping = original_config['metric_to_optimize_early_stopping'],
		metric_to_optimize_best_epoch_selection = 'loss',

		instance_branch_architecture = original_config['instance_branch_architecture'],
		use_instance_features = False,
		instance_branch_params = {
			'instance_branch_nodes_reducing_factor': 2,
			'instance_branch_nodes_per_layer': [123, 100],
			'instance_branch_layers': None,
			# 'instance_branch_conv_architecture': 'resnet',
			# 'instance_branch_conv_architecture_version': 'resnet101',
			# 'instance_branch_conv_architecture_dense_layers': 1,
			# 'instance_branch_conv_architecture_last_layer_trained': 'last',
		},

		target_branch_architecture = original_config['target_branch_architecture'],
		use_target_features = False,
		target_branch_params = {
			'target_branch_nodes_reducing_factor': 2,
			'target_branch_nodes_per_layer': [132, 100],
			'target_branch_layers': None,
			# 'target_branch_conv_architecture': 'resnet',
			# 'target_branch_conv_architecture_version': 'resnet101',
			# 'target_branch_conv_architecture_dense_layers': 1,
			# 'target_branch_conv_architecture_last_layer_trained': 'last',
		},
		
		embedding_size = 30,
		comb_mlp_nodes_reducing_factor = 2,
		comb_mlp_nodes_per_layer = [2048, 2048, 2048],
		comb_mlp_layers = None, 

		save_model = True,

		eval_every_n_epochs = 10,

		additional_info = {}
	)

	for k,v in config.items():
		assert k in original_config
		assert v == original_config[k]


test_generate_config_fails_data = [
	{'pass_fail': 'fail', 'temp_config_data': {'metrics': ['aupr'], 'metric_to_optimize_early_stopping': 'auroc'}, 'data_to_be_added':{}},
	{'pass_fail': 'fail', 'temp_config_data': {'problem_mode': 'regression', 'metrics': ['recall']}, 'data_to_be_added':{}},
	{'pass_fail': 'fail', 'temp_config_data': {'problem_mode': 'classification', 'metrics': ['MSE']}, 'data_to_be_added':{}},
	{'pass_fail': 'pass', 'temp_config_data': {'problem_mode': 'regression', 'metrics': ['MSE']}, 'data_to_be_added':{}},
	{'pass_fail': 'fail', 'temp_config_data': {'metrics_average': ['mocro']}, 'data_to_be_added':{}},
	{'pass_fail': 'fail', 'temp_config_data': {'validation_setting': 'K'}, 'data_to_be_added':{}},
	{'pass_fail': 'pass', 'temp_config_data': {'validation_setting': 'A', 'metrics_average': ['micro']}, 'data_to_be_added':{}},
	{'pass_fail': 'pass', 'temp_config_data': {'validation_setting': 'A', 'metrics_average': ['macro']}, 'data_to_be_added':{'metrics_average': ['macro', 'micro']}},
	{'pass_fail': 'pass', 'temp_config_data': {'validation_setting': 'D', 'metrics_average': ['micro']}, 'data_to_be_added':{}},
	{'pass_fail': 'pass', 'temp_config_data': {'validation_setting': 'D', 'metrics_average': ['macro']}, 'data_to_be_added':{'metrics_average': ['macro', 'micro']}},
 	{'pass_fail': 'pass', 'temp_config_data': {'general_architecture_version': 'mlp'}, 'data_to_be_added':{}},
 	{'pass_fail': 'pass', 'temp_config_data': {'general_architecture_version': 'kronecker'}, 'data_to_be_added':{}},
 	{'pass_fail': 'pass', 'temp_config_data': {'instance_branch_architecture': 'CONV'}, 'data_to_be_added':{
																											'instance_branch_conv_architecture': 'resnet',
																											'instance_branch_conv_architecture_version': 'resnet101',
																											'instance_branch_conv_architecture_dense_layers': 1,
																											'instance_branch_conv_architecture_last_layer_trained': 'last',
																										},},
 	{'pass_fail': 'pass', 'temp_config_data': {'target_branch_architecture': 'CONV'}, 'data_to_be_added':{
																										  'target_branch_conv_architecture': 'resnet',
																										  'target_branch_conv_architecture_version': 'resnet101',
																										  'target_branch_conv_architecture_dense_layers': 1,
																										  'target_branch_conv_architecture_last_layer_trained': 'last',
																										},},
	{'pass_fail': 'fail', 'temp_config_data': {'instance_branch_architecture': None}, 'data_to_be_added':{}},
 	{'pass_fail': 'fail', 'temp_config_data': {'target_branch_architecture': None}, 'data_to_be_added':{}},
	{'pass_fail': 'fail', 'temp_config_data': {'instance_branch_architecture': 'lalala'}, 'data_to_be_added':{}},
 	{'pass_fail': 'fail', 'temp_config_data': {'target_branch_architecture': 'lalala'}, 'data_to_be_added':{}},

]

@pytest.mark.parametrize('test_generate_config_fails_data', test_generate_config_fails_data)    
def test_generate_config_fails(test_generate_config_fails_data):
	fail_pass = test_generate_config_fails_data['pass_fail']
 
	if fail_pass == 'fail':
		with pytest.raises(Exception):
			original_config = get_default_config_dict()
			original_config.update(test_generate_config_fails_data['temp_config_data'])
			config = generate_config(    
				instance_branch_input_dim = original_config['instance_branch_input_dim'],
				target_branch_input_dim = original_config['target_branch_input_dim'],
				validation_setting = original_config['validation_setting'],
				general_architecture_version = original_config['general_architecture_version'],
				problem_mode = original_config['problem_mode'],
				learning_rate = 0.001,
				decay = 0,
				batch_norm = False,
				dropout_rate = 0,
				momentum = 0.9,
				weighted_loss = False,
				compute_mode = 'cuda:7',
				train_batchsize = 512,
				val_batchsize = 512,
				num_epochs = 50,
				num_workers = 4,
				# metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
				# metrics_average = ['macro', 'micro'],
				metrics = original_config['metrics'],
				metrics_average = original_config['metrics_average'],
				patience = 10,

				evaluate_train = True,
				evaluate_val = True,

				verbose = True,
				results_verbose = False,
				use_early_stopping = True,
				use_tensorboard_logger = True,
				wandb_project_name = 'Dummy_Project',
				wandb_project_entity = 'username',
				metric_to_optimize_early_stopping = original_config['metric_to_optimize_early_stopping'],
				metric_to_optimize_best_epoch_selection = 'loss',

				instance_branch_architecture = original_config['instance_branch_architecture'],
				use_instance_features = False,
				instance_branch_params = {
					'instance_branch_nodes_reducing_factor': 2,
					'instance_branch_nodes_per_layer': [123, 100],
					'instance_branch_layers': None,
					# 'instance_branch_conv_architecture': 'resnet',
					# 'instance_branch_conv_architecture_version': 'resnet101',
					# 'instance_branch_conv_architecture_dense_layers': 1,
					# 'instance_branch_conv_architecture_last_layer_trained': 'last',
				},


				target_branch_architecture = original_config['target_branch_architecture'],
				use_target_features = False,
				target_branch_params = {
					'target_branch_nodes_reducing_factor': 2,
					'target_branch_nodes_per_layer': [132, 100],
					'target_branch_layers': None,
					# 'target_branch_conv_architecture': 'resnet',
					# 'target_branch_conv_architecture_version': 'resnet101',
					# 'target_branch_conv_architecture_dense_layers': 1,
					# 'target_branch_conv_architecture_last_layer_trained': 'last',
				},
				
				embedding_size = 30,
				comb_mlp_nodes_reducing_factor = 2,
				comb_mlp_nodes_per_layer = [2048, 2048, 2048],
				comb_mlp_layers = None, 

				save_model = True,

				eval_every_n_epochs = 10,

				additional_info = {}
			)

	else:
		try:
			original_config = get_default_config_dict()
			original_config.update(test_generate_config_fails_data['temp_config_data'])
			config = generate_config(    
				instance_branch_input_dim = original_config['instance_branch_input_dim'],
				target_branch_input_dim = original_config['target_branch_input_dim'],
				validation_setting = original_config['validation_setting'],
				general_architecture_version = original_config['general_architecture_version'],
				problem_mode = original_config['problem_mode'],
				learning_rate = 0.001,
				decay = 0,
				batch_norm = False,
				dropout_rate = 0,
				momentum = 0.9,
				weighted_loss = False,
				compute_mode = 'cuda:7',
				train_batchsize = 512,
				val_batchsize = 512,
				num_epochs = 50,
				num_workers = 4,
				# metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
				# metrics_average = ['macro', 'micro'],
				metrics = original_config['metrics'],
				metrics_average = original_config['metrics_average'],
				patience = 10,

				evaluate_train = True,
				evaluate_val = True,

				verbose = True,
				results_verbose = False,
				use_early_stopping = True,
				use_tensorboard_logger = True,
				wandb_project_name = 'Dummy_Project',
				wandb_project_entity = 'username',
				metric_to_optimize_early_stopping = original_config['metric_to_optimize_early_stopping'],
				metric_to_optimize_best_epoch_selection = 'loss',

				instance_branch_architecture = original_config['instance_branch_architecture'],
				use_instance_features = False,
				instance_branch_params = {
					'instance_branch_nodes_reducing_factor': 2,
					'instance_branch_nodes_per_layer': [123, 100],
					'instance_branch_layers': None,
					# 'instance_branch_conv_architecture': 'resnet',
					# 'instance_branch_conv_architecture_version': 'resnet101',
					# 'instance_branch_conv_architecture_dense_layers': 1,
					# 'instance_branch_conv_architecture_last_layer_trained': 'last',
				},


				target_branch_architecture = original_config['target_branch_architecture'],
				use_target_features = False,
				target_branch_params = {
					'target_branch_nodes_reducing_factor': 2,
					'target_branch_nodes_per_layer': [132, 100],
					'target_branch_layers': None,
					# 'target_branch_conv_architecture': 'resnet',
					# 'target_branch_conv_architecture_version': 'resnet101',
					# 'target_branch_conv_architecture_dense_layers': 1,
					# 'target_branch_conv_architecture_last_layer_trained': 'last',
				},
				
				embedding_size = 30,
				comb_mlp_nodes_reducing_factor = 2,
				comb_mlp_nodes_per_layer = [2048, 2048, 2048],
				comb_mlp_layers = None, 

				save_model = True,

				eval_every_n_epochs = 10,

				additional_info = {}
			)
			
			original_config.update(test_generate_config_fails_data['data_to_be_added'])
			
			for k,v in config.items():
				if True not in [kt in k for kt in ['train_transforms', 'inference_transforms']]:
					assert k in original_config
					assert v == original_config[k]
		except Exception as exc:
			assert False