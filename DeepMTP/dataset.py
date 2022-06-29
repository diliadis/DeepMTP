import os
import wget
import scipy
import requests, zipfile, hashlib, tarfile, io
import numpy as np
import pandas as pd
import arff
import pickle
import json
from prettytable import PrettyTable
from skmultilearn.dataset import load_dataset_dump, save_dataset_dump, available_data_sets, load_dataset
from sklearn.model_selection import train_test_split


def load_process_MLC(path='./data', dataset_name='bibtex', variant='undivided', features_type='numpy', print_mode='basic'):
	'''Load multi-label classfication datasets from the mulan repository.

	Args:
		path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
		dataset_name (str, optional): The name of the multi-label classification dataset. Defaults to 'bibtex'.
		variant (str, optional): The version of the dataset. If value set to 'undivided' the whole dataset is returned as a training set. Defaults to 'undivided'.
		features_type (str, optional): The format of the instance features. There are two possible values, numpy and dataframe. This is intended to test the functionality of the data_process function. Defaults to 'numpy'.
		print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.


	Raises:
		AttributeError: if the dataset_name doesn't exist in the mulan repository

	Returns:
		dict: A dictionary with all the available data for the multi-label classification dataset.
	'''
	# The current version just uses the datasets from the skmultilearn library. In the future I will be hosting everything from the university server
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	path += '/multi-label_classification'

	print(('info: ' if print_mode=='dev' else '')+'Processing...')
	available_datasets = set([x[0] for x in available_data_sets().keys()])

	if not os.path.exists(path):
		os.makedirs(path)

	if dataset_name not in available_datasets:
		raise AttributeError('Please use one of the valid dataset names: '+str(list(available_data_sets().keys())))
	else:
		if variant == 'undivided':
			X_train_instance, y_train, _, _ = load_dataset(dataset_name, variant, data_home=path)
		else:
			X_train_instance, y_train, _, _ = load_dataset(dataset_name, 'train')
			X_test_instance, y_test, _, _ = load_dataset(dataset_name, 'test')

	if scipy.sparse.issparse(X_train_instance):
		X_train_instance = X_train_instance.toarray()
	if scipy.sparse.issparse(X_test_instance):
		X_test_instance = X_test_instance.toarray()
	
	if features_type == 'dataframe':
		temp_df_train = pd.DataFrame(np.arange(len(X_train_instance)), columns=['id'])
		temp_df_train['features'] = [r for r in X_train_instance]
		X_train_instance = temp_df_train

		if X_test_instance is not None:
			temp_df_test = pd.DataFrame(np.arange(len(X_test_instance)), columns=['id'])
			temp_df_test['features'] = [r for r in X_test_instance]
			X_test_instance = temp_df_test

	if scipy.sparse.issparse(y_train):
		y_train = y_train.toarray()

	if scipy.sparse.issparse(y_test):
		y_test = y_test.toarray()

	print(('info: ' if print_mode=='dev' else '')+'Done')

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def process_dummy_MLC(num_features=10, num_instances=50, num_targets=5):
	'''Generates a dummy multi-label classification dataset

	Args:
		num_features (int, optional): The number of instance features. Defaults to 10.
		num_instances (int, optional): The number of instances. Defaults to 50.
		num_targets (int, optional): The number of targets. Defaults to 5.

	Returns:
		dict: A dictionary with all the available data for the multi-label classification dataset.
	'''
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	X_train_instance = np.random.random((num_instances, num_features))
	for i in range(num_instances):
		X_train_instance[i, 0] = i
	y_train = np.random.random((num_instances, num_targets)).astype(int)

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def load_process_MTR(path='./data', dataset_name='enb', features_type='numpy', print_mode='basic'):
	'''Load multivariate regression datasets from the mulan repository.

	Args:
		path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
		dataset_name (str, optional): The name of the multivariate regression dataset. Defaults to 'enb'.
		features_type (str, optional): The format of the instance features. There are two possible values, numpy and dataframe. This is intended to test the functionality of the data_process function. Defaults to 'numpy'.
		print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.


	Raises:
		AttributeError: if the dataset_name doesn't exist in the mulan repository

	Returns:
		dict: A dictionary with all the available data for the multivariate regression dataset.
	'''
	# The current version downloads all the datasets from a personal github repo
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 
	
	print(('info: ' if print_mode=='dev' else '')+'Processing...')

	'''
	+--------+------------+--------------------+----------+
	|  name  | #instances | #instance_features | #targets |
	+--------+------------+--------------------+----------+
	| atp1d  |    337     |        411         |    6     |
	| atp7d  |    296     |        411         |    6     |
	| oes97  |    334     |        263         |    16    |
	| oes10  |    403     |        298         |    16    |
	|  rf1   |    9125    |         64         |    8     |
	|  rf2   |    9125    |        576         |    8     |
	| scm1d  |    9803    |        280         |    16    |
	| scm20d |    8966    |         61         |    16    |
	|  edm   |    154     |         16         |    2     |
	|  sf1   |    323     |         10         |    3     |
	|  sf2   |    1066    |         10         |    3     |
	|  jura  |    359     |         15         |    3     |
	|   wq   |    1060    |         16         |    14    |
	|  enb   |    768     |         8          |    2     |
	| slump  |    103     |         7          |    3     |
	| andro  |     49     |         30         |    6     |
	| osales |    639     |        413         |    12    |
	|  scfp  |    1137    |         23         |    3     |
	+--------+------------+--------------------+----------+
	'''

	labels_per_dataset = {
		'atp1d': 6,
		'atp7d': 6,
		'oes97': 16,
		'oes10': 16,
		'rf1': 8,
		'rf2': 8,
		'scm1d': 16,
		'scm20d': 16,
		'edm': 2,
		'sf1': 3,
		'sf2': 3,
		'jura': 3,
		'wq': 14,
		'enb': 2,
		'slump': 3,
		'andro': 6,
		'osales': 12,
		'scpf': 3
	}

	available_data_sets = list(labels_per_dataset.keys())

	if dataset_name not in available_data_sets:
		raise AttributeError('Please use one of the valid dataset names: '+str(available_data_sets))

	if not os.path.exists(path):
		os.makedirs(path)
	
	if not os.path.exists(path+'/mtr-datasets'):
		print(('info: ' if print_mode=='dev' else '')+'Downloading and extracting dataset from scratch... ', end="")
		r = requests.get('https://github.com/diliadis/MTR/blob/main/mtr-datasets.zip?raw=true')
		z = zipfile.ZipFile(io.BytesIO(r.content))
		z.extractall(path)
		print(('info: ' if print_mode=='dev' else '')+'Done')

	dataset = arff.load(open(path+'/mtr-datasets/'+dataset_name+'.arff'))
	data = np.array(dataset['data'])

	X_train_instance = np.array(data[:, [i for i in range(0, data.shape[1]-labels_per_dataset[dataset_name])]])
	if features_type == 'dataframe':
		temp_df = pd.DataFrame(np.arange(len(X_train_instance)), columns=['id'])
		temp_df['features'] = [r for r in X_train_instance]
		X_train_instance = temp_df

	y_train = np.array(data[:, [i for i in range(data.shape[1]-labels_per_dataset[dataset_name], data.shape[1])]])

	print(('info: ' if print_mode=='dev' else '')+'Done')

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def process_dummy_MTR(num_features=10, num_instances=50, num_targets=5):
	'''Generates a dummy multivariate regression dataset

	Args:
		num_features (int, optional): The number of instance features. Defaults to 10.
		num_instances (int, optional): The number of instances. Defaults to 50.
		num_targets (int, optional): The number of targets. Defaults to 5.

	Returns:
		dict: A dictionary with all the available data for the multivariate regression dataset.
	'''
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	X_train_instance = np.random.random((num_instances, num_features))
	for i in range(num_instances):
		X_train_instance[i, 0] = i
	y_train = np.random.randint(10, size=(num_instances, num_targets))

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def print_MTR_datasets():
	'''Prints a table with the basic info for every multivariate regression dataset available in the Mulan repository
	'''
	table = PrettyTable(['name', '#instances', '#instance_features', '#targets'])
	table.add_row(['atp1d', '337', '411', '6'])
	table.add_row(['atp7d', '296', '411', '6'])
	table.add_row(['oes97', '334', '263', '16'])
	table.add_row(['oes10', '403', '298', '16'])
	table.add_row(['rf1', '9125', '64', '8'])
	table.add_row(['rf2', '9125', '576', '8'])
	table.add_row(['scm1d', '9803', '280', '16'])
	table.add_row(['scm20d', '8966', '61', '16'])
	table.add_row(['edm', '154', '16', '2'])
	table.add_row(['sf1', '323', '10', '3'])
	table.add_row(['sf2', '1066', '10', '3'])
	table.add_row(['jura', '359', '15', '3'])
	table.add_row(['wq', '1060', '16', '14'])
	table.add_row(['enb', '768', '8', '2'])
	table.add_row(['slump', '103', '7', '3'])
	table.add_row(['andro', '49', '30', '6'])
	table.add_row(['osales', '639', '413', '12'])
	table.add_row(['scpf', '1137', '23', '3'])
	print(table.get_string())

def load_process_DP(path='./data', dataset_name='ern', print_mode='basic'):
	'''Load dyadic prediction datasets from the following repository: https://people.montefiore.uliege.be.

	Args:
		path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
		dataset_name (str, optional): The name of the dyadic prediction dataset. Defaults to 'ern'.
		print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.


	Raises:
		AttributeError: if the dataset_name doesn't exist in the repository

	Returns:
		dict: A dictionary with all the available data for the dyadic prediction dataset.
	'''
	# The current version downloads all the datasets from https://people.montefiore.uliege.be
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 
	
	print(('info: ' if print_mode=='dev' else '')+'Processing...')

	available_data_sets = ['ern', 'srn', 'dpie', 'dpii']

	if dataset_name.startswith('dp'):
		url = 'https://people.montefiore.uliege.be/schrynemackers/dpix/'+dataset_name+'_'
	else:
		url = 'https://people.montefiore.uliege.be/schrynemackers/'+dataset_name+'/'

	if dataset_name not in available_data_sets:
		raise AttributeError('Please use one of the valid dataset names: '+str(available_data_sets))

	else:

		if not os.path.exists(path+'/dyadic_prediction-datasets/'+dataset_name):
			os.makedirs(path+'/dyadic_prediction-datasets/'+dataset_name)
			print(('info: ' if print_mode=='dev' else '')+'Downloading dataset from scratch... ', end="")
			wget.download(url+'Y.txt', path+'/dyadic_prediction-datasets/'+dataset_name+'/')
			wget.download(url+'X1.txt', path+'/dyadic_prediction-datasets/'+dataset_name+'/')
			wget.download(url+'X2.txt', path+'/dyadic_prediction-datasets/'+dataset_name+'/')
			print(('info: ' if print_mode=='dev' else '')+'Done')

		if (dataset_name == 'ern') or (dataset_name == 'srn'):
			y_train = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/Y.txt', delimiter=',')
			X_train_instance = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/X1.txt', delimiter=',')
			X_train_target = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/X2.txt', delimiter=',')
		else:
			y_train = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/'+dataset_name+'_Y.txt')
			X_train_instance = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/'+dataset_name+'_X1.txt')
			X_train_target = np.genfromtxt(path+'/dyadic_prediction-datasets/'+dataset_name+'/'+dataset_name+'_X2.txt')

	print(('info: ' if print_mode=='dev' else '')+'Done')

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def process_dummy_DP(num_instance_features=10, num_target_features=3, num_instances=50, num_targets=5):
	'''Generates a dummy multivariate regression dataset

	Args:
		num_instance_features (int, optional): The number of instance features. Defaults to 10.
		num_target_features (int, optional): The number of target features. Defaults to 3.
		num_instances (int, optional): The number of instances. Defaults to 50.
		num_targets (int, optional): The number of targets. Defaults to 5.
	Returns:
		dict: A dictionary with all the available data for the dyadic prediction dataset.
	'''
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	X_train_instance = np.random.random((num_instances, num_instance_features))
	for i in range(num_instances):
		X_train_instance[i, 0] = i

	X_train_target = np.random.random((num_targets, num_target_features))
	for i in range(num_targets):
		X_train_target[i, 0] = i

	y_train = np.random.randint(10, size=(num_instances, num_targets))

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def load_process_MC(path='./data', dataset_name='ml-100k', print_mode='basic'):
	'''Load matrix completion datasets from the Movielens repository.

	Args:
		path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
		dataset_name (str, optional): The name of the matrix completion dataset. Defaults to 'ml-100k'.
		print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

	Raises:
		AttributeError: if the dataset_name doesn't exist in the repository

	Returns:
		dict: A dictionary with all the available data for the matrix completion dataset.
	'''
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	path += '/matrix-completion-datasets'
	urls_checksums_per_dataset = {
		'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
		# 'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
		# 'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
	}
	
	if dataset_name not in urls_checksums_per_dataset:
		raise AttributeError('Please use one of the valid dataset names: '+str(list(urls_checksums_per_dataset.keys())))

	url = urls_checksums_per_dataset[dataset_name]

	os.makedirs(path, exist_ok=True)
	fname = os.path.join(path, url.split('/')[-1])
	# print('fname: '+str(fname))

	base_dir = os.path.dirname(fname)
	data_dir, ext = os.path.splitext(fname)
	# print('base_dir: '+str(base_dir))
	# print('data_dir: '+str(data_dir))
	# print('ext: '+str(ext))

	if not os.path.exists(data_dir):
		print(('info: ' if print_mode=='dev' else '')+'Downloading dataset '+dataset_name+' from scratch... ', end="")
		r = requests.get(url, stream=True, verify=True)
		with open(fname, 'wb') as f:
			f.write(r.content)
		print(('info: ' if print_mode=='dev' else '')+'Done')

	fp = zipfile.ZipFile(fname, 'r')
	fp.extractall(base_dir)

	names = ['instance_id', 'target_id', 'value', 'timestamp']
	y_train = pd.read_csv(filepath_or_buffer=os.path.join(data_dir, 'u.data'), sep='\t', names=names, engine='python')

	y_train.drop(['timestamp'], axis=1, inplace=True)
	unique_instance_ids = y_train['instance_id'].unique()
	unique_target_ids = y_train['target_id'].unique()

	old_to_new_instance_ids_map = {old_id: new_id for new_id, old_id in enumerate(unique_instance_ids)}
	old_to_new_target_ids_map = {old_id: new_id for new_id, old_id in enumerate(unique_target_ids)}

	y_train['instance_id'] = y_train['instance_id'].map(old_to_new_instance_ids_map)
	y_train['target_id'] = y_train['target_id'].map(old_to_new_target_ids_map)

	# return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}


def load_process_MTL(path='./data', dataset_name='dog', print_mode='basic'):
	'''Load multi-task learning datasets from my custom repository.

	Args:
		path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
		dataset_name (str, optional): The name of the multi-task learning dataset. Defaults to 'dog'.
		print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

	Raises:
		AttributeError: if the dataset_name doesn't exist in the repository

	Returns:
		dict: A dictionary with all the available data for the multi-task learning dataset.
	'''
	X_train_instance, X_val_instance, X_test_instance = None, None, None
	X_train_target, X_val_target, X_test_target = None, None, None
	y_train, y_val, y_test = None, None, None 

	if dataset_name not in ['dog', 'bird']:
		raise AttributeError('Please use one of the valid dataset names: '+str(['dog', 'bird']))

	path += '/multi-task_learning-datasets'

	os.makedirs(path, exist_ok=True)
	data_dir = path+'/'+dataset_name 
	base_dir = os.path.dirname(data_dir)
	# print('base_dir: '+str(base_dir))
	# print('data_dir: '+str(data_dir))

	if not os.path.exists(data_dir):
		print(('info: ' if print_mode=='dev' else '')+'Downloading dataset '+dataset_name+' from scratch... ', end="")
		r = requests.get('https://github.com/diliadis/MTL/blob/main/'+dataset_name+'.zip?raw=true') 
		z = zipfile.ZipFile(io.BytesIO(r.content))
		z.extractall(base_dir)
		print(('info: ' if print_mode=='dev' else '')+'Done')
		generate_interaction_matrix(data_dir+'/data.json', data_dir+'/y_dog.pkl')
	
	scores_matrix = pickle.load(open(data_dir+'/y_dog.pkl', 'rb'))
	instance_ids = np.array([data_dir+'/image/'+str(i+1)+'.jpg' for i in range(0, scores_matrix.shape[0])])
	train_ids, test_ids = train_test_split(np.arange(scores_matrix.shape[0]), test_size=0.25, shuffle=True, random_state=42)

	train_arr = [(i, j, scores_matrix[i, j]) for i in train_ids for j in range(scores_matrix.shape[1]) if not np.isnan(scores_matrix[i, j])]
	test_arr = [(i, j, scores_matrix[i, j]) for i in test_ids for j in range(scores_matrix.shape[1]) if not np.isnan(scores_matrix[i, j])]

	y_train = pd.DataFrame(train_arr, columns=['instance_id', 'target_id', 'value'])
	y_test = pd.DataFrame(test_arr, columns=['instance_id', 'target_id', 'value'])

	X_train_instance = pd.DataFrame([(i, instance_ids[i]) for i in train_ids], columns=['id', 'dir'])
	X_test_instance = pd.DataFrame([(i, instance_ids[i]) for i in test_ids], columns=['id', 'dir'])

	return {'train': {'y': y_train, 'X_instance': X_train_instance, 'X_target': X_train_target}, 'test': {'y': y_test, 'X_instance': X_test_instance, 'X_target': X_test_target}, 'val': {'y': y_val, 'X_instance': X_val_instance, 'X_target': X_val_target}}

def generate_interaction_matrix(input_path, output_path):
	'''Helper function that transforms the multi-task learning dataset into a basic interaction matrix

	Args:
		input_path (_type_): _description_
		output_path (_type_): _description_
	'''
	with open(input_path) as f:
		data = json.load(f)
		
	labels = data['true_labels']
	y = np.array(data['WorkerLabels']).astype('float')
	if "dog" in input_path:
		y = y.reshape(800, -1)
		
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			if y[i, j] != -1:
				if y[i, j] == labels[i]:
					y[i, j] = 1.0
				else:
					y[i, j] = 0.0
					
	y[y == -1] = 'nan'
	pickle.dump(y, open(output_path, 'wb'))