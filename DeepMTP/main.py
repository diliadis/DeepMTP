import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable
import os
from datetime import datetime
import json
import wandb

from DeepMTP.utils.data_utils import *
from DeepMTP.branch_models import *
from DeepMTP.utils.model_utils import EarlyStopping
from DeepMTP.utils.eval_utils import get_performance_results

class TwoBranchDotProductModel(nn.Sequential):
	'''Implements a two branch neural network that uses a dot product to combine the two embeddings.
	'''	
	def __init__(self, config, instance_branch_model, target_branch_model):
		super(TwoBranchDotProductModel, self).__init__()
		self.instance_branch_model = instance_branch_model
		self.target_branch_model = target_branch_model

	def forward(self, instance_features, target_features):
		instance_embedding = self.instance_branch_model(instance_features)
		target_embedding = self.target_branch_model(target_features)

		output = torch.unsqueeze((instance_embedding*target_embedding).sum(1), 1)
		return output

class TwoBranchMLPModel(nn.Sequential):
	'''Implements a two branch neural network that uses an MLP on top of the two branches. The two embedding vectors are just concatenated
	'''
	def __init__(self, config, instance_branch_model, target_branch_model):
		super(TwoBranchMLPModel, self).__init__()
		self.instance_branch_model = instance_branch_model
		self.target_branch_model = target_branch_model
		comb_dim = instance_branch_model[0][-2].out_features + target_branch_model[0][-2].out_features
		self.comb_branch = MLP(config, comb_dim, 1, config['comb_mlp_nodes_per_layer'], config['comb_mlp_branch_layers'], config['dropout_rate'], config['batch_norm'])

	def forward(self, instance_features, target_features):
		instance_embedding = self.instance_branch_model(instance_features)
		target_embedding = self.target_branch_model(target_features)
		# concatenate the two embedding vectors
		v_comb = torch.cat((instance_embedding, target_embedding), 1)
		output = self.comb_branch(v_comb)
		return output


class DeepMTP:
	''' Implements the training and inference logic of the DeepMTP framework. 
	'''
	def __init__(self, config, instance_branch_model=None, target_branch_model=None, checkpoint_dir=None):
		self.checkpoint_dict = None
		self.wandb_run = None
		self.tensorboard_logger = None

		# Load a checkpoint file, if it exists
		if checkpoint_dir is not None:
			if not os.path.isfile(checkpoint_dir):
				raise AttributeError('The file directory: '+checkpoint_dir+' does not exist!!!')
			else:
				if config['verbose']: print('Loading checkpoint from '+str(checkpoint_dir)+'... ', '')
				self.checkpoint_dict = torch.load(checkpoint_dir)
				if config['verbose']: print('Done')

			# when a checkpoint is loaded, it's config can (or sometimes has to) be modified
			self.config = self.checkpoint_dict['config']
			self.config.update(config)
		else:
			self.config = config

		# initialize the device that will be used of training-inference
		self.device = torch.device(self.config['compute_mode'] if (torch.cuda.is_available() and 'cuda' in self.config['compute_mode']) else 'cpu')
		if self.config['verbose']: print('Selected device: '+str(self.device))

		# setup the directory that will be used to save all relevant experiment info
		if not os.path.exists(self.config['results_path']):
			os.mkdir(self.config['results_path'])
		if self.config['experiment_name'] is None:
			self.experiment_dir = self.config['results_path']+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
		else:
			self.experiment_dir = self.config['results_path']+self.config['experiment_name']
		os.mkdir(self.experiment_dir)
		self.config['experiment_dir'] = self.experiment_dir

		'''
			Two main variants of the two-branch neural network are currently supported:
			1) The dot-product version: the two embedding vectors have the same size, so their dot-product is just computed.
			2) The MLP version: the two embedding vectors (don't have to be of the same size) are concatenated and then passed through a series or fully connected layers.
		'''
		if self.config['enable_dot_product_version']:
			instance_branch_output_dim = self.config['embedding_size']
			target_branch_output_dim = self.config['embedding_size']
		else:
			instance_branch_output_dim = self.config['instance_branch_nodes_per_layer'][-1] if isinstance(self.config['instance_branch_nodes_per_layer'], list) else self.config['instance_branch_nodes_per_layer']
			target_branch_output_dim = self.config['target_branch_nodes_per_layer'][-1] if isinstance(self.config['target_branch_nodes_per_layer'], list) else self.config['target_branch_nodes_per_layer']
		
		# define models that will be used in the two branches
		if instance_branch_model is None:
			if self.config['instance_branch_architecture'] == 'MLP':
				self.instance_branch_model = MLP(self.config, self.config['instance_branch_input_dim'], instance_branch_output_dim, self.config['instance_branch_nodes_per_layer'], self.config['instance_branch_layers'], self.config['dropout_rate'], self.config['batch_norm'])
			elif self.config['instance_branch_architecture'] == 'CONV':
				self.instance_branch_model = ConvNet(self.config, self.config['instance_branch_input_dim'], instance_branch_output_dim, self.config['instance_branch_conv_architecture'], self.config['instance_branch_conv_architecture_version'], self.config['instance_branch_conv_architecture_last_layer_trained'], self.config['instance_branch_conv_architecture_dense_layers'])
		else:
			# support for custom instance branch models
			self.instance_branch_model = instance_branch_model(self.config)

		if target_branch_model is None:
			if self.config['target_branch_architecture'] == 'MLP':
				self.target_branch_model = MLP(self.config, self.config['target_branch_input_dim'], target_branch_output_dim, self.config['target_branch_nodes_per_layer'], self.config['target_branch_layers'], self.config['dropout_rate'], self.config['batch_norm'])
			elif self.config['target_branch_architecture'] == 'CONV':
				self.target_branch_model = ConvNet(self.config, self.config['target_branch_input_dim'], target_branch_output_dim, self.config['target_branch_conv_architecture'], self.config['target_branch_conv_architecture_version'], self.config['target_branch_conv_architecture_last_layer_trained'], self.config['target_branch_conv_architecture_dense_layers'])
		else:
			# support for custom target branch models
			self.target_branch_model = target_branch_model(self.config)

		# initalize the two-branch neural network
		if self.config['enable_dot_product_version']:
			self.deepMTP_model = TwoBranchDotProductModel(self.config, self.instance_branch_model, self.target_branch_model)
		else:
			self.deepMTP_model = TwoBranchMLPModel(self.config, self.instance_branch_model, self.target_branch_model)

		self.deepMTP_model.to(self.device)

		# define the optimizer
		self.optimizer = torch.optim.Adam(
			self.deepMTP_model.parameters(),
			lr=self.config['learning_rate'],
			weight_decay=self.config['decay'],
		)
		
		if self.checkpoint_dict is not None: 
			# apply the loaded checkpoint
			if self.config['verbose']: print('Applying saved weights... ', end='')
			self.deepMTP_model.load_state_dict(self.checkpoint_dict['model_state_dict'])
			# update the device of the optimizer
			self.optimizer.load_state_dict(self.checkpoint_dict['optimizer_state_dict'])
			for state in self.optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.to(self.device)
			if self.config['verbose']: print('Done')
		
		# initialize the loss function that will be used. BCE for classification tasks and MSE for regression tasks.
		if self.config['problem_mode'] == 'classification':
			self.criterion = nn.BCELoss(reduction='mean')
			self.act = torch.nn.Sigmoid()
		elif self.config['problem_mode'] == 'regression':
			self.criterion = nn.MSELoss(reduction='mean')
		else:
			raise AttributeError('Invalid problem model: '+str(self.config['problem_mode']))

		'''
			the user can specify wether or not they want to use early stopping.
			In both cases we create the early stopping object so that we can track the best epoch, model etc. 
		'''
		self.early_stopping = EarlyStopping(
			use_early_stopping=self.config['use_early_stopping'],
			patience=self.config['patience'],
			verbose=self.config['verbose'],
			metric_to_track=self.config['metric_to_optimize_early_stopping'] if self.config['use_early_stopping'] else self.config['metric_to_optimize_best_epoch_selection']
		)

	def inference(self, model, dataloader, mode, epoch=0, return_predictions=False, verbose=False):
		model.eval()
		with torch.no_grad():
			loss_arr = []
			true_scores_arr = []
			pred_scores_arr = []
			instance_ids_arr = []
			target_ids_arr = []
			results = {}

			# iterate over batches
			for batch_id, batch in enumerate(dataloader):
				instance_features = batch['instance_features'].to(self.device)
				target_features = batch['target_features'].to(self.device)
				true_score = batch['score'].float().to(self.device)

				pred_score = self.deepMTP_model(instance_features, target_features)
				# for the binary targets, pass the network's output through a sigmoid
				if self.config['problem_mode'] == 'classification':
					pred_score = torch.squeeze(self.act(pred_score), 1)
				else:
					pred_score = torch.squeeze(pred_score, 1)

				if mode != 'test': # calculating the loss on the test set is pointless
					loss = self.criterion(pred_score, true_score)
					loss_arr.append(loss.item())

				if ((mode=='test') or ((epoch % self.config['eval_every_n_epochs'] == 0) and (self.config['evaluate_val'])) or (self.config['metric_to_optimize_early_stopping']!='loss') or (self.config['num_epochs']-1 == epoch)):
					true_scores_arr.extend(true_score.cpu().numpy())
					pred_scores_arr.extend(pred_score.detach().cpu().numpy())
					instance_ids_arr.extend(batch['instance_id'].numpy())
					target_ids_arr.extend(batch['target_id'].numpy())

			# calculate the performance
			if ((mode=='test') or ((epoch % self.config['eval_every_n_epochs'] == 0) and (self.config['evaluate_val'])) or (self.config['metric_to_optimize_early_stopping']!='loss') or (self.config['num_epochs']-1 == epoch)):
				if verbose: print('Calculating '+mode+' performance... ', end='')
				results = get_performance_results(
					mode, 
					epoch,
					instance_ids_arr, 
					target_ids_arr, 
					true_scores_arr, 
					pred_scores_arr, 
					self.config['validation_setting'],
					self.config['problem_mode'],
					self.config['metrics'],
					self.config['metrics_average'],
					verbose=self.config['results_verbose'],
					train_true_value=None,
					scaler_per_target=None,
				)
				if verbose: print('Done')
			
			if mode != 'test':
				results[mode+'_loss'] = np.mean(loss_arr)
			if return_predictions:
				return results, pd.DataFrame({'instance_id': instance_ids_arr, 'target_id': target_ids_arr, 'true_values': true_scores_arr, 'predicted_values': pred_scores_arr})
			else:
				return results

	def train(self, train_data, val_data, test_data, verbose=False):
		self.deepMTP_model.train()

		# train_dataset = BaseDataset(self.config, train_data['y'], train_data['X_instance'], train_data['X_target'])
		# test_dataset = BaseDataset(self.config, test_data['y'], test_data['X_instance'], test_data['X_target'])
		# val_dataset = BaseDataset(self.config, val_data['y'], val_data['X_instance'], val_data['X_target'])

		# initialize the dataloaders
		train_dataloader = DataLoader(BaseDataset(self.config, train_data['y'], train_data['X_instance'], train_data['X_target'], instance_transform=self.config['instance_train_transforms'], target_transform=self.config['target_train_transforms']), self.config['train_batchsize'], shuffle=True, num_workers=self.config['num_workers'])
		if val_data is not None:
			val_dataloader = DataLoader(BaseDataset(self.config, val_data['y'], val_data['X_instance'], val_data['X_target'], instance_transform=self.config['instance_inference_transforms'], target_transform=self.config['target_inference_transforms']), self.config['val_batchsize'], shuffle=False, num_workers=self.config['num_workers'])
		if test_data is not None:
			test_dataloader = DataLoader(BaseDataset(self.config, test_data['y'], test_data['X_instance'], test_data['X_target'], instance_transform=self.config['instance_inference_transforms'], target_transform=self.config['target_inference_transforms']), self.config['val_batchsize'], shuffle=False, num_workers=self.config['num_workers'])

		if self.config['verbose']: print('Starting training...')

		# initialize the tables used to show all the results
		run_story_header = ['mode', '#epoch', 'loss'] + [m+'_'+av for m in self.config['metrics'] for av in self.config['metrics_average']]
		if self.config['evaluate_train']:    
			train_run_story = []
			train_run_story_table = PrettyTable(run_story_header)
		val_run_story = []
		val_run_story_table = PrettyTable(run_story_header+['early_stopping'])
		test_run_story_table = PrettyTable(run_story_header)

		# update the possible data loggers. The currently supported loggers are wandb and tensorboard
		if self.config['use_tensorboard_logger']:
			self.tensorboard_logger = SummaryWriter(self.experiment_dir)
			for key in self.config.keys():
				self.tensorboard_logger.add_text(
					key, str(self.config[key])
				)
		if self.config['wandb_project_name'] is not None and self.config['wandb_project_entity'] is not None:
			self.wandb_run = wandb.init(project=self.config['wandb_project_name'], entity=self.config['wandb_project_entity'], reinit=True)
			self.wandb_run.watch(self.deepMTP_model)
			self.wandb_run.config.update(self.config)

		# iterate over epochs
		for epoch in range(self.config['num_epochs']):
			train_results = {}
			train_run_story = []
			loss_arr = []
			true_scores_arr = []
			pred_scores_arr = []
			instance_ids_arr = []
			target_ids_arr = []
			
			if self.config['verbose']: print('Epoch:'+str(epoch)+'... ', end='')
			# iterate over batches
			for batch_id, batch in enumerate(train_dataloader):
				instance_features = batch['instance_features'].to(self.device)
				target_features = batch['target_features'].to(self.device)
				true_score = batch['score'].float().to(self.device)

				pred_score = self.deepMTP_model(instance_features, target_features)
				# for the binary targets, pass the network's output through a sigmoid
				if self.config['problem_mode'] == 'classification':
					loss = self.criterion(torch.squeeze(self.act(pred_score), 1), true_score)
				else:
					loss = self.criterion(torch.squeeze(pred_score, 1), true_score)

				# back propagate
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				# keep train of train loss, as well as all other info needed to compute performance metrics(if the user specifies it )
				loss_arr.append(loss.item())
				if self.config['evaluate_train']:
					true_scores_arr.extend(true_score.cpu().numpy())
					pred_scores_arr.extend(pred_score.detach().cpu().numpy())
					instance_ids_arr.extend(batch['instance_id'].numpy())
					target_ids_arr.extend(batch['target_id'].numpy())

			# calculate performance metrics on the training set
			if ((self.config['evaluate_train']) and ((epoch % self.config['eval_every_n_epochs'] == 0) or (self.config['num_epochs']-1 == epoch ))):
				train_results = get_performance_results(
					'train', 
					epoch,
					instance_ids_arr, 
					target_ids_arr, 
					true_scores_arr, 
					pred_scores_arr, 
					self.config['validation_setting'],
					self.config['problem_mode'],
					self.config['metrics'],
					self.config['metrics_average'],
					verbose=self.config['results_verbose'],
					train_true_value=None,
					scaler_per_target=None,
				)

			mean_loss = np.mean(loss_arr)
			# update train_run_story_table
			train_run_story.append(['train', str(epoch), round(mean_loss, 4)] + [round(train_results['train_'+m+'_'+av], 4) if 'train_'+m+'_'+av in train_results else '-' for m in self.config['metrics'] for av in self.config['metrics_average']])
			# update the loggers
			results_to_log = {'train_'+m+'_'+av: train_results['train_'+m+'_'+av] for m in self.config['metrics'] for av in self.config['metrics_average'] if 'train_'+m+'_'+av in train_results}
			results_to_log['train_loss'] = mean_loss
			if self.wandb_run is not None:
				self.wandb_run.log(results_to_log, step=epoch)
			if self.tensorboard_logger is not None:
				for k, v in results_to_log.items():
					self.tensorboard_logger.add_scalar(
						k.replace('_', '/'), v, epoch
					)

			if self.config['verbose']: print('Done') 

			# generate predictions for the validation set. This needs to be done in order to early stop and to select between configurations
			if self.config['verbose']: print('  Validating... ', end='')
			val_results = self.inference(self.deepMTP_model, val_dataloader, 'val', epoch, verbose=self.config['verbose'])
			if self.config['verbose']: print('Done')

			# update val_run_story_table
			val_run_story.append(['val', str(epoch), round(val_results['val_loss'], 4)] + [round(val_results['val_'+m+'_'+av], 4) if 'val_'+m+'_'+av in val_results else '-' for m in self.config['metrics'] for av in self.config['metrics_average']])
			# update the loggers
			results_to_log = {'val_loss': val_results['val_loss']}
			results_to_log.update({'val_'+m+'_'+av: val_results['val_'+m+'_'+av] for m in self.config['metrics'] for av in self.config['metrics_average'] if 'val_'+m+'_'+av in val_results})
			if self.wandb_run is not None:
				self.wandb_run.log(results_to_log, step=epoch)
			if self.tensorboard_logger is not None:
				for k, v in results_to_log.items():
					self.tensorboard_logger.add_scalar(
						k.replace('_', '/'), v, epoch
					)

			# update early stopping and keep track of best model 
			self.early_stopping(
				val_results,
				copy.deepcopy(self.deepMTP_model),
				epoch,
				optimizer_state_dict=copy.deepcopy(self.optimizer.state_dict())
			)
			if self.early_stopping.early_stop_flag and self.config['use_early_stopping']:
				print('Early stopping criterion met. Training stopped!!!')
				break
			else:
				val_run_story[-1].append(str(self.early_stopping.counter)+'/'+str(self.early_stopping.patience))

			# update the result tables with the current epoch
			train_run_story_table.add_row(train_run_story[-1])
			val_run_story_table.add_row(val_run_story[-1])


		self.deepMTP_model = self.early_stopping.best_model        

		# log the performance validation results of the best model
		results_to_log = {'best_val_'+m+'_'+av: self.early_stopping.best_performance_results['val_'+m+'_'+av] for m in self.config['metrics'] for av in self.config['metrics_average'] if 'val_'+m+'_'+av in self.early_stopping.best_performance_results}
		if self.wandb_run is not None:
			self.wandb_run.log(results_to_log)
		if self.tensorboard_logger is not None:
			for k, v in results_to_log.items():
				self.tensorboard_logger.add_scalar(
					k.replace('_', '/'), v, epoch
				)
		# training is done (either completed all epochs or early stopping kicked in). Now testing starts using the best model
		if self.config['verbose']: print('Starting testing... ', end='')
		test_results = self.inference(self.deepMTP_model, test_dataloader, 'test', epoch, verbose=self.config['verbose'])
		if self.config['verbose']: print('Done')

		# update test_run_story_table
		test_run_story_table.add_row(['test', self.early_stopping.best_epoch, '-'] + [round(test_results['test_'+m+'_'+av], 4) for m in self.config['metrics'] for av in self.config['metrics_average']])
		# update the loggers
		results_to_log = {'test_'+m+'_'+av: test_results['test_'+m+'_'+av] for m in self.config['metrics'] for av in self.config['metrics_average']}
		if self.wandb_run is not None:
			self.wandb_run.log(results_to_log)
		if self.tensorboard_logger is not None:
			for k, v in results_to_log.items():
				self.tensorboard_logger.add_scalar(
					k.replace('_', '/'), v, epoch
				)

		# terminate loggers
		if self.wandb_run is not None:
			self.wandb_run.finish()
		if self.tensorboard_logger is not None:
			self.tensorboard_logger.flush()
			self.tensorboard_logger.close()

		# print and save the train-val-test summaries to a single .txt file
		if self.config['verbose']:
			if self.config['evaluate_train']: print(train_run_story_table.get_string())
			print(20*'=')
			if self.config['evaluate_val']: print(val_run_story_table.get_string())
			print(20*'=')
			print(test_run_story_table.get_string())

		with open(self.experiment_dir+'/summary.txt', 'w') as f:
			f.write(train_run_story_table.get_string())
			f.write('\n')
			f.write(val_run_story_table.get_string())
			f.write('\n')
			f.write(test_run_story_table.get_string())

		# save the best model 
		if self.config['save_model']:
			self.save_model(verbose=self.config['verbose'])
		
		# save the configuration in a json file
		if self.config['instance_train_transforms'] is not None and self.config['target_train_transforms'] is not None:
			with open(self.experiment_dir+'/config.json', 'w') as fp:
				json.dump(self.config, fp, indent=4)
		else:
			pickle.dump(self.config, open(self.experiment_dir+'/config.pkl', 'wb'))

		return self.early_stopping.best_performance_results

	def predict(self, data, return_predictions=False, verbose=False):
		self.deepMTP_model.to(self.device)
		dataloader = DataLoader(BaseDataset(self.config, data['y'], data['X_instance'], data['X_target']), self.config['val_batchsize'], shuffle=False, num_workers=self.config['num_workers'])
		return self.inference(self.deepMTP_model, dataloader, '', 0, return_predictions=True, verbose=verbose)

	def save_model(self, verbose=False):
		self.config['use_tensorboard_logger'] = False
		if verbose: print('Saving the best model... ', end='')
		torch.save({
			'model_state_dict': self.early_stopping.best_model.state_dict(),
			'optimizer_state_dict': self.early_stopping.best_optimizer_state_dict,
			# 'optimizer_state_dict': self.optimizer.state_dict(),
			'config': self.config
			}, self.experiment_dir+'/model.pt')
		if verbose: print('Done')

def initialize_mode(config):
	return DeepMTP(config)