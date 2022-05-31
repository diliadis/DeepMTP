from pickletools import optimize
import random
import math
import sys
import sys
from typing import Optional

sys.path.insert(0, '../../../..')
import pickle
import random
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
import pprint

from DeepMTP.main import DeepMTP
from DeepMTP.utils.utils import generate_config

def get_optimization_direction(metric_name):
    metrics_to_max = ['sensitivity', 'f1_score', 'recall', 'positive_predictive_value']
    if True in [n in metric_name for n in metrics_to_max]:
        return 'max'
    return 'min'

class BaseWorker:
    def __init__(
        self, train, val, test, data_info, base_config, metric_to_optimize
    ):
        if not os.path.exists(base_config['hpo_results_path']):
            os.mkdir(base_config['hpo_results_path'])

        self.project_name = base_config['hpo_results_path']+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')+'/'
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)

        self.config_to_model = {}
        self.older_model_dir = None
        self.current_model_dir = None
        self.older_model_budget = None
        self.older_model = None
        self.optimize = metric_to_optimize

        self.train = train
        self.val = val
        self.test = test

        self.base_config = base_config

    def compute(self, budget, config):

        '''
        The input parameter 'config' (dictionary) contains the sampled configurations passed by the bohb optimizer
        '''
        temp_config = dict(config)
        original_budget = int(budget)
        current_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

        self.master_config = generate_config(    
            instance_branch_input_dim = self.base_config['instance_branch_input_dim'],
            target_branch_input_dim = self.base_config['target_branch_input_dim'],
            validation_setting = self.base_config['validation_setting'],
            enable_dot_product_version = self.base_config['enable_dot_product_version'],
            problem_mode = self.base_config['problem_mode'],
            learning_rate = temp_config['learning_rate'],
            decay = 0,
            batch_norm = temp_config['batch_norm'] if 'batch_norm' in temp_config else 0,
            dropout_rate = temp_config['dropout_rate'] if 'dropout_rate' in temp_config else 0,
            momentum = 0.9,
            weighted_loss = False,
            compute_mode = self.base_config['compute_mode'],
            train_batchsize = self.base_config['train_batchsize'],
            val_batchsize = self.base_config['val_batchsize'],
            num_epochs = original_budget,
            num_workers = self.base_config['num_workers'],

            metrics = self.base_config['metrics'],
            metrics_average = self.base_config['metrics_average'],
            patience = self.base_config['patience'],

            evaluate_train = self.base_config['evaluate_train'],
            evaluate_val = self.base_config['evaluate_val'],

            verbose = self.base_config['verbose'],
            results_verbose = self.base_config['results_verbose'],
            use_early_stopping = self.base_config['use_early_stopping'],
            use_tensorboard_logger = self.base_config['use_tensorboard_logger'],
            wandb_project_name = self.base_config['wandb_project_name'],
            wandb_project_entity = self.base_config['wandb_project_entity'],
            results_path = self.project_name,
            experiment_name = current_time,
            metric_to_optimize_early_stopping = self.base_config['metric_to_optimize_early_stopping'],
            metric_to_optimize_best_epoch_selection = self.base_config['metric_to_optimize_best_epoch_selection'],

            instance_branch_architecture = self.base_config['instance_branch_architecture'],
            instance_branch_nodes_per_layer = temp_config['instance_branch_nodes_per_layer'],
            instance_branch_layers = temp_config['instance_branch_layers'],

            target_branch_architecture = self.base_config['target_branch_architecture'],
            target_branch_nodes_per_layer = temp_config['target_branch_nodes_per_layer'],
            target_branch_layers = temp_config['target_branch_layers'],

            embedding_size = temp_config['embedding_size'],

            save_model = self.base_config['save_model'],

            eval_every_n_epochs = self.base_config['eval_every_n_epochs'],

            additional_info = self.base_config['additional_info'])

        self.older_model_dir = None
        self.older_model_budget = None

        self.master_config.update(
            {'budget': budget, 'budget_int': int(budget)}
        )

        # create a key from the given configuration
        model_config_key = tuple(sorted(temp_config.items()))
        # check if the configuration has already been seeen. If so extract relevant info from the last experiment with that configuration
        if model_config_key in self.config_to_model:
            self.older_model_dir = self.config_to_model[model_config_key]['model_dir'][-1]
            self.older_model_budget = self.config_to_model[model_config_key]['budget'][-1]
            budget = budget - self.older_model_budget
        else:
            self.config_to_model[model_config_key] = {
                'budget': [],
                'model_dir': [],
                'run_name': [],
                'config': self.master_config,
            }

        # update the actual budget that will be used to train the model
        self.master_config.update({'num_epochs': int(budget), 'actuall_budget': int(budget)})

        # initialize a new model or continue training from an older version with the same configuration
        if len(self.config_to_model[model_config_key]['model_dir']) != 0:
            model = DeepMTP(self.master_config, self.older_model_dir)
        else:
            model = DeepMTP(self.master_config)

        # train, validate and test all at once
        val_results = model.train(self.train, self.val, self.test)

        # append all the latest relevant info for the given configuration
        self.config_to_model[model_config_key]['budget'].append(original_budget) 
        self.config_to_model[model_config_key]['model_dir'].append(self.project_name+'/'+current_time+'/model.pt')
        self.config_to_model[model_config_key]['run_name'].append(current_time)

        # output_file = open(self.project_name+'/'+current_time+ '/run_results.pkl', 'wb')
        # pickle.dump(self.config_to_model, output_file)
        # output_file.close()

        # output_file = open('hyperopt/older_models/'+wandb_run_project_name+'/run_results.json', 'w')
        # json.dump(self.config_to_model, output_file)
        # output_file.close()

        return {
            'loss': val_results['val_'+self.optimize if 'val' not in self.optimize else self.optimize],  # remember: always minimizes!
            'info': {},
        }


class BaseExperimentInfo:
    def __init__(self, config, budget):
        self.config = config
        self.score = 0
        self.budget = budget
        self.info = {}

    def update_score(self, score):
        self.score = score

    def get_config(self):
        return self.config

    def get_budget(self):
        return self.budget

    def __repr__(self):
        return (
            'config: '
            + str(self.config)
            + '  |  budget: '
            + str(self.budget)
            + '  |  score: '
            + str(self.score)
            + '\n \n'
        )

class HyperBand:
    def __init__(
        self,
        base_worker,
        configspace,
        eta=3,
        max_budget=1,
        direction='min',
        verbose=False,
    ):
        self.base_worker = base_worker
        self.configspace = configspace
        self.verbose = verbose
        self.direction = direction

        # Hyperband related stuff
        self.eta = eta
        self.max_budget = max_budget

        # calculate the hyperband run using the max_budget the eta parameters that are provided by the user
        self.budgets_per_bracket = self.calculate_hyperband_iters(
            R=self.max_budget, eta=eta, verbose=False
        )
        if self.verbose:
            print('These are the pre-calculate brackets and successive halving runs:')
            print(str(self.budgets_per_bracket))

        self.best_experiments_per_bracket = {}
        self.experiment_history = {}
        # self.starting_configs_per_bracket = {num_configs:[BaseExperimentInfo(config=configspace.sample_configuration(), budget=d['r_i'][0]) for c in range(num_configs)] for num_configs, d in self.budgets_per_bracket.items()}

    def get_run_summary(self):
        return self.experiment_history

    def run_optimizer(self):

        # iterate over the calculated brackets
        for bracket, d in self.budgets_per_bracket.items():
            if self.verbose:
                print('-- Running bracket with starting budget: ' + str(bracket))
            self.experiment_history[bracket] = {}

            # you first start with as many randomly selected configurations as the current bracket defines
            self.configs_to_evaluate = [
                BaseExperimentInfo(
                    config=self.configspace.sample_configuration(), budget=d['r_i'][0]
                )
                for c in range(d['n_i'][0])
            ]
            # print('original configs list: '+str(self.configs_to_evaluate))

            # this is basically the successive halving routine
            for iteration in range(d['num_iters']):
                self.configs_to_evaluate = self.configs_to_evaluate[
                    : d['n_i'][iteration]
                ]

                # pass every configuration to the worker and store its returned score. The scores will be used to determine which configurations graduate to the next round of the successive halving subroutine
                for exp_idx, experiment in enumerate(self.configs_to_evaluate):
                    # time.sleep(5)
                    if self.verbose:
                        print('---- Evaluating configuration... ')
                    experiment.score, experiment.info = self.base_worker.compute(
                        d['r_i'][iteration], experiment.config
                    )
                    if self.verbose:
                        print(
                            '---- Finished evaluating configuration with score: '
                            + str(experiment.score)
                        )
                    experiment.budget = d['r_i'][iteration]

                # print('Selecting the '+str(d['n_i'][iteration])+' best performing configs')

                self.configs_to_evaluate = sorted(
                    self.configs_to_evaluate,
                    key=lambda x: x.score,
                    reverse=False if self.direction == 'min' else True,
                )
                self.experiment_history[bracket][
                    iteration
                ] = self.configs_to_evaluate.copy()
                # print('evaluated configs list: '+str(self.configs_to_evaluate))
                # print('=========== Finished with bracket: '+str(bracket)+' ===========')

    def calculate_hyperband_iters(self, R, eta, verbose=False):

        result_dict = {}

        smax = math.floor(math.log(R, eta))
        B = (smax + 1) * R
        if verbose:
            print('smax: ' + str(smax))
            print('B: ' + str(B))
            print('')
        for s in reversed((range(smax + 1))):

            # n = int(math.ceil(int((B/R) * ((hta**s)/(s+1)))))
            n = int(math.ceil(int(B / R / (s + 1)) * eta ** s))
            r = int(R * (eta ** (-s)))
            result_dict[n] = {'n_i': [], 'r_i': [], 'num_iters': s + 1}

            if verbose:
                print('s: ' + str(s))
                print('     n: ' + str(n) + '   r: ' + str(r))
                print('---------------------------')
            for i in range(s + 1):
                ni = math.floor(n * (eta ** (-i)))
                ri = r * (eta ** i)
                if verbose:
                    print('     ni: ' + str(ni) + '   ri (epochs): ' + str(ri))
                result_dict[n]['n_i'].append(ni)
                result_dict[n]['r_i'].append(ri)
            if verbose:
                print('')
                print('===========================')
        return result_dict
