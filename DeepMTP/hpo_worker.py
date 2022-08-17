from pickletools import optimize
import sys
import sys
sys.path.insert(0, '../../../..')
import os
from datetime import datetime
from DeepMTP.main import DeepMTP
from DeepMTP.main_streamlit import DeepMTP as DeepMTP_st
from DeepMTP.utils.utils import generate_config


class BaseWorker:
    ''' Implements a basic worker that can be used by HPO methods. The basic idea is that an HPO methods just has to pass a config and then it gets back the performance of the best epoch on the validation set
    '''    
    def __init__(
        self, train, val, test, data_info, base_config, metric_to_optimize, mode='standard'
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

        if mode not in ['standard', 'streamlit']:
            raise Exception('Invalid mode value for the BaseWorker (select between standard and streamlit)')
        else:
            self.mode = mode

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
            batch_norm = temp_config['batch_norm'] if 'batch_norm' in temp_config else False,
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
            target_branch_architecture = self.base_config['target_branch_architecture'],

            embedding_size = temp_config['embedding_size'],

            save_model = self.base_config['save_model'],

            eval_every_n_epochs = self.base_config['eval_every_n_epochs'],
            running_hpo = self.base_config['running_hpo'],
            additional_info = self.base_config['additional_info'],

            instance_branch_params = {p_name: p_val for p_name, p_val in temp_config.items() if p_name.startswith('instance_') and p_name not in ['instance_branch_input_dim', 'instance_branch_architecture']},
            target_branch_params = {p_name: p_val for p_name, p_val in temp_config.items() if p_name.startswith('target_') and p_name not in ['target_branch_input_dim', 'target_branch_architecture']},
            )

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
            if self.mode == 'standard':
                model = DeepMTP(config=self.master_config, checkpoint_dir=self.older_model_dir)
            else:
                model = DeepMTP_st(config=self.master_config, checkpoint_dir=self.older_model_dir)
        else:
            if self.mode == 'standard':
                model = DeepMTP(config=self.master_config)
            else:
                model = DeepMTP_st(config=self.master_config)

        # train, validate and test all at once
        val_results = model.train(self.train, self.val, self.test)
        print('val_results: '+str(val_results))

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
            'info': {'model_dir': self.config_to_model[model_config_key]['model_dir'][-1], 'config': self.master_config},
        }