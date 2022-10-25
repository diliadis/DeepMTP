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
        if not os.path.exists(base_config['hpo_results_path']):   # pragma: no cover
            os.mkdir(base_config['hpo_results_path'])

        self.project_name = base_config['hpo_results_path']+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')+'/'
        if not os.path.exists(self.project_name):   # pragma: no cover
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

        if mode not in ['standard', 'streamlit']:   # pragma: no cover
            raise Exception('Invalid mode value for the BaseWorker (select between standard and streamlit)')
        else:
            self.mode = mode

    def compute(self, budget, config):    # pragma: no cover
        '''The input parameter 'config' (dictionary) contains the sampled configurations passed by the bohb optimizer
        '''
        current_config = self.base_config.copy()
        temp_config = dict(config)
        original_budget = int(budget)
        current_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

        # isolate the instance and target specific parameters that are stored in specific instance_branch_params and target_branch_params nested directories
        instance_specific_param_keys = [p_name for p_name in temp_config.keys() if p_name.startswith('instance_') and p_name not in ['instance_branch_input_dim', 'instance_branch_architecture']]
        target_specific_param_keys = [p_name for p_name in temp_config.keys() if p_name.startswith('target_') and p_name not in ['target_branch_input_dim', 'target_branch_architecture']]

        current_config['instance_branch_params'] = {p_name: temp_config[p_name] for p_name in instance_specific_param_keys}
        current_config['target_branch_params'] = {p_name: temp_config[p_name] for p_name in target_specific_param_keys}
        current_config.update({p_name: temp_config[p_name] for p_name in temp_config.keys() if p_name not in instance_specific_param_keys+target_specific_param_keys})
        current_config['results_path'] = self.project_name
        current_config['experiment_name'] = current_time
        current_config = generate_config(**current_config)

        self.older_model_dir = None
        self.older_model_budget = None

        current_config.update(
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
                'config': current_config,
            }

        # update the actual budget that will be used to train the model
        current_config.update({'num_epochs': int(budget), 'actuall_budget': int(budget)})

        # initialize a new model or continue training from an older version with the same configuration
        if len(self.config_to_model[model_config_key]['model_dir']) != 0:
            if self.mode == 'standard':
                model = DeepMTP(config=current_config, checkpoint_dir=self.older_model_dir)
            else:
                model = DeepMTP_st(config=current_config, checkpoint_dir=self.older_model_dir)
        else:
            if self.mode == 'standard':
                model = DeepMTP(config=current_config)
            else:
                model = DeepMTP_st(config=current_config)

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
            'info': {'model_dir': self.config_to_model[model_config_key]['model_dir'][-1], 'config': current_config},
        }