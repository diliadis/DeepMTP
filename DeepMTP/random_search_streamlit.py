import sys
sys.path.insert(0, '../../../..')
from DeepMTP.utils.utils import BaseExperimentInfo, get_optimization_direction
import streamlit as st

class RandomSearch:
    '''Implements the basic Random search HPO method. Nothing fancy, just a for loop over randomly generated configurations.
    '''
    def __init__(
        self,
        base_worker,
        configspace,
        budget=1,
        max_num_epochs=100,
        direction='min',
        verbose=False,
    ):
        self.base_worker = base_worker
        self.configspace = configspace
        self.verbose = verbose
        self.direction = direction
        self.budget = budget
        self.max_num_epochs = max_num_epochs
        self.experiment_history = {}

    def get_run_summary(self):
        return self.experiment_history

    def get_norm_val(self, val, min_val, max_val):
        return (val-min_val) / (max_val-min_val) * 100

    def run_optimizer(self):
        random_search_iter_info_update = st.empty()
        random_search_iter_progress_bar = st.progress(0)
        config_info_update = st.empty()

        self.configs_to_evaluate = [BaseExperimentInfo(config=self.configspace.sample_configuration(), budget=self.max_num_epochs) for c in range(self.budget)]
        for exp_counter, experiment in enumerate(self.configs_to_evaluate):
            if self.verbose:
                random_search_iter_info_update.write('---- Evaluating configuration: ['+str(exp_counter)+'/'+str(len(self.configs_to_evaluate))+']')
                config_info_update.json(experiment.config.get_dictionary())
            temp_result_dict = self.base_worker.compute(
                self.max_num_epochs, experiment.config
            )    
            experiment.score = temp_result_dict['loss']
            experiment.info = temp_result_dict['info']
            if self.verbose:
                random_search_iter_info_update.write(
                    '---- Finished evaluating configuration with score: '
                    + str(experiment.score)
                )
            random_search_iter_progress_bar.progress(int(self.get_norm_val(exp_counter, 0, self.budget)))
        random_search_iter_progress_bar.progress(100)
        random_search_iter_info_update.empty()
        config_info_update.empty()
        
        self.configs_to_evaluate = sorted(
            self.configs_to_evaluate,
            key=lambda x: x.score,
            reverse=False if self.direction == 'min' else True,
        )
        best_overall_config = self.configs_to_evaluate[0]
        best_overall_config.info['config']['experiment_name'] = 'best_model'

        return best_overall_config