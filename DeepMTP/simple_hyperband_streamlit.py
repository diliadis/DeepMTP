from pickletools import optimize
import random
import math
import sys
import sys
from typing import Optional
from DeepMTP.utils.utils import BaseExperimentInfo, get_optimization_direction

sys.path.insert(0, '../../../..')
import random
import streamlit as st

class HyperBand:
    '''Implements a basic version of the Hyperband HPO method. One cool thing about it is that I reduced the training time by continuing to train later configurations instead of starting from scratch each time.
    '''
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
            st.write('These are the pre-calculate brackets and successive halving runs:')
            st.write(str(self.budgets_per_bracket))

        self.best_experiments_per_bracket = {}
        self.experiment_history = {}
        # self.starting_configs_per_bracket = {num_configs:[BaseExperimentInfo(config=configspace.sample_configuration(), budget=d['r_i'][0]) for c in range(num_configs)] for num_configs, d in self.budgets_per_bracket.items()}

    def get_run_summary(self):
        return self.experiment_history

    def get_norm_val(self, val, min_val, max_val):
        return (val-min_val) / (max_val-min_val) * 100

    def run_optimizer(self):
        # iterate over the calculated brackets
        bracket_counter = 0
        bracket_info_update = st.empty()
        bracket_progress_bar = st.progress(0)
        iteration_info_update = st.empty()
        config_info_update = st.empty()
        iteration_progress_bar = st.progress(0)
        for bracket, d in self.budgets_per_bracket.items():
            iteration_counter = 0
            bracket_counter += 1
            # if self.verbose:
            bracket_info_update.write('-- Running bracket with starting budget: '+str(bracket)+'['+str(bracket_counter)+'/'+str(len(self.budgets_per_bracket))+']')
            self.experiment_history[bracket] = {}

            # you first start with as many randomly selected configurations as the current bracket defines
            self.configs_to_evaluate = [
                BaseExperimentInfo(
                    config=self.configspace.sample_configuration(), budget=d['r_i'][0]
                )
                for c in range(d['n_i'][0])
            ]
            # this is basically the successive halving routine
            for iteration in range(d['num_iters']):
                self.configs_to_evaluate = self.configs_to_evaluate[
                    : d['n_i'][iteration]
                ]
                # pass every configuration to the worker and store its returned score. The scores will be used to determine which configurations graduate to the next round of the successive halving subroutine
                for exp_idx, experiment in enumerate(self.configs_to_evaluate):
                    # time.sleep(5)
                    # if self.verbose:
                    iteration_info_update.write('---- Evaluating configuration: ['+str(exp_idx)+'/'+str(len(self.configs_to_evaluate))+']')
                    config_info_update.json(experiment.config.get_dictionary())
                    temp_result_dict = self.base_worker.compute(
                        d['r_i'][iteration], experiment.config
                    )
                    experiment.score = temp_result_dict['loss']
                    experiment.info = temp_result_dict['info']
                    # if self.verbose:
                    iteration_info_update.write(
                        '---- Finished evaluating configuration with score: '
                        + str(experiment.score)
                    )
                    experiment.budget = d['r_i'][iteration]

                self.configs_to_evaluate = sorted(
                    self.configs_to_evaluate,
                    key=lambda x: x.score,
                    reverse=False if self.direction == 'min' else True,
                )
                self.experiment_history[bracket][
                    iteration
                ] = self.configs_to_evaluate.copy()

                iteration_counter += 1
                iteration_progress_bar.progress(int(self.get_norm_val(iteration_counter, 0, d['num_iters'])))

            bracket_progress_bar.progress(int(self.get_norm_val(bracket_counter, 0, len(self.budgets_per_bracket))))
        bracket_progress_bar.progress(100)
        iteration_progress_bar.progress(100)
        bracket_info_update.empty()
        iteration_info_update.empty()
        config_info_update.empty()

        # get the best performing model from the "complete" runs
        if self.direction == 'min':
            best_overall_config = min([experiment for bracket_id, bracket in self.experiment_history.items() for experiment in bracket[max(list(bracket.keys()))]])
        else:
            best_overall_config = max([experiment for bracket_id, bracket in self.experiment_history.items() for experiment in bracket[max(list(bracket.keys()))]])
        best_overall_config.info['config']['experiment_name'] = 'best_model'

        return best_overall_config

    def calculate_hyperband_iters(self, R, eta, verbose=False):
        result_dict = {}
        smax = math.floor(math.log(R, eta))
        B = (smax + 1) * R
        if verbose:
            st.write('smax: ' + str(smax))
            st.write('B: ' + str(B))
            st.write('')
        for s in reversed((range(smax + 1))):
            # n = int(math.ceil(int((B/R) * ((hta**s)/(s+1)))))
            n = int(math.ceil(int(B / R / (s + 1)) * eta ** s))
            r = int(R * (eta ** (-s)))
            result_dict[n] = {'n_i': [], 'r_i': [], 'num_iters': s + 1}
            if verbose:
                st.write('s: ' + str(s))
                st.write('     n: ' + str(n) + '   r: ' + str(r))
                st.write('---------------------------')
            for i in range(s + 1):
                ni = math.floor(n * (eta ** (-i)))
                ri = r * (eta ** i)
                if verbose:
                    st.write('     ni: ' + str(ni) + '   ri (epochs): ' + str(ri))
                result_dict[n]['n_i'].append(ni)
                result_dict[n]['r_i'].append(ri)
            if verbose:
                st.write('')
                st.write('===========================')
        return result_dict
