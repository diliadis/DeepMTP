from pickletools import optimize
import random
import math
import sys
from typing import Optional
from DeepMTP.utils.utils import BaseExperimentInfo, get_optimization_direction

sys.path.insert(0, '../../../..')
import random

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
                    temp_result_dict = self.base_worker.compute(
                        d['r_i'][iteration], experiment.config
                    )
                    experiment.score = temp_result_dict['loss']
                    experiment.info = temp_result_dict['info']
                    if self.verbose:
                        print(
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
        
        # get the best performing model from the "complete" runs
        if self.direction == 'min':
            best_overall_config = min([experiment for bracket_id, bracket in self.experiment_history.items() for experiment in bracket[max(list(bracket.keys()))]])
        else:
            best_overall_config = max([experiment for bracket_id, bracket in self.experiment_history.items() for experiment in bracket[max(list(bracket.keys()))]])
        best_overall_config.info['config']['experiment_name'] = 'best_model'

        if self.verbose:
            print('Best overall configuration: ')
            print(best_overall_config)
        return best_overall_config

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
